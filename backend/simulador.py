"""Motor de simulação mensal do balanço hídrico.

Este módulo não acessa o banco de dados nem a API: recebe séries e parâmetros já
preparados e devolve, para cada reservatório, os registros mensais e os
indicadores de desempenho. A conversão de vazão para volume usa o mês
convencional de 30 dias (2,592 hm³ por m³/s).
"""
import unicodedata

import numpy as np
from numba import njit

from optimizer_engine import dinamica_mensal_fast

SEGUNDOS_MES_PADRAO = 2_592_000.0
HM3_POR_M3S = SEGUNDOS_MES_PADRAO / 1e6

MESES = ("JAN", "FEV", "MAR", "ABR", "MAI", "JUN", "JUL", "AGO", "SET", "OUT", "NOV", "DEZ")
ORDEM_MESES = {mes: i + 1 for i, mes in enumerate(MESES)}

# ---------------------------------------------------------------------------
# Configuração específica do hidrossistema Fogareiro–Quixeramobim (PGPS, cenário 1)
# ---------------------------------------------------------------------------
FOGAREIRO_QUIXERAMOBIM_CENARIO_1_ID = "pgps_fogareiro_quixeramobim_cenario_1"
FOGAREIRO_QUIXERAMOBIM_CENARIO_1 = {
    "id": FOGAREIRO_QUIXERAMOBIM_CENARIO_1_ID,
    "controlador_cod": "119",
    "receptor_cod": "16",
    "gatilho_receptor_percent": 30.0,
    "limites_percent": {
        "Alerta": [49.152542, 46.610169, 45.762712, 50.000000, 67.796610, 66.101695, 64.406780, 62.711864, 60.169492, 57.627119, 54.237288, 51.694915],
        "Seca": [39.830508, 38.135593, 37.288136, 41.525424, 57.627119, 55.084746, 53.389831, 51.694915, 50.000000, 47.457627, 44.915254, 42.372881],
        "Seca Severa": [27.966102, 26.271186, 25.423729, 30.508475, 41.525424, 39.830508, 38.135593, 37.288136, 35.593220, 33.898305, 31.355932, 29.661017],
    },
    # Retiradas locais. A Tabela 5.3 do PGPS soma a retirada do Fogareiro com a
    # transferência: 272 + 500 = 772 L/s no estado Normal, por exemplo.
    "demandas_lps": {
        "119": {"Normal": 272.0, "Alerta": 220.0, "Seca": 139.6, "Seca Severa": 6.0},
        "16": {"Normal": 342.0, "Alerta": 301.8, "Seca": 213.3, "Seca Severa": 70.5},
    },
    "transferencias_lps": {
        "Normal": 500.0,
        "Alerta": 400.0,
        "Seca": 300.0,
        "Seca Severa": 85.0,
    },
}


def faixas_fogareiro_quixeramobim_percentuais():
    demandas = FOGAREIRO_QUIXERAMOBIM_CENARIO_1["demandas_lps"]["119"]
    normal = demandas["Normal"]
    faixas = []
    for nome in ("Seca Severa", "Seca", "Alerta"):
        linha = {
            "Faixa": nome,
            "Racionamento": round((1.0 - demandas[nome] / normal) * 100.0, 2),
        }
        linha.update({
            mes: round(valor, 2)
            for mes, valor in zip(MESES, FOGAREIRO_QUIXERAMOBIM_CENARIO_1["limites_percent"][nome])
        })
        faixas.append(linha)
    faixas.append({"Faixa": "Normal", "Racionamento": 0.0, **{mes: 100.0 for mes in MESES}})
    return faixas


def estado_fogareiro_quixeramobim(volume_hm3: float, capacidade_hm3: float, mes: int) -> str:
    limites = FOGAREIRO_QUIXERAMOBIM_CENARIO_1["limites_percent"]
    indice_mes = max(1, min(int(mes), 12)) - 1
    volume_percent = volume_hm3 / capacidade_hm3 * 100.0 if capacidade_hm3 > 0 else 0.0
    tolerancia_percent = 1e-6
    if volume_percent <= limites["Seca Severa"][indice_mes] + tolerancia_percent:
        return "Seca Severa"
    if volume_percent <= limites["Seca"][indice_mes] + tolerancia_percent:
        return "Seca"
    if volume_percent <= limites["Alerta"][indice_mes] + tolerancia_percent:
        return "Alerta"
    return "Normal"


def normalizar_modo_simulacao(modo: str) -> str:
    texto = unicodedata.normalize("NFKD", str(modo or "")).encode("ascii", "ignore").decode("ascii").lower()
    if "serie" in texto:
        return "serie"
    if "paral" in texto:
        return "paralelo"
    return "individual"


# ---------------------------------------------------------------------------
# Simulação
# ---------------------------------------------------------------------------
def _abaixo_do_gatilho(volume, vol_gatilho, vol_desligamento, ativo_anterior):
    """Regra de gatilho com histerese.

    Sem histerese (vol_desligamento == vol_gatilho), a ação fica ativa sempre que
    o volume está abaixo do gatilho. Com histerese, uma ação já ativa só é
    encerrada quando o volume alcança o limite de desligamento, maior que o gatilho.
    """
    limite = vol_desligamento if ativo_anterior else vol_gatilho
    return volume < limite


def simular_sistema_n(
    series,
    params,
    modo,
    vazao_conjunta,
    atendimento_transferencia=100.0,
    cenario_hidrossistema=None,
    histerese_percent=0.0,
):
    """Simula mês a mês todos os reservatórios.

    histerese_percent: pontos percentuais da capacidade acima do gatilho que o
    volume precisa alcançar para encerrar uma transferência (modo Série) ou para
    devolver a demanda conjunta à unidade anterior (modo Paralelo). Com 0, o
    comportamento é o da regra simples de gatilho.
    """
    n_res = len(series)
    if n_res == 0:
        return []

    n_meses = len(series[0]["datas"])
    modo_id = normalizar_modo_simulacao(modo)
    vazao_conjunta = float(vazao_conjunta)
    histerese_percent = max(0.0, float(histerese_percent or 0.0))
    atendimento_transferencia = max(0.0, min(float(atendimento_transferencia), 100.0)) / 100.0
    cenario_fq_ativo = cenario_hidrossistema == FOGAREIRO_QUIXERAMOBIM_CENARIO_1_ID and modo_id == "serie"
    indices_por_codigo = {str(param.get("cod", "")): indice for indice, param in enumerate(params)}
    indice_controlador_fq = indices_por_codigo.get(FOGAREIRO_QUIXERAMOBIM_CENARIO_1["controlador_cod"])
    indice_receptor_fq = indices_por_codigo.get(FOGAREIRO_QUIXERAMOBIM_CENARIO_1["receptor_cod"])
    if cenario_fq_ativo and (indice_controlador_fq is None or indice_receptor_fq is None):
        raise ValueError("O cenário PGPS requer os reservatórios Fogareiro e Quixeramobim.")

    for serie in series[1:]:
        if len(serie["datas"]) != n_meses or serie["datas"] != series[0]["datas"]:
            raise ValueError("As séries dos reservatórios precisam cobrir os mesmos meses.")

    saidas = []
    for _ in range(n_res):
        saidas.append({
            "armazenamento_inicial": np.zeros(n_meses, dtype=float),
            "armazenamento_final": np.zeros(n_meses, dtype=float),
            "demanda_solicitada": np.zeros(n_meses, dtype=float),
            "demanda_aplicada": np.zeros(n_meses, dtype=float),
            "demanda_atendida": np.zeros(n_meses, dtype=float),
            "retirada_total": np.zeros(n_meses, dtype=float),
            "racionamento": np.zeros(n_meses, dtype=float),
            "transferencia_recebida": np.zeros(n_meses, dtype=float),
            "transferencia_enviada": np.zeros(n_meses, dtype=float),
            "evaporacao_hm3": np.zeros(n_meses, dtype=float),
            "vertimento_hm3": np.zeros(n_meses, dtype=float),
            "falha": np.full(n_meses, "Não", dtype=object),
            "modo_operacao": np.full(n_meses, "Normal", dtype=object),
            "responsavel_conjunta": np.zeros(n_meses, dtype=bool),
            "conjunta_solicitada": np.zeros(n_meses, dtype=float),
            "conjunta_atendida": np.zeros(n_meses, dtype=float),
        })

    volumes_atuais = [float(p["vol_ini"]) for p in params]
    afluencias_hm3 = [serie["vazoes_m3s"] * HM3_POR_M3S for serie in series]
    evaporacoes_m = [serie["evaporacao_mm"] / 1000.0 for serie in series]
    # estado das regras de gatilho no mês anterior (usado pela histerese)
    transferencia_ativa = [False] * n_res
    conjunta_deslocada = [False] * n_res

    for t in range(n_meses):
        estado_sistema_fq = None
        if cenario_fq_ativo:
            estado_sistema_fq = estado_fogareiro_quixeramobim(
                volumes_atuais[indice_controlador_fq],
                float(params[indice_controlador_fq]["capacidade"]),
                series[indice_controlador_fq]["meses_num"][t],
            )

        demandas_iniciais = []
        racionamentos = []
        nomes_faixas_atuais = []
        prev_volumes_pos_natureza = []

        for i in range(n_res):
            p = params[i]
            vol_ini = volumes_atuais[i]
            capacidade = float(p["capacidade"])
            pct_vol = (vol_ini / capacidade) * 100.0 if capacidade > 0 else 0.0
            mes_atual = series[i]["meses"][t]

            rac = 0.0
            nome_faixa = "Normal"
            codigo = str(p.get("cod", ""))
            demandas_cenario = FOGAREIRO_QUIXERAMOBIM_CENARIO_1["demandas_lps"].get(codigo)
            if cenario_fq_ativo and demandas_cenario:
                demanda_normal = demandas_cenario["Normal"] / 1000.0
                demanda_estado = demandas_cenario[estado_sistema_fq] / 1000.0
                rac = (1.0 - demanda_estado / demanda_normal) * 100.0
                nome_faixa = estado_sistema_fq
            else:
                demanda_normal = float(p["demanda_nominal"])

            regras = p["regras_secas"].get(mes_atual, []) if p["regras_secas"] else []
            if regras and not cenario_fq_ativo:
                nome_faixa = p.get("nome_faixa_normal", "Acima do Teto")
                for limite, rac_regra, faixa in regras:
                    if pct_vol <= limite:
                        rac = float(rac_regra)
                        nome_faixa = faixa
                        break

            demandas_iniciais.append(demanda_normal)
            racionamentos.append(rac)
            nomes_faixas_atuais.append(nome_faixa)

            vol_pos_natureza, _, _, _ = dinamica_mensal_fast(
                float(vol_ini),
                float(afluencias_hm3[i][t]),
                float(evaporacoes_m[i][t]),
                0.0,
                0.0,
                capacidade,
                p["cav_vol"],
                p["cav_area"],
            )
            prev_volumes_pos_natureza.append(vol_pos_natureza)

        demandas_finais = [0.0] * n_res
        transferencias_registradas = [0.0] * n_res
        transferencias_enviadas = [0.0] * n_res
        demandas_solicitadas_paralelo = [0.0] * n_res
        responsabilidade_especifica = [float(p["demanda_nominal"]) for p in params]
        conjunta_aplicada = [0.0] * n_res
        conjunta_bruta = [0.0] * n_res

        if modo_id == "paralelo":
            alocacao_conjunta_bruta = [0.0] * n_res
            alocacao_conjunta_bruta[0] = vazao_conjunta

            for i in range(n_res - 1):
                p = params[i]
                capacidade = float(p["capacidade"])
                vol_gatilho = capacidade * (float(p["gatilho"]) / 100.0)
                vol_retorno = capacidade * ((float(p["gatilho"]) + histerese_percent) / 100.0)
                carga_para_mover_bruta = alocacao_conjunta_bruta[i]
                deslocar = False

                if carga_para_mover_bruta > 0 and _abaixo_do_gatilho(
                    volumes_atuais[i], vol_gatilho, vol_retorno, conjunta_deslocada[i]
                ):
                    dem_esp_prox = responsabilidade_especifica[i + 1] * (1.0 - racionamentos[i + 1] / 100.0)
                    carga_conj_prox = carga_para_mover_bruta * (1.0 - racionamentos[i + 1] / 100.0)
                    demanda_total_prox_hm3 = (dem_esp_prox + carga_conj_prox) * HM3_POR_M3S
                    if prev_volumes_pos_natureza[i + 1] >= demanda_total_prox_hm3:
                        alocacao_conjunta_bruta[i] = 0.0
                        alocacao_conjunta_bruta[i + 1] += carga_para_mover_bruta
                        deslocar = True
                conjunta_deslocada[i] = deslocar

            for i in range(n_res):
                dem_esp = responsabilidade_especifica[i] * (1.0 - racionamentos[i] / 100.0)
                dem_conj = alocacao_conjunta_bruta[i] * (1.0 - racionamentos[i] / 100.0)
                demandas_finais[i] = dem_esp + dem_conj
                demandas_solicitadas_paralelo[i] = responsabilidade_especifica[i] + alocacao_conjunta_bruta[i]
                conjunta_aplicada[i] = dem_conj
                conjunta_bruta[i] = alocacao_conjunta_bruta[i]

            for i in range(1, n_res):
                if alocacao_conjunta_bruta[i] > 0:
                    transferencia = alocacao_conjunta_bruta[i] * (1.0 - racionamentos[i] / 100.0)
                    transferencias_registradas[i] = transferencia
                    transferencias_enviadas[i - 1] = transferencia

        elif modo_id == "serie":
            for i in range(n_res):
                demandas_finais[i] = demandas_iniciais[i] * (1.0 - racionamentos[i] / 100.0)

            if cenario_fq_ativo:
                idx_sender = indice_controlador_fq
                idx_receiver = indice_receptor_fq
                capacidade_receptor = float(params[idx_receiver]["capacidade"])
                gatilho_percent = FOGAREIRO_QUIXERAMOBIM_CENARIO_1["gatilho_receptor_percent"]
                vol_gatilho = capacidade_receptor * gatilho_percent / 100.0
                vol_desligamento = capacidade_receptor * (gatilho_percent + histerese_percent) / 100.0
                ativa = _abaixo_do_gatilho(
                    prev_volumes_pos_natureza[idx_receiver], vol_gatilho, vol_desligamento,
                    transferencia_ativa[idx_receiver],
                )
                transferencia_ativa[idx_receiver] = ativa
                if ativa:
                    transferencia_normal = FOGAREIRO_QUIXERAMOBIM_CENARIO_1["transferencias_lps"]["Normal"]
                    fator_estado = (
                        FOGAREIRO_QUIXERAMOBIM_CENARIO_1["transferencias_lps"][estado_sistema_fq]
                        / transferencia_normal
                    )
                    fluxo_enviado_alvo = vazao_conjunta * fator_estado * atendimento_transferencia
                    volume_enviado_alvo = fluxo_enviado_alvo * HM3_POR_M3S
                    volume_enviado = min(volume_enviado_alvo, max(prev_volumes_pos_natureza[idx_sender], 0.0))
                    volume_recebido = volume_enviado

                    prev_volumes_pos_natureza[idx_receiver] += volume_recebido
                    prev_volumes_pos_natureza[idx_sender] -= volume_enviado

                    transferencias_registradas[idx_receiver] = volume_recebido * (1e6 / SEGUNDOS_MES_PADRAO)
                    transferencias_enviadas[idx_sender] = volume_enviado * (1e6 / SEGUNDOS_MES_PADRAO)
            else:
                for i in range(1, n_res):
                    idx_sender = i
                    idx_receiver = i - 1
                    capacidade_receptor = float(params[idx_receiver]["capacidade"])
                    gatilho_percent = float(params[idx_receiver]["gatilho"])
                    vol_gatilho = capacidade_receptor * (gatilho_percent / 100.0)
                    vol_desligamento = capacidade_receptor * ((gatilho_percent + histerese_percent) / 100.0)
                    ativa = _abaixo_do_gatilho(
                        prev_volumes_pos_natureza[idx_receiver], vol_gatilho, vol_desligamento,
                        transferencia_ativa[idx_receiver],
                    )
                    transferencia_ativa[idx_receiver] = ativa
                    if ativa:
                        fluxo_transferencia = vazao_conjunta * atendimento_transferencia
                        vol_demanda_hm3 = fluxo_transferencia * HM3_POR_M3S
                        disponivel_sender = prev_volumes_pos_natureza[idx_sender]
                        qtd_transferir_hm3 = min(vol_demanda_hm3, disponivel_sender)
                        prev_volumes_pos_natureza[idx_receiver] += qtd_transferir_hm3
                        prev_volumes_pos_natureza[idx_sender] -= qtd_transferir_hm3

                        fluxo_transferido = qtd_transferir_hm3 * (1e6 / SEGUNDOS_MES_PADRAO)
                        transferencias_registradas[idx_receiver] += fluxo_transferido
                        transferencias_enviadas[idx_sender] += fluxo_transferido

        else:
            for i in range(n_res):
                demandas_finais[i] = demandas_iniciais[i] * (1.0 - racionamentos[i] / 100.0)

        falhas_do_mes = []
        for i in range(n_res):
            p = params[i]
            saida = saidas[i]
            vol_ini = volumes_atuais[i]
            demanda_hm3 = demandas_finais[i] * HM3_POR_M3S

            if modo_id == "paralelo":
                saida["demanda_solicitada"][t] = demandas_solicitadas_paralelo[i]
            else:
                saida["demanda_solicitada"][t] = demandas_iniciais[i]
                saida["transferencia_recebida"][t] = transferencias_registradas[i]
                saida["transferencia_enviada"][t] = transferencias_enviadas[i]

            saida["armazenamento_inicial"][t] = vol_ini
            saida["racionamento"][t] = racionamentos[i]
            saida["modo_operacao"][t] = nomes_faixas_atuais[i]
            saida["demanda_aplicada"][t] = demandas_finais[i]

            delta_transferencia_hm3 = 0.0
            if modo_id == "serie":
                delta_transferencia_hm3 = (transferencias_registradas[i] - transferencias_enviadas[i]) * HM3_POR_M3S

            vol_final, demanda_atendida_hm3, vertimento, evap_hm3 = dinamica_mensal_fast(
                float(vol_ini),
                float(afluencias_hm3[i][t] + delta_transferencia_hm3),
                float(evaporacoes_m[i][t]),
                float(demanda_hm3),
                0.0,
                float(p["capacidade"]),
                p["cav_vol"],
                p["cav_area"],
            )

            falhou = round(demanda_atendida_hm3, 6) < round(demanda_hm3, 6)
            saida["falha"][t] = "Sim" if falhou else "Não"
            falhas_do_mes.append(falhou)
            saida["demanda_atendida"][t] = demanda_atendida_hm3 * (1e6 / SEGUNDOS_MES_PADRAO)
            saida["retirada_total"][t] = saida["demanda_atendida"][t] + transferencias_enviadas[i]
            saida["evaporacao_hm3"][t] = evap_hm3
            saida["vertimento_hm3"][t] = vertimento
            saida["armazenamento_final"][t] = vol_final
            volumes_atuais[i] = vol_final

            if modo_id == "paralelo" and conjunta_bruta[i] > 0:
                # a retirada da unidade responsável atende, na mesma proporção, a
                # parcela própria e a parcela conjunta da demanda
                saida["responsavel_conjunta"][t] = True
                saida["conjunta_solicitada"][t] = conjunta_bruta[i]
                fracao = demanda_atendida_hm3 / demanda_hm3 if demanda_hm3 > 0 else 1.0
                saida["conjunta_atendida"][t] = conjunta_aplicada[i] * min(1.0, fracao)

        if modo_id in {"paralelo", "serie"} and falhas_do_mes and all(falhas_do_mes):
            for saida in saidas:
                saida["modo_operacao"][t] = "FALHA SISTÊMICA"

    return saidas


# ---------------------------------------------------------------------------
# Indicadores de desempenho
# ---------------------------------------------------------------------------
def indicadores_desempenho(saida):
    """Confiabilidade, resiliência e vulnerabilidade (Hashimoto et al., 1982).

    - confiabilidade: fração dos meses sem falha;
    - resiliência: probabilidade de o sistema sair da falha no mês seguinte,
      isto é, número de recuperações dividido pelo número de meses em falha;
    - vulnerabilidade: média, entre os eventos de falha, do maior déficit
      relativo (déficit / demanda aplicada) de cada evento, em %.
    """
    falha = np.asarray(saida["falha"]) == "Sim"
    n = len(falha)
    aplicada = np.asarray(saida["demanda_aplicada"], dtype=float)
    atendida = np.asarray(saida["demanda_atendida"], dtype=float)
    solicitada = np.asarray(saida["demanda_solicitada"], dtype=float)
    deficit = np.clip(aplicada - atendida, 0.0, None)
    deficit_relativo = np.where(aplicada > 0, deficit / np.where(aplicada > 0, aplicada, 1.0), 0.0)

    meses_falha = int(falha.sum())
    eventos = []
    inicio = None
    for t in range(n):
        if falha[t] and inicio is None:
            inicio = t
        if inicio is not None and (not falha[t] or t == n - 1):
            fim = t if falha[t] else t - 1
            eventos.append((inicio, fim))
            inicio = None
    recuperacoes = sum(1 for _, fim in eventos if fim < n - 1)

    if meses_falha:
        resiliencia = recuperacoes / meses_falha
        vulnerabilidade = float(np.mean([deficit_relativo[a:b + 1].max() for a, b in eventos])) * 100.0
    else:
        resiliencia = 1.0
        vulnerabilidade = 0.0

    total_aplicada = float(aplicada.sum())
    total_solicitada = float(solicitada.sum())
    total_atendida = float(atendida.sum())
    return {
        "meses": n,
        "meses_falha": meses_falha,
        "confiabilidade_percent": round((1.0 - meses_falha / n) * 100.0, 4) if n else 100.0,
        "resiliencia_percent": round(resiliencia * 100.0, 4),
        "vulnerabilidade_percent": round(vulnerabilidade, 4),
        "eventos_falha": len(eventos),
        "duracao_maxima_falha_meses": max((b - a + 1 for a, b in eventos), default=0),
        "deficit_acumulado_hm3": round(float(deficit.sum()) * HM3_POR_M3S, 6),
        "atendimento_demanda_aplicada_percent": round(100.0 * total_atendida / total_aplicada, 4) if total_aplicada > 0 else 100.0,
        "atendimento_demanda_solicitada_percent": round(100.0 * total_atendida / total_solicitada, 4) if total_solicitada > 0 else 100.0,
        "meses_racionamento": int((np.asarray(saida["racionamento"], dtype=float) > 0).sum()),
    }


def indicadores_sistema(saidas, modo):
    """Indicadores do conjunto de reservatórios (modos Série e Paralelo)."""
    modo_id = normalizar_modo_simulacao(modo)
    if not saidas:
        return {}
    n = len(saidas[0]["falha"])
    falhas = np.array([np.asarray(s["falha"]) == "Sim" for s in saidas])
    resultado = {
        "modo": modo_id,
        "meses": n,
        "falhas_sistemicas": int(falhas.all(axis=0).sum()) if len(saidas) > 1 else int(falhas[0].sum()),
    }
    if modo_id == "paralelo":
        solicitada = sum(np.asarray(s["conjunta_solicitada"]) for s in saidas)
        atendida = sum(np.asarray(s["conjunta_atendida"]) for s in saidas)
        aplicada_total = sum(
            np.asarray(s["conjunta_solicitada"]) * (1.0 - np.asarray(s["racionamento"]) / 100.0) for s in saidas
        )
        falha_conjunta = (solicitada > 0) & (np.round(atendida * HM3_POR_M3S, 6) < np.round(aplicada_total * HM3_POR_M3S, 6))
        responsavel = np.array([np.asarray(s["responsavel_conjunta"]) for s in saidas])
        indice_resp = np.where(responsavel.any(axis=0), responsavel.argmax(axis=0), -1)
        resultado.update({
            "meses_falha_demanda_conjunta": int(falha_conjunta.sum()),
            "atendimento_demanda_conjunta_percent": round(100.0 * float(atendida.sum()) / float(solicitada.sum()), 4) if solicitada.sum() > 0 else 100.0,
            "meses_por_unidade_responsavel": [int((indice_resp == i).sum()) for i in range(len(saidas))],
            "mudancas_de_responsavel": int((np.diff(indice_resp) != 0).sum()) if n > 1 else 0,
        })
    elif modo_id == "serie":
        enviada = sum(np.asarray(s["transferencia_enviada"]) for s in saidas)
        resultado.update({
            "meses_com_transferencia": int((enviada > 1e-12).sum()),
            "volume_transferido_hm3": round(float(enviada.sum()) * HM3_POR_M3S, 6),
            "acionamentos_transferencia": int(np.sum(np.diff(np.r_[0, (enviada > 1e-12).astype(int)]) == 1)),
        })
    return resultado


# ---------------------------------------------------------------------------
# Vazões de garantia (permanência)
# ---------------------------------------------------------------------------
@njit(cache=True)
def contar_falhas_demanda(demanda_hm3, aflu_hm3, evap_m, cap_hm3, cav_vol, cav_area, vol_inicial):
    """Número de meses em que uma demanda constante não é integralmente atendida."""
    vol = vol_inicial
    falhas = 0
    for i in range(len(aflu_hm3)):
        vol, retirada_efetiva, _, _ = dinamica_mensal_fast(
            vol, aflu_hm3[i], evap_m[i], demanda_hm3, 0.0, cap_hm3, cav_vol, cav_area
        )
        if retirada_efetiva + 1e-7 < demanda_hm3:
            falhas += 1
    return falhas


def garantia_demanda(demanda_m3s, aflu_hm3, evap_m, cap_hm3, cav_vol, cav_area, vol_inicial_percent):
    if len(aflu_hm3) == 0:
        return 0.0, 0
    demanda_hm3 = max(0.0, float(demanda_m3s)) * 2.592
    vol = max(0.0, min(100.0, float(vol_inicial_percent))) / 100.0 * float(cap_hm3)
    falhas = int(contar_falhas_demanda(
        float(demanda_hm3),
        np.ascontiguousarray(aflu_hm3, dtype=np.float64),
        np.ascontiguousarray(evap_m, dtype=np.float64),
        float(cap_hm3),
        np.ascontiguousarray(cav_vol, dtype=np.float64),
        np.ascontiguousarray(cav_area, dtype=np.float64),
        float(vol),
    ))
    garantia = 1.0 - (falhas / len(aflu_hm3))
    return float(max(0.0, min(1.0, garantia))), falhas


def buscar_vazao_por_garantia(garantia_alvo, aflu_hm3, evap_m, cap_hm3, cav_vol, cav_area, vol_inicial_percent):
    """Maior demanda constante atendida com a garantia requerida (busca por bisseção)."""
    alvo = max(0.0, min(1.0, float(garantia_alvo)))
    high = max(0.001, float(np.nanmax(aflu_hm3 / 2.592)) + (float(cap_hm3) / 2.592))

    garantia_high, _ = garantia_demanda(high, aflu_hm3, evap_m, cap_hm3, cav_vol, cav_area, vol_inicial_percent)
    expansoes = 0
    while garantia_high >= alvo and high < 1e5 and expansoes < 20:
        high *= 2.0
        garantia_high, _ = garantia_demanda(high, aflu_hm3, evap_m, cap_hm3, cav_vol, cav_area, vol_inicial_percent)
        expansoes += 1

    low = 0.0
    for _ in range(28):
        mid = (low + high) / 2.0
        garantia_mid, _ = garantia_demanda(mid, aflu_hm3, evap_m, cap_hm3, cav_vol, cav_area, vol_inicial_percent)
        if garantia_mid >= alvo:
            low = mid
        else:
            high = mid

    garantia_final, falhas = garantia_demanda(low, aflu_hm3, evap_m, cap_hm3, cav_vol, cav_area, vol_inicial_percent)
    return float(low), garantia_final, falhas
