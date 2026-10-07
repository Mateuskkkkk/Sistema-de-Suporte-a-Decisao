# API do Sistema de Suporte à Decisão: leitura da base de dados, validação das
# requisições e montagem das respostas. O cálculo do balanço hídrico fica em
# simulador.py.
import os
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, model_validator

from forecast_engine import router as previsao_router
from optimizer_engine import router as otimizador_router
from simulador import (  # noqa: F401  (reexportados para os testes e scripts)
    FOGAREIRO_QUIXERAMOBIM_CENARIO_1,
    FOGAREIRO_QUIXERAMOBIM_CENARIO_1_ID,
    HM3_POR_M3S,
    MESES,
    ORDEM_MESES,
    SEGUNDOS_MES_PADRAO,
    buscar_vazao_por_garantia,
    estado_fogareiro_quixeramobim,
    faixas_fogareiro_quixeramobim_percentuais,
    indicadores_desempenho,
    indicadores_sistema,
    normalizar_modo_simulacao,
    simular_sistema_n,
)

app = FastAPI(title="API do Simulador Hidrológico", version="1.1")

# Origens autorizadas: defina CORS_ORIGINS (separadas por vírgula) na implantação.
# Sem a variável, qualquer origem é aceita, mas sem envio de credenciais, que o
# frontend não utiliza.
_origens = [o.strip() for o in os.environ.get("CORS_ORIGINS", "").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origens or ["*"],
    allow_credentials=bool(_origens),
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(otimizador_router)
app.include_router(previsao_router)

# banco de dados ao lado deste arquivo, independentemente da pasta de execução
DB_PATH = os.environ.get("BANCO_SITE_DB", str(Path(__file__).resolve().parent / "banco_site.db"))

ordem_meses = ORDEM_MESES  # nome mantido por compatibilidade

CENARIOS_HIDROLOGICOS = {
    "historico": "Série histórica",
    "afluencia_zero": "Afluência nula",
    "seco_50": "50% da afluência histórica",
    "umido_120": "120% da afluência histórica",
    "fator_personalizado": "Percentual personalizado da afluência histórica",
    "seca_repetida": "Repetição de uma seca histórica",
    "reamostragem_anual": "Reamostragem aleatória de anos históricos",
}
FATORES_FIXOS = {"historico": 1.0, "afluencia_zero": 0.0, "seco_50": 0.5, "umido_120": 1.2}


def corrigir_mojibake(valor):
    """Corrige textos com acentuação duplamente codificada (ex.: 'SÃ©rie')."""
    if not isinstance(valor, str):
        return valor
    texto = valor
    for _ in range(3):
        if not any(marca in texto for marca in ("Ã", "Â", "â")):
            break
        try:
            novo = texto.encode("cp1252").decode("utf-8")
        except UnicodeError:
            try:
                novo = texto.encode("latin1").decode("utf-8")
            except UnicodeError:
                break
        if novo == texto:
            break
        texto = novo
    return texto


# ---------------------------------------------------------------------------
# Modelos de requisição, com validação e mensagens em português
# ---------------------------------------------------------------------------
class FaixaCustom(BaseModel):
    """Limites mensais (% da capacidade) e racionamento de um nível meta."""
    Faixa: str
    Racionamento: float
    NomeFaixaNormal: Optional[str] = None
    JAN: float = 100
    FEV: float = 100
    MAR: float = 100
    ABR: float = 100
    MAI: float = 100
    JUN: float = 100
    JUL: float = 100
    AGO: float = 100
    SET: float = 100
    OUT: float = 100
    NOV: float = 100
    DEZ: float = 100

    @model_validator(mode="after")
    def validar(self):
        if not 0.0 <= self.Racionamento <= 100.0:
            raise ValueError(f"Faixa {self.Faixa}: o racionamento deve estar entre 0 e 100%.")
        for mes in MESES:
            valor = getattr(self, mes)
            if valor < 0:
                raise ValueError(f"Faixa {self.Faixa}: o limite de {mes} não pode ser negativo.")
            if valor > 100.0:
                # acima de 100% equivale a 100%, pois o volume nunca excede a capacidade
                setattr(self, mes, 100.0)
        return self


class Reservatorio(BaseModel):
    nome: str
    cod: str
    capacidade: float
    est_evap: str
    vol_inicial: float
    demanda: float
    gatilho: float
    plano_secas_custom: Optional[List[FaixaCustom]] = None  # faixas editadas na sessão

    @model_validator(mode="after")
    def validar(self):
        nome = self.nome or "reservatório"
        if not str(self.nome).strip():
            raise ValueError("Selecione o reservatório.")
        if self.capacidade <= 0:
            raise ValueError(f"{nome}: a capacidade deve ser maior que zero.")
        if self.vol_inicial < 0:
            raise ValueError(f"{nome}: o volume inicial não pode ser negativo.")
        if self.vol_inicial > self.capacidade * (1 + 1e-9) + 1e-12:
            raise ValueError(f"{nome}: o volume inicial não pode ser maior que a capacidade.")
        if self.demanda < 0:
            raise ValueError(f"{nome}: a demanda não pode ser negativa.")
        if not 0.0 <= self.gatilho <= 100.0:
            raise ValueError(f"{nome}: o gatilho deve estar entre 0 e 100% da capacidade.")
        return self


class SimulacaoRequest(BaseModel):
    reservatorios: List[Reservatorio]
    modo: str
    vazao_conjunta: float
    atendimento_transferencia: float = 100.0
    mes_inicial: str
    ano_inicial: int
    mes_final: str
    ano_final: int
    cenario_hidrologico: str = "historico"
    usar_niveis_meta: bool = False
    cenario_hidrossistema: Optional[str] = None
    # opções adicionadas na versão 1.1; os valores padrão reproduzem a versão 1.0
    fator_afluencia_percent: float = 100.0
    seca_ano_inicial: Optional[int] = None
    seca_ano_final: Optional[int] = None
    semente: int = 1

    @model_validator(mode="after")
    def validar(self):
        self.mes_inicial = str(self.mes_inicial).upper()[:3]
        self.mes_final = str(self.mes_final).upper()[:3]
        erros = []
        if not self.reservatorios:
            erros.append("Informe ao menos um reservatório.")
        if self.mes_inicial not in ORDEM_MESES or self.mes_final not in ORDEM_MESES:
            erros.append("Mês inicial ou final inválido.")
        elif self.ano_inicial * 12 + ORDEM_MESES[self.mes_inicial] > self.ano_final * 12 + ORDEM_MESES[self.mes_final]:
            erros.append("O período inicial deve ser anterior ao período final.")
        if self.vazao_conjunta < 0:
            erros.append("A vazão conjunta ou de transferência não pode ser negativa.")
        if not 0.0 <= self.atendimento_transferencia <= 100.0:
            erros.append("O atendimento da transferência deve estar entre 0 e 100%.")
        if self.cenario_hidrologico not in CENARIOS_HIDROLOGICOS:
            erros.append(f"Cenário hidrológico desconhecido: {self.cenario_hidrologico}.")
        if not 0.0 <= self.fator_afluencia_percent <= 500.0:
            erros.append("O percentual da afluência deve estar entre 0 e 500%.")
        if self.cenario_hidrologico == "seca_repetida":
            if self.seca_ano_inicial is None or self.seca_ano_final is None:
                erros.append("Informe os anos inicial e final da seca a repetir.")
            elif self.seca_ano_inicial > self.seca_ano_final:
                erros.append("O ano inicial da seca deve ser anterior ao ano final.")
        if erros:
            raise ValueError(" ".join(erros))
        return self


class VazoesBaseRequest(BaseModel):
    reservatorio: str
    mes_inicial: int = 1
    ano_inicial: int = 1911
    mes_final: int = 12
    ano_final: int = 2021


class PermanenciaRequest(VazoesBaseRequest):
    vol_inicial_percent: float = 100.0


# ---------------------------------------------------------------------------
# Acesso à base de dados
# ---------------------------------------------------------------------------
def get_db_connection():
    if not os.path.exists(DB_PATH):
        raise HTTPException(status_code=500, detail="Base de dados não encontrada.")
    return sqlite3.connect(DB_PATH)


def float_seguro(valor, padrao=0.0):
    try:
        numero = float(valor)
    except (TypeError, ValueError):
        return float(padrao)
    return numero if np.isfinite(numero) else float(padrao)


def ler_vazoes_brutas(conexao, nome: str) -> Dict[Tuple[int, int], float]:
    """Vazões médias mensais (m³/s) de um reservatório, indexadas por (ano, mês)."""
    linhas = conexao.execute(
        'SELECT Ano, "Mês", "Vazão (m³/s)" FROM vazoes WHERE nome_reservatorio = ?',
        (nome,),
    ).fetchall()
    vazoes = {}
    for ano, mes, vazao in linhas:
        mes_num = ORDEM_MESES.get(str(mes).upper()[:3])
        if mes_num is None or ano is None or vazao is None:
            continue
        vazoes[(int(float_seguro(ano)), mes_num)] = float_seguro(vazao)
    return vazoes


def resolver_nome_reservatorio(conexao, nome: str) -> str:
    """Nome exato na tabela de vazões; aceita busca parcial apenas se ela for inequívoca."""
    nome = corrigir_mojibake(str(nome)).strip()
    if conexao.execute("SELECT 1 FROM vazoes WHERE nome_reservatorio = ? LIMIT 1", (nome,)).fetchone():
        return nome
    candidatos = [r[0] for r in conexao.execute(
        "SELECT DISTINCT nome_reservatorio FROM vazoes WHERE nome_reservatorio LIKE ?", (f"%{nome}%",)
    )]
    if len(candidatos) == 1:
        return candidatos[0]
    if not candidatos:
        raise HTTPException(status_code=404, detail=f"Vazões não encontradas para {nome}.")
    raise HTTPException(status_code=400, detail=f"O nome '{nome}' corresponde a mais de um reservatório: {', '.join(candidatos)}.")


def ler_evaporacao_mensal(conexao, est_evap) -> np.ndarray:
    codigo = str(est_evap).strip()
    if codigo.endswith(".0"):
        codigo = codigo[:-2]
    linha = conexao.execute(
        'SELECT JAN, FEV, MAR, ABR, MAI, JUN, JUL, AGO, "SET", OUT, NOV, DEZ FROM evaporacao WHERE COD = ? LIMIT 1',
        (codigo,),
    ).fetchone()
    evap = np.zeros(12, dtype=float)
    if linha is not None:
        evap[:] = [float_seguro(v) for v in linha]
    return evap


def ler_cav(conexao, cod, capacidade_hm3):
    linhas = conexao.execute(
        'SELECT "VOLUME (m³)", "AREA (km²)" FROM cav WHERE COD = ?', (str(cod),)
    ).fetchall()
    if len(linhas) < 2:
        return (
            np.array([0.0, max(float(capacidade_hm3), 0.01)], dtype=float),
            np.array([0.0, 0.0], dtype=float),
        )
    return (
        np.fromiter((float_seguro(v) / 1e6 for v, _ in linhas), dtype=float),
        np.fromiter((float_seguro(a) for _, a in linhas), dtype=float),
    )


def ler_plano_secas(conexao, cod):
    colunas = ", ".join(f'"{m}"' for m in MESES)
    return conexao.execute(
        f'SELECT Faixa, "Racionamento (%)", {colunas} FROM plano_secas WHERE COD = ?', (str(cod),)
    ).fetchall()


# ---------------------------------------------------------------------------
# Rotas de consulta
# ---------------------------------------------------------------------------
@app.get("/api/reservatorios")
def listar_reservatorios():
    conexao = get_db_connection()
    try:
        linhas = conexao.execute('SELECT CORPO, COD, "CAPAC (m³)", "Est. Evap." FROM acudes').fetchall()
    finally:
        conexao.close()
    registros = []
    for corpo, cod, capac, est_evap in linhas:
        cap_hm3 = float(capac) / 1e6 if capac is not None else None
        registros.append({
            "CORPO": corrigir_mojibake(corpo),
            "COD": cod,
            "CAPAC (m³)": cap_hm3,  # mantido por compatibilidade: valor em hm³
            "Est. Evap.": est_evap,
            "capacidade_hm3": cap_hm3,
        })
    return registros


@app.get("/api/presets")
def listar_presets():
    conexao = get_db_connection()
    try:
        df_hidro = pd.read_sql_query("SELECT * FROM hidrossistemas", conexao)
    finally:
        conexao.close()
    coluna_operacao = next(c for c in df_hidro.columns if c.lower().startswith("opera"))
    presets = []
    for nome_sis, group in df_hidro.groupby("hidrossistema"):
        modo = str(group[coluna_operacao].iloc[0]).lower()
        modo_operacao = "Série" if "ser" in modo or "sér" in modo else "Paralelo" if "paral" in modo else "Individual"
        preset = {
            "nome": corrigir_mojibake(nome_sis),
            "modo": modo_operacao,
            "reservatorios": group["cod_acude"].astype(str).tolist(),
        }
        codigos = set(preset["reservatorios"])
        if {"16", "119"}.issubset(codigos) and modo_operacao == "Série":
            preset.update({
                "nome": "Fogareiro/Quixeramobim - PGPS Cenário 1",
                "reservatorios": ["119", "16"],
                "cenario_hidrossistema": FOGAREIRO_QUIXERAMOBIM_CENARIO_1_ID,
                "fonte": "Plano de Gestão Proativa de Seca, cenário 1 escolhido",
                "periodo": {"mes_inicial": "JAN", "ano_inicial": 1911, "mes_final": "DEZ", "ano_final": 2019},
                "defaults": {
                    "16": {"demanda_lps": 342.0, "vol_inicial_percent": 100.0, "gatilho_percent": 30.0},
                    "119": {"demanda_lps": 272.0, "vol_inicial_percent": 100.0, "gatilho_percent": 0.0},
                },
                "niveis_meta": {"reservatorio_cod": "119", "faixas": faixas_fogareiro_quixeramobim_percentuais()},
                "vazao_transferencia_lps": 500.0,
                "atendimento_transferencia_percent": 100.0,
                "transferencias_lps": FOGAREIRO_QUIXERAMOBIM_CENARIO_1["transferencias_lps"],
            })
        presets.append(preset)
    return presets


@app.get("/api/cenarios-hidrologicos")
def listar_cenarios_hidrologicos():
    return [{"id": chave, "nome": nome} for chave, nome in CENARIOS_HIDROLOGICOS.items()]


@app.get("/api/plano-secas/{cod_acude}")
def obter_plano_secas(cod_acude: str):
    try:
        conexao = get_db_connection()
        try:
            linhas = ler_plano_secas(conexao, cod_acude)
        finally:
            conexao.close()
    except sqlite3.Error as erro:
        raise HTTPException(status_code=500, detail=f"Não foi possível ler o plano de secas: {erro}")
    registros = []
    for linha in linhas:
        registro = {"COD": cod_acude, "Faixa": corrigir_mojibake(linha[0]), "Racionamento": linha[1]}
        registro.update({mes: linha[2 + i] for i, mes in enumerate(MESES)})
        registros.append(registro)
    return registros


# ---------------------------------------------------------------------------
# Simulação
# ---------------------------------------------------------------------------
def mapear_fonte_vazoes(req: SimulacaoRequest, periodo, vazoes_por_res):
    """Para cada mês simulado, o (ano, mês) da série histórica de onde vem a vazão.

    Nos cenários de fator (histórico, 50%, 120%...), cada mês usa a própria vazão.
    Na repetição de seca, os anos da seca são repetidos em sequência a partir do
    início da simulação. Na reamostragem, cada ano simulado recebe um ano histórico
    sorteado (o mesmo para todos os reservatórios, preservando a coerência espacial).
    """
    cenario = req.cenario_hidrologico
    if cenario == "seca_repetida":
        anos_seca = list(range(int(req.seca_ano_inicial), int(req.seca_ano_final) + 1))
        ano0 = periodo[0][0]
        return {(a, m): (anos_seca[(a - ano0) % len(anos_seca)], m) for a, m in periodo}
    if cenario == "reamostragem_anual":
        anos_completos = None
        for vazoes in vazoes_por_res:
            anos = {a for a in {a for a, _ in vazoes} if all((a, m) in vazoes for m in range(1, 13))}
            anos_completos = anos if anos_completos is None else anos_completos & anos
        anos_completos = sorted(anos_completos or [])
        if not anos_completos:
            raise HTTPException(status_code=400, detail="Não há anos completos comuns a todos os reservatórios para reamostrar.")
        anos_simulados = sorted({a for a, _ in periodo})
        sorteio = np.random.default_rng(int(req.semente)).choice(anos_completos, size=len(anos_simulados), replace=True)
        mapa_anos = dict(zip(anos_simulados, (int(a) for a in sorteio)))
        return {(a, m): (mapa_anos[a], m) for a, m in periodo}
    return None


def montar_series(req: SimulacaoRequest, conexao):
    nomes = [resolver_nome_reservatorio(conexao, r.nome) for r in req.reservatorios]
    vazoes_por_res = [ler_vazoes_brutas(conexao, nome) for nome in nomes]
    p_ini = int(req.ano_inicial) * 12 + ORDEM_MESES[req.mes_inicial]
    p_fim = int(req.ano_final) * 12 + ORDEM_MESES[req.mes_final]

    fator = FATORES_FIXOS.get(req.cenario_hidrologico, 1.0)
    if req.cenario_hidrologico == "fator_personalizado":
        fator = float(req.fator_afluencia_percent) / 100.0

    periodo_ref = sorted(k for k in vazoes_por_res[0] if p_ini <= k[0] * 12 + k[1] <= p_fim)
    if not periodo_ref:
        raise HTTPException(status_code=404, detail=f"Não há vazões no período selecionado para {nomes[0]}.")
    mapa = mapear_fonte_vazoes(req, periodo_ref, vazoes_por_res)

    series = []
    for reservatorio, nome, vazoes in zip(req.reservatorios, nomes, vazoes_por_res):
        periodo = sorted(k for k in vazoes if p_ini <= k[0] * 12 + k[1] <= p_fim)
        if not periodo:
            raise HTTPException(status_code=404, detail=f"Não há vazões no período selecionado para {nome}.")
        valores, anos_fonte = [], []
        for ano, mes in periodo:
            fonte = mapa[(ano, mes)] if mapa else (ano, mes)
            if fonte not in vazoes:
                raise HTTPException(status_code=400, detail=f"A série de {nome} não possui vazão em {MESES[fonte[1] - 1]}/{fonte[0]}.")
            valores.append(vazoes[fonte] * fator)
            anos_fonte.append(fonte[0])
        evap_mensal = ler_evaporacao_mensal(conexao, reservatorio.est_evap)
        meses_num = np.fromiter((m for _, m in periodo), dtype=np.int16)
        series.append({
            "nome_reservatorio": nome,
            "anos": np.fromiter((a for a, _ in periodo), dtype=np.int32),
            "meses": [MESES[m - 1] for _, m in periodo],
            "meses_num": meses_num,
            "datas": [f"{a:04d}-{m:02d}" for a, m in periodo],
            "vazoes_m3s": np.array(valores, dtype=float),
            "evaporacao_mm": evap_mensal[meses_num - 1],
            "anos_fonte": anos_fonte if mapa else None,
        })
    return series


def carregar_regras_simulador(conexao, reservatorio, usar_niveis_meta):
    if not usar_niveis_meta:
        return {}
    regras_mes = {}
    if reservatorio.plano_secas_custom:
        for mes in MESES:
            regras = [(float(getattr(f, mes)), float(f.Racionamento), f.Faixa) for f in reservatorio.plano_secas_custom]
            regras.sort(key=lambda item: item[0])
            regras_mes[mes] = regras
        return regras_mes
    linhas = ler_plano_secas(conexao, reservatorio.cod)
    for indice_mes, mes in enumerate(MESES):
        regras = [
            (float_seguro(linha[2 + indice_mes]), float_seguro(linha[1]), corrigir_mojibake(linha[0]))
            for linha in linhas
        ]
        regras.sort(key=lambda item: item[0])
        regras_mes[mes] = regras
    return regras_mes


def montar_registros_simulador(serie, saida, modo_id):
    registros = []
    for i, data in enumerate(serie["datas"]):
        registro = {
            "nome_reservatorio": serie["nome_reservatorio"],
            "Ano": int(serie["anos"][i]),
            "Mês": serie["meses"][i],
            "Vazão (m³/s)": float(serie["vazoes_m3s"][i]),
            "Ordem_Mês": int(serie["meses_num"][i]),
            "Data": data,
            "Evaporação (m)": float(serie["evaporacao_mm"][i]),
            "Armazenamento Inicial": float(saida["armazenamento_inicial"][i]),
            "Armazenamento Final": float(saida["armazenamento_final"][i]),
            "Demanda Solicitada (m³/s)": float(saida["demanda_solicitada"][i]),
            "Demanda Aplicada (m³/s)": float(saida["demanda_aplicada"][i]),
            "Demanda Atendida (m³/s)": float(saida["demanda_atendida"][i]),
            "Retirada Total (m³/s)": float(saida["retirada_total"][i]),
            "Racionamento (%)": float(saida["racionamento"][i]),
            "Transferência Recebida (m³/s)": float(saida["transferencia_recebida"][i]),
            "Transferência Enviada (m³/s)": float(saida["transferencia_enviada"][i]),
            "Evaporação (hm³)": float(saida["evaporacao_hm3"][i]),
            "Vertimento (hm³)": float(saida["vertimento_hm3"][i]),
            "Falha": saida["falha"][i],
            "Modo Operação": saida["modo_operacao"][i],
            "Afluências (hm³/mês)": float(serie["vazoes_m3s"][i] * HM3_POR_M3S),
        }
        if modo_id == "paralelo":
            registro["Responsável Demanda Conjunta"] = "Sim" if saida["responsavel_conjunta"][i] else "Não"
        if serie.get("anos_fonte"):
            registro["Ano de Origem da Vazão"] = int(serie["anos_fonte"][i])
        registros.append(registro)
    return registros


@app.post("/api/simular")
def processar_simulacao_api(req: SimulacaoRequest):
    if not req.reservatorios:
        raise HTTPException(status_code=400, detail="Informe ao menos um reservatório.")

    conexao = get_db_connection()
    params = []
    try:
        series = montar_series(req, conexao)
        for reservatorio in req.reservatorios:
            cav_vol, cav_area = ler_cav(conexao, reservatorio.cod, reservatorio.capacidade)
            params.append({
                "cod": str(reservatorio.cod),
                "cav_vol": cav_vol,
                "cav_area": cav_area,
                "regras_secas": carregar_regras_simulador(conexao, reservatorio, req.usar_niveis_meta),
                "nome_faixa_normal": (
                    reservatorio.plano_secas_custom[0].NomeFaixaNormal
                    if reservatorio.plano_secas_custom and reservatorio.plano_secas_custom[0].NomeFaixaNormal
                    else "Acima do Teto"
                ),
                "capacidade": float(reservatorio.capacidade),
                "vol_ini": float(reservatorio.vol_inicial),
                "demanda_nominal": float(reservatorio.demanda),
                "gatilho": float(reservatorio.gatilho),
            })
    finally:
        conexao.close()

    cenario_hidrossistema_ativo = req.cenario_hidrossistema if req.usar_niveis_meta else None
    try:
        saidas = simular_sistema_n(
            series, params, req.modo, req.vazao_conjunta, req.atendimento_transferencia,
            cenario_hidrossistema_ativo,
        )
    except ValueError as erro:
        raise HTTPException(status_code=400, detail=str(erro))

    modo_id = normalizar_modo_simulacao(req.modo)
    resultados = [
        {
            "reservatorio": req.reservatorios[i].nome,
            "dados": montar_registros_simulador(series[i], saidas[i], modo_id),
            "indicadores": indicadores_desempenho(saidas[i]),
        }
        for i in range(len(series))
    ]
    return {
        "status": "sucesso",
        "resultados": resultados,
        "indicadores_sistema": indicadores_sistema(saidas, req.modo),
        "cenario_hidrossistema": cenario_hidrossistema_ativo,
        "cenario_hidrologico": {
            "id": req.cenario_hidrologico,
            "nome": CENARIOS_HIDROLOGICOS[req.cenario_hidrologico],
        },
    }


# ---------------------------------------------------------------------------
# Vazões de garantia (permanência)
# ---------------------------------------------------------------------------
def carregar_serie_vazoes(reservatorio: str, mes_ini: int, ano_ini: int, mes_fim: int, ano_fim: int) -> pd.DataFrame:
    conexao = get_db_connection()
    try:
        nome = resolver_nome_reservatorio(conexao, reservatorio)
        vazoes = ler_vazoes_brutas(conexao, nome)
    finally:
        conexao.close()
    p_ini, p_fim = int(ano_ini) * 12 + int(mes_ini), int(ano_fim) * 12 + int(mes_fim)
    periodo = sorted(k for k in vazoes if p_ini <= k[0] * 12 + k[1] <= p_fim)
    if not periodo:
        raise HTTPException(status_code=404, detail="Não há vazões no período selecionado.")
    return pd.DataFrame({
        "Data": pd.to_datetime([f"{a}-{m}-01" for a, m in periodo]),
        "mes_num": [m for _, m in periodo],
        "Vazão (m³/s)": [vazoes[k] for k in periodo],
    })


def carregar_parametros_regularizacao(reservatorio: str):
    conexao = get_db_connection()
    try:
        nome = corrigir_mojibake(str(reservatorio)).strip()
        linha = conexao.execute(
            'SELECT COD, "CAPAC (m³)", "Est. Evap." FROM acudes WHERE CORPO = ? LIMIT 1', (nome,)
        ).fetchone()
        if linha is None:
            candidatos = conexao.execute(
                'SELECT COD, "CAPAC (m³)", "Est. Evap." FROM acudes WHERE CORPO LIKE ?', (f"%{nome}%",)
            ).fetchall()
            if len(candidatos) != 1:
                raise HTTPException(status_code=404, detail=f"Reservatório não encontrado: {reservatorio}.")
            linha = candidatos[0]
        cod, capac, est_evap = linha
        cap_hm3 = float(capac) / 1e6
        linhas_cav = conexao.execute(
            'SELECT "VOLUME (m³)", "AREA (km²)" FROM cav WHERE COD = ? OR CAST(COD AS REAL) = ?', (str(cod), cod)
        ).fetchall()
        if len(linhas_cav) < 2:
            cav_vol = np.array([0.0, max(cap_hm3, 0.01)], dtype=float)
            cav_area = np.array([0.0, 0.0], dtype=float)
        else:
            linhas_cav.sort(key=lambda item: float_seguro(item[0]))
            cav_vol = np.array([float_seguro(v) for v, _ in linhas_cav], dtype=float) / 1e6
            cav_area = np.array([float_seguro(a) for _, a in linhas_cav], dtype=float)
        evap_mm = ler_evaporacao_mensal(conexao, est_evap)
    finally:
        conexao.close()
    return {"cod": str(cod), "cap_hm3": cap_hm3, "cav_vol": cav_vol, "cav_area": cav_area, "evap_mm": evap_mm}


def vazao_para_hm3_mes(vazao_m3s: float) -> float:
    return float(vazao_m3s) * 2.592


@app.post("/api/vazoes/permanencia")
def calcular_permanencias_api(req: PermanenciaRequest):
    df = carregar_serie_vazoes(req.reservatorio, req.mes_inicial, req.ano_inicial, req.mes_final, req.ano_final)
    params = carregar_parametros_regularizacao(req.reservatorio)
    valores_m3s = df["Vazão (m³/s)"].astype(float).to_numpy()
    aflu_hm3 = valores_m3s * 2.592
    evap_m = np.array([params["evap_mm"][int(m) - 1] / 1000.0 for m in df["mes_num"].astype(int).to_numpy()], dtype=float)

    resultados = []
    for q in range(1, 101):
        vazao, garantia_obtida, falhas = buscar_vazao_por_garantia(
            q / 100.0, aflu_hm3, evap_m, params["cap_hm3"], params["cav_vol"], params["cav_area"], req.vol_inicial_percent,
        )
        resultados.append({
            "referencia": f"Q{q}",
            "garantia_requerida": q,
            "garantia_obtida": round(garantia_obtida * 100, 6),
            "falhas": falhas,
            "meses": int(len(df)),
            "vazao_m3s": round(vazao, 6),
            "vazao_hm3_mes": round(vazao_para_hm3_mes(vazao), 6),
        })

    destaques = {row["referencia"]: row for row in resultados if row["referencia"] in {"Q100", "Q99", "Q98", "Q95", "Q90"}}
    curva = [{"garantia": row["garantia_requerida"], "vazao_m3s": row["vazao_m3s"], "falhas": row["falhas"]} for row in resultados]

    return {
        "status": "sucesso",
        "metodo": "Vazões de garantia calculadas por garantia mensal: garantia = 1 - falhas/meses. "
                  "Cada Qxx é a maior demanda constante atendida com a garantia requerida.",
        "reservatorio": req.reservatorio,
        "periodo": {
            "inicio": df["Data"].min().strftime("%Y-%m"),
            "fim": df["Data"].max().strftime("%Y-%m"),
            "meses": int(len(df)),
        },
        "volume_inicial_percent": req.vol_inicial_percent,
        "capacidade_hm3": round(float(params["cap_hm3"]), 6),
        "vazao_plena": destaques.get("Q100"),
        "destaques": destaques,
        "resultados": resultados,
        "curva": curva,
    }
