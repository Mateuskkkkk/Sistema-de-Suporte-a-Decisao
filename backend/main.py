# importaÃ§Ãµes necessÃ¡rias pra API funcionar
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import sqlite3
import os
import unicodedata
from typing import List, Dict, Optional
from forecast_engine import router as previsao_router
from optimizer_engine import router as otimizador_router, dinamica_mensal_fast

# cria a aplicaÃ§Ã£o FastAPI
app = FastAPI(title="API do Simulador HidrolÃ³gico", version="1.0")

# libera acesso de qualquer origem (CORS aberto)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(otimizador_router)
app.include_router(previsao_router)

# caminho do banco de dados SQLite na mesma pasta do script
DB_PATH = os.path.join(os.path.abspath("."), "banco_site.db")


def normalizar_colunas_db(df: pd.DataFrame) -> pd.DataFrame:
    renomear = {}
    for col in df.columns:
        if col.startswith("CAPAC"):
            renomear[col] = "CAPAC (mÂ³)"
        elif col.startswith("VOLUME"):
            renomear[col] = "VOLUME (mÂ³)"
        elif col.startswith("AREA"):
            renomear[col] = "AREA (kmÂ²)"
        elif col.startswith("Vaz"):
            renomear[col] = "VazÃ£o (mÂ³/s)"
        elif col.startswith("M"):
            renomear[col] = "MÃªs"
        elif col.startswith("opera"):
            renomear[col] = "operacao"
    return df.rename(columns=renomear)


def corrigir_texto_db(valor):
    if not isinstance(valor, str):
        return valor
    if not any(marca in valor for marca in ("Ãƒ", "Ã‚", "Ã¢")):
        return valor
    try:
        return valor.encode("latin1").decode("utf-8")
    except UnicodeError:
        return valor


def texto_para_legado(valor: str) -> str:
    try:
        return valor.encode("utf-8").decode("latin1")
    except UnicodeError:
        return valor


def normalizar_textos_db(df: pd.DataFrame) -> pd.DataFrame:
    for coluna in df.select_dtypes(include=["object"]).columns:
        df[coluna] = df[coluna].map(corrigir_texto_db)
    return df


def corrigir_mojibake(valor):
    if not isinstance(valor, str):
        return valor
    texto = valor
    for _ in range(3):
        if not any(marca in texto for marca in ("Ã", "Â", "â")):
            break
        try:
            novo = texto.encode("latin1").decode("utf-8")
        except UnicodeError:
            break
        if novo == texto:
            break
        texto = novo
    return texto


def normalizar_registros_saida(registros):
    normalizados = []
    for registro in registros:
        normalizados.append({
            corrigir_mojibake(chave): corrigir_mojibake(valor)
            for chave, valor in registro.items()
        })
    return normalizados

# dicionÃ¡rio pra converter nome do mÃªs em nÃºmero (JAN=1, FEV=2, ...)
ordem_meses = {
    'JAN': 1, 'FEV': 2, 'MAR': 3, 'ABR': 4, 'MAI': 5, 'JUN': 6,
    'JUL': 7, 'AGO': 8, 'SET': 9, 'OUT': 10, 'NOV': 11, 'DEZ': 12
}


MESES_ORDEM = tuple(ordem_meses.keys())
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
    # Retiradas locais. A Tabela 5.3 soma a retirada do Fogareiro com a
    # transferencia: 272 + 500 = 772 L/s no estado Normal, por exemplo.
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
            for mes, valor in zip(
                MESES_ORDEM,
                FOGAREIRO_QUIXERAMOBIM_CENARIO_1["limites_percent"][nome],
            )
        })
        faixas.append(linha)
    faixas.append({
        "Faixa": "Normal",
        "Racionamento": 0.0,
        **{mes: 100.0 for mes in MESES_ORDEM},
    })
    return faixas



# recebe os limites mensais (% do volume) e o racionamento de cada nÃ­vel meta
class FaixaCustom(BaseModel):
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


# modelo de dados de um reservatÃ³rio individual
class Reservatorio(BaseModel):
    nome: str
    cod: str
    capacidade: float
    est_evap: str
    vol_inicial: float
    demanda: float
    gatilho: float
    plano_secas_custom: Optional[List[FaixaCustom]] = None  # faixas editadas na sessÃ£o do frontend


# modelo de dados que o front manda pra iniciar uma simulaÃ§Ã£o
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


class VazoesBaseRequest(BaseModel):
    reservatorio: str
    mes_inicial: int = 1
    ano_inicial: int = 1911
    mes_final: int = 12
    ano_final: int = 2021


class PermanenciaRequest(VazoesBaseRequest):
    vol_inicial_percent: float = 100.0


class KnnRequest(VazoesBaseRequest):
    k: int = 5
    lags: int = 12
    horizonte: int = 12
    teste_meses: int = 24


def get_db_connection():
    if not os.path.exists(DB_PATH):
        raise HTTPException(status_code=500, detail="Base de dados nÃ£o encontrada.")
    return sqlite3.connect(DB_PATH)


def carregar_serie_vazoes(reservatorio: str, mes_ini: int, ano_ini: int, mes_fim: int, ano_fim: int) -> pd.DataFrame:
    conexao = get_db_connection()
    try:
        nomes_busca = [reservatorio, texto_para_legado(reservatorio)]
        df = normalizar_colunas_db(pd.read_sql_query(
            "SELECT * FROM vazoes WHERE nome_reservatorio IN (?, ?) OR nome_reservatorio LIKE ? OR nome_reservatorio LIKE ?",
            conexao,
            params=(nomes_busca[0], nomes_busca[1], f"%{nomes_busca[0]}%", f"%{nomes_busca[1]}%"),
        ))
    finally:
        conexao.close()

    if df.empty:
        raise HTTPException(status_code=404, detail=f"VazÃµes nÃ£o encontradas para {reservatorio}")

    df = normalizar_textos_db(df)
    df["mes_num"] = df["MÃªs"].map(lambda m: ordem_meses.get(str(m).upper()[:3], None))
    df = df.dropna(subset=["mes_num", "VazÃ£o (mÂ³/s)", "Ano"]).copy()
    df["mes_num"] = df["mes_num"].astype(int)
    df["Ano"] = df["Ano"].astype(int)
    df["VazÃ£o (mÂ³/s)"] = pd.to_numeric(df["VazÃ£o (mÂ³/s)"], errors="coerce")
    df = df.dropna(subset=["VazÃ£o (mÂ³/s)"])
    df["Data"] = pd.to_datetime(df["Ano"].astype(str) + "-" + df["mes_num"].astype(str) + "-01")

    data_inicio = pd.to_datetime(f"{ano_ini}-{mes_ini}-01")
    data_fim = pd.to_datetime(f"{ano_fim}-{mes_fim}-01") + pd.offsets.MonthEnd(0)
    df = df[(df["Data"] >= data_inicio) & (df["Data"] <= data_fim)].sort_values("Data").reset_index(drop=True)
    if df.empty:
        raise HTTPException(status_code=404, detail="NÃ£o hÃ¡ vazÃµes no perÃ­odo selecionado.")
    return df


def carregar_parametros_regularizacao(reservatorio: str):
    conexao = get_db_connection()
    try:
        df_acudes = normalizar_colunas_db(pd.read_sql_query(
            "SELECT * FROM acudes WHERE CORPO = ? OR CORPO LIKE ? OR CORPO = ? OR CORPO LIKE ? LIMIT 1",
            conexao,
            params=(reservatorio, f"%{reservatorio}%", texto_para_legado(reservatorio), f"%{texto_para_legado(reservatorio)}%"),
        ))
        if df_acudes.empty:
            raise HTTPException(status_code=404, detail=f"ReservatÃ³rio nÃ£o encontrado: {reservatorio}")

        acude = df_acudes.iloc[0]
        cod = acude["COD"]
        cap_hm3 = float(acude["CAPAC (mÂ³)"]) / 1e6
        est_evap = acude["Est. Evap."]

        df_cav = normalizar_colunas_db(pd.read_sql_query(
            "SELECT * FROM cav WHERE COD = ? OR CAST(COD AS REAL) = ?",
            conexao,
            params=(str(cod), cod),
        ))
        if len(df_cav) < 2:
            cav_vol = np.array([0.0, max(cap_hm3, 0.01)], dtype=float)
            cav_area = np.array([0.0, 0.0], dtype=float)
        else:
            df_cav = df_cav.sort_values("VOLUME (mÂ³)")
            cav_vol = df_cav["VOLUME (mÂ³)"].astype(float).to_numpy() / 1e6
            cav_area = df_cav["AREA (kmÂ²)"].astype(float).to_numpy()

        est_evap_str = str(int(float(est_evap))) if str(est_evap).strip() else ""
        df_evap = pd.read_sql_query(
            'SELECT JAN, FEV, MAR, ABR, MAI, JUN, JUL, AGO, "SET", OUT, NOV, DEZ '
            'FROM evaporacao WHERE COD = ? OR COD = ? LIMIT 1',
            conexao,
            params=(est_evap_str, str(est_evap)),
        )
        if df_evap.empty:
            evap_mm = np.zeros(12, dtype=float)
        else:
            evap_mm = df_evap.iloc[0].fillna(0).astype(float).to_numpy()
    finally:
        conexao.close()

    return {
        "cod": str(cod),
        "cap_hm3": cap_hm3,
        "cav_vol": cav_vol.astype(float),
        "cav_area": cav_area.astype(float),
        "evap_mm": evap_mm.astype(float),
    }


def vazao_para_hm3_mes(vazao_m3s: float) -> float:
    return float(vazao_m3s) * 2.592


def calcular_q_permanencia(valores: np.ndarray, permanencia: float) -> float:
    valores = valores[np.isfinite(valores)]
    if len(valores) == 0:
        return 0.0
    prob_nao_excedencia = max(0.0, min(1.0, 1.0 - float(permanencia) / 100.0))
    return float(np.quantile(valores, prob_nao_excedencia))


def simular_garantia_demanda(demanda_m3s: float, aflu_hm3: np.ndarray, evap_m: np.ndarray, cap_hm3: float, cav_vol: np.ndarray, cav_area: np.ndarray, vol_inicial_percent: float):
    if len(aflu_hm3) == 0:
        return 0.0, 0

    demanda_hm3 = max(0.0, float(demanda_m3s)) * 2.592
    vol = max(0.0, min(100.0, float(vol_inicial_percent))) / 100.0 * float(cap_hm3)
    falhas = 0

    for i in range(len(aflu_hm3)):
        vol, retirada_efetiva, _, _ = dinamica_mensal_fast(
            float(vol),
            float(aflu_hm3[i]),
            float(evap_m[i]),
            float(demanda_hm3),
            0.0,
            float(cap_hm3),
            cav_vol,
            cav_area,
        )
        if retirada_efetiva + 1e-7 < demanda_hm3:
            falhas += 1

    garantia = 1.0 - (falhas / len(aflu_hm3))
    return float(max(0.0, min(1.0, garantia))), int(falhas)


def buscar_vazao_por_garantia(garantia_alvo: float, aflu_hm3: np.ndarray, evap_m: np.ndarray, cap_hm3: float, cav_vol: np.ndarray, cav_area: np.ndarray, vol_inicial_percent: float):
    alvo = max(0.0, min(1.0, float(garantia_alvo)))
    high = max(0.001, float(np.nanmax(aflu_hm3 / 2.592)) + (float(cap_hm3) / 2.592))

    garantia_high, _ = simular_garantia_demanda(high, aflu_hm3, evap_m, cap_hm3, cav_vol, cav_area, vol_inicial_percent)
    expansoes = 0
    while garantia_high >= alvo and high < 1e5 and expansoes < 20:
        high *= 2.0
        garantia_high, _ = simular_garantia_demanda(high, aflu_hm3, evap_m, cap_hm3, cav_vol, cav_area, vol_inicial_percent)
        expansoes += 1

    low = 0.0
    for _ in range(28):
        mid = (low + high) / 2.0
        garantia_mid, _ = simular_garantia_demanda(mid, aflu_hm3, evap_m, cap_hm3, cav_vol, cav_area, vol_inicial_percent)
        if garantia_mid >= alvo:
            low = mid
        else:
            high = mid

    garantia_final, falhas = simular_garantia_demanda(low, aflu_hm3, evap_m, cap_hm3, cav_vol, cav_area, vol_inicial_percent)
    return float(low), garantia_final, falhas


def montar_features_knn(valores: np.ndarray, meses: np.ndarray, lags: int, indices: np.ndarray):
    features = []
    targets = []
    for i in indices:
        if i < lags:
            continue
        mes_alvo = int(meses[i])
        sazonal = [np.sin(2 * np.pi * mes_alvo / 12), np.cos(2 * np.pi * mes_alvo / 12)]
        janela = valores[i - lags:i]
        features.append(np.concatenate([janela, sazonal]))
        targets.append(valores[i])
    return np.array(features, dtype=float), np.array(targets, dtype=float)


def prever_knn(features_treino: np.ndarray, targets_treino: np.ndarray, feature_alvo: np.ndarray, k: int):
    if len(features_treino) == 0:
        return 0.0
    escala = np.std(features_treino, axis=0)
    escala = np.where(escala == 0, 1.0, escala)
    dist = np.linalg.norm((features_treino - feature_alvo) / escala, axis=1)
    vizinhos = np.argsort(dist)[:max(1, min(k, len(dist)))]
    pesos = 1.0 / (dist[vizinhos] + 1e-9)
    return float(np.sum(targets_treino[vizinhos] * pesos) / np.sum(pesos))


# funÃ§Ã£o principal que roda a simulaÃ§Ã£o mÃªs a mÃªs pra todos os reservatÃ³rios
# recebe os dataframes com as vazÃµes, os parÃ¢metros de cada aÃ§ude, o modo de operaÃ§Ã£o
# (SÃ©rie, Paralelo ou Individual) e a vazÃ£o conjunta do sistema
# funÃ§Ã£o principal que roda a simulaÃ§Ã£o mÃªs a mÃªs pra todos os reservatÃ³rios
SEGUNDOS_MES_PADRAO = 2_592_000.0


def normalizar_modo_simulacao(modo: str) -> str:
    texto = corrigir_mojibake(str(modo or ""))
    texto = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("ascii").lower()
    if "serie" in texto:
        return "serie"
    if "paral" in texto:
        return "paralelo"
    return "individual"


def estado_fogareiro_quixeramobim(
    volume_hm3: float,
    capacidade_hm3: float,
    mes: int,
) -> str:
    limites = FOGAREIRO_QUIXERAMOBIM_CENARIO_1["limites_percent"]
    indice_mes = max(1, min(int(mes), 12)) - 1
    volume_percent = (
        volume_hm3 / capacidade_hm3 * 100.0
        if capacidade_hm3 > 0
        else 0.0
    )
    tolerancia_percent = 1e-6
    if volume_percent <= limites["Seca Severa"][indice_mes] + tolerancia_percent:
        return "Seca Severa"
    if volume_percent <= limites["Seca"][indice_mes] + tolerancia_percent:
        return "Seca"
    if volume_percent <= limites["Alerta"][indice_mes] + tolerancia_percent:
        return "Alerta"
    return "Normal"


def simular_sistema_n(
    series,
    params,
    modo,
    vazao_conjunta,
    atendimento_transferencia=100.0,
    cenario_hidrossistema=None,
):
    n_res = len(series)
    if n_res == 0:
        return []

    n_meses = len(series[0]["datas"])
    modo_id = normalizar_modo_simulacao(modo)
    vazao_conjunta = float(vazao_conjunta)
    atendimento_transferencia = max(0.0, min(float(atendimento_transferencia), 100.0)) / 100.0
    cenario_fq_ativo = (
        cenario_hidrossistema == FOGAREIRO_QUIXERAMOBIM_CENARIO_1_ID
        and modo_id == "serie"
    )
    indices_por_codigo = {
        str(param.get("cod", "")): indice
        for indice, param in enumerate(params)
    }
    indice_controlador_fq = indices_por_codigo.get(
        FOGAREIRO_QUIXERAMOBIM_CENARIO_1["controlador_cod"]
    )
    indice_receptor_fq = indices_por_codigo.get(
        FOGAREIRO_QUIXERAMOBIM_CENARIO_1["receptor_cod"]
    )
    if cenario_fq_ativo and (
        indice_controlador_fq is None or indice_receptor_fq is None
    ):
        raise HTTPException(
            status_code=400,
            detail="O cenário PGPS requer os reservatórios Fogareiro e Quixeramobim.",
        )

    for serie in series[1:]:
        if len(serie["datas"]) != n_meses or serie["datas"] != series[0]["datas"]:
            raise HTTPException(
                status_code=400,
                detail="As séries dos reservatórios precisam cobrir os mesmos meses.",
            )

    saidas = []
    for _ in range(n_res):
        saidas.append({
            "armazenamento_inicial": np.zeros(n_meses, dtype=float),
            "armazenamento_final": np.zeros(n_meses, dtype=float),
            "demanda_solicitada": np.zeros(n_meses, dtype=float),
            "demanda_atendida": np.zeros(n_meses, dtype=float),
            "retirada_total": np.zeros(n_meses, dtype=float),
            "racionamento": np.zeros(n_meses, dtype=float),
            "transferencia_recebida": np.zeros(n_meses, dtype=float),
            "transferencia_enviada": np.zeros(n_meses, dtype=float),
            "evaporacao_hm3": np.zeros(n_meses, dtype=float),
            "vertimento_hm3": np.zeros(n_meses, dtype=float),
            "falha": np.full(n_meses, "Não", dtype=object),
            "modo_operacao": np.full(n_meses, "Normal", dtype=object),
        })

    volumes_atuais = [float(p["vol_ini"]) for p in params]
    afluencias_hm3 = [serie["vazoes_m3s"] * (SEGUNDOS_MES_PADRAO / 1e6) for serie in series]
    evaporacoes_m = [serie["evaporacao_mm"] / 1000.0 for serie in series]

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

        if modo_id == "paralelo":
            alocacao_conjunta_bruta = [0.0] * n_res
            alocacao_conjunta_bruta[0] = vazao_conjunta

            for i in range(n_res - 1):
                p = params[i]
                vol_gatilho = float(p["capacidade"]) * (float(p["gatilho"]) / 100.0)
                carga_para_mover_bruta = alocacao_conjunta_bruta[i]

                if carga_para_mover_bruta > 0 and volumes_atuais[i] < vol_gatilho:
                    dem_esp_prox = responsabilidade_especifica[i + 1] * (1.0 - racionamentos[i + 1] / 100.0)
                    carga_conj_prox = carga_para_mover_bruta * (1.0 - racionamentos[i + 1] / 100.0)
                    demanda_total_prox_hm3 = (dem_esp_prox + carga_conj_prox) * (SEGUNDOS_MES_PADRAO / 1e6)
                    if prev_volumes_pos_natureza[i + 1] >= demanda_total_prox_hm3:
                        alocacao_conjunta_bruta[i] = 0.0
                        alocacao_conjunta_bruta[i + 1] += carga_para_mover_bruta

            for i in range(n_res):
                dem_esp = responsabilidade_especifica[i] * (1.0 - racionamentos[i] / 100.0)
                dem_conj = alocacao_conjunta_bruta[i] * (1.0 - racionamentos[i] / 100.0)
                demandas_finais[i] = dem_esp + dem_conj
                demandas_solicitadas_paralelo[i] = responsabilidade_especifica[i] + alocacao_conjunta_bruta[i]

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
                vol_gatilho = (
                    float(params[idx_receiver]["capacidade"])
                    * FOGAREIRO_QUIXERAMOBIM_CENARIO_1["gatilho_receptor_percent"]
                    / 100.0
                )
                if prev_volumes_pos_natureza[idx_receiver] < vol_gatilho:
                    transferencia_normal = (
                        FOGAREIRO_QUIXERAMOBIM_CENARIO_1["transferencias_lps"]["Normal"]
                    )
                    fator_estado = (
                        FOGAREIRO_QUIXERAMOBIM_CENARIO_1["transferencias_lps"][estado_sistema_fq]
                        / transferencia_normal
                    )
                    fluxo_enviado_alvo = (
                        vazao_conjunta * fator_estado * atendimento_transferencia
                    )
                    volume_enviado_alvo = fluxo_enviado_alvo * (SEGUNDOS_MES_PADRAO / 1e6)
                    volume_enviado = min(
                        volume_enviado_alvo,
                        max(prev_volumes_pos_natureza[idx_sender], 0.0),
                    )
                    volume_recebido = volume_enviado

                    prev_volumes_pos_natureza[idx_receiver] += volume_recebido
                    prev_volumes_pos_natureza[idx_sender] -= volume_enviado

                    transferencias_registradas[idx_receiver] = (
                        volume_recebido * (1e6 / SEGUNDOS_MES_PADRAO)
                    )
                    transferencias_enviadas[idx_sender] = (
                        volume_enviado * (1e6 / SEGUNDOS_MES_PADRAO)
                    )
            else:
                for i in range(1, n_res):
                    idx_sender = i
                    idx_receiver = i - 1
                    vol_gatilho = float(params[idx_receiver]["capacidade"]) * (
                        float(params[idx_receiver]["gatilho"]) / 100.0
                    )
                    if prev_volumes_pos_natureza[idx_receiver] < vol_gatilho:
                        fluxo_transferencia = vazao_conjunta * atendimento_transferencia
                        vol_demanda_hm3 = fluxo_transferencia * (SEGUNDOS_MES_PADRAO / 1e6)
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
            demanda_hm3 = demandas_finais[i] * (SEGUNDOS_MES_PADRAO / 1e6)

            if modo_id == "paralelo":
                saida["demanda_solicitada"][t] = demandas_solicitadas_paralelo[i]
            else:
                saida["demanda_solicitada"][t] = demandas_iniciais[i]
                saida["transferencia_recebida"][t] = transferencias_registradas[i]
                saida["transferencia_enviada"][t] = transferencias_enviadas[i]

            saida["armazenamento_inicial"][t] = vol_ini
            saida["racionamento"][t] = racionamentos[i]
            saida["modo_operacao"][t] = nomes_faixas_atuais[i]

            delta_transferencia_hm3 = 0.0
            if modo_id == "serie":
                delta_transferencia_hm3 = (
                    transferencias_registradas[i] - transferencias_enviadas[i]
                ) * (SEGUNDOS_MES_PADRAO / 1e6)

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
            saida["retirada_total"][t] = (
                saida["demanda_atendida"][t] + transferencias_enviadas[i]
            )
            saida["evaporacao_hm3"][t] = evap_hm3
            saida["vertimento_hm3"][t] = vertimento
            saida["armazenamento_final"][t] = vol_final
            volumes_atuais[i] = vol_final

        if modo_id in {"paralelo", "serie"} and falhas_do_mes and all(falhas_do_mes):
            for saida in saidas:
                saida["modo_operacao"][t] = "FALHA SISTÊMICA"

    return saidas
# rota que retorna a lista de todos os reservatÃ³rios cadastrados no banco
@app.get("/api/reservatorios")
def listar_reservatorios():
    if not os.path.exists(DB_PATH):
        raise HTTPException(status_code=500, detail="Base de dados nÃ£o encontrada.")
    conexao = sqlite3.connect(DB_PATH)
    df = normalizar_colunas_db(pd.read_sql_query(
        "SELECT * FROM acudes", conexao))
    conexao.close()
    df = df[["CORPO", "COD", "CAPAC (mÂ³)", "Est. Evap."]]
    # converte capacidade de mÂ³ pra hmÂ³
    df['CAPAC (mÂ³)'] = df['CAPAC (mÂ³)'] / 1e6
    df['capacidade_hm3'] = df['CAPAC (mÂ³)']
    df = normalizar_textos_db(df)
    df = df.replace({np.nan: None})
    return normalizar_registros_saida(df.to_dict(orient="records"))


# rota que retorna os hidrossistemas prÃ©-configurados (presets de simulaÃ§Ã£o)
@app.get("/api/presets")
def listar_presets():
    conexao = sqlite3.connect(DB_PATH)
    try:
        df_hidro = normalizar_textos_db(
            normalizar_colunas_db(pd.read_sql_query("SELECT * FROM hidrossistemas", conexao))
        )
        presets  = []
        for nome_sis, group in df_hidro.groupby('hidrossistema'):
            # detecta o modo de operaÃ§Ã£o pelo texto salvo no banco
            modo = group['operacao'].iloc[0]
            modo_operacao = ("Série"     if 'ser'   in str(modo).lower() else
                             "Paralelo" if 'paral' in str(modo).lower() else
                             "Individual")
            preset = {
                "nome":          nome_sis,
                "modo":          modo_operacao,
                "reservatorios": group['cod_acude'].astype(str).tolist()
            }
            codigos = set(preset["reservatorios"])
            if {"16", "119"}.issubset(codigos) and modo_operacao == "Série":
                preset.update({
                    "nome": "Fogareiro/Quixeramobim - PGPS Cenário 1",
                    "reservatorios": ["119", "16"],
                    "cenario_hidrossistema": FOGAREIRO_QUIXERAMOBIM_CENARIO_1_ID,
                    "fonte": "Plano de Gestão Proativa de Seca, cenário 1 escolhido",
                    "periodo": {
                        "mes_inicial": "JAN",
                        "ano_inicial": 1911,
                        "mes_final": "DEZ",
                        "ano_final": 2019,
                    },
                    "defaults": {
                        "16": {
                            "demanda_lps": 342.0,
                            "vol_inicial_percent": 100.0,
                            "gatilho_percent": 30.0,
                        },
                        "119": {
                            "demanda_lps": 272.0,
                            "vol_inicial_percent": 100.0,
                            "gatilho_percent": 0.0,
                        },
                    },
                    "niveis_meta": {
                        "reservatorio_cod": "119",
                        "faixas": faixas_fogareiro_quixeramobim_percentuais(),
                    },
                    "vazao_transferencia_lps": 500.0,
                    "atendimento_transferencia_percent": 100.0,
                    "transferencias_lps": FOGAREIRO_QUIXERAMOBIM_CENARIO_1["transferencias_lps"],
                })
            presets.append(preset)
        return presets
    finally:
        conexao.close()


# rota principal que executa a simulaÃ§Ã£o e devolve os resultados mÃªs a mÃªs
def float_seguro(valor, padrao=0.0):
    try:
        numero = float(valor)
    except (TypeError, ValueError):
        return float(padrao)
    return numero if np.isfinite(numero) else float(padrao)


def carregar_serie_simulador(conexao, reservatorio, req, fator_cenario):
    nomes_busca = [reservatorio.nome, texto_para_legado(reservatorio.nome)]
    linhas = conexao.execute(
        "SELECT * FROM vazoes WHERE nome_reservatorio IN (?, ?)",
        tuple(nomes_busca),
    ).fetchall()
    if not linhas:
        raise HTTPException(
            status_code=404,
            detail=f"Vazões não encontradas para {reservatorio.nome}",
        )

    mes_inicial = ordem_meses[req.mes_inicial]
    mes_final = ordem_meses[req.mes_final]
    periodo_inicial = int(req.ano_inicial) * 12 + mes_inicial
    periodo_final = int(req.ano_final) * 12 + mes_final
    registros = []

    for linha in linhas:
        ano = int(float_seguro(linha[1]))
        mes = corrigir_mojibake(str(linha[2])).upper()[:3]
        mes_num = ordem_meses.get(mes)
        if mes_num is None:
            continue
        periodo = ano * 12 + mes_num
        if periodo_inicial <= periodo <= periodo_final:
            registros.append((
                corrigir_mojibake(linha[0]),
                ano,
                mes,
                mes_num,
                float_seguro(linha[3]) * fator_cenario,
            ))

    registros.sort(key=lambda item: (item[1], item[3]))
    if not registros:
        raise HTTPException(
            status_code=404,
            detail=f"Não há vazões no período selecionado para {reservatorio.nome}.",
        )

    codigo_evap = str(reservatorio.est_evap).replace(".0", "")
    linha_evap = conexao.execute(
        "SELECT * FROM evaporacao WHERE COD = ? LIMIT 1",
        (codigo_evap,),
    ).fetchone()
    evap_mensal = np.zeros(12, dtype=float)
    if linha_evap is not None:
        evap_mensal[:] = [float_seguro(valor) for valor in linha_evap[2:14]]

    meses_num = np.fromiter((item[3] for item in registros), dtype=np.int16)
    return {
        "nome_reservatorio": registros[0][0],
        "anos": np.fromiter((item[1] for item in registros), dtype=np.int32),
        "meses": [item[2] for item in registros],
        "meses_num": meses_num,
        "datas": [f"{item[1]:04d}-{item[3]:02d}" for item in registros],
        "vazoes_m3s": np.fromiter((item[4] for item in registros), dtype=float),
        "evaporacao_mm": evap_mensal[meses_num - 1],
    }


def carregar_cav_simulador(conexao, reservatorio):
    linhas = conexao.execute(
        "SELECT * FROM cav WHERE COD = ?",
        (str(reservatorio.cod),),
    ).fetchall()
    if len(linhas) < 2:
        return (
            np.array([0.0, max(float(reservatorio.capacidade), 0.01)], dtype=float),
            np.array([0.0, 0.0], dtype=float),
        )
    return (
        np.fromiter((float_seguro(linha[2]) / 1e6 for linha in linhas), dtype=float),
        np.fromiter((float_seguro(linha[3]) for linha in linhas), dtype=float),
    )


def carregar_regras_simulador(conexao, reservatorio, usar_niveis_meta):
    if not usar_niveis_meta:
        return {}

    regras_mes = {}
    if reservatorio.plano_secas_custom:
        for mes in ordem_meses:
            regras = [
                (float(getattr(faixa, mes)), float(faixa.Racionamento), faixa.Faixa)
                for faixa in reservatorio.plano_secas_custom
            ]
            regras.sort(key=lambda item: item[0])
            regras_mes[mes] = regras
        return regras_mes

    linhas = conexao.execute(
        "SELECT * FROM plano_secas WHERE COD = ?",
        (str(reservatorio.cod),),
    ).fetchall()
    for indice_mes, mes in enumerate(ordem_meses):
        regras = [
            (
                float_seguro(linha[3 + indice_mes]),
                float_seguro(linha[2]),
                corrigir_mojibake(linha[1]),
            )
            for linha in linhas
        ]
        regras.sort(key=lambda item: item[0])
        regras_mes[mes] = regras
    return regras_mes


def montar_registros_simulador(serie, saida):
    registros = []
    for i, data in enumerate(serie["datas"]):
        registros.append({
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
            "Demanda Atendida (m³/s)": float(saida["demanda_atendida"][i]),
            "Retirada Total (m³/s)": float(saida["retirada_total"][i]),
            "Racionamento (%)": float(saida["racionamento"][i]),
            "Transferência Recebida (m³/s)": float(saida["transferencia_recebida"][i]),
            "Transferência Enviada (m³/s)": float(saida["transferencia_enviada"][i]),
            "Evaporação (hm³)": float(saida["evaporacao_hm3"][i]),
            "Vertimento (hm³)": float(saida["vertimento_hm3"][i]),
            "Falha": saida["falha"][i],
            "Modo Operação": saida["modo_operacao"][i],
            "Afluências (hm³/mês)": float(serie["vazoes_m3s"][i] * (SEGUNDOS_MES_PADRAO / 1e6)),
        })
    return normalizar_registros_saida(registros)


@app.post("/api/simular")
def processar_simulacao_api(req: SimulacaoRequest):
    if not req.reservatorios:
        raise HTTPException(status_code=400, detail="Informe ao menos um reservatório.")

    cenario_hidrologico = str(req.cenario_hidrologico or "historico")
    fator_cenario = {
        "historico": 1.0,
        "afluencia_zero": 0.0,
        "seco_50": 0.5,
        "umido_120": 1.2,
    }.get(cenario_hidrologico, 1.0)

    conexao = sqlite3.connect(DB_PATH)
    series = []
    params = []
    try:
        for reservatorio in req.reservatorios:
            serie = carregar_serie_simulador(conexao, reservatorio, req, fator_cenario)
            cav_vol, cav_area = carregar_cav_simulador(conexao, reservatorio)
            regras_mes = carregar_regras_simulador(
                conexao,
                reservatorio,
                req.usar_niveis_meta,
            )
            series.append(serie)
            params.append({
                "cod": str(reservatorio.cod),
                "cav_vol": cav_vol,
                "cav_area": cav_area,
                "regras_secas": regras_mes,
                "nome_faixa_normal": (
                    reservatorio.plano_secas_custom[0].NomeFaixaNormal
                    if reservatorio.plano_secas_custom
                    and reservatorio.plano_secas_custom[0].NomeFaixaNormal
                    else "Acima do Teto"
                ),
                "capacidade": float(reservatorio.capacidade),
                "vol_ini": float(reservatorio.vol_inicial),
                "demanda_nominal": float(reservatorio.demanda),
                "gatilho": float(reservatorio.gatilho),
            })
    finally:
        conexao.close()

    cenario_hidrossistema_ativo = (
        req.cenario_hidrossistema if req.usar_niveis_meta else None
    )
    saidas = simular_sistema_n(
        series,
        params,
        req.modo,
        req.vazao_conjunta,
        req.atendimento_transferencia,
        cenario_hidrossistema_ativo,
    )
    resultados = [
        {
            "reservatorio": req.reservatorios[i].nome,
            "dados": montar_registros_simulador(series[i], saidas[i]),
        }
        for i in range(len(series))
    ]
    return {
        "status": "sucesso",
        "resultados": resultados,
        "cenario_hidrossistema": cenario_hidrossistema_ativo,
    }


# rota que retorna o plano de secas (faixas de racionamento) de um aÃ§ude especÃ­fico
@app.post("/api/vazoes/permanencia")
def calcular_permanencias_api(req: PermanenciaRequest):
    df = carregar_serie_vazoes(req.reservatorio, req.mes_inicial, req.ano_inicial, req.mes_final, req.ano_final)
    params = carregar_parametros_regularizacao(req.reservatorio)
    valores_m3s = df["VazÃ£o (mÂ³/s)"].astype(float).to_numpy()
    aflu_hm3 = valores_m3s * 2.592
    evap_m = np.array([params["evap_mm"][int(m) - 1] / 1000.0 for m in df["mes_num"].astype(int).to_numpy()], dtype=float)

    resultados = []
    for q in range(1, 101):
        vazao, garantia_obtida, falhas = buscar_vazao_por_garantia(
            q / 100.0,
            aflu_hm3,
            evap_m,
            params["cap_hm3"],
            params["cav_vol"],
            params["cav_area"],
            req.vol_inicial_percent,
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
    curva = [{
        "garantia": row["garantia_requerida"],
        "vazao_m3s": row["vazao_m3s"],
        "falhas": row["falhas"],
    } for row in resultados]

    return {
        "status": "sucesso",
        "metodo": "Vazoes de garantia calculadas por garantia mensal: garantia = 1 - falhas/meses. Cada Qxx e a maior demanda constante atendida com a garantia requerida.",
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


@app.post("/api/vazoes/previsao-knn")
def prever_afluencia_knn_api(req: KnnRequest):
    df = carregar_serie_vazoes(req.reservatorio, req.mes_inicial, req.ano_inicial, req.mes_final, req.ano_final)
    valores = df["VazÃ£o (mÂ³/s)"].astype(float).to_numpy()
    meses = df["mes_num"].astype(int).to_numpy()

    lags = max(1, min(int(req.lags), 24))
    horizonte = max(1, min(int(req.horizonte), 36))
    k = max(1, min(int(req.k), 50))
    teste_meses = max(0, min(int(req.teste_meses), max(0, len(valores) - lags - 2)))

    if len(valores) < lags + 6:
        raise HTTPException(status_code=400, detail=f"Serie insuficiente para KNN com {lags} defasagens.")

    limite_treino = len(valores) - teste_meses
    if limite_treino <= lags + 1:
        limite_treino = len(valores)
        teste_meses = 0

    idx_treino = np.arange(lags, limite_treino)
    x_treino, y_treino = montar_features_knn(valores, meses, lags, idx_treino)
    if len(x_treino) == 0:
        raise HTTPException(status_code=400, detail="Nao foi possivel montar amostras de treino para o KNN.")

    validacao = []
    if teste_meses > 0:
        for i in range(limite_treino, len(valores)):
            mes_alvo = int(meses[i])
            sazonal = np.array([np.sin(2 * np.pi * mes_alvo / 12), np.cos(2 * np.pi * mes_alvo / 12)])
            feature = np.concatenate([valores[i - lags:i], sazonal])
            previsto = prever_knn(x_treino, y_treino, feature, k)
            observado = float(valores[i])
            erro = previsto - observado
            validacao.append({
                "data": df.loc[i, "Data"].strftime("%Y-%m"),
                "observado_m3s": round(observado, 6),
                "previsto_m3s": round(previsto, 6),
                "erro_m3s": round(erro, 6),
            })

    historico = [{
        "data": row["Data"].strftime("%Y-%m"),
        "vazao_m3s": round(float(row["VazÃ£o (mÂ³/s)"]), 6),
        "afluencia_hm3_mes": round(vazao_para_hm3_mes(float(row["VazÃ£o (mÂ³/s)"])), 6),
    } for _, row in df.iterrows()]

    x_total, y_total = montar_features_knn(valores, meses, lags, np.arange(lags, len(valores)))
    serie_expandida = valores.astype(float).tolist()
    ultima_data = df["Data"].max()
    previsao = []
    for passo in range(1, horizonte + 1):
        data_prevista = ultima_data + pd.DateOffset(months=passo)
        mes_alvo = int(data_prevista.month)
        sazonal = np.array([np.sin(2 * np.pi * mes_alvo / 12), np.cos(2 * np.pi * mes_alvo / 12)])
        feature = np.concatenate([np.array(serie_expandida[-lags:], dtype=float), sazonal])
        previsto = max(0.0, prever_knn(x_total, y_total, feature, k))
        serie_expandida.append(previsto)
        previsao.append({
            "data": data_prevista.strftime("%Y-%m"),
            "vazao_m3s": round(previsto, 6),
            "afluencia_hm3_mes": round(vazao_para_hm3_mes(previsto), 6),
        })

    metricas = {}
    if validacao:
        erros = np.array([v["erro_m3s"] for v in validacao], dtype=float)
        observados = np.array([v["observado_m3s"] for v in validacao], dtype=float)
        metricas = {
            "mae_m3s": round(float(np.mean(np.abs(erros))), 6),
            "rmse_m3s": round(float(np.sqrt(np.mean(erros ** 2))), 6),
            "mape_percent": round(float(np.mean(np.abs(erros) / np.maximum(np.abs(observados), 1e-9)) * 100), 4),
        }

    return {
        "status": "sucesso",
        "metodo": "KNN mensal com defasagens da vazao e sazonalidade do mes.",
        "reservatorio": req.reservatorio,
        "parametros": {
            "k": k,
            "lags": lags,
            "horizonte": horizonte,
            "teste_meses": teste_meses,
        },
        "periodo": {
            "inicio": df["Data"].min().strftime("%Y-%m"),
            "fim": df["Data"].max().strftime("%Y-%m"),
            "meses": int(len(df)),
        },
        "metricas": metricas,
        "historico": historico,
        "validacao": validacao,
        "previsao": previsao,
    }


@app.get("/api/plano-secas/{cod_acude}")
def obter_plano_secas(cod_acude: str):
    try:
        conexao = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(
            "SELECT * FROM plano_secas WHERE COD = ?",
            conexao, params=(cod_acude,))
        conexao.close()
        df = df.replace({np.nan: None})
        # renomeia a coluna pra simplificar o nome no JSON de resposta
        if "Racionamento (%)" in df.columns:
            df.rename(columns={"Racionamento (%)": "Racionamento"}, inplace=True)
        return normalizar_registros_saida(df.to_dict(orient="records"))
    except Exception:
        return []
