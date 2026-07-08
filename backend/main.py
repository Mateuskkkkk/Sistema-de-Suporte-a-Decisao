# importações necessárias pra API funcionar
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import sqlite3
import os
from scipy import interpolate
from typing import List, Dict, Optional
from optimizer_engine import router as otimizador_router, dinamica_mensal_fast

# cria a aplicação FastAPI
app = FastAPI(title="API do Simulador Hidrológico", version="1.0")

# libera acesso de qualquer origem (CORS aberto)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(otimizador_router)

# caminho do banco de dados SQLite na mesma pasta do script
DB_PATH = os.path.join(os.path.abspath("."), "banco_site.db")


def normalizar_colunas_db(df: pd.DataFrame) -> pd.DataFrame:
    renomear = {}
    for col in df.columns:
        if col.startswith("CAPAC"):
            renomear[col] = "CAPAC (m³)"
        elif col.startswith("VOLUME"):
            renomear[col] = "VOLUME (m³)"
        elif col.startswith("AREA"):
            renomear[col] = "AREA (km²)"
        elif col.startswith("Vaz"):
            renomear[col] = "Vazão (m³/s)"
        elif col.startswith("M"):
            renomear[col] = "Mês"
        elif col.startswith("opera"):
            renomear[col] = "operacao"
    return df.rename(columns=renomear)


def corrigir_texto_db(valor):
    if not isinstance(valor, str):
        return valor
    if not any(marca in valor for marca in ("Ã", "Â", "â")):
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

# dicionário pra converter nome do mês em número (JAN=1, FEV=2, ...)
ordem_meses = {
    'JAN': 1, 'FEV': 2, 'MAR': 3, 'ABR': 4, 'MAI': 5, 'JUN': 6,
    'JUL': 7, 'AGO': 8, 'SET': 9, 'OUT': 10, 'NOV': 11, 'DEZ': 12
}



# recebe os limites mensais (% do volume) e o racionamento de cada nível meta
class FaixaCustom(BaseModel):
    Faixa: str
    Racionamento: float
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


# modelo de dados de um reservatório individual
class Reservatorio(BaseModel):
    nome: str
    cod: str
    capacidade: float
    est_evap: str
    vol_inicial: float
    demanda: float
    gatilho: float
    plano_secas_custom: Optional[List[FaixaCustom]] = None  # faixas editadas na sessão do frontend


# modelo de dados que o front manda pra iniciar uma simulação
class SimulacaoRequest(BaseModel):
    reservatorios: List[Reservatorio]
    modo: str
    vazao_conjunta: float
    mes_inicial: str
    ano_inicial: int
    mes_final: str
    ano_final: int


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
        raise HTTPException(status_code=500, detail="Base de dados não encontrada.")
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
        raise HTTPException(status_code=404, detail=f"Vazões não encontradas para {reservatorio}")

    df = normalizar_textos_db(df)
    df["mes_num"] = df["Mês"].map(lambda m: ordem_meses.get(str(m).upper()[:3], None))
    df = df.dropna(subset=["mes_num", "Vazão (m³/s)", "Ano"]).copy()
    df["mes_num"] = df["mes_num"].astype(int)
    df["Ano"] = df["Ano"].astype(int)
    df["Vazão (m³/s)"] = pd.to_numeric(df["Vazão (m³/s)"], errors="coerce")
    df = df.dropna(subset=["Vazão (m³/s)"])
    df["Data"] = pd.to_datetime(df["Ano"].astype(str) + "-" + df["mes_num"].astype(str) + "-01")

    data_inicio = pd.to_datetime(f"{ano_ini}-{mes_ini}-01")
    data_fim = pd.to_datetime(f"{ano_fim}-{mes_fim}-01") + pd.offsets.MonthEnd(0)
    df = df[(df["Data"] >= data_inicio) & (df["Data"] <= data_fim)].sort_values("Data").reset_index(drop=True)
    if df.empty:
        raise HTTPException(status_code=404, detail="Não há vazões no período selecionado.")
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
            raise HTTPException(status_code=404, detail=f"Reservatório não encontrado: {reservatorio}")

        acude = df_acudes.iloc[0]
        cod = acude["COD"]
        cap_hm3 = float(acude["CAPAC (m³)"]) / 1e6
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
            df_cav = df_cav.sort_values("VOLUME (m³)")
            cav_vol = df_cav["VOLUME (m³)"].astype(float).to_numpy() / 1e6
            cav_area = df_cav["AREA (km²)"].astype(float).to_numpy()

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


# função principal que roda a simulação mês a mês pra todos os reservatórios
# recebe os dataframes com as vazões, os parâmetros de cada açude, o modo de operação
# (Série, Paralelo ou Individual) e a vazão conjunta do sistema
# função principal que roda a simulação mês a mês pra todos os reservatórios
def simular_sistema_n(dfs, params, modo, vazao_conjunta):
    n_res    = len(dfs)
    n_meses  = len(dfs[0])
    segundos_mes = 2.592e6  

    colunas_init = [
        'Armazenamento Inicial', 'Armazenamento Final',
        'Demanda Solicitada (m³/s)', 'Demanda Atendida (m³/s)',
        'Racionamento (%)', 'Transferência Recebida (m³/s)',
        'Transferência Enviada (m³/s)', 'Evaporação (hm³)',
        'Vertimento (hm³)', 'Falha', 'Modo Operação'
    ]
    for df in dfs:
        for col in colunas_init:
            df[col] = 0.0
        df['Falha']        = 'Não'
        df['Modo Operação'] = 'Normal'

    volumes_atuais = [p['vol_ini'] for p in params]

    for t in range(n_meses):
        demandas_iniciais        = []
        racionamentos            = []
        nomes_faixas_atuais      = []
        prev_volumes_pos_natureza = []

        for i in range(n_res):
            p       = params[i]
            vol_ini = volumes_atuais[i]
            pct_vol = (vol_ini / p['capacidade']) * 100 
            mes_atual = dfs[i].loc[t, 'Mês']

            rac        = 0.0
            nome_faixa = "Normal"
            if p['regras_secas']:
                regras = p['regras_secas'].get(mes_atual, [])
                if regras:
                    nome_faixa = "Acima do Teto"
                    for lim, r_val, n_faixa in regras:
                        if pct_vol <= lim:
                            rac        = r_val
                            nome_faixa = n_faixa
                            break

            demandas_iniciais.append(p['demanda_nominal'])
            racionamentos.append(rac)
            nomes_faixas_atuais.append(nome_faixa)

            afluencia_hm3 = dfs[i].loc[t, 'Vazão (m³/s)'] * (segundos_mes / 1e6)
            evap_taxa_m   = float(dfs[i].loc[t, 'Evaporação (m)']) / 1000.0

            vol_pos_natureza, _, _, evap_hm3 = dinamica_mensal_fast(
                float(vol_ini),
                float(afluencia_hm3),
                float(evap_taxa_m),
                0.0,
                0.0,
                float(p['capacidade']),
                p['cav_vol'],
                p['cav_area'],
            )

            dfs[i].loc[t, 'Evaporação (hm³)']    = evap_hm3
            dfs[i].loc[t, 'Afluências (hm³/mês)'] = afluencia_hm3

            prev_volumes_pos_natureza.append(vol_pos_natureza)

        demandas_finais            = [0.0] * n_res
        transferencias_registradas = [0.0] * n_res
        transferencias_enviadas    = [0.0] * n_res
        demandas_solicitadas_paralelo = [0.0] * n_res

        responsabilidade_especifica = [p['demanda_nominal'] for p in params]

        if modo == "Paralelo":
            alocacao_conjunta_bruta = [0.0] * n_res
            if n_res > 0:
                alocacao_conjunta_bruta[0] = vazao_conjunta

            for i in range(n_res - 1):
                p = params[i]
                vol_gatilho          = p['capacidade'] * (p['gatilho'] / 100)
                carga_para_mover_bruta = alocacao_conjunta_bruta[i]

                if carga_para_mover_bruta > 0 and volumes_atuais[i] < vol_gatilho:
                    dem_esp_prox_teorica   = responsabilidade_especifica[i+1] * (1 - racionamentos[i+1] / 100.0)
                    carga_conj_prox_racionada = carga_para_mover_bruta * (1 - racionamentos[i+1] / 100.0)
                    demanda_total_prox_hm3 = (dem_esp_prox_teorica + carga_conj_prox_racionada) * (segundos_mes / 1e6)

                    if prev_volumes_pos_natureza[i+1] >= demanda_total_prox_hm3:
                        alocacao_conjunta_bruta[i]     = 0.0
                        alocacao_conjunta_bruta[i+1]  += carga_para_mover_bruta

            for k in range(n_res):
                dem_esp  = responsabilidade_especifica[k] * (1 - racionamentos[k] / 100.0)
                dem_conj = alocacao_conjunta_bruta[k]     * (1 - racionamentos[k] / 100.0)
                demandas_finais[k] = dem_esp + dem_conj

            demandas_solicitadas_paralelo = [
                responsabilidade_especifica[k] + alocacao_conjunta_bruta[k]
                for k in range(n_res)
            ]

            for k in range(1, n_res):
                if alocacao_conjunta_bruta[k] > 0:
                    val = alocacao_conjunta_bruta[k] * (1 - racionamentos[k] / 100.0)
                    transferencias_registradas[k]   = val
                    transferencias_enviadas[k - 1]  = val

        elif modo == "Série":
            for k in range(n_res):
                base_demand = demandas_iniciais[k] + (vazao_conjunta if k == 0 else 0)
                demandas_finais[k] = base_demand * (1 - racionamentos[k] / 100.0)

            for i in range(1, n_res):
                idx_sender   = i
                idx_receiver = i - 1
                vol_gatilho_A = params[idx_receiver]['capacidade'] * (params[idx_receiver]['gatilho'] / 100.0)

                if prev_volumes_pos_natureza[idx_receiver] < vol_gatilho_A:
                    vol_demanda_hm3    = demandas_finais[idx_receiver] * (segundos_mes / 1e6)
                    disponivel_sender  = prev_volumes_pos_natureza[idx_sender]
                    qtd_transferir_hm3 = min(vol_demanda_hm3, disponivel_sender)

                    prev_volumes_pos_natureza[idx_receiver] += qtd_transferir_hm3
                    prev_volumes_pos_natureza[idx_sender]   -= qtd_transferir_hm3

                    fluxo_transf = qtd_transferir_hm3 * (1e6 / segundos_mes)
                    transferencias_registradas[idx_receiver] += fluxo_transf
                    transferencias_enviadas[idx_sender]       += fluxo_transf
        else:
            for k in range(n_res):
                demandas_finais[k] = demandas_iniciais[k] * (1 - racionamentos[k] / 100.0)

        falhas_do_mes = []

        for i in range(n_res):
            p   = params[i]
            df  = dfs[i]
            vol_ini     = volumes_atuais[i]
            
            demanda_hm3 = demandas_finais[i] * (segundos_mes / 1e6)

            if modo == "Paralelo":
                df.loc[t, 'Demanda Solicitada (m³/s)']     = demandas_solicitadas_paralelo[i]
                df.loc[t, 'Transferência Recebida (m³/s)'] = 0.0
                df.loc[t, 'Transferência Enviada (m³/s)']  = 0.0
            else:
                df.loc[t, 'Demanda Solicitada (m³/s)']     = demandas_iniciais[i] + (vazao_conjunta if i == 0 else 0.0)
                df.loc[t, 'Transferência Recebida (m³/s)'] = transferencias_registradas[i]
                df.loc[t, 'Transferência Enviada (m³/s)']  = transferencias_enviadas[i]

            df.loc[t, 'Armazenamento Inicial'] = vol_ini
            df.loc[t, 'Racionamento (%)']      = racionamentos[i]
            df.loc[t, 'Modo Operação']         = nomes_faixas_atuais[i]

            delta_transferencia_hm3 = 0.0
            if modo == "Série":
                delta_transferencia_hm3 = (
                    transferencias_registradas[i] - transferencias_enviadas[i]
                ) * (segundos_mes / 1e6)

            evap_taxa_m = float(df.loc[t, 'Evaporação (m)']) / 1000.0
            vol_final, demanda_atendida_real_hm3, vertimento, evap_hm3 = dinamica_mensal_fast(
                float(vol_ini),
                float(df.loc[t, 'Afluências (hm³/mês)'] + delta_transferencia_hm3),
                float(evap_taxa_m),
                float(demanda_hm3),
                0.0,
                float(p['capacidade']),
                p['cav_vol'],
                p['cav_area'],
            )

            if round(demanda_atendida_real_hm3, 6) < round(demanda_hm3, 6):
                df.loc[t, 'Falha'] = 'Sim'
                falhas_do_mes.append(True)
            else:
                df.loc[t, 'Falha'] = 'Não'
                falhas_do_mes.append(False)

            df.loc[t, 'Demanda Atendida (m³/s)'] = demanda_atendida_real_hm3 * (1e6 / segundos_mes)

            df.loc[t, 'Evaporação (hm³)']   = evap_hm3
            df.loc[t, 'Vertimento (hm³)']   = vertimento
            df.loc[t, 'Armazenamento Final'] = vol_final
            volumes_atuais[i]                = vol_final 

        # Carimba visualmente a operação falha no sistema se todos caíram
        if modo in ["Paralelo", "Série"] and all(falhas_do_mes) and len(falhas_do_mes) > 0:
            for i in range(n_res):
                dfs[i].loc[t, 'Modo Operação'] = 'FALHA SISTÊMICA'

    return dfs
# rota que retorna a lista de todos os reservatórios cadastrados no banco
@app.get("/api/reservatorios")
def listar_reservatorios():
    if not os.path.exists(DB_PATH):
        raise HTTPException(status_code=500, detail="Base de dados não encontrada.")
    conexao = sqlite3.connect(DB_PATH)
    df = normalizar_colunas_db(pd.read_sql_query(
        "SELECT * FROM acudes", conexao))
    conexao.close()
    df = df[["CORPO", "COD", "CAPAC (m³)", "Est. Evap."]]
    # converte capacidade de m³ pra hm³
    df['CAPAC (m³)'] = df['CAPAC (m³)'] / 1e6
    df['capacidade_hm3'] = df['CAPAC (m³)']
    df = normalizar_textos_db(df)
    df = df.replace({np.nan: None})
    return df.to_dict(orient="records")


# rota que retorna os hidrossistemas pré-configurados (presets de simulação)
@app.get("/api/presets")
def listar_presets():
    conexao = sqlite3.connect(DB_PATH)
    try:
        df_hidro = normalizar_textos_db(
            normalizar_colunas_db(pd.read_sql_query("SELECT * FROM hidrossistemas", conexao))
        )
        presets  = []
        for nome_sis, group in df_hidro.groupby('hidrossistema'):
            # detecta o modo de operação pelo texto salvo no banco
            modo = group['operacao'].iloc[0]
            modo_operacao = ("Série"    if 'ser'   in str(modo).lower() else
                             "Paralelo" if 'paral' in str(modo).lower() else
                             "Individual")
            presets.append({
                "nome":          nome_sis,
                "modo":          modo_operacao,
                "reservatorios": group['cod_acude'].astype(str).tolist()
            })
        return presets
    finally:
        conexao.close()


# rota principal que executa a simulação e devolve os resultados mês a mês
@app.post("/api/simular")
def processar_simulacao_api(req: SimulacaoRequest):
    conexao          = sqlite3.connect(DB_PATH)
    lista_dfs_input  = []
    lista_params     = []

    # carrega tabelas auxiliares uma vez só
    df_evap  = pd.read_sql_query("SELECT * FROM evaporacao", conexao)
    df_cav   = normalizar_colunas_db(pd.read_sql_query("SELECT * FROM cav", conexao))
    df_plano = pd.read_sql_query("SELECT * FROM plano_secas", conexao)

    for res in req.reservatorios:
        # busca as vazões históricas do reservatório
        nomes_busca = [res.nome, texto_para_legado(res.nome)]
        df_vazoes = normalizar_colunas_db(pd.read_sql_query(
            "SELECT * FROM vazoes WHERE nome_reservatorio IN (?, ?)",
            conexao, params=tuple(nomes_busca)))

        if df_vazoes.empty:
            conexao.close()
            raise HTTPException(status_code=404,
                                detail=f"Vazões não encontradas para {res.nome}")

        # monta coluna de data pra filtrar pelo período solicitado
        df_vazoes['Ordem_Mês'] = df_vazoes['Mês'].map(ordem_meses)
        df_vazoes['Data'] = pd.to_datetime(
            df_vazoes['Ano'].astype(str) + '-' +
            df_vazoes['Ordem_Mês'].astype(str) + '-01')

        mi, mf = ordem_meses[req.mes_inicial], ordem_meses[req.mes_final]
        d_ini  = pd.to_datetime(f"{req.ano_inicial}-{mi}-01")
        d_fim  = pd.to_datetime(f"{req.ano_final}-{mf}-01") + pd.offsets.MonthEnd(0)

        # filtra só o período pedido e ordena por data
        df_long = (df_vazoes[(df_vazoes['Data'] >= d_ini) & (df_vazoes['Data'] <= d_fim)]
                   .sort_values('Data').reset_index(drop=True))

        # adiciona a evaporação mensal correspondente a cada linha
        evap_row = df_evap[df_evap["COD"] == str(res.est_evap).replace('.0', '')]
        df_long["Evaporação (m)"] = (
            df_long["Mês"].map(evap_row.iloc[0][list(ordem_meses.keys())])
            if not evap_row.empty else 0.0)

        # monta função de interpolação volume → área usando a curva cota-área-volume (CAV)
        cav_res = df_cav[df_cav["COD"] == str(res.cod)]
        if len(cav_res) < 2:
            x_vol = np.array([0.0, max(float(res.capacidade), 0.01)])
            y_area = np.array([0.0, 0.0])
            func_interp = lambda v: 0.0  # sem dados suficientes, retorna área zero
        else:
            x_vol  = (cav_res["VOLUME (m³)"].astype(float).values / 1e6).astype(np.float64)
            y_area = cav_res["AREA (km²)"].astype(float).values.astype(np.float64)
            
            func_interp = interpolate.interp1d(
                x_vol, 
                y_area, 
                kind='linear',
                bounds_error=False, 
                fill_value=(float(y_area[0]), float(y_area[-1]))
            )
                
        # Prioridade: faixas customizadas enviadas pelo frontend (editadas na sessão)
        # Fallback: dados do banco de dados
        regras_mes = {}

        if res.plano_secas_custom:
            # usa as faixas que o usuário editou na sessão do frontend
            for m in ordem_meses.keys():
                regras = [
                    (getattr(f, m), f.Racionamento, f.Faixa)
                    for f in res.plano_secas_custom
                ]
                regras.sort(key=lambda x: x[0])
                regras_mes[m] = regras
        else:
            # comportamento original: busca do banco de dados
            plano_res = df_plano[df_plano['COD'].astype(str) == str(res.cod)]
            if not plano_res.empty:
                for m in ordem_meses.keys():
                    regras = [(row[m], row['Racionamento (%)'], row['Faixa'])
                              for _, row in plano_res.iterrows()]
                    regras.sort(key=lambda x: x[0])
                    regras_mes[m] = regras

        lista_dfs_input.append(df_long)
        lista_params.append({
            'func_area':       func_interp,
            'cav_vol':         x_vol,
            'cav_area':        y_area,
            'regras_secas':    regras_mes,
            'capacidade':      res.capacidade,
            'vol_ini':         res.vol_inicial,
            'demanda_nominal': res.demanda,
            'gatilho':         res.gatilho
        })

    conexao.close()

    # roda a simulação com todos os reservatórios
    dfs_resultados = simular_sistema_n(lista_dfs_input, lista_params,
                                       req.modo, req.vazao_conjunta)

    # formata os resultados e devolve como JSON
    resultados_json = []
    for i, df in enumerate(dfs_resultados):
        df_limpo = df.replace({np.nan: None})
        df_limpo['Data'] = df_limpo['Data'].dt.strftime('%Y-%m')
        resultados_json.append({
            "reservatorio": req.reservatorios[i].nome,
            "dados":        df_limpo.to_dict(orient="records")
        })

    return {"status": "sucesso", "resultados": resultados_json}


# rota que retorna o plano de secas (faixas de racionamento) de um açude específico
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
    valores = df["Vazão (m³/s)"].astype(float).to_numpy()
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
        "vazao_m3s": round(float(row["Vazão (m³/s)"]), 6),
        "afluencia_hm3_mes": round(vazao_para_hm3_mes(float(row["Vazão (m³/s)"])), 6),
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
        return df.to_dict(orient="records")
    except Exception:
        return []
