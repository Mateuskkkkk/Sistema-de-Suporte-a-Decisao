from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import sqlite3
import os
from scipy import interpolate
from scipy.interpolate import PchipInterpolator
from typing import List, Dict, Optional

# ── DEAP (algoritmo genético NSGA-II) ────────────────────────────────────────
from deap import base, creator, tools, algorithms
import random

app = FastAPI(title="API do Simulador Hidrológico", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = os.path.join(os.path.abspath("."), "banco_site.db")

ordem_meses = {
    'JAN': 1, 'FEV': 2, 'MAR': 3, 'ABR': 4, 'MAI': 5, 'JUN': 6,
    'JUL': 7, 'AGO': 8, 'SET': 9, 'OUT': 10, 'NOV': 11, 'DEZ': 12
}

# ─────────────────────────────────────────────────────────────────────────────
# MODELOS DE DADOS
# ─────────────────────────────────────────────────────────────────────────────

class Reservatorio(BaseModel):
    nome: str
    cod: str
    capacidade: float
    est_evap: str
    vol_inicial: float
    demanda: float
    gatilho: float

class SimulacaoRequest(BaseModel):
    reservatorios: List[Reservatorio]
    modo: str
    vazao_conjunta: float
    mes_inicial: str
    ano_inicial: int
    mes_final: str
    ano_final: int

class OtimizacaoRequest(BaseModel):
    """Parâmetros para o optimizador NSGA-II de curvas guia."""
    cod: str
    ano_inicial: int
    ano_final: int
    demanda_alvo: float           # m³/s
    vol_inicial_pct: float = 50.0 # % da capacidade para iniciar
    n_gen: int = 80               # número de gerações
    pop_size: int = 200           # tamanho da população (múltiplo de 4)

# ─────────────────────────────────────────────────────────────────────────────
# FUNÇÕES MATEMÁTICAS ORIGINAIS
# ─────────────────────────────────────────────────────────────────────────────

def obter_k_dinamico(area_km2):
    ha = area_km2 * 100.0
    if ha <= 5:    return 0.90
    elif ha <= 10: return 0.85
    elif ha <= 20: return 0.80
    elif ha <= 50: return 0.75
    else:          return 0.70

def simular_sistema_n(dfs, params, modo, vazao_conjunta):
    """Motor original de simulação Pandas — mantido sem alterações."""
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

    volumes_atueis = [p['vol_ini'] for p in params]

    for t in range(n_meses):
        demandas_iniciais        = []
        racionamentos            = []
        nomes_faixas_atuais      = []
        prev_volumes_pos_natureza = []

        for i in range(n_res):
            p       = params[i]
            vol_ini = volumes_atueis[i]
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

            area          = p['func_area'](vol_ini)
            kp_dinamico   = obter_k_dinamico(area)
            evap_tanque_mm = dfs[i].loc[t, 'Evaporação (m)']
            evap_hm3      = (evap_tanque_mm * kp_dinamico * area) / 1000.0
            afluencia_hm3 = dfs[i].loc[t, 'Vazão (m³/s)'] * (segundos_mes / 1e6)

            dfs[i].loc[t, 'Evaporação (hm³)']    = evap_hm3
            dfs[i].loc[t, 'Afluências (hm³/mês)'] = afluencia_hm3

            vol_pos_natureza = max(0.0, vol_ini + afluencia_hm3 - evap_hm3)
            prev_volumes_pos_natureza.append(vol_pos_natureza)

        total_vol_disponivel = sum(prev_volumes_pos_natureza)

        rac_inicial_conjunta      = racionamentos[0] if racionamentos else 0.0
        demanda_conjunta_estimada = vazao_conjunta * (1 - rac_inicial_conjunta / 100.0) * (segundos_mes / 1e6)

        total_demanda_necessaria = demanda_conjunta_estimada
        for i in range(n_res):
            dem_esp_hm3 = params[i]['demanda_nominal'] * (1 - racionamentos[i] / 100.0) * (segundos_mes / 1e6)
            total_demanda_necessaria += dem_esp_hm3

        sistema_em_falha = False
        if modo == "Paralelo" and total_vol_disponivel < total_demanda_necessaria:
            sistema_em_falha = True
            for i in range(n_res):
                dfs[i].loc[t, 'Falha']        = 'Sim'
                dfs[i].loc[t, 'Modo Operação'] = 'FALHA SISTÊMICA'

        demandas_finais           = [0.0] * n_res
        transferencias_registradas = [0.0] * n_res
        transferencias_enviadas    = [0.0] * n_res

        if not sistema_em_falha:
            responsabilidade_especifica = [p['demanda_nominal'] for p in params]

            if modo == "Paralelo":
                alocacao_conjunta_bruta = [0.0] * n_res
                if n_res > 0:
                    alocacao_conjunta_bruta[0] = vazao_conjunta

                for i in range(n_res - 1):
                    p = params[i]
                    vol_gatilho          = p['capacidade'] * (p['gatilho'] / 100)
                    carga_para_mover_bruta = alocacao_conjunta_bruta[i]

                    if carga_para_mover_bruta > 0 and volumes_atueis[i] < vol_gatilho:
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
        else:
            for k in range(n_res):
                base = demandas_iniciais[k] + (vazao_conjunta if modo == "Paralelo" and k == 0 else 0)
                demandas_finais[k] = base * (1 - racionamentos[k] / 100.0)
            if modo == "Paralelo":
                demandas_solicitadas_paralelo = [
                    demandas_iniciais[k] + (vazao_conjunta if k == 0 else 0.0)
                    for k in range(n_res)
                ]

        for i in range(n_res):
            p   = params[i]
            df  = dfs[i]
            vol_ini     = volumes_atueis[i]
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
            if not sistema_em_falha:
                df.loc[t, 'Modo Operação'] = nomes_faixas_atuais[i]

            vol_disp = (prev_volumes_pos_natureza[i] if modo == "Série"
                        else vol_ini + dfs[i].loc[t, 'Afluências (hm³/mês)'] - dfs[i].loc[t, 'Evaporação (hm³)'])

            if sistema_em_falha:
                demanda_atendida_real_hm3 = max(0, min(vol_disp, demanda_hm3))
            else:
                if vol_disp < demanda_hm3:
                    demanda_atendida_real_hm3 = max(0, vol_disp)
                    df.loc[t, 'Falha'] = 'Sim'
                else:
                    demanda_atendida_real_hm3 = demanda_hm3

            df.loc[t, 'Demanda Atendida (m³/s)'] = demanda_atendida_real_hm3 * (1e6 / segundos_mes)

            vol_final  = vol_disp - demanda_atendida_real_hm3
            vertimento = 0.0
            if vol_final > p['capacidade']:
                vertimento = vol_final - p['capacidade']
                vol_final  = p['capacidade']
            vol_final = max(0.0, vol_final)

            df.loc[t, 'Vertimento (hm³)']   = vertimento
            df.loc[t, 'Armazenamento Final'] = vol_final
            volumes_atueis[i]                = vol_final

    return dfs

# ─────────────────────────────────────────────────────────────────────────────
# SIMULADOR NUMPY — usado exclusivamente dentro do DEAP (zero Pandas no loop)
# ─────────────────────────────────────────────────────────────────────────────

def simular_numpy(individuo, vazoes_np, evap_np, areas_np, kp_np,
                  capacidade_hm3, vol_inicial_hm3, demanda_alvo_hm3):
    """
    Balanço hídrico mês-a-mês em NumPy puro para a função de fitness do DEAP.

    Cromossomo (48 genes):
        [0:12]  → Normal      (% cap)
        [12:24] → Atenção     (% cap)
        [24:36] → Seca        (% cap)
        [36:48] → Seca Severa (% cap)

    Factores de liberação:
        Vol ≥ Normal      → 100%
        Normal > Vol ≥ Atenção  → 80%
        Atenção > Vol ≥ Seca    → 60%
        Seca > Vol ≥ Seca Sev.  → 30%
        Vol < Seca Severa       → 10%

    Retorna (F1, F2):
        F1 = Σ (demanda_alvo − liberado)²
        F2 = Σ vertimento
    """
    ind = individuo   # lista de 48 floats

    n_meses = len(vazoes_np)
    vol     = vol_inicial_hm3
    f1      = 0.0
    f2      = 0.0

    FATORES = (1.0, 0.80, 0.60, 0.30, 0.10)

    for t in range(n_meses):
        m = t % 12   # índice do mês (0 = JAN)

        # Evaporação
        area = float(np.interp(vol, areas_np[0], areas_np[1]))
        evap = (evap_np[t] * kp_np[t] * area) / 1000.0   # hm³
        vol_pos = max(0.0, vol + vazoes_np[t] - evap)

        # Nível da curva guia pelo volume actual em %
        pct = (vol_pos / capacidade_hm3) * 100.0 if capacidade_hm3 > 0 else 0.0

        if   pct >= ind[m]:       fator = FATORES[0]
        elif pct >= ind[12 + m]:  fator = FATORES[1]
        elif pct >= ind[24 + m]:  fator = FATORES[2]
        elif pct >= ind[36 + m]:  fator = FATORES[3]
        else:                     fator = FATORES[4]

        liberado = min(vol_pos, demanda_alvo_hm3 * fator)
        deficit  = demanda_alvo_hm3 - liberado
        f1      += deficit * deficit

        vol_pos -= liberado
        if vol_pos > capacidade_hm3:
            f2     += vol_pos - capacidade_hm3
            vol_pos = capacidade_hm3

        vol = max(0.0, vol_pos)

    return f1, f2


# ─────────────────────────────────────────────────────────────────────────────
# FUNÇÃO REPAIR — garante Normal ≥ Atenção ≥ Seca ≥ Seca Severa por mês
# ─────────────────────────────────────────────────────────────────────────────

def repair(individuo):
    """Ordena decrescentemente os 4 limites de cada mês."""
    for m in range(12):
        vals = sorted(
            [individuo[m], individuo[12+m], individuo[24+m], individuo[36+m]],
            reverse=True
        )
        individuo[m]      = vals[0]
        individuo[12 + m] = vals[1]
        individuo[24 + m] = vals[2]
        individuo[36 + m] = vals[3]
    return individuo


# ─────────────────────────────────────────────────────────────────────────────
# SETUP DEAP — executado uma vez no arranque do servidor
# ─────────────────────────────────────────────────────────────────────────────

if not hasattr(creator, "FitnessMulti"):
    creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMulti)

_toolbox = base.Toolbox()

def _criar_individuo():
    ind = [random.uniform(0.0, 100.0) for _ in range(48)]
    repair(ind)
    return creator.Individual(ind)

_toolbox.register("individual", _criar_individuo)
_toolbox.register("population", tools.initRepeat, list, _toolbox.individual)
_toolbox.register("select",     tools.selNSGA2)
_toolbox.register("mate",       tools.cxSimulatedBinaryBounded,
                  low=0.0, up=100.0, eta=20.0)
_toolbox.register("mutate",     tools.mutPolynomialBounded,
                  low=0.0, up=100.0, eta=20.0, indpb=1.0 / 48.0)


# ─────────────────────────────────────────────────────────────────────────────
# ROTAS ORIGINAIS — mantidas sem alterações
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/reservatorios")
def listar_reservatorios():
    if not os.path.exists(DB_PATH):
        raise HTTPException(status_code=500, detail="Base de dados não encontrada.")
    conexao = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT CORPO, COD, [CAPAC (m³)], [Est. Evap.] FROM acudes", conexao)
    conexao.close()
    df['CAPAC (m³)'] = df['CAPAC (m³)'] / 1e6
    df = df.replace({np.nan: None})
    return df.to_dict(orient="records")


@app.get("/api/presets")
def listar_presets():
    conexao = sqlite3.connect(DB_PATH)
    try:
        df_hidro = pd.read_sql_query("SELECT * FROM hidrossistemas", conexao)
        presets  = []
        for nome_sis, group in df_hidro.groupby('hidrossistema'):
            modo = group['operação'].iloc[0]
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


@app.post("/api/simular")
def processar_simulacao_api(req: SimulacaoRequest):
    conexao          = sqlite3.connect(DB_PATH)
    lista_dfs_input  = []
    lista_params     = []

    df_evap  = pd.read_sql_query("SELECT * FROM evaporacao", conexao)
    df_cav   = pd.read_sql_query("SELECT * FROM cav", conexao)
    df_plano = pd.read_sql_query("SELECT * FROM plano_secas", conexao)

    for res in req.reservatorios:
        df_vazoes = pd.read_sql_query(
            "SELECT * FROM vazoes WHERE nome_reservatorio = ?",
            conexao, params=(res.nome,))

        if df_vazoes.empty:
            conexao.close()
            raise HTTPException(status_code=404,
                                detail=f"Vazões não encontradas para {res.nome}")

        df_vazoes['Ordem_Mês'] = df_vazoes['Mês'].map(ordem_meses)
        df_vazoes['Data'] = pd.to_datetime(
            df_vazoes['Ano'].astype(str) + '-' +
            df_vazoes['Ordem_Mês'].astype(str) + '-01')

        mi, mf = ordem_meses[req.mes_inicial], ordem_meses[req.mes_final]
        d_ini  = pd.to_datetime(f"{req.ano_inicial}-{mi}-01")
        d_fim  = pd.to_datetime(f"{req.ano_final}-{mf}-01") + pd.offsets.MonthEnd(0)

        df_long = (df_vazoes[(df_vazoes['Data'] >= d_ini) & (df_vazoes['Data'] <= d_fim)]
                   .sort_values('Data').reset_index(drop=True))

        evap_row = df_evap[df_evap["COD"] == str(res.est_evap).replace('.0', '')]
        df_long["Evaporação (m)"] = (
            df_long["Mês"].map(evap_row.iloc[0][list(ordem_meses.keys())])
            if not evap_row.empty else 0.0)

        cav_res = df_cav[df_cav["COD"] == str(res.cod)]
        if len(cav_res) < 2:
            func_interp = lambda v: 0.0
        else:
            x_vol  = cav_res["VOLUME (m³)"].values / 1e6
            y_area = cav_res["AREA (km²)"].values
            func_interp = interpolate.interp1d(x_vol, y_area, fill_value="extrapolate")

        plano_res  = df_plano[df_plano['COD'].astype(str) == str(res.cod)]
        regras_mes = {}
        if not plano_res.empty:
            for m in ordem_meses.keys():
                regras = [(row[m], row['Racionamento (%)'], row['Faixa'])
                          for _, row in plano_res.iterrows()]
                regras.sort(key=lambda x: x[0])
                regras_mes[m] = regras

        lista_dfs_input.append(df_long)
        lista_params.append({
            'func_area':       func_interp,
            'regras_secas':    regras_mes,
            'capacidade':      res.capacidade,
            'vol_ini':         res.vol_inicial,
            'demanda_nominal': res.demanda,
            'gatilho':         res.gatilho
        })

    conexao.close()

    dfs_resultados = simular_sistema_n(lista_dfs_input, lista_params,
                                       req.modo, req.vazao_conjunta)

    resultados_json = []
    for i, df in enumerate(dfs_resultados):
        df_limpo = df.replace({np.nan: None})
        df_limpo['Data'] = df_limpo['Data'].dt.strftime('%Y-%m')
        resultados_json.append({
            "reservatorio": req.reservatorios[i].nome,
            "dados":        df_limpo.to_dict(orient="records")
        })

    return {"status": "sucesso", "resultados": resultados_json}


@app.get("/api/plano-secas/{cod_acude}")
def obter_plano_secas(cod_acude: str):
    try:
        conexao = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(
            "SELECT * FROM plano_secas WHERE COD = ?",
            conexao, params=(cod_acude,))
        conexao.close()
        df = df.replace({np.nan: None})
        if "Racionamento (%)" in df.columns:
            df.rename(columns={"Racionamento (%)": "Racionamento"}, inplace=True)
        return df.to_dict(orient="records")
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────────────
# NOVA ROTA — /api/otimizar-curvas  (NSGA-II)
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/otimizar-curvas")
def otimizar_curvas(req: OtimizacaoRequest):
    """
    Optimiza as 4 curvas guia mensais (Normal, Atenção, Seca, Seca Severa)
    usando NSGA-II (DEAP).

    Fluxo:
      1. Carrega dados do SQLite UMA vez → converte para NumPy.
      2. Função de fitness chama simular_numpy (zero Pandas).
      3. NSGA-II corre por n_gen gerações com pop_size indivíduos.
      4. Extrai a Fronteira de Pareto e devolve o indivíduo com menor F1.
    """
    if not os.path.exists(DB_PATH):
        raise HTTPException(status_code=500, detail="Base de dados não encontrada.")

    SEGUNDOS_MES = 2.592e6

    # ── 1. Carregar dados (única vez) ─────────────────────────────────────────
    conexao = sqlite3.connect(DB_PATH)
    try:
        # Reservatório
        df_acude = pd.read_sql_query(
            "SELECT CORPO, [CAPAC (m³)], [Est. Evap.] FROM acudes WHERE COD = ?",
            conexao, params=(req.cod,))
        if df_acude.empty:
            raise HTTPException(status_code=404,
                                detail=f"Reservatório {req.cod} não encontrado.")

        capacidade_m3  = float(df_acude.iloc[0]['CAPAC (m³)'])
        capacidade_hm3 = capacidade_m3 / 1e6
        cod_evap       = str(df_acude.iloc[0]['Est. Evap.']).replace('.0', '').strip()
        nome_res       = df_acude.iloc[0]['CORPO']

        # Vazões
        df_vaz = pd.read_sql_query(
            "SELECT * FROM vazoes WHERE nome_reservatorio = ?",
            conexao, params=(nome_res,))
        if df_vaz.empty:
            raise HTTPException(status_code=404,
                                detail=f"Vazões não encontradas para {nome_res}.")

        df_vaz['Ordem_Mês'] = df_vaz['Mês'].map(ordem_meses)
        df_vaz['Data'] = pd.to_datetime(
            df_vaz['Ano'].astype(str) + '-' +
            df_vaz['Ordem_Mês'].astype(str) + '-01')

        d_ini  = pd.to_datetime(f"{req.ano_inicial}-01-01")
        d_fim  = pd.to_datetime(f"{req.ano_final}-12-01") + pd.offsets.MonthEnd(0)
        df_vaz = (df_vaz[(df_vaz['Data'] >= d_ini) & (df_vaz['Data'] <= d_fim)]
                  .sort_values('Data').reset_index(drop=True))

        if df_vaz.empty:
            raise HTTPException(status_code=404,
                                detail="Nenhuma vazão no período solicitado.")

        # Arrays NumPy da série histórica
        vazoes_np  = df_vaz['Vazão (m³/s)'].values.astype(float) * SEGUNDOS_MES / 1e6
        mes_idx_np = df_vaz['Ordem_Mês'].values.astype(int) - 1   # 0-based

        # Evaporação mensal (metros) → array por passo de tempo
        df_evap  = pd.read_sql_query("SELECT * FROM evaporacao", conexao)
        evap_row = df_evap[df_evap["COD"] == cod_evap]
        evap_mensal_m = (evap_row.iloc[0][list(ordem_meses.keys())].values.astype(float)
                         if not evap_row.empty else np.zeros(12))
        evap_np = evap_mensal_m[mes_idx_np]

        # CAV → arrays de interpolação (volume hm³, área km²)
        df_cav  = pd.read_sql_query("SELECT * FROM cav", conexao)
        cav_res = df_cav[df_cav["COD"] == str(req.cod)]
        if len(cav_res) >= 2:
            cav_sorted = cav_res.sort_values("VOLUME (m³)")
            x_vol_hm3  = cav_sorted["VOLUME (m³)"].values.astype(float) / 1e6
            y_area_km2 = cav_sorted["AREA (km²)"].values.astype(float)
        else:
            x_vol_hm3  = np.array([0.0, capacidade_hm3])
            y_area_km2 = np.array([0.0, max(1.0, capacidade_hm3 * 0.5)])

        areas_np = np.array([x_vol_hm3, y_area_km2])   # shape (2, M)

        # Kp pré-calculado: volume médio como referência para cada mês
        area_media   = float(np.interp(capacidade_hm3 * 0.5, areas_np[0], areas_np[1]))
        kp_fixo      = obter_k_dinamico(area_media)
        kp_np        = np.full(len(vazoes_np), kp_fixo)   # constante por simplicidade

    finally:
        conexao.close()

    # ── 2. Parâmetros derivados ───────────────────────────────────────────────
    vol_inicial_hm3  = capacidade_hm3 * (req.vol_inicial_pct / 100.0)
    demanda_alvo_hm3 = req.demanda_alvo * SEGUNDOS_MES / 1e6
    pop_size         = max(40, (req.pop_size // 4) * 4)   # múltiplo de 4

    # ── 3. Função de avaliação (closure sobre arrays NumPy) ───────────────────
    def evaluate(individuo):
        return simular_numpy(
            individuo,
            vazoes_np, evap_np, areas_np, kp_np,
            capacidade_hm3, vol_inicial_hm3, demanda_alvo_hm3
        )

    _toolbox.register("evaluate", evaluate)

    # ── 4. Executar NSGA-II ───────────────────────────────────────────────────
    random.seed(42)
    pop = _toolbox.population(n=pop_size)

    # Avaliação inicial
    for ind, fit in zip(pop, map(_toolbox.evaluate, pop)):
        ind.fitness.values = fit

    for gen in range(req.n_gen):
        # Selecção de pais (torneio binário com crowding distance)
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = [_toolbox.clone(ind) for ind in offspring]

        # Cruzamento SBX (prob. 90%)
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.9:
                _toolbox.mate(c1, c2)
                repair(c1); repair(c2)
                del c1.fitness.values
                del c2.fitness.values

        # Mutação polinomial (prob. 10% por indivíduo)
        for mut in offspring:
            if random.random() < 0.1:
                _toolbox.mutate(mut)
                repair(mut)
                del mut.fitness.values

        # Reavaliação dos indivíduos modificados
        invalidos = [ind for ind in offspring if not ind.fitness.valid]
        for ind, fit in zip(invalidos, map(_toolbox.evaluate, invalidos)):
            ind.fitness.values = fit

        # Próxima geração via NSGA-II
        pop = _toolbox.select(pop + offspring, pop_size)

    # ── 5. Fronteira de Pareto → melhor F1 ───────────────────────────────────
    pareto = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
    melhor = min(pareto, key=lambda ind: ind.fitness.values[0])
    repair(melhor)

    ind = list(melhor)

    # ── 6. Resposta ───────────────────────────────────────────────────────────
    return {
        "status":            "sucesso",
        "reservatorio_cod":  req.cod,
        "reservatorio_nome": nome_res,
        "demanda_alvo":      req.demanda_alvo,
        "periodo": {
            "ano_inicial": req.ano_inicial,
            "ano_final":   req.ano_final,
            "n_meses":     int(len(vazoes_np))
        },
        "fitness_vencedor": {
            "f1_deficit_quadratico": round(melhor.fitness.values[0], 4),
            "f2_vertimento_hm3":     round(melhor.fitness.values[1], 4)
        },
        "n_solucoes_pareto": len(pareto),
        "curvas_guia_percentual": {
            "normal":      [round(v, 2) for v in ind[0:12]],
            "atencao":     [round(v, 2) for v in ind[12:24]],
            "seca":        [round(v, 2) for v in ind[24:36]],
            "seca_severa": [round(v, 2) for v in ind[36:48]]
        }
    }
