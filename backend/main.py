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

# caminho do banco de dados SQLite na mesma pasta do script
DB_PATH = os.path.join(os.path.abspath("."), "banco_site.db")

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


# função principal que roda a simulação mês a mês pra todos os reservatórios
# recebe os dataframes com as vazões, os parâmetros de cada açude, o modo de operação
# (Série, Paralelo ou Individual) e a vazão conjunta do sistema
def simular_sistema_n(dfs, params, modo, vazao_conjunta):
    n_res    = len(dfs)
    n_meses  = len(dfs[0])
    segundos_mes = 2.592e6  # quantos segundos tem num mês médio (30 dias)

    # inicializa todas as colunas de resultado com zero
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

    # começa com o volume inicial de cada reservatório
    volumes_atueis = [p['vol_ini'] for p in params]

    # loop mês a mês
    for t in range(n_meses):
        demandas_iniciais        = []
        racionamentos            = []
        nomes_faixas_atuais      = []
        prev_volumes_pos_natureza = []

        # processa cada reservatório no mês atual
        for i in range(n_res):
            p       = params[i]
            vol_ini = volumes_atueis[i]
            pct_vol = (vol_ini / p['capacidade']) * 100  # volume atual em % da capacidade
            mes_atual = dfs[i].loc[t, 'Mês']

            # verifica se tem racionamento ativo baseado no plano de secas
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

            # 1. Converte afluência e evaporação para as unidades base (hm³ e m)
            afluencia_hm3 = dfs[i].loc[t, 'Vazão (m³/s)'] * (segundos_mes / 1e6)
            evap_taxa_m   = float(dfs[i].loc[t, 'Evaporação (m)']) / 1000.0

            # 2. Calcula a "Retirada" total (Demanda do açude + Vazão Conjunta, se aplicável)
            retirada_prevista_m3s = p['demanda_nominal']
            if i == 0 and modo in ["Paralelo", "Série"]:
                retirada_prevista_m3s += vazao_conjunta
            
            # Aplica o racionamento à retirada
            retirada_hm3 = retirada_prevista_m3s * (1 - rac / 100.0) * (segundos_mes / 1e6)

            # 3. Aplica os passos exatos do modelo matemático:
            area_ini = float(p['func_area'](vol_ini))
            
            # Chute do volume final intermédio
            vol_est = vol_ini + afluencia_hm3 - retirada_hm3 - (evap_taxa_m * area_ini)
            vol_est = max(0.0, vol_est) 
            
            area_fin = float(p['func_area'](vol_est))
            area_med = (area_ini + area_fin) / 2.0
            evap_hm3 = evap_taxa_m * area_med

            # Guarda os valores
            dfs[i].loc[t, 'Evaporação (hm³)']    = evap_hm3
            dfs[i].loc[t, 'Afluências (hm³/mês)'] = afluencia_hm3

            # Volume parcial 
            vol_pos_natureza = max(0.0, vol_ini + afluencia_hm3 - evap_hm3)
            prev_volumes_pos_natureza.append(vol_pos_natureza)

        total_vol_disponivel = sum(prev_volumes_pos_natureza)

        rac_inicial_conjunta      = racionamentos[0] if racionamentos else 0.0
        demanda_conjunta_estimada = vazao_conjunta * (1 - rac_inicial_conjunta / 100.0) * (segundos_mes / 1e6)

        total_demanda_necessaria = demanda_conjunta_estimada
        for i in range(n_res):
            dem_esp_hm3 = params[i]['demanda_nominal'] * (1 - racionamentos[i] / 100.0) * (segundos_mes / 1e6)
            total_demanda_necessaria += dem_esp_hm3

        # verifica se o sistema entrou em falha 
        sistema_em_falha = False
        if modo == "Paralelo" and total_vol_disponivel < total_demanda_necessaria:
            sistema_em_falha = True
            for i in range(n_res):
                dfs[i].loc[t, 'Modo Operação'] = 'FALHA SISTÊMICA'

        demandas_finais            = [0.0] * n_res
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

        # Lista de controle para segurar as falhas do mês antes de escrever no DataFrame
        falhas_do_mes = []

        # grava os resultados de cada açude no dataframe e calcula o volume final do mês
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

            # Avalia a falha e guarda na lista de controle (não mais no df.loc direto)
            falhou_agora = False
            if sistema_em_falha:
                demanda_atendida_real_hm3 = max(0, min(vol_disp, demanda_hm3))
                falhou_agora = True
            else:
                if vol_disp < demanda_hm3:
                    demanda_atendida_real_hm3 = max(0, vol_disp)
                    falhou_agora = True
                else:
                    demanda_atendida_real_hm3 = demanda_hm3

            falhas_do_mes.append(falhou_agora)

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

        # Só registra a falha se todos os açudes do sistema caíram simultaneamente
        if modo in ["Paralelo", "Série"]:
            falha_conjunta = all(falhas_do_mes)
            for i in range(n_res):
                dfs[i].loc[t, 'Falha'] = 'Sim' if falha_conjunta else 'Não'
        else:
            # No modo individual, cada um é responsável pelo seu próprio déficit
            for i in range(n_res):
                dfs[i].loc[t, 'Falha'] = 'Sim' if falhas_do_mes[i] else 'Não'

    return dfs


# rota que retorna a lista de todos os reservatórios cadastrados no banco
@app.get("/api/reservatorios")
def listar_reservatorios():
    if not os.path.exists(DB_PATH):
        raise HTTPException(status_code=500, detail="Base de dados não encontrada.")
    conexao = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT CORPO, COD, [CAPAC (m³)], [Est. Evap.] FROM acudes", conexao)
    conexao.close()
    # converte capacidade de m³ pra hm³
    df['CAPAC (m³)'] = df['CAPAC (m³)'] / 1e6
    df = df.replace({np.nan: None})
    return df.to_dict(orient="records")


# rota que retorna os hidrossistemas pré-configurados (presets de simulação)
@app.get("/api/presets")
def listar_presets():
    conexao = sqlite3.connect(DB_PATH)
    try:
        df_hidro = pd.read_sql_query("SELECT * FROM hidrossistemas", conexao)
        presets  = []
        for nome_sis, group in df_hidro.groupby('hidrossistema'):
            # detecta o modo de operação pelo texto salvo no banco
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


# rota principal que executa a simulação e devolve os resultados mês a mês
@app.post("/api/simular")
def processar_simulacao_api(req: SimulacaoRequest):
    conexao          = sqlite3.connect(DB_PATH)
    lista_dfs_input  = []
    lista_params     = []

    # carrega tabelas auxiliares uma vez só
    df_evap  = pd.read_sql_query("SELECT * FROM evaporacao", conexao)
    df_cav   = pd.read_sql_query("SELECT * FROM cav", conexao)
    df_plano = pd.read_sql_query("SELECT * FROM plano_secas", conexao)

    for res in req.reservatorios:
        # busca as vazões históricas do reservatório
        df_vazoes = pd.read_sql_query(
            "SELECT * FROM vazoes WHERE nome_reservatorio = ?",
            conexao, params=(res.nome,))

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
            func_interp = lambda v: 0.0  # sem dados suficientes, retorna área zero
        else:
            x_vol  = cav_res["VOLUME (m³)"].values / 1e6
            y_area = cav_res["AREA (km²)"].values
            
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
