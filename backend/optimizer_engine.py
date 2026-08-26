import os
import json
import sqlite3
import numpy as np
import pandas as pd
import pyswarms as ps
from numba import njit
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

router = APIRouter(prefix="/api/otimizador", tags=["otimizador"])

class LimitesPayload(BaseModel):
    reservatorio: str

class SimularPayload(BaseModel):
    cenario_id: str = 'padrao'
    reservatorio: str
    durb_m3s: float
    dsupl_m3s: float
    prob: float = 0.25
    iters: int = 50
    ninicio: int = 1
    mes_inicio: int = 1
    ano_inicio: int = 1911
    mes_fim: int = 12
    ano_fim: int = 2021
    frac_durb: List[float]
    frac_dsup: List[float]
    garantia_req: List[float]
    quantidade_faixas: int = 4
    faixas_nomes: Optional[List[str]] = None
    seed: Optional[int] = None


def nomes_faixas(
    quantidade_faixas: int,
    nomes_personalizados: Optional[List[str]] = None,
) -> list[str]:
    nomes_padrao = ["Normal", "Alerta", "Seca", "Seca Severa"]
    if nomes_personalizados is None:
        return nomes_padrao[:quantidade_faixas]

    nomes = [str(nome).strip() for nome in nomes_personalizados]
    if len(nomes) != quantidade_faixas:
        raise ValueError(f"Informe exatamente {quantidade_faixas} nomes de faixas.")
    if any(not nome for nome in nomes):
        raise ValueError("Os nomes das faixas não podem ficar vazios.")
    if len({nome.casefold() for nome in nomes}) != len(nomes):
        raise ValueError("Os nomes das faixas devem ser diferentes entre si.")
    return nomes


def validar_faixas_payload(payload: SimularPayload) -> None:
    quantidade = int(payload.quantidade_faixas)
    if not 2 <= quantidade <= 4:
        raise ValueError("A quantidade de faixas deve estar entre 2 e 4.")
    tamanhos = {
        "atendimento da demanda": len(payload.frac_durb),
        "atendimento suplementar": len(payload.frac_dsup),
        "permanências requeridas": len(payload.garantia_req),
    }
    invalidos = [nome for nome, tamanho in tamanhos.items() if tamanho != quantidade]
    if invalidos:
        raise ValueError(
            "Cada vetor de faixas deve possuir "
            f"{quantidade} valores: " + ", ".join(invalidos) + "."
        )
    nomes_faixas(quantidade, payload.faixas_nomes)


def get_db_path():
    # Na web, o banco de dados geralmente fica na mesma pasta do main.py
    if os.path.exists('banco_site.db'):
        return 'banco_site.db'
    raise Exception("Arquivo banco_site.db não foi encontrado na raiz do projeto.")


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

@router.get("/reservatorios")
def listar_reservatorios():
    try:
        db_path = get_db_path()
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT DISTINCT CORPO FROM acudes WHERE CORPO IS NOT NULL ORDER BY CORPO')
        rows = cursor.fetchall()
        conn.close()
        lista_acudes = [corrigir_texto_db(str(r[0]).strip()) for r in rows if str(r[0]).strip()]
        return {"status": "reservatorios", "lista": lista_acudes}
    except Exception as e:
        return {"status": "erro", "mensagem": f"Erro ao listar açudes: {str(e)}"}

@router.post("/limites")
def buscar_limites_bd(payload: LimitesPayload):
    try:
        reservatorio = payload.reservatorio
        meses_map = {"JAN":1, "FEV":2, "MAR":3, "ABR":4, "MAI":5, "JUN":6, "JUL":7, "AGO":8, "SET":9, "OUT":10, "NOV":11, "DEZ":12}
        db_path = get_db_path()
        conn = sqlite3.connect(db_path)
        df = normalizar_colunas_db(pd.read_sql_query(
            'SELECT * FROM vazoes WHERE nome_reservatorio LIKE ? OR nome_reservatorio LIKE ?',
            conn,
            params=(f"%{reservatorio}%", f"%{texto_para_legado(reservatorio)}%")
        ))
        conn.close()

        if df.empty:
            return {"status": "limites", "ano_min": 1911, "mes_min": 1, "ano_max": 2021, "mes_max": 12}

        df['mes_num'] = df['Mês'].map(lambda m: meses_map.get(str(m).upper()[:3], 1))
        df = df.sort_values(['Ano', 'mes_num'])
        row_min = df.iloc[0]
        row_max = df.iloc[-1]
        return {"status": "limites", "ano_min": int(row_min['Ano']), "mes_min": int(row_min['mes_num']), "ano_max": int(row_max['Ano']), "mes_max": int(row_max['mes_num'])}
    except Exception as e:
        return {"status": "erro", "mensagem": str(e)}

def carregar_dados_fisicos(reservatorio, mes_ini, ano_ini, mes_fim, ano_fim):
    meses_map = {"JAN":1, "FEV":2, "MAR":3, "ABR":4, "MAI":5, "JUN":6, "JUL":7, "AGO":8, "SET":9, "OUT":10, "NOV":11, "DEZ":12}
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    df_acudes = normalizar_colunas_db(pd.read_sql_query(
        'SELECT * FROM acudes WHERE CORPO = ? OR CORPO LIKE ? OR CORPO = ? OR CORPO LIKE ? LIMIT 1',
        conn,
        params=(reservatorio, f"%{reservatorio}%", texto_para_legado(reservatorio), f"%{texto_para_legado(reservatorio)}%")
    ))
    if df_acudes.empty:
        conn.close()
        raise Exception(f"Açude '{reservatorio}' não encontrado na tabela 'acudes'.")

    acude_row = df_acudes.iloc[0]
    cod_acude = acude_row['COD']
    capac_m3 = acude_row['CAPAC (m³)']
    est_evap = acude_row['Est. Evap.']
    cap_hm3 = float(capac_m3) / 1e6

    df_cav = normalizar_colunas_db(pd.read_sql_query(
        'SELECT * FROM cav WHERE COD = ? OR CAST(COD AS REAL) = ?',
        conn,
        params=(str(cod_acude), cod_acude)
    ))
    if df_cav.empty:
        conn.close()
        raise Exception(f"CAV não encontrada para o açude COD {cod_acude}.")

    df_cav = df_cav.sort_values('VOLUME (m³)')
    cav_vol = df_cav['VOLUME (m³)'].astype(float).values / 1e6
    cav_area = df_cav['AREA (km²)'].astype(float).values
    
    est_evap_str = str(int(float(est_evap))) if est_evap else ""
    df_evap = pd.read_sql_query(
        'SELECT JAN, FEV, MAR, ABR, MAI, JUN, JUL, AGO, "SET", OUT, NOV, DEZ '
        'FROM evaporacao WHERE COD = ? OR COD = ? LIMIT 1',
        conn,
        params=(est_evap_str, str(est_evap)),
    )
    if df_evap.empty:
        evap_mensal = np.ones(12) * 150.0
    else:
        evap_mensal = df_evap.iloc[0].fillna(0).astype(float).values
        
    df_vazoes = normalizar_colunas_db(pd.read_sql_query(
        'SELECT * FROM vazoes WHERE nome_reservatorio LIKE ? OR nome_reservatorio LIKE ?',
        conn,
        params=(f"%{reservatorio}%", f"%{texto_para_legado(reservatorio)}%")
    ))
    conn.close()

    vazoes_filtradas = []
    for _, row in df_vazoes.iterrows():
        ano = int(row['Ano'])
        mes_num = meses_map.get(str(row['Mês']).upper()[:3], 1)
        data_atual = ano * 12 + mes_num
        data_inicio = ano_ini * 12 + mes_ini
        data_fim = ano_fim * 12 + mes_fim
        if data_inicio <= data_atual <= data_fim:
            vazoes_filtradas.append((ano, mes_num, float(row['Vazão (m³/s)'])))

    vazoes_filtradas.sort(key=lambda x: (x[0], x[1]))
    aflu_serie = np.array([x[2] for x in vazoes_filtradas])
    evap_serie = np.array([evap_mensal[x[1] - 1] for x in vazoes_filtradas])
    
    return aflu_serie, evap_serie, cap_hm3, cav_vol, cav_area


@njit
def interpola(x_vals, y_vals, x):
    if x <= x_vals[0]: return y_vals[0]
    if x >= x_vals[-1]: return y_vals[-1]
    return np.interp(x, x_vals, y_vals)

@njit
def dinamica_mensal_fast(vol_ini, aflu_hm3, evap_m, ret_hm3, vol_min, vol_max, cav_vol, cav_area):
    area_ini = interpola(cav_vol, cav_area, vol_ini)
    vol = vol_ini + aflu_hm3 - ret_hm3 - (evap_m * area_ini)
    area_fin = interpola(cav_vol, cav_area, vol)
    area_med = (area_ini + area_fin) / 2.0
    vol = vol_ini + aflu_hm3 - ret_hm3 - (evap_m * area_med)
    
    retirada_efetiva = ret_hm3
    if vol < vol_min:
        retirada_orig = retirada_efetiva
        retirada_efetiva = retirada_efetiva - (vol_min - vol)
        if retirada_efetiva >= 0:
            vol = vol_min
        else:
            vol = vol + retirada_orig
            retirada_efetiva = 0.0
            if vol < 0: vol = 0.0
            
    vertimento = 0.0
    if vol > vol_max:
        vertimento = vol - vol_max
        vol = vol_max
        
    return vol, retirada_efetiva, vertimento, evap_m * area_med

@njit
def calculo_volume_meta_fast(niveis_metas, aflu_prob, evap_ano, dem_total_hm3, cap_hm3, cav_vol, cav_area, ninicio):
    aflu_zero = np.zeros(12)
    vmeta = np.zeros((len(niveis_metas), 12))
    vol_util = cap_hm3 
    
    for vm in range(len(niveis_metas)):
        vol_ini = niveis_metas[vm] * vol_util
        v2_fim = np.zeros(12)
        v = vol_ini
        for i in range(12):
            v, _, _, _ = dinamica_mensal_fast(v, -aflu_zero[i], -evap_ano[i], -dem_total_hm3, 0.0, cap_hm3, cav_vol, cav_area)
            v2_fim[i] = v
        v2_rev = np.zeros(12)
        for i in range(12): v2_rev[i] = v2_fim[11 - i]
            
        v3_fim = np.zeros(12)
        v = vol_ini
        for i in range(12):
            v, _, _, _ = dinamica_mensal_fast(v, aflu_prob[i], evap_ano[i], dem_total_hm3, 0.0, cap_hm3, cav_vol, cav_area)
            v3_fim[i] = v
        v3_shifted = np.zeros(12)
        v3_shifted[0] = vol_ini
        for i in range(11): v3_shifted[i+1] = v3_fim[i]
            
        vmeta_aux = np.zeros(12)
        for i in range(12):
            vmeta_aux[i] = min(v2_rev[i], v3_shifted[i]) / vol_util
            
        if ninicio != 1:
            idx_split = 13 - ninicio
            for i in range(13 - ninicio): vmeta[vm, (ninicio - 1) + i] = vmeta_aux[i]
            for i in range(ninicio - 1): vmeta[vm, i] = vmeta_aux[idx_split + i]
        else:
            for i in range(12): vmeta[vm, i] = vmeta_aux[i]
    return vmeta

@njit
def engine_simulacao_temporal(nmetas, aflu_hm3, evap_serie_m, ret_vec_hm3, cap_hm3, cav_vol, cav_area, mes_inicio):
    num_meses = len(aflu_hm3)
    num_estados = len(ret_vec_hm3)
    falhas = np.zeros(num_estados)
    vol = cap_hm3 * 0.5 
    
    for i in range(num_meses):
        idx_mes = (mes_inicio - 1 + i) % 12
        vol_perc = vol / cap_hm3
        coluna_meta = np.empty(nmetas.shape[0])
        for k in range(nmetas.shape[0]): coluna_meta[k] = nmetas[k, idx_mes]
            
        est_hidr = np.searchsorted(coluna_meta, vol_perc)
        idx_alvo = (num_estados - 1) - est_hidr
        if idx_alvo < 0: idx_alvo = 0
        if idx_alvo >= num_estados: idx_alvo = num_estados - 1
        
        vol, ret_efetiva, _, _ = dinamica_mensal_fast(vol, aflu_hm3[i], evap_serie_m[i], ret_vec_hm3[idx_alvo], 0.0, cap_hm3, cav_vol, cav_area)
        for k in range(num_estados):
            if ret_efetiva < ret_vec_hm3[k]: falhas[k] += 1
                
    garantias = np.empty(num_estados)
    for k in range(num_estados): garantias[k] = 1.0 - (falhas[k] / num_meses)
    return garantias

@njit
def simular_serie_historica_fast(nmetas, aflu_hm3, evap_serie_m, ret_vec_hm3, cap_hm3, cav_vol, cav_area, mes_inicio):
    num_meses = len(aflu_hm3)
    num_estados = len(ret_vec_hm3)
    vol = cap_hm3 * 0.5 
    historico_vol = np.zeros(num_meses) 
    
    for i in range(num_meses):
        idx_mes = (mes_inicio - 1 + i) % 12
        vol_perc = vol / cap_hm3
        coluna_meta = np.empty(nmetas.shape[0])
        for k in range(nmetas.shape[0]): coluna_meta[k] = nmetas[k, idx_mes]
            
        est_hidr = np.searchsorted(coluna_meta, vol_perc)
        idx_alvo = (num_estados - 1) - est_hidr
        if idx_alvo < 0: idx_alvo = 0
        if idx_alvo >= num_estados: idx_alvo = num_estados - 1
        
        vol, _, _, _ = dinamica_mensal_fast(vol, aflu_hm3[i], evap_serie_m[i], ret_vec_hm3[idx_alvo], 0.0, cap_hm3, cav_vol, cav_area)
        historico_vol[i] = vol
    return historico_vol


@njit
def simular_serie_historica_detalhada_fast(nmetas, aflu_hm3, evap_serie_m, ret_vec_hm3, cap_hm3, cav_vol, cav_area, mes_inicio):
    num_meses = len(aflu_hm3)
    num_estados = len(ret_vec_hm3)
    vol = cap_hm3 * 0.5
    dados = np.zeros((num_meses, 8))

    for i in range(num_meses):
        idx_mes = (mes_inicio - 1 + i) % 12
        vol_ini = vol
        vol_perc = vol / cap_hm3
        coluna_meta = np.empty(nmetas.shape[0])
        for k in range(nmetas.shape[0]):
            coluna_meta[k] = nmetas[k, idx_mes]

        est_hidr = np.searchsorted(coluna_meta, vol_perc)
        idx_alvo = (num_estados - 1) - est_hidr
        if idx_alvo < 0:
            idx_alvo = 0
        if idx_alvo >= num_estados:
            idx_alvo = num_estados - 1

        ret_solicitada = ret_vec_hm3[idx_alvo]
        vol, ret_efetiva, vertimento, evap_hm3 = dinamica_mensal_fast(vol, aflu_hm3[i], evap_serie_m[i], ret_solicitada, 0.0, cap_hm3, cav_vol, cav_area)
        dados[i, 0] = vol_ini
        dados[i, 1] = aflu_hm3[i]
        dados[i, 2] = evap_hm3
        dados[i, 3] = ret_solicitada
        dados[i, 4] = ret_efetiva
        dados[i, 5] = vertimento
        dados[i, 6] = vol
        dados[i, 7] = idx_alvo

    return dados

def gerar_resultado_final(niveis_metas, aflu_hm3, evap_serie_m, dem_total_hm3, ret_vec_hm3, cap_hm3, cav_vol, cav_area, aflu_prob, evap_ano, ninicio, mes_inicio):
    nmetas = calculo_volume_meta_fast(niveis_metas, aflu_prob, evap_ano, dem_total_hm3, cap_hm3, cav_vol, cav_area, ninicio)
    garantias = engine_simulacao_temporal(nmetas, aflu_hm3, evap_serie_m, ret_vec_hm3, cap_hm3, cav_vol, cav_area, mes_inicio)
    return garantias, nmetas


def diferenciar_retiradas_equivalentes(ret_vec_hm3, diferenca_relativa=0.0001):
    retiradas = np.asarray(ret_vec_hm3, dtype=float).copy()
    if retiradas.size < 2:
        return retiradas

    escala = max(1.0, float(np.max(np.abs(retiradas))))
    inicio = 0
    while inicio < retiradas.size:
        fim = inicio
        while fim + 1 < retiradas.size and np.isclose(
            ret_vec_hm3[inicio],
            ret_vec_hm3[fim + 1],
            rtol=1e-10,
            atol=1e-12 * escala,
        ):
            fim += 1

        quantidade = fim - inicio + 1
        valor = float(ret_vec_hm3[inicio])
        if quantidade > 1 and valor > 0:
            proximo = float(ret_vec_hm3[fim + 1]) if fim + 1 < retiradas.size else 0.0
            espaco = max(0.0, valor - proximo)
            passo = min(valor * diferenca_relativa, espaco / quantidade)
            for deslocamento in range(1, quantidade):
                retiradas[inicio + deslocamento] = valor - (passo * deslocamento)
        inicio = fim + 1

    return retiradas


def calcular_erro_garantias(garantias, garantia_req):
    obtidas = np.asarray(garantias, dtype=float)
    requeridas = np.asarray(garantia_req, dtype=float)
    if obtidas.size != requeridas.size:
        raise ValueError("Garantias obtidas e requeridas devem possuir o mesmo tamanho.")
    denominadores = np.maximum(np.abs(requeridas), 1e-9)
    return float(np.sum(((obtidas - requeridas) / denominadores) ** 2))

def funcao_objetivo_pso(x_matrix, aflu_hm3, evap_serie_m, dem_total_hm3, ret_vec_hm3, cap_hm3, cav_vol, cav_area, garantia_req, aflu_prob, evap_ano, ninicio, mes_inicio):
    n_particles = x_matrix.shape[0]
    resultados = np.zeros(n_particles)
    for i in range(n_particles):
        niveis_metas = x_matrix[i] / 10.0
        if not np.all(np.diff(niveis_metas) >= 0.02):
            resultados[i] = 1e7
            continue
        nmetas = calculo_volume_meta_fast(niveis_metas, aflu_prob, evap_ano, dem_total_hm3, cap_hm3, cav_vol, cav_area, ninicio)
        if np.min(nmetas) <= 0.05:
            resultados[i] = 1e6
            continue
        garantias = engine_simulacao_temporal(nmetas, aflu_hm3, evap_serie_m, ret_vec_hm3, cap_hm3, cav_vol, cav_area, mes_inicio)
        resultados[i] = calcular_erro_garantias(garantias, garantia_req)
    return resultados



def simular_generator(payload: SimularPayload):
    try:
        validar_faixas_payload(payload)
        faixas_nomes = nomes_faixas(payload.quantidade_faixas, payload.faixas_nomes)
        aflu, evap, cap_hm3, cav_vol, cav_area = carregar_dados_fisicos(
            payload.reservatorio, payload.mes_inicio, payload.ano_inicio, payload.mes_fim, payload.ano_fim
        )

        fator_conv = 2.592
        aflu_hm3 = aflu * fator_conv
        dem_total_hm3 = (payload.durb_m3s + payload.dsupl_m3s) * fator_conv
        ret_vec_hm3 = (payload.durb_m3s * fator_conv * np.array(payload.frac_durb)) + (payload.dsupl_m3s * fator_conv * np.array(payload.frac_dsup))
        ret_vec_otimizacao_hm3 = diferenciar_retiradas_equivalentes(ret_vec_hm3)
        evap_serie_m = evap / 1000.0

        meses_serie = np.array([(payload.mes_inicio - 1 + i) % 12 + 1 for i in range(len(aflu_hm3))])
        aflu_prob_12 = np.zeros(12)
        evap_ano_12 = np.zeros(12)
        for m in range(1, 13):
            idx = (meses_serie == m)
            aflu_prob_12[m-1] = np.quantile(aflu_hm3[idx], payload.prob) if np.any(idx) else 0.0
            evap_ano_12[m-1] = evap_serie_m[idx][0] if np.any(idx) else 0.0
            
        if payload.ninicio != 1:
            idx_start = payload.ninicio - 1
            aflu_prob = np.concatenate([aflu_prob_12[idx_start:], aflu_prob_12[:idx_start]])
            evap_ano = np.concatenate([evap_ano_12[idx_start:], evap_ano_12[:idx_start]])
        else:
            aflu_prob, evap_ano = aflu_prob_12, evap_ano_12

        n_vars = len(payload.frac_durb) - 1 
        bounds = (np.ones(n_vars), np.ones(n_vars) * 9.0)
        if payload.seed is not None:
            np.random.seed(payload.seed)
        optimizer = ps.single.GlobalBestPSO(n_particles=200, dimensions=n_vars, options={'c1': 0.5, 'c2': 0.3, 'w': 0.9}, bounds=bounds)

        kwargs = dict(aflu_hm3=aflu_hm3, evap_serie_m=evap_serie_m, dem_total_hm3=dem_total_hm3, ret_vec_hm3=ret_vec_otimizacao_hm3, cap_hm3=cap_hm3, cav_vol=cav_vol, cav_area=cav_area, garantia_req=payload.garantia_req, aflu_prob=aflu_prob, evap_ano=evap_ano, ninicio=payload.ninicio, mes_inicio=payload.mes_inicio)

        passos_por_bloco = 5
        total_blocos = max(1, payload.iters // passos_por_bloco)
        resto = payload.iters % passos_por_bloco
        best_cost, best_pos = float('inf'), None

        # Gerador: envia blocos de progresso aos poucos
        for bloco in range(1, total_blocos + 1):
            cost, pos = optimizer.optimize(funcao_objetivo_pso, iters=passos_por_bloco, verbose=False, **kwargs)
            best_cost, best_pos = cost, pos
            progresso_data = {
                "status": "progresso", "cenario_id": payload.cenario_id,
                "iteracao": bloco * passos_por_bloco, "total_iteracoes": payload.iters
            }
            yield f"data: {json.dumps(progresso_data)}\n\n"

        if resto > 0:
            cost, pos = optimizer.optimize(funcao_objetivo_pso, iters=resto, verbose=False, **kwargs)
            best_cost, best_pos = cost, pos
            progresso_data = {"status": "progresso", "cenario_id": payload.cenario_id, "iteracao": payload.iters, "total_iteracoes": payload.iters}
            yield f"data: {json.dumps(progresso_data)}\n\n"

        melhores_metas = np.sort(best_pos / 10.0)
        garantias_finais, curvas_finais = gerar_resultado_final(melhores_metas, aflu_hm3, evap_serie_m, dem_total_hm3, ret_vec_otimizacao_hm3, cap_hm3, cav_vol, cav_area, aflu_prob, evap_ano, payload.ninicio, payload.mes_inicio)
        volumes_hist = simular_serie_historica_fast(curvas_finais, aflu_hm3, evap_serie_m, ret_vec_hm3, cap_hm3, cav_vol, cav_area, payload.mes_inicio)
        volumes_hist = np.where(np.isfinite(volumes_hist), volumes_hist, 0.0)
        sim_detalhada = simular_serie_historica_detalhada_fast(curvas_finais, aflu_hm3, evap_serie_m, ret_vec_hm3, cap_hm3, cav_vol, cav_area, payload.mes_inicio)
        sim_detalhada = np.where(np.isfinite(sim_detalhada), sim_detalhada, 0.0)
        meses_rotulo = ["JAN", "FEV", "MAR", "ABR", "MAI", "JUN", "JUL", "AGO", "SET", "OUT", "NOV", "DEZ"]
        segundos_mes = 2.592
        simulacao_historica = []
        for i in range(len(sim_detalhada)):
            mes_idx = (payload.mes_inicio - 1 + i) % 12
            ano_atual = payload.ano_inicio + ((payload.mes_inicio - 1 + i) // 12)
            ret_sol_m3s = float(sim_detalhada[i, 3] / segundos_mes)
            ret_ef_m3s = float(sim_detalhada[i, 4] / segundos_mes)
            rac = 0.0 if ret_sol_m3s <= 0 else max(0.0, (1.0 - (ret_ef_m3s / ret_sol_m3s)) * 100.0)
            estado_idx = int(sim_detalhada[i, 7])
            modo_operacao = faixas_nomes[estado_idx]
            simulacao_historica.append({
                "Data": f"{meses_rotulo[mes_idx]}/{ano_atual}",
                "Armazenamento Inicial": float(sim_detalhada[i, 0]),
                "Afluências (hm³/mês)": float(aflu_hm3[i]),
                "Evaporação (hm³)": float(sim_detalhada[i, 2]),
                "Demanda Solicitada (m³/s)": ret_sol_m3s,
                "Demanda Atendida (m³/s)": ret_ef_m3s,
                "Racionamento (%)": float(rac),
                "Vertimento (hm³)": float(sim_detalhada[i, 5]),
                "Armazenamento Final": float(sim_detalhada[i, 6]),
                "Falha": "Sim" if ret_ef_m3s + 1e-9 < ret_sol_m3s else "Não",
                "Modo Operação": modo_operacao,
            })
        
        resultado_final = {
            "status": "sucesso", "cenario_id": payload.cenario_id, "custo_final": float(best_cost),
            "quantidade_faixas": int(payload.quantidade_faixas),
            "faixas_nomes": faixas_nomes,
            "niveis_meta": melhores_metas[::-1].tolist(), "garantias_obtidas": garantias_finais.tolist(),
            "matriz_curvas": curvas_finais[::-1].tolist(), "volumes_historicos": volumes_hist.tolist(),
            "simulacao_historica": simulacao_historica,
            "mes_inicio": payload.mes_inicio, "ano_inicio": payload.ano_inicio, "capacidade_hm3": cap_hm3
        }
        yield f"data: {json.dumps(resultado_final)}\n\n"

    except Exception as e:
        erro_data = {"status": "erro", "cenario_id": payload.cenario_id, "mensagem": str(e)}
        yield f"data: {json.dumps(erro_data)}\n\n"

@router.post("/simular")
def simular_endpoint(payload: SimularPayload):
    # StreamingResponse mantém a ligação aberta enquanto o PSO calcula
    return StreamingResponse(simular_generator(payload), media_type="text/event-stream")
