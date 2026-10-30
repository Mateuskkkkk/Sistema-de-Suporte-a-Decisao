"""Otimização multiobjetivo dos níveis meta (NSGA-II).

Enquanto o otimizador por PSO procura curvas que atinjam garantias exigidas em
cada nível, este módulo explora o compromisso entre dois objetivos:

* segurança: minimizar o número de meses com falha, isto é, meses em que a
  retirada do nível vigente não pôde ser atendida integralmente;
* água entregue: maximizar o volume efetivamente retirado ao longo da série.
  Há dois critérios. No "gradual" (padrão) minimiza-se a soma dos quadrados dos
  déficits mensais em relação à demanda Normal: cortes pequenos e frequentes
  pesam menos que cortes raros e profundos, o que favorece o racionamento
  escalonado (hedging) e afasta as curvas umas das outras. No "total" maximiza-se
  o volume entregue; para esse critério a política ótima é manter o atendimento
  pleno até o limite e cortar de uma vez, e os níveis intermediários ficam estreitos.

Curvas altas antecipam o racionamento e reduzem as falhas, mas entregam menos
água; curvas baixas fazem o contrário. O resultado é a fronteira de Pareto: o
conjunto de curvas em que não é possível melhorar um objetivo sem piorar o
outro. As curvas usam a mesma parametrização do PSO (um nível meta por estado
restritivo, convertido em curvas mensais por ``calculo_volume_meta_fast``).
"""
import json
from typing import List, Literal, Optional

import numpy as np
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from numba import njit
from pydantic import BaseModel, Field
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.termination import get_termination

from optimizer_engine import (
    calculo_volume_meta_fast,
    carregar_dados_fisicos,
    dinamica_mensal_fast,
    engine_simulacao_temporal,
    nomes_faixas,
    simular_serie_historica_fast,
)

router = APIRouter(prefix="/api/otimizador", tags=["otimizador"])

HM3_POR_M3S = 2.592
NIVEL_MIN, NIVEL_MAX = 0.10, 0.90  # posição dos níveis meta, como no PSO
FOLGA_ENTRE_NIVEIS = 0.02
VOLUME_META_MIN = 0.05


class ParetoPayload(BaseModel):
    cenario_id: str = "padrao"
    reservatorio: str
    durb_m3s: float
    dsupl_m3s: float = 0.0
    prob: float = 0.25
    ninicio: int = 1
    mes_inicio: int = 1
    ano_inicio: int = 1911
    mes_fim: int = 12
    ano_fim: int = 2021
    frac_durb: List[float]
    frac_dsup: List[float]
    quantidade_faixas: int = 4
    faixas_nomes: Optional[List[str]] = None
    # distância mínima entre curvas vizinhas, em fração da capacidade, em todos os meses;
    # sem ela o NSGA-II tende a espremer os níveis intermediários (ir direto do Normal ao mais severo)
    largura_minima: float = Field(0.05, ge=0.0, le=0.25)
    criterio_entrega: Literal["gradual", "total"] = "gradual"
    populacao: int = Field(80, ge=20, le=400)
    geracoes: int = Field(60, ge=5, le=500)
    seed: Optional[int] = 1


@njit(cache=False)
def avaliar_operacao(nmetas, aflu_hm3, evap_serie_m, ret_vec_hm3, cap_hm3, cav_vol, cav_area, mes_inicio):
    """Simula a série com as curvas e devolve os indicadores dos dois objetivos.

    Retorna (meses com falha, volume entregue, déficit, meses com racionamento,
    volume mínimo, soma dos quadrados dos déficits relativos à demanda Normal).
    A regra de estado é a mesma do PSO e da simulação detalhada.
    """
    num_meses = len(aflu_hm3)
    num_estados = len(ret_vec_hm3)
    vol = cap_hm3 * 0.5
    falhas = 0
    entregue = 0.0
    deficit = 0.0
    racionamento = 0
    vol_min = vol
    deficit_quad = 0.0
    for i in range(num_meses):
        idx_mes = (mes_inicio - 1 + i) % 12
        coluna = np.empty(nmetas.shape[0])
        for k in range(nmetas.shape[0]):
            coluna[k] = nmetas[k, idx_mes]
        est = np.searchsorted(coluna, vol / cap_hm3)
        idx = (num_estados - 1) - est
        if idx < 0:
            idx = 0
        if idx >= num_estados:
            idx = num_estados - 1
        solicitada = ret_vec_hm3[idx]
        if solicitada < ret_vec_hm3[0] - 1e-12:
            racionamento += 1
        vol, efetiva, _, _ = dinamica_mensal_fast(vol, aflu_hm3[i], evap_serie_m[i], solicitada, 0.0, cap_hm3, cav_vol, cav_area)
        entregue += efetiva
        if ret_vec_hm3[0] > 0:
            deficit_quad += (max(0.0, ret_vec_hm3[0] - efetiva) / ret_vec_hm3[0]) ** 2
        if efetiva + 1e-9 < solicitada:
            falhas += 1
            deficit += solicitada - efetiva
        if vol < vol_min:
            vol_min = vol
    return falhas, entregue, deficit, racionamento, vol_min, deficit_quad


def largura_minima_curvas(nmetas):
    """Menor distância, em fração da capacidade, entre duas curvas vizinhas em qualquer mês."""
    if len(nmetas) < 2:
        return 1.0
    return float(np.min(np.diff(np.sort(nmetas, axis=0), axis=0)))


def objetivo_entrega(d, entregue, deficit_quad):
    """Valor do segundo objetivo (menor é melhor) conforme o critério escolhido."""
    return float(deficit_quad) if d["criterio_entrega"] == "gradual" else -float(entregue)


class ProblemaNiveisMeta(Problem):
    """Variáveis: um nível meta por estado restritivo (ordenados dentro da avaliação)."""

    def __init__(self, dados):
        self.d = dados
        n_var = len(dados["ret_vec_hm3"]) - 1
        super().__init__(n_var=n_var, n_obj=2, n_ieq_constr=3, xl=NIVEL_MIN, xu=NIVEL_MAX)

    def curvas(self, x):
        d = self.d
        niveis = np.sort(np.asarray(x, dtype=float))
        nmetas = calculo_volume_meta_fast(niveis, d["aflu_prob"], d["evap_ano"], d["dem_total_hm3"], d["cap_hm3"],
                                          d["cav_vol"], d["cav_area"], d["ninicio"])
        return niveis, nmetas

    def _evaluate(self, X, out, *args, **kwargs):
        d = self.d
        F = np.zeros((len(X), 2))
        G = np.zeros((len(X), 3))
        for i, x in enumerate(X):
            niveis, nmetas = self.curvas(x)
            folga = np.min(np.diff(niveis)) if len(niveis) > 1 else 1.0
            G[i, 0] = FOLGA_ENTRE_NIVEIS - folga
            G[i, 1] = VOLUME_META_MIN - float(np.min(nmetas))
            G[i, 2] = d["largura_minima"] - largura_minima_curvas(nmetas)
            falhas, entregue, _, _, _, deficit_quad = avaliar_operacao(
                nmetas, d["aflu_hm3"], d["evap_m"], d["ret_vec_hm3"], d["cap_hm3"], d["cav_vol"], d["cav_area"],
                d["mes_inicio"])
            F[i, 0] = falhas
            F[i, 1] = objetivo_entrega(d, entregue, deficit_quad)
        out["F"] = F
        out["G"] = G


def preparar_dados(payload: ParetoPayload):
    if not 2 <= int(payload.quantidade_faixas) <= 4:
        raise ValueError("A quantidade de níveis deve estar entre 2 e 4.")
    if len(payload.frac_durb) != payload.quantidade_faixas or len(payload.frac_dsup) != payload.quantidade_faixas:
        raise ValueError("Informe uma fração de retirada para cada nível.")
    if payload.durb_m3s + payload.dsupl_m3s <= 0:
        raise ValueError("Informe uma demanda maior que zero.")
    aflu, evap, cap_hm3, cav_vol, cav_area = carregar_dados_fisicos(
        payload.reservatorio, payload.mes_inicio, payload.ano_inicio, payload.mes_fim, payload.ano_fim)
    if len(aflu) < 24:
        raise ValueError("O período escolhido precisa ter pelo menos 24 meses de vazões.")
    aflu_hm3 = aflu * HM3_POR_M3S
    evap_m = evap / 1000.0
    ret_vec_hm3 = (payload.durb_m3s * HM3_POR_M3S * np.array(payload.frac_durb, dtype=float)
                   + payload.dsupl_m3s * HM3_POR_M3S * np.array(payload.frac_dsup, dtype=float))
    meses = np.array([(payload.mes_inicio - 1 + i) % 12 + 1 for i in range(len(aflu_hm3))])
    aflu_prob_12, evap_ano_12 = np.zeros(12), np.zeros(12)
    for m in range(1, 13):
        sel = meses == m
        aflu_prob_12[m - 1] = np.quantile(aflu_hm3[sel], payload.prob) if np.any(sel) else 0.0
        evap_ano_12[m - 1] = evap_m[sel][0] if np.any(sel) else 0.0
    k = payload.ninicio - 1
    return {
        "aflu_hm3": aflu_hm3, "evap_m": evap_m, "cap_hm3": float(cap_hm3), "cav_vol": cav_vol, "cav_area": cav_area,
        "ret_vec_hm3": ret_vec_hm3, "dem_total_hm3": (payload.durb_m3s + payload.dsupl_m3s) * HM3_POR_M3S,
        "aflu_prob": np.concatenate([aflu_prob_12[k:], aflu_prob_12[:k]]),
        "evap_ano": np.concatenate([evap_ano_12[k:], evap_ano_12[:k]]),
        "ninicio": int(payload.ninicio), "mes_inicio": int(payload.mes_inicio),
        "largura_minima": float(payload.largura_minima),
        "criterio_entrega": payload.criterio_entrega,
    }


def descrever_solucao(problema, x, nomes):
    d = problema.d
    niveis, nmetas = problema.curvas(x)
    falhas, entregue, deficit, racionamento, vol_min, deficit_quad = avaliar_operacao(
        nmetas, d["aflu_hm3"], d["evap_m"], d["ret_vec_hm3"], d["cap_hm3"], d["cav_vol"], d["cav_area"], d["mes_inicio"])
    garantias = engine_simulacao_temporal(nmetas, d["aflu_hm3"], d["evap_m"], d["ret_vec_hm3"], d["cap_hm3"],
                                          d["cav_vol"], d["cav_area"], d["mes_inicio"])
    n = len(d["aflu_hm3"])
    demanda_normal = d["ret_vec_hm3"][0] * n
    return {
        "niveis_meta": niveis[::-1].tolist(),
        "matriz_curvas": nmetas[::-1].tolist(),
        "faixas_nomes": nomes,
        "meses_falha": int(falhas),
        "garantia_percent": round(100.0 * (1 - falhas / n), 3),
        "entregue_hm3": round(float(entregue), 4),
        "entregue_percent": round(100.0 * entregue / demanda_normal, 3) if demanda_normal > 0 else 0.0,
        "vazao_media_lps": round(1000.0 * entregue / n / HM3_POR_M3S, 2),
        "deficit_hm3": round(float(deficit), 4),
        "meses_racionamento": int(racionamento),
        "volume_minimo_percent": round(100.0 * vol_min / d["cap_hm3"], 2),
        # raiz da média dos déficits relativos ao quadrado: o critério "gradual" em % da demanda
        "deficit_quadratico_percent": round(100.0 * np.sqrt(deficit_quad / n), 4),
        "objetivo_entrega": objetivo_entrega(d, entregue, deficit_quad),
        "garantias_obtidas": [float(g) for g in garantias],
        # volume ao fim de cada mês, para o gráfico da série histórica da solução escolhida
        "volumes_historicos": np.round(simular_serie_historica_fast(
            nmetas, d["aflu_hm3"], d["evap_m"], d["ret_vec_hm3"], d["cap_hm3"], d["cav_vol"], d["cav_area"],
            d["mes_inicio"]), 3).tolist(),
    }


def escolher_destaques(solucoes):
    """Índices das soluções de segurança total, máxima entrega e equilíbrio (joelho)."""
    if not solucoes:
        return {}
    sem_falha = [i for i, s in enumerate(solucoes) if s["meses_falha"] == 0]
    seg = min(sem_falha, key=lambda i: solucoes[i]["objetivo_entrega"]) if sem_falha else None
    maxi = min(range(len(solucoes)), key=lambda i: (solucoes[i]["objetivo_entrega"], solucoes[i]["meses_falha"]))
    f = np.array([[s["meses_falha"], s["objetivo_entrega"]] for s in solucoes], dtype=float)
    faixa = np.where(np.ptp(f, axis=0) > 0, np.ptp(f, axis=0), 1.0)
    norm = (f - f.min(axis=0)) / faixa
    equilibrio = int(np.argmin(np.hypot(norm[:, 0], norm[:, 1])))
    return {"seguranca_total": seg, "maxima_entrega": maxi, "equilibrio": equilibrio}


def pareto_generator(payload: ParetoPayload):
    try:
        nomes = nomes_faixas(payload.quantidade_faixas, payload.faixas_nomes)
        problema = ProblemaNiveisMeta(preparar_dados(payload))
        algoritmo = NSGA2(pop_size=payload.populacao, crossover=SBX(prob=0.9, eta=15), mutation=PM(eta=20),
                          eliminate_duplicates=True)
        algoritmo.setup(problema, termination=get_termination("n_gen", payload.geracoes), seed=payload.seed, verbose=False)
        while algoritmo.has_next():
            algoritmo.next()
            gen = algoritmo.n_gen
            if gen % 5 == 0 or not algoritmo.has_next():
                yield f"data: {json.dumps({'status': 'progresso', 'cenario_id': payload.cenario_id, 'iteracao': min(gen, payload.geracoes), 'total_iteracoes': payload.geracoes})}\n\n"

        opt = algoritmo.opt
        viaveis = [ind for ind in opt if ind.CV is None or float(np.max(ind.CV)) <= 0]
        vistos, solucoes = set(), []
        for ind in viaveis:
            s = descrever_solucao(problema, ind.X, nomes)
            chave = (s["meses_falha"], round(s["objetivo_entrega"], 6))
            if chave in vistos:
                continue
            vistos.add(chave)
            solucoes.append(s)
        solucoes.sort(key=lambda s: (-s["objetivo_entrega"], s["meses_falha"]))
        # referência: operação sem curvas-guia (estado Normal o tempo todo)
        d = problema.d
        # curvas abaixo de zero: nem o reservatório vazio entra em racionamento
        sem_curvas = np.full((len(d["ret_vec_hm3"]) - 1, 12), -1.0)
        f_ref, e_ref, def_ref, _, vmin_ref, dq_ref = avaliar_operacao(sem_curvas, d["aflu_hm3"], d["evap_m"], d["ret_vec_hm3"],
                                                               d["cap_hm3"], d["cav_vol"], d["cav_area"], d["mes_inicio"])
        n = len(d["aflu_hm3"])
        referencia = {
            "meses_falha": int(f_ref),
            "garantia_percent": round(100.0 * (1 - f_ref / n), 3),
            "entregue_hm3": round(float(e_ref), 4),
            "entregue_percent": round(100.0 * e_ref / (d["ret_vec_hm3"][0] * n), 3),
            "vazao_media_lps": round(1000.0 * e_ref / n / HM3_POR_M3S, 2),
            "deficit_hm3": round(float(def_ref), 4),
            "volume_minimo_percent": round(100.0 * vmin_ref / d["cap_hm3"], 2),
            "deficit_quadratico_percent": round(100.0 * np.sqrt(dq_ref / n), 4),
        }
        for i, s in enumerate(solucoes):
            s["id"] = i
        resultado = {
            "status": "sucesso",
            "metodo": "nsga2",
            "cenario_id": payload.cenario_id,
            "reservatorio": payload.reservatorio,
            "quantidade_faixas": int(payload.quantidade_faixas),
            "faixas_nomes": nomes,
            "meses": int(len(problema.d["aflu_hm3"])),
            "mes_inicio": int(payload.mes_inicio),
            "ano_inicio": int(payload.ano_inicio),
            "capacidade_hm3": problema.d["cap_hm3"],
            "demanda_normal_lps": round(1000.0 * problema.d["ret_vec_hm3"][0] / HM3_POR_M3S, 3),
            "avaliacoes": int(algoritmo.evaluator.n_eval),
            "parametros": {"populacao": payload.populacao, "geracoes": payload.geracoes, "seed": payload.seed,
                           "largura_minima": payload.largura_minima, "criterio_entrega": payload.criterio_entrega},
            "solucoes": solucoes,
            "destaques": escolher_destaques(solucoes),
            "referencia_sem_racionamento": referencia,
        }
        yield f"data: {json.dumps(resultado)}\n\n"
    except Exception as erro:  # mensagem exibida na interface
        yield f"data: {json.dumps({'status': 'erro', 'cenario_id': payload.cenario_id, 'mensagem': str(erro)})}\n\n"


@router.post("/pareto")
def pareto_endpoint(payload: ParetoPayload):
    return StreamingResponse(pareto_generator(payload), media_type="text/event-stream")
