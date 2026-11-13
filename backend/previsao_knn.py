"""Previsão de afluências por reamostragem dos k vizinhos mais próximos.

Método de Lall e Sharma (1996, doi:10.1029/95WR02966), na forma usada por
Souza Filho e Lall (2003, doi:10.1029/2002WR001373) para reservatórios do Ceará:

1. cada ano do histórico é descrito por um vetor de estado conhecido no mês de
   emissão (índices climáticos médios dos três meses anteriores e, opcionalmente,
   a afluência da quadra chuvosa do ano anterior);
2. as variáveis são padronizadas com a média e o desvio dos anos do histórico;
3. calcula-se a distância euclidiana entre o ano a prever e cada ano do histórico;
4. os k anos mais próximos recebem pesos (1/j) / soma(1/i, i = 1..k);
5. sorteiam-se N cenários com esses pesos; cada cenário é a sequência mensal
   observada do ano sorteado, do mês de emissão até dezembro do ano-alvo, sem
   interpolação nem suavização.

A climatologia (todos os anos com o mesmo peso) sai da mesma função.

Convenções:
- o ano-alvo Y é o ano civil da quadra prevista (fevereiro a maio);
- emissões de julho a dezembro ocorrem no ano Y-1; de janeiro a junho, no ano Y;
- a emissão é no início do mês: os dados do próprio mês ainda não existem e o
  mês de emissão já faz parte do horizonte previsto;
- volumes em hm³ (vazão em m³/s vezes 2,592, mês de 30 dias, como no simulador).

O módulo não acessa o banco nem a API e não usa biblioteca de aprendizado de máquina.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

HM3_POR_M3S = 2.592
MESES_QUADRA = (2, 3, 4, 5)
MESES_NOME = ("JAN", "FEV", "MAR", "ABR", "MAI", "JUN", "JUL", "AGO", "SET", "OUT", "NOV", "DEZ")
SEMENTE_PADRAO = 1
N_CENARIOS_PADRAO = 1000
QUANTIS = (0.10, 0.50, 0.90)

PREDITORES = {
    "nino34": "Niño 3.4 médio dos três meses anteriores à emissão (°C)",
    "gradiente": "TNA menos TSA médio dos três meses anteriores à emissão (°C)",
    "quadra_anterior": "ln(1 + afluência de fevereiro a maio do ano anterior, em hm³)",
    "acumulada": "ln(1 + afluência de janeiro até o mês anterior à emissão, em hm³); só para emissões de fevereiro a junho",
}
CONJUNTOS = {
    "nino": ("nino34",),
    "atlantico": ("gradiente",),
    "ambos": ("nino34", "gradiente"),
    "ambos_quadra": ("nino34", "gradiente", "quadra_anterior"),
}

Mes = Tuple[int, int]  # (deslocamento em anos em relação ao ano-alvo, mês 1–12)


# ---------------------------------------------------------------------------
# Calendário relativo ao ano-alvo
# ---------------------------------------------------------------------------
def emissao_relativa(mes_emissao: int) -> Mes:
    """Mês de emissão em relação ao ano-alvo: julho a dezembro caem no ano anterior."""
    if not 1 <= int(mes_emissao) <= 12:
        raise ValueError("O mês de emissão deve estar entre 1 e 12.")
    return (-1 if mes_emissao >= 7 else 0, int(mes_emissao))


def somar_meses(mes: Mes, n: int) -> Mes:
    total = mes[0] * 12 + (mes[1] - 1) + n
    return (total // 12, total % 12 + 1)


def meses_anteriores(mes_emissao: int, n: int = 3) -> List[Mes]:
    """Os n meses anteriores à emissão, em ordem cronológica."""
    e = emissao_relativa(mes_emissao)
    return [somar_meses(e, -i) for i in range(n, 0, -1)]


def horizonte(mes_emissao: int) -> List[Mes]:
    """Meses previstos: do mês de emissão até dezembro do ano-alvo."""
    e = emissao_relativa(mes_emissao)
    meses, atual = [], e
    while atual <= (0, 12):
        meses.append(atual)
        atual = somar_meses(atual, 1)
    return meses


def rotulos_horizonte(mes_emissao: int, ano_alvo: Optional[int] = None) -> List[str]:
    rotulos = []
    for dy, m in horizonte(mes_emissao):
        if ano_alvo is None:
            rotulos.append(MESES_NOME[m - 1] + ("/Y-1" if dy < 0 else ""))
        else:
            rotulos.append(f"{MESES_NOME[m - 1]}/{ano_alvo + dy}")
    return rotulos


def indices_quadra(mes_emissao: int) -> List[int]:
    """Posições dos meses de fevereiro a maio do ano-alvo dentro do horizonte."""
    return [i for i, (dy, m) in enumerate(horizonte(mes_emissao)) if dy == 0 and m in MESES_QUADRA]


# ---------------------------------------------------------------------------
# Montagem do vetor de estado e das sequências
# ---------------------------------------------------------------------------
def gradiente_atlantico(indices: Dict[str, Dict[Mes, float]]) -> Dict[Mes, float]:
    tna, tsa = indices.get("tna", {}), indices.get("tsa", {})
    return {k: tna[k] - tsa[k] for k in tna if k in tsa}


def _media(serie: Dict[Mes, float], ano: int, meses: Sequence[Mes]) -> Optional[float]:
    valores = [serie.get((ano + dy, m)) for dy, m in meses]
    if any(v is None or not math.isfinite(v) for v in valores):
        return None
    return float(np.mean(valores))


def _soma(serie: Dict[Mes, float], ano: int, meses: Sequence[Mes]) -> Optional[float]:
    valores = [serie.get((ano + dy, m)) for dy, m in meses]
    if any(v is None or not math.isfinite(v) for v in valores):
        return None
    return float(np.sum(valores))


def validar_preditores(preditores: Sequence[str], mes_emissao: int) -> Tuple[str, ...]:
    preditores = tuple(preditores)
    if not preditores:
        raise ValueError("Informe ao menos um preditor.")
    desconhecidos = [p for p in preditores if p not in PREDITORES]
    if desconhecidos:
        raise ValueError(f"Preditores desconhecidos: {', '.join(desconhecidos)}.")
    if len(set(preditores)) != len(preditores):
        raise ValueError("Preditores repetidos.")
    if "acumulada" in preditores and emissao_relativa(mes_emissao) not in [(0, m) for m in range(2, 7)]:
        raise ValueError("A afluência acumulada só existe para emissões de fevereiro a junho.")
    return preditores


def vetor_estado(
    ano: int,
    mes_emissao: int,
    preditores: Sequence[str],
    indices: Dict[str, Dict[Mes, float]],
    vazoes_hm3: Dict[Mes, float],
) -> Optional[np.ndarray]:
    """Vetor de estado do ano-alvo `ano`, ou None se faltar algum dado."""
    anteriores = meses_anteriores(mes_emissao)
    grad = None
    valores = []
    for p in preditores:
        if p == "nino34":
            v = _media(indices.get("nino34", {}), ano, anteriores)
        elif p == "gradiente":
            grad = grad if grad is not None else gradiente_atlantico(indices)
            v = _media(grad, ano, anteriores)
        elif p == "quadra_anterior":
            s = _soma(vazoes_hm3, ano, [(-1, m) for m in MESES_QUADRA])
            v = None if s is None else math.log1p(max(s, 0.0))
        elif p == "acumulada":
            e = emissao_relativa(mes_emissao)
            s = _soma(vazoes_hm3, ano, [(0, m) for m in range(1, e[1])])
            v = None if s is None else math.log1p(max(s, 0.0))
        else:
            raise ValueError(f"Preditor desconhecido: {p}")
        if v is None:
            return None
        valores.append(v)
    return np.array(valores, dtype=float)


def sequencia(ano: int, mes_emissao: int, vazoes_hm3: Dict[Mes, float]) -> Optional[np.ndarray]:
    """Afluências observadas (hm³) do ano `ano` ao longo do horizonte, ou None se houver falha."""
    valores = [vazoes_hm3.get((ano + dy, m)) for dy, m in horizonte(mes_emissao)]
    if any(v is None or not math.isfinite(v) for v in valores):
        return None
    return np.array(valores, dtype=float)


def montar_historico(
    anos: Iterable[int],
    mes_emissao: int,
    preditores: Sequence[str],
    indices: Dict[str, Dict[Mes, float]],
    vazoes_hm3: Dict[Mes, float],
) -> dict:
    """Anos com vetor de estado e sequência completos; os demais ficam fora, com o motivo."""
    anos_ok, X, S, excluidos = [], [], [], []
    for ano in sorted(set(int(a) for a in anos)):
        x = vetor_estado(ano, mes_emissao, preditores, indices, vazoes_hm3)
        s = sequencia(ano, mes_emissao, vazoes_hm3)
        if x is None:
            excluidos.append({"ano": ano, "motivo": "preditor ausente"})
        elif s is None:
            excluidos.append({"ano": ano, "motivo": "falha na série de afluência"})
        else:
            anos_ok.append(ano)
            X.append(x)
            S.append(s)
    p, h = len(preditores), len(horizonte(mes_emissao))
    return {
        "anos": np.array(anos_ok, dtype=int),
        "X": np.array(X, dtype=float).reshape(len(anos_ok), p),
        "S": np.array(S, dtype=float).reshape(len(anos_ok), h),
        "excluidos": excluidos,
    }


# ---------------------------------------------------------------------------
# Núcleo do método
# ---------------------------------------------------------------------------
def k_padrao(n: int) -> int:
    """k = arredondamento de raiz(n), com meio arredondado para cima."""
    return max(1, int(math.floor(math.sqrt(n) + 0.5)))


def pesos_lall_sharma(k: int) -> np.ndarray:
    """Peso do vizinho de ordem j: (1/j) / soma(1/i, i = 1..k)."""
    inv = 1.0 / np.arange(1, k + 1, dtype=float)
    return inv / inv.sum()


def padronizar(X_hist: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Padroniza com média e desvio (ddof = 1) só dos anos do histórico."""
    media = X_hist.mean(axis=0)
    desvio = X_hist.std(axis=0, ddof=1) if X_hist.shape[0] > 1 else np.ones(X_hist.shape[1])
    desvio = np.where(np.isfinite(desvio) & (desvio > 0), desvio, 1.0)
    return (X_hist - media) / desvio, (np.asarray(x, dtype=float) - media) / desvio, media, desvio


def probabilidades(
    X_hist: np.ndarray,
    anos_hist: np.ndarray,
    x: np.ndarray,
    k: Optional[int] = None,
    climatologia: bool = False,
    pesos: str = "lall_sharma",
) -> dict:
    """Probabilidade de cada ano do histórico e a lista ordenada dos vizinhos."""
    n = X_hist.shape[0]
    if n == 0:
        raise ValueError("O histórico não tem anos com dados completos.")
    anos_hist = np.asarray(anos_hist, dtype=int)
    if climatologia:
        prob = np.full(n, 1.0 / n)
        ordem = np.argsort(anos_hist, kind="stable")
        return {"k": n, "prob": prob, "ordem": ordem, "distancias": np.zeros(n), "pesos": prob[ordem]}
    k = k_padrao(n) if k is None else int(k)
    if not 1 <= k <= n:
        raise ValueError(f"k deve estar entre 1 e {n} (anos disponíveis).")
    Z, z, _, _ = padronizar(X_hist, x)
    dist = np.sqrt(((Z - z) ** 2).sum(axis=1))
    # empate de distância: o ano mais antigo vem primeiro, para o resultado ser determinístico
    ordem = np.lexsort((anos_hist, dist))[:k]
    w = pesos_lall_sharma(k) if pesos == "lall_sharma" else np.full(k, 1.0 / k)
    prob = np.zeros(n)
    prob[ordem] = w
    return {"k": k, "prob": prob, "ordem": ordem, "distancias": dist[ordem], "pesos": w}


def quantis(matriz: np.ndarray, qs: Sequence[float] = QUANTIS) -> Dict[str, List[float]]:
    q = np.quantile(matriz, qs, axis=0)
    return {f"p{int(round(v * 100))}": q[i].tolist() for i, v in enumerate(qs)}


def prever(
    X_hist: np.ndarray,
    anos_hist: np.ndarray,
    S_hist: np.ndarray,
    x: Optional[np.ndarray],
    k: Optional[int] = None,
    n_cenarios: int = N_CENARIOS_PADRAO,
    semente: int = SEMENTE_PADRAO,
    climatologia: bool = False,
    pesos: str = "lall_sharma",
    indices_quadra_horizonte: Optional[Sequence[int]] = None,
) -> dict:
    """Gera os cenários e os resumos de uma previsão (ou da climatologia).

    X_hist (n x p), anos_hist (n) e S_hist (n x h) vêm de montar_historico; x é o
    vetor de estado do ano a prever (ignorado na climatologia).
    """
    if n_cenarios < 1:
        raise ValueError("O número de cenários deve ser positivo.")
    res = probabilidades(X_hist, anos_hist, x, k=k, climatologia=climatologia, pesos=pesos)
    ordem, w = res["ordem"], res["pesos"]
    rng = np.random.default_rng(int(semente))
    sorteio = rng.choice(len(ordem), size=int(n_cenarios), replace=True, p=w)
    linhas = ordem[sorteio]
    cenarios = S_hist[linhas].copy()
    acumulado = np.cumsum(cenarios, axis=1)
    saida = {
        "anos_usados": int(X_hist.shape[0]),
        "k": int(res["k"]),
        "vizinhos": [
            {"ano": int(anos_hist[i]), "distancia": float(d), "peso": float(p)}
            for i, d, p in zip(ordem, res["distancias"], w)
        ],
        "probabilidade_por_ano": {int(a): float(p) for a, p in zip(anos_hist, res["prob"]) if p > 0},
        "anos_sorteados": anos_hist[linhas].astype(int).tolist(),
        "cenarios": cenarios,
        "quantis_mensais": quantis(cenarios),
        "quantis_acumulados": quantis(acumulado),
    }
    if indices_quadra_horizonte:
        quadra = cenarios[:, list(indices_quadra_horizonte)].sum(axis=1)
        saida["quadra"] = quadra
        saida["quantis_quadra"] = {k_: v[0] for k_, v in quantis(quadra[:, None]).items()}
    return saida


# ---------------------------------------------------------------------------
# Distribuição discreta (usada na validação)
# ---------------------------------------------------------------------------
def crps_discreto(valores: np.ndarray, prob: np.ndarray, observado: float) -> float:
    """CRPS exato de uma distribuição discreta: E|X - y| - E|X - X'| / 2."""
    valores = np.asarray(valores, dtype=float)
    prob = np.asarray(prob, dtype=float)
    termo1 = np.sum(prob * np.abs(valores - observado))
    termo2 = 0.5 * np.sum(prob[:, None] * prob[None, :] * np.abs(valores[:, None] - valores[None, :]))
    return float(termo1 - termo2)


def prob_abaixo(valores: np.ndarray, prob: np.ndarray, limite: float) -> float:
    return float(np.sum(np.asarray(prob)[np.asarray(valores) < limite]))


def area_roc(prob_evento: Sequence[float], evento: Sequence[bool]) -> float:
    """Área sob a curva ROC pela estatística de Mann-Whitney (empates contam meio)."""
    p = np.asarray(prob_evento, dtype=float)
    e = np.asarray(evento, dtype=bool)
    pos, neg = p[e], p[~e]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    maior = (pos[:, None] > neg[None, :]).sum()
    empate = (pos[:, None] == neg[None, :]).sum()
    return float((maior + 0.5 * empate) / (len(pos) * len(neg)))


# ---------------------------------------------------------------------------
# Serialização
# ---------------------------------------------------------------------------
def para_json(obj):
    """Converte arrays e números do NumPy para tipos do Python e troca NaN e Inf por None."""
    if isinstance(obj, dict):
        return {str(k): para_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [para_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return para_json(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        return v if math.isfinite(v) else None
    return obj
