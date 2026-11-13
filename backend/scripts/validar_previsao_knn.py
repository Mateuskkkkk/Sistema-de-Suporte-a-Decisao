"""Validação retrospectiva da previsão k-NN, deixando um ano de fora por vez.

Uso (na pasta backend):
    python scripts/validar_previsao_knn.py [--reservatorios Mundaú Carnaubal ...] [--ano-final 2021]

Para cada reservatório, período do histórico, mês de emissão, conjunto de
preditores, valor de k e variável-alvo, o script retira cada ano do histórico,
faz a previsão com os demais anos (inclusive a padronização) e compara com o
observado.

Variáveis-alvo: volume afluente da quadra (fevereiro a maio) e do ano civil
(janeiro a dezembro) do ano-alvo, em hm³.

Métricas:
- CRPS da previsão e da climatologia (todos os anos de treino com o mesmo peso),
  calculado de forma exata sobre a distribuição discreta dos anos e seus pesos,
  equivalente a infinitos cenários sorteados;
- CRPSS = 1 - CRPS médio da previsão / CRPS médio da climatologia;
- área sob a curva ROC para o evento "volume abaixo do tercil inferior", com o
  tercil calculado só com os anos de treino;
- correlação de Spearman entre a mediana prevista e o observado;
- intervalos de 90% por reamostragem dos anos (bootstrap, semente fixa).

Saída em resultados/previsao_knn/: validacao_<reservatorio>.csv (uma linha por
configuração), validacao_<reservatorio>_anos.csv (uma linha por ano e
configuração) e metadados_validacao.json.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import sqlite3
import subprocess
import sys
import unicodedata
from pathlib import Path

import numpy as np

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
import previsao_knn as pk  # noqa: E402

RESERVATORIOS = ["Mundaú", "Carnaubal", "Barragem do Batalhão", "Fogareiro", "Quixeramobim"]
MESES_EMISSAO = [10, 11, 12, 1]
PERIODOS_INICIO = [1912, 1951]
ALVOS = {"quadra": [(0, m) for m in pk.MESES_QUADRA], "ano": [(0, m) for m in range(1, 13)]}
N_BOOTSTRAP = 2000
SEMENTE_BOOTSTRAP = 2026
SAIDA = BACKEND / "resultados" / "previsao_knn"
ORDEM_MESES = {nome: i + 1 for i, nome in enumerate(pk.MESES_NOME)}


def slug(texto: str) -> str:
    t = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode().lower()
    return "_".join("".join(c if c.isalnum() else " " for c in t).split())


def ler_dados(db: Path, reservatorio: str):
    con = sqlite3.connect(db)
    try:
        vazoes = {}
        for ano, mes, v in con.execute(
            'SELECT Ano, "Mês", "Vazão (m³/s)" FROM vazoes WHERE nome_reservatorio = ?', (reservatorio,)
        ):
            m = ORDEM_MESES.get(str(mes).upper()[:3])
            if m is not None and v is not None and math.isfinite(float(v)):
                vazoes[(int(ano), m)] = float(v) * pk.HM3_POR_M3S
        indices = {}
        for ano, mes, nome, v in con.execute("SELECT ano, mes, indice, valor FROM indices_climaticos"):
            indices.setdefault(nome, {})[(int(ano), int(mes))] = float(v)
        fonte = [dict(zip(("indice", "descricao", "fonte", "url", "unidade", "periodo_base", "primeiro_mes",
                           "ultimo_mes", "atualizado_em", "sha256_arquivos"), linha))
                 for linha in con.execute("SELECT * FROM indices_climaticos_fonte")]
    finally:
        con.close()
    if not vazoes:
        raise SystemExit(f"Sem vazões para {reservatorio}.")
    return vazoes, indices, fonte


def alvo_por_ano(anos, vazoes, meses):
    valores = []
    for a in anos:
        v = [vazoes.get((a + dy, m)) for dy, m in meses]
        valores.append(np.nan if any(x is None for x in v) else float(np.sum(v)))
    return np.array(valores)


def spearman(a, b):
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def quantil_ponderado(valores, prob, q):
    ordem = np.argsort(valores, kind="stable")
    v, p = np.asarray(valores)[ordem], np.asarray(prob)[ordem]
    acumulada = np.cumsum(p)
    return float(v[np.searchsorted(acumulada, q - 1e-12)])


def bootstrap(crps_f, crps_c, prob, evento, rng):
    n = len(crps_f)
    crpss, aucs = [], []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, n)
        cc = crps_c[idx].mean()
        crpss.append(1.0 - crps_f[idx].mean() / cc if cc > 0 else np.nan)
        aucs.append(pk.area_roc(prob[idx], evento[idx]))
    def ic(v):
        v = np.array(v, dtype=float)
        v = v[np.isfinite(v)]
        return (float(np.quantile(v, 0.05)), float(np.quantile(v, 0.95))) if len(v) else (np.nan, np.nan)
    return ic(crpss), ic(aucs)


def validar_configuracao(hist, alvos_hist, k):
    """Previsões deixando um ano de fora; devolve as linhas por ano e o resumo."""
    anos, X = hist["anos"], hist["X"]
    n = len(anos)
    linhas = []
    for i in range(n):
        treino = np.arange(n) != i
        res = pk.probabilidades(X[treino], anos[treino], X[i], k=k)
        prob = res["prob"]
        valores = alvos_hist[treino]
        p_clim = np.full(len(valores), 1.0 / len(valores))
        y = float(alvos_hist[i])
        limiar = float(np.quantile(valores, 1.0 / 3.0))
        linhas.append({
            "ano": int(anos[i]),
            "observado": y,
            "p10": quantil_ponderado(valores, prob, 0.10),
            "mediana": quantil_ponderado(valores, prob, 0.50),
            "p90": quantil_ponderado(valores, prob, 0.90),
            "limiar_tercil_inferior": limiar,
            "evento_seco": int(y < limiar),
            "prob_tercil_inferior": pk.prob_abaixo(valores, prob, limiar),
            "prob_tercil_inferior_clim": pk.prob_abaixo(valores, p_clim, limiar),
            "crps": pk.crps_discreto(valores, prob, y),
            "crps_clim": pk.crps_discreto(valores, p_clim, y),
            "vizinhos": " ".join(str(int(anos[treino][j])) for j in res["ordem"]),
        })
    return linhas


def resumir(linhas, rng):
    crps_f = np.array([l["crps"] for l in linhas])
    crps_c = np.array([l["crps_clim"] for l in linhas])
    prob = np.array([l["prob_tercil_inferior"] for l in linhas])
    evento = np.array([bool(l["evento_seco"]) for l in linhas])
    obs = np.array([l["observado"] for l in linhas])
    med = np.array([l["mediana"] for l in linhas])
    (c_inf, c_sup), (a_inf, a_sup) = bootstrap(crps_f, crps_c, prob, evento, rng)
    return {
        "n_anos": len(linhas),
        "crps_medio": float(crps_f.mean()),
        "crps_clim_medio": float(crps_c.mean()),
        "crpss": float(1.0 - crps_f.mean() / crps_c.mean()),
        "crpss_ic90_inf": c_inf,
        "crpss_ic90_sup": c_sup,
        "auc_tercil_inferior": pk.area_roc(prob, evento),
        "auc_ic90_inf": a_inf,
        "auc_ic90_sup": a_sup,
        "n_anos_secos": int(evento.sum()),
        "spearman_mediana_observado": spearman(med, obs),
    }


def commit_atual():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=BACKEND, capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reservatorios", nargs="+", default=RESERVATORIOS)
    ap.add_argument("--ano-final", type=int, default=2021, help="último ano-alvo do histórico (comum aos reservatórios)")
    ap.add_argument("--db", default=str(BACKEND / "banco_site.db"))
    ap.add_argument("--saida", default=str(SAIDA))
    args = ap.parse_args()

    saida = Path(args.saida)
    saida.mkdir(parents=True, exist_ok=True)
    campos_resumo = ["reservatorio", "periodo", "mes_emissao", "conjunto", "preditores", "k", "k_relativo", "alvo",
                     "n_anos", "crps_medio", "crps_clim_medio", "crpss", "crpss_ic90_inf", "crpss_ic90_sup",
                     "auc_tercil_inferior", "auc_ic90_inf", "auc_ic90_sup", "n_anos_secos",
                     "spearman_mediana_observado", "anos_excluidos"]
    campos_anos = ["reservatorio", "periodo", "mes_emissao", "conjunto", "k", "alvo", "ano", "observado", "p10",
                   "mediana", "p90", "limiar_tercil_inferior", "evento_seco", "prob_tercil_inferior",
                   "prob_tercil_inferior_clim", "crps", "crps_clim", "vizinhos"]
    fonte_indices = None
    for reservatorio in args.reservatorios:
        vazoes, indices, fonte_indices = ler_dados(Path(args.db), reservatorio)
        rng = np.random.default_rng(SEMENTE_BOOTSTRAP)
        resumo_linhas, ano_linhas = [], []
        for inicio in PERIODOS_INICIO:
            periodo = f"{inicio}-{args.ano_final}"
            anos = range(inicio, args.ano_final + 1)
            for mes in MESES_EMISSAO:
                for conjunto, preditores in pk.CONJUNTOS.items():
                    hist = pk.montar_historico(anos, mes, preditores, indices, vazoes)
                    n = len(hist["anos"])
                    r = pk.k_padrao(n - 1)
                    for k_rel, k in (("raiz(n)-2", r - 2), ("raiz(n)", r), ("raiz(n)+2", r + 2)):
                        if not 1 <= k <= n - 1:
                            continue
                        for alvo, meses_alvo in ALVOS.items():
                            alvos_hist = alvo_por_ano(hist["anos"], vazoes, meses_alvo)
                            linhas = validar_configuracao(hist, alvos_hist, k)
                            res = resumir(linhas, rng)
                            base = {"reservatorio": reservatorio, "periodo": periodo, "mes_emissao": pk.MESES_NOME[mes - 1],
                                    "conjunto": conjunto, "k": k, "alvo": alvo}
                            resumo_linhas.append({**base, "preditores": "+".join(preditores), "k_relativo": k_rel,
                                                  **res, "anos_excluidos": len(hist["excluidos"])})
                            ano_linhas.extend({**base, **l} for l in linhas)
            print(f"{reservatorio} {periodo}: ok", file=sys.stderr)
        nome = slug(reservatorio)
        with open(saida / f"validacao_{nome}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=campos_resumo)
            w.writeheader()
            for l in resumo_linhas:
                w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in l.items()})
        with open(saida / f"validacao_{nome}_anos.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=campos_anos)
            w.writeheader()
            for l in ano_linhas:
                w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in l.items()})

    metadados = {
        "gerado_em": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "commit": commit_atual(),
        "numpy": np.__version__,
        "python": sys.version.split()[0],
        "reservatorios": args.reservatorios,
        "periodos": [f"{i}-{args.ano_final}" for i in PERIODOS_INICIO],
        "meses_emissao": [pk.MESES_NOME[m - 1] for m in MESES_EMISSAO],
        "conjuntos": {k: list(v) for k, v in pk.CONJUNTOS.items()},
        "preditores": pk.PREDITORES,
        "k": "arredondamento de raiz(n-1), com n-1 anos de treino, e esse valor -2 e +2",
        "alvos": {"quadra": "fevereiro a maio do ano-alvo", "ano": "janeiro a dezembro do ano-alvo"},
        "crps": "exato sobre a distribuição discreta dos anos de treino e seus pesos",
        "evento_roc": "volume abaixo do tercil inferior dos anos de treino",
        "bootstrap": {"n": N_BOOTSTRAP, "semente": SEMENTE_BOOTSTRAP, "intervalo": "5% a 95%"},
        "fonte_indices": fonte_indices,
    }
    (saida / "metadados_validacao.json").write_text(json.dumps(metadados, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Resultados em {saida}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
