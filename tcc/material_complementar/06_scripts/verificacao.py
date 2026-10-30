"""Verificação numérica do simulador: comparação com a planilha e conferência das regras."""
from collections import Counter
import openpyxl

TOL = 1e-9          # tolerância numérica (hm³)
K = 2.592           # conversão de m³/s para hm³/mês (mês de 30 dias)
COMPONENTES = ["Vi", "I", "E", "R", "S", "Vf"]


def ler_planilha(arquivo):
    """Lê, da aba Verificacao, as linhas mensais calculadas pela planilha."""
    aba = openpyxl.load_workbook(arquivo, data_only=True)["Verificacao"]
    meses = []
    for lin in aba.iter_rows(min_row=24, max_row=1367, values_only=True):
        if lin[21] not in ("Sim", "Não"):          # coluna V: falha
            continue
        meses.append({"Vi": lin[7], "I": lin[4], "E": lin[14], "R": lin[16],
                      "S": lin[19], "Vf": lin[20], "falha": lin[21]})
    return meses


def ler_simulador(df):
    """Converte a tabela mensal do simulador para os mesmos componentes."""
    return [{"Vi": m["Armazenamento Inicial"], "I": m["Afluências (hm³/mês)"],
             "E": m["Evaporação (hm³)"], "R": m["Demanda Atendida (m³/s)"] * K,
             "S": m["Vertimento (hm³)"], "Vf": m["Armazenamento Final"],
             "falha": m["Falha"]} for _, m in df.iterrows()]


def comparar(planilha, simulador):
    """Conta os meses em que planilha e simulador divergem."""
    assert len(planilha) == len(simulador)
    divergentes = 0
    for p, s in zip(planilha, simulador):
        dif = max(abs(p[c] - s[c]) for c in COMPONENTES)
        if dif > TOL or p["falha"] != s["falha"]:
            divergentes += 1
    return divergentes


def estado_esperado(pct, faixas, mes):
    """Estado e racionamento definidos pelas curvas mensais (faixas)."""
    for limite, rac, estado in sorted((f["limites"][mes], f["rac"], f["estado"])
                                      for f in faixas):
        if pct <= limite:
            return estado, rac
    return "Normal", 0.0


def conferir_regras(df, faixas, cap):
    """Confere, mês a mês, a tabela exportada pelo simulador."""
    v = Counter()
    vf_anterior = None
    for _, m in df.iterrows():
        vi, vf = m["Armazenamento Inicial"], m["Armazenamento Final"]
        estado, rac = estado_esperado(100 * vi / cap, faixas, m["Ordem_Mês"] - 1)
        v["estado"] += estado != m["Modo Operação"]
        v["racionamento"] += abs(rac - m["Racionamento (%)"]) > 1e-9
        d_apl = m["Demanda Solicitada (m³/s)"] * (1 - m["Racionamento (%)"] / 100) * K
        r = m["Demanda Atendida (m³/s)"] * K
        v["atendida"] += r > d_apl + TOL
        v["falha"] += (round(r, 6) < round(d_apl, 6)) != (m["Falha"] == "Sim")
        eps = vf - (vi + m["Afluências (hm³/mês)"] - r
                    - m["Evaporação (hm³)"] - m["Vertimento (hm³)"])
        v["massa"] += abs(eps) > TOL
        v["limites"] += not (-TOL <= vf <= cap + TOL)
        if vf_anterior is not None:
            v["continuidade"] += abs(vi - vf_anterior) > TOL
        vf_anterior = vf
    return v
