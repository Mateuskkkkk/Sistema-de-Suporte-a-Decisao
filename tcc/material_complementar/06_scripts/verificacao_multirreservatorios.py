"""Recalculo independente dos modos Serie (Fogareiro-Quixeramobim) e Paralelo
(Carnaubal-Barragem do Batalhao) a partir da descricao do TCC (Secoes 3.6-3.8,
Tabelas 4 e 6) e do banco, sem importar o codigo do simulador. Compara mes a mes
com as tabelas exportadas pelo simulador (material complementar, 04_aplicacoes)."""
import sqlite3, sys
import numpy as np, pandas as pd

DB = sys.argv[1] if len(sys.argv) > 1 else "../../../backend/banco_site.db"
APL = "../04_aplicacoes/"  # tabelas mensais exportadas pelo simulador
K = 2.592
TOL = 1e-9
MESES = {'JAN': 1, 'FEV': 2, 'MAR': 3, 'ABR': 4, 'MAI': 5, 'JUN': 6,
         'JUL': 7, 'AGO': 8, 'SET': 9, 'OUT': 10, 'NOV': 11, 'DEZ': 12}
db = sqlite3.connect(DB)


def dados(nome, fim):
    cod, cap, est = db.execute('select COD, "CAPAC (m³)", "Est. Evap." from acudes where CORPO=?', (nome,)).fetchone()
    serie = {}
    for _, a, m, q in db.execute('select * from vazoes where nome_reservatorio=?', (nome,)):
        mm = MESES.get(str(m).upper()[:3]) or int(m)
        a = int(float(a))
        if 1911 <= a <= fim:
            serie[(a, mm)] = float(q)
    chaves = sorted(serie)
    pts = list(db.execute('select "VOLUME (m³)", "AREA (km²)" from cav where COD=? order by "VOLUME (m³)"', (cod,)))
    ev = db.execute("select * from evaporacao where COD=?", (est,)).fetchone()
    return dict(cap=cap / 1e6, Q=np.array([serie[c] for c in chaves]), mes=np.array([c[1] for c in chaves]),
                cv=np.array([p[0] / 1e6 for p in pts]), ca=np.array([p[1] for p in pts]),
                lam=np.array(ev[2:14]) / 1000.0)


def area(r, v):
    return float(np.interp(min(max(v, r["cv"][0]), r["cv"][-1]), r["cv"], r["ca"]))


def evap(r, V, I, T, D, e):
    """Evaporacao pela area media (Eq. 3.3)."""
    a0 = area(r, V)
    return e * (a0 + area(r, V + I + T - D - e * a0)) / 2.0


def balanco(r, V, I, T, D, e):
    """Eq. 3.2 a 3.6: devolve volume final, retirada, vertimento e evaporacao."""
    E = evap(r, V, I, T, D, e)
    P = V + I + T - D - E
    R = D if P >= 0 else max(0.0, D + P)
    Vf = max(0.0, P + D - R)
    S = max(0.0, Vf - r["cap"])
    return min(Vf, r["cap"]), R, S, E


def comparar(nome, calc, sim):
    cols = {"Vi": "Armazenamento Inicial", "Vf": "Armazenamento Final", "E": "Evaporação (hm³)",
            "S": "Vertimento (hm³)"}
    dif = {k: float(np.max(np.abs(np.array(calc[k]) - sim[c].values))) for k, c in cols.items()}
    dif["R"] = float(np.max(np.abs(np.array(calc["R"]) - sim["Demanda Atendida (m³/s)"].values * K)))
    div = 0
    for i in range(len(sim)):
        d = max(abs(calc[k][i] - sim[c].iloc[i]) for k, c in cols.items())
        d = max(d, abs(calc["R"][i] - sim["Demanda Atendida (m³/s)"].iloc[i] * K))
        div += d > TOL
    print(f"  {nome}: meses divergentes = {div} de {len(sim)}; máx. dif. " +
          ", ".join(f"{k} {v:.1e}" for k, v in dif.items()))
    return div


# ---------------------------------------------------------------- Paralelo
def paralelo():
    c, b = dados("Carnaubal", 2021), dados("Barragem do Batalhão", 2021)
    Dj = 0.160 * K
    gat = 0.10 * c["cap"]
    Vc, Vb = 0.5 * c["cap"], 0.5 * b["cap"]
    rc = {k: [] for k in ("Vi", "Vf", "R", "E", "S")}
    rb = {k: [] for k in ("Vi", "Vf", "R", "E", "S")}
    resp_b = 0
    for t in range(len(c["Q"])):
        e = c["lam"][c["mes"][t] - 1]
        Ic, Ib = c["Q"][t] * K, b["Q"][t] * K
        Dc, Db = Dj, 0.0
        if Vc < gat:  # Secao 3.7.3: troca se a 2a unidade tiver volume apos afluencia e evaporacao
            if Vb + Ib - evap(b, Vb, Ib, 0.0, 0.0, e) >= Dj:
                Dc, Db = 0.0, Dj
                resp_b += 1
        for r, V, I, D, out in ((c, Vc, Ic, Dc, rc), (b, Vb, Ib, Db, rb)):
            Vf, R, S, E = balanco(r, V, I, 0.0, D, e)
            for k, v in zip(("Vi", "Vf", "R", "E", "S"), (V, Vf, R, E, S)):
                out[k].append(v)
        Vc, Vb = rc["Vf"][-1], rb["Vf"][-1]
    print(f"Paralelo (Carnaubal–Batalhão): meses com o Batalhão responsável = {resp_b}")
    div = comparar("Carnaubal", rc, pd.read_excel(APL + "Carnaubal_Batalhao_tabela_mensal.xlsx", "Carnaubal"))
    div += comparar("Batalhão", rb, pd.read_excel(APL + "Carnaubal_Batalhao_tabela_mensal.xlsx", "Batalhao"))
    return div


# ---------------------------------------------------------------- Serie
LIM = {"Seca Severa": [27.966102, 26.271186, 25.423729, 30.508475, 41.525424, 39.830508, 38.135593, 37.288136, 35.593220, 33.898305, 31.355932, 29.661017],
       "Seca": [39.830508, 38.135593, 37.288136, 41.525424, 57.627119, 55.084746, 53.389831, 51.694915, 50.000000, 47.457627, 44.915254, 42.372881],
       "Alerta": [49.152542, 46.610169, 45.762712, 50.000000, 67.796610, 66.101695, 64.406780, 62.711864, 60.169492, 57.627119, 54.237288, 51.694915]}
DEM_F = {"Normal": 272.0, "Alerta": 220.0, "Seca": 139.6, "Seca Severa": 6.0}     # Tabela 6
DEM_Q = {"Normal": 342.0, "Alerta": 301.8, "Seca": 213.3, "Seca Severa": 70.5}
TRF = {"Normal": 500.0, "Alerta": 400.0, "Seca": 300.0, "Seca Severa": 85.0}


def serie():
    f, q = dados("Fogareiro", 2019), dados("Quixeramobim", 2019)
    Vf_, Vq = f["cap"], q["cap"]  # volume inicial de 100% (Tabela 4)
    gat = 0.30 * q["cap"]
    rf = {k: [] for k in ("Vi", "Vf", "R", "E", "S")}
    rq = {k: [] for k in ("Vi", "Vf", "R", "E", "S")}
    est_calc, trf_calc = [], []
    for t in range(len(f["Q"])):
        m = f["mes"][t] - 1
        e = f["lam"][m]
        p = 100 * Vf_ / f["cap"]
        estado = "Normal"
        for nome in ("Seca Severa", "Seca", "Alerta"):  # ordem crescente de limite (Secao 3.8)
            if p <= LIM[nome][m]:
                estado = nome
                break
        If, Iq = f["Q"][t] * K, q["Q"][t] * K
        Df, Dq = DEM_F[estado] / 1000 * K, DEM_Q[estado] / 1000 * K
        T = 0.0
        if Vq + Iq - evap(q, Vq, Iq, 0.0, 0.0, e) < gat:  # receptor apos afluencia e evaporacao, sem retirada
            disp = max(0.0, Vf_ + If - Df - evap(f, Vf_, If, 0.0, Df, e))
            T = min(TRF[estado] / 1000 * K, disp)  # limitada a disponibilidade do fornecedor
        for r, V, I, Tt, D, out in ((f, Vf_, If, -T, Df, rf), (q, Vq, Iq, T, Dq, rq)):
            Vn, R, S, E = balanco(r, V, I, Tt, D, e)
            for k, v in zip(("Vi", "Vf", "R", "E", "S"), (V, Vn, R, E, S)):
                out[k].append(v)
        Vf_, Vq = rf["Vf"][-1], rq["Vf"][-1]
        est_calc.append(estado)
        trf_calc.append(T)
    sf = pd.read_excel(APL + "Fogareiro_Quixeramobim_tabela_mensal.xlsx", "Fogareiro")
    sq = pd.read_excel(APL + "Fogareiro_Quixeramobim_tabela_mensal.xlsx", "Quixeramobim")
    n_est = int(sum(a != b for a, b in zip(est_calc, sf["Modo Operação"])))
    n_trf = int(np.sum(np.abs(np.array(trf_calc) - sf["Transferência Enviada (m³/s)"].values * K) > TOL))
    print(f"Série (Fogareiro–Quixeramobim): meses com transferência = {sum(x > 0 for x in trf_calc)}; "
          f"volume transferido = {sum(trf_calc):.2f} hm³; estados divergentes = {n_est}; transferências divergentes = {n_trf}")
    div = comparar("Fogareiro", rf, sf) + comparar("Quixeramobim", rq, sq)
    return div + n_est + n_trf


if __name__ == "__main__":
    total = paralelo() + serie()
    print("TOTAL de divergências:", total)
