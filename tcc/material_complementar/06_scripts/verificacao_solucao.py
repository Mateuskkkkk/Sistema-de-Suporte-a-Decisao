"""Verificação da solução: erro do esquema de área média do motor.

A) Caso com solução analítica: área proporcional ao volume, A = a V, sem
   afluência nem retirada. Então dV/dt = -a e(t) V e V_n = V_0 exp(-a sum e).
   O esquema mensal do motor (área média com uma estimativa preliminar) é
   aplicado mês a mês e comparado com a solução exata.
B) Flor do Campo (curva cota-área-volume real, 1911-2021, 60 L/s, sem
   racionamento, volume inicial de 50%): o passo mensal do motor é comparado
   com (B2) a área média iterada até a convergência e com (B3) 30 subpassos
   diários por mês, que distribuem afluência, retirada e evaporação ao longo
   do mês.
"""
import json
from simlib import *
from optimizer_engine import dinamica_mensal_fast, interpola

TOL = 1e-9
COD, Q_LPS = 172, 60.0

s = simular([res(COD, 50, Q_LPS)])["Flor do Campo"]
afl = s["Afluências (hm³/mês)"].to_numpy(float)
lam = s["Evaporação (m)"].to_numpy(float) / 1000.0          # lâmina mensal (m)
cap = acude(COD)["capacidade"]
con = sqlite3.connect(main.DB_PATH)
cav_vol, cav_area = main.ler_cav(con, COD, cap)
con.close()
order = np.argsort(cav_vol); cav_vol, cav_area = cav_vol[order], cav_area[order]
dem = Q_LPS / 1000.0 * K
out = {}

# ---------------- A) solução analítica ----------------
a = float(interpola(cav_vol, cav_area, cap)) / cap          # km²/hm³
cv = np.array([0.0, 2 * cap]); ca = a * cv
lam12 = lam[:12]
for nome, meses in (("12_meses", 12), ("120_meses", 120)):
    v, ac, erros = cap, 0.0, []
    for i in range(meses):
        e = lam12[i % 12]
        v, _, _, _ = dinamica_mensal_fast(v, 0.0, e, 0.0, 0.0, 2 * cap, cv, ca)
        ac += e
        exato = cap * np.exp(-a * ac)
        erros.append(abs(v - exato) / exato)
    out["A_" + nome] = dict(a_km2_por_hm3=a, x_max=float(a * lam12.max()), V_final=v,
                            V_exato=float(cap * np.exp(-a * ac)), erro_rel_final=erros[-1],
                            erro_rel_max=max(erros))

# ---------------- B) Flor do Campo ----------------
def mensal_convergido(v0, i, d, e):
    ai = interpola(cav_vol, cav_area, v0); am = ai
    for _ in range(100):
        v = v0 + i - d - e * am
        an = (ai + interpola(cav_vol, cav_area, max(v, 0.0))) / 2.0
        if abs(an - am) < 1e-14: break
        am = an
    ev = e * am; v = v0 + i - d - ev; ret = d
    if v < 0:
        ret = max(d + v, 0.0); v = v0 + i - ret - ev
        if v < 0: ev += v; v = 0.0
    vert = max(v - cap, 0.0); v = min(v, cap)
    return v, ret, vert, ev

def rodar(passo):
    v = 0.5 * cap; falhas = 0; ev_t = vert_t = 0.0; vols = []
    for t in range(len(afl)):
        if passo == "mensal":
            v, r, vt, ev = dinamica_mensal_fast(v, afl[t], lam[t], dem, 0.0, cap, cav_vol, cav_area)
        elif passo == "convergido":
            v, r, vt, ev = mensal_convergido(v, afl[t], dem, lam[t])
        else:
            r = vt = ev = 0.0
            for _ in range(30):
                v, rd, vd, ed = dinamica_mensal_fast(v, afl[t] / 30, lam[t] / 30, dem / 30, 0.0, cap, cav_vol, cav_area)
                r += rd; vt += vd; ev += ed
        falhas += r < dem - TOL; ev_t += ev; vert_t += vt; vols.append(v)
    return dict(falhas=int(falhas), evaporacao_hm3=ev_t, vertimento_hm3=vert_t,
                volume_medio_pct=100 * float(np.mean(vols)) / cap)

for p in ("mensal", "convergido", "diario"):
    out["B_" + p] = rodar(p)
out["B_simulador"] = dict(falhas=int((s["Falha"] == "Sim").sum()),
                          evaporacao_hm3=float(s["Evaporação (hm³)"].sum()))
json.dump(out, open("verificacao_solucao.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(out, indent=1, ensure_ascii=False))
