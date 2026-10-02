from simlib import *
import json, copy
FATOR = {"evap": 1.0}
_orig = main.carregar_serie_simulador
def _patched(*a, **k):
    s = _orig(*a, **k); s["evaporacao_mm"] = s["evaporacao_mm"] * FATOR["evap"]; return s
main.carregar_serie_simulador = _patched
def mund(v0=50, dem=250):
    d = simular([res(61, v0, dem)])["Mundaú"]; i = indicadores(d, 21.3)
    return dict(atend=i["atend_aplicada_pct"], falhas=i["falhas"], vmin=i["vmin_pct"], deficit=i["deficit_hm3"], vert=i["vert_hm3"], evap=i["evap_hm3"])
S = {"v0": [], "dem": [], "evap": [], "gat": [], "trf": []}
for v0 in [0, 25, 50, 75, 100]: S["v0"].append(dict(x=v0, **mund(v0=v0)))
for dem in [150, 200, 250, 300, 350, 400]: S["dem"].append(dict(x=dem, **mund(dem=dem)))
for fe in [0.8, 0.9, 1.0, 1.1, 1.2]:
    FATOR["evap"] = fe; S["evap"].append(dict(x=fe, **mund())); FATOR["evap"] = 1.0
cfg0 = copy.deepcopy(main.FOGAREIRO_QUIXERAMOBIM_CENARIO_1)
def fq(gat=30.0, trf=500.0):
    main.FOGAREIRO_QUIXERAMOBIM_CENARIO_1["gatilho_receptor_percent"] = gat
    f = res(119, 100, 272); q = res(16, 100, 342, gatilho=gat)
    dd = simular([f, q], modo="Série", conjunta_lps=trf, fim=("DEZ", 2019), niveis_meta=True, cenario=main.FOGAREIRO_QUIXERAMOBIM_CENARIO_1_ID)
    main.FOGAREIRO_QUIXERAMOBIM_CENARIO_1.update(copy.deepcopy(cfg0))
    fo, qx = dd["Fogareiro"], dd["Quixeramobim"]
    iq, i_f = indicadores(qx, 7.88), indicadores(fo, 118.0)
    return dict(atend=iq["atend_aplicada_pct"], falhas=iq["falhas"], vmin=iq["vmin_pct"], deficit=iq["deficit_hm3"],
                meses_transf=iq["meses_transf_rec"], vol_transf=float(qx["Transferência Recebida (m³/s)"].sum() * K),
                vmin_fog=i_f["vmin_pct"], falhas_fog=i_f["falhas"], vert_q=iq["vert_hm3"])
for g in [10, 20, 30, 40, 50]: S["gat"].append(dict(x=g, **fq(gat=g)))
for t in [0, 100, 200, 300, 400, 500, 600, 700]: S["trf"].append(dict(x=t, **fq(trf=t)))
json.dump(S, open("sens.json", "w"), indent=1)
for k, v in S.items():
    print(k); print(pd.DataFrame(v).round(3).to_string())
