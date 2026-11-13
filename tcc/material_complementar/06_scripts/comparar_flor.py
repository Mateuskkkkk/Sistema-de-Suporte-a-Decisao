from simlib import *
from verificacao import ler_planilha, ler_simulador, comparar
import json, openpyxl
out = {}
for q in (60, 120):
    sim = simular([res(172, 50, q)])["Flor do Campo"]
    sim.to_csv(f"flor_sem_{q}.csv", index=False)
    pl = ler_planilha(f"/tmp/lo_work/Verificacao_Flor_do_Campo_{q}Ls_CAV_banco.xlsx"); si = ler_simulador(sim)
    difs = {c: max(abs(p[c] - s[c]) for p, s in zip(pl, si)) for c in ["Vi", "I", "E", "R", "S", "Vf"]}
    wb = openpyxl.load_workbook(f"/tmp/lo_work/Verificacao_Flor_do_Campo_{q}Ls_CAV_banco.xlsx", data_only=True)["Verificacao"]
    res_pl = max(abs(r[22]) for r in wb.iter_rows(min_row=24, max_row=1367, values_only=True) if r[21] in ("Sim", "Não"))
    tot = lambda L, c: sum(x[c] for x in L)
    dem = q / 1000 * K * len(si)
    out[q] = dict(divergentes=comparar(pl, si), difs=difs, res_planilha=res_pl, res_sim=float(residuo(sim).abs().max()),
                  falhas_pl=sum(p["falha"] == "Sim" for p in pl), falhas_si=sum(s["falha"] == "Sim" for s in si),
                  afl=(tot(pl, "I"), tot(si, "I")), ret=(tot(pl, "R"), tot(si, "R")), evap=(tot(pl, "E"), tot(si, "E")),
                  vert=(tot(pl, "S"), tot(si, "S")), atend=(100*tot(pl, "R")/dem, 100*tot(si, "R")/dem),
                  vmin=(min(p["Vf"] for p in pl), min(s["Vf"] for s in si)), meses=len(si))
    print(q, json.dumps(out[q], indent=None, default=float))
json.dump(out, open("comparacao_flor.json", "w"), indent=1, default=float)
