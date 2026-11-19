"""Executa as duas partes da verificação para os 25 hidrossistemas."""
import json, os
from simlib import acude, simular, MESES
from verificacao import ler_planilha, ler_simulador, comparar, conferir_regras
PASTA = "/tmp/lo_work/todos"
cfg = json.load(open("cfg_todos.json"))
tot = {"meses": 0, "div": 0}; regras = {}
for h in cfg:
    a = acude(h["cod"]); cap = a["capacidade"]
    ini, fim = tuple(h["ini"]), tuple(h["fim"])
    # parte 1: balanço hídrico (sem racionamento)
    res = dict(a, vol_inicial=0.5 * cap, demanda=h["q"] / 1000, gatilho=0.0, plano_secas_custom=None)
    sim = list(simular([res], ini=ini, fim=fim).values())[0]
    nome = h["arq"].replace(".xlsx", "")
    arq = os.path.join(PASTA, f"Verificacao_{nome}_CAV_banco.xlsx" if "Canoas" in nome else f"Verificacao_{nome}.xlsx")
    n_div = comparar(ler_planilha(arq), ler_simulador(sim))
    tot["meses"] += len(sim); tot["div"] += n_div
    # parte 2: regras de níveis meta e racionamento
    faixas = [{"estado": e, "rac": r, "limites": l} for e, r, l in h["curvas"]]
    plano = [{"Faixa": f["estado"], "Racionamento": f["rac"], "NomeFaixaNormal": "Normal",
              **dict(zip(MESES, f["limites"]))} for f in faixas]
    res["plano_secas_custom"] = plano
    sim2 = list(simular([res], ini=ini, fim=fim, niveis_meta=True).values())[0]
    v = conferir_regras(sim2, faixas, cap)
    regras[h["nome"]] = dict(meses=len(sim2), **v)
    print(f"{h['nome']:32s} meses={len(sim):5d} divergentes={n_div}  regras={dict(v)}")
print("TOTAL", tot, "violacoes regras:", sum(sum(x[k] for k in x if k != 'meses') for x in regras.values()),
      "meses regras:", sum(x['meses'] for x in regras.values()))
