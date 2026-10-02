from simlib import *
import json
out = {}
# --- Mundaú: Individual com níveis meta do Cenário 1 do PGPS como dado de entrada
vm1=[55,52,50,51,54,79,78,75,71,67,63,59]; vm2=[35,32,30,31,35,58,57,54,50,47,43,39]; vm3=[21,18,16,17,21,43,42,39,36,32,28,25]
d = simular([res(61, 50, 250, plano=faixas(vm1, vm2, vm3))], niveis_meta=True)["Mundaú"]
d.to_csv("mundau.csv", index=False)
ind = indicadores(d, 21.3); ind["res"] = float(residuo(d).abs().max())
dmin = d.loc[d["Armazenamento Final"].idxmin(), "Data"]
# episódios de seca severa (sequências)
ss = (d["Modo Operação"] == "Seca Severa").astype(int).values
eps = []; i = 0
while i < len(ss):
    if ss[i]:
        j = i
        while j < len(ss) and ss[j]: j += 1
        eps.append((d.Data[i], d.Data[j-1], j-i)); i = j
    else: i += 1
eps.sort(key=lambda e: -e[2])
ind["data_vmin"] = dmin; ind["episodios_ss_top"] = eps[:5]; ind["n_episodios_ss"] = len(eps)
ind["meses_rac"] = int((d["Racionamento (%)"] > 0).sum())
# meses em que a vazão atendida foi 250/125/75
ind["vazoes"] = d["Demanda Atendida (m³/s)"].round(4).value_counts().to_dict()
ind["meses_cheio"] = int((d["Vertimento (hm³)"] > 0).sum())
out["mundau"] = ind
# --- Carnaubal–Barragem do Batalhão: Paralelo
r1 = res(53, 50, 0, gatilho=10); r2 = res(202, 50, 0, gatilho=0)
dd = simular([r1, r2], modo="Paralelo", conjunta_lps=160)
c, b = dd["Carnaubal"], dd["Barragem do Batalhão"]
c.to_csv("carnaubal.csv", index=False); b.to_csv("batalhao.csv", index=False)
resp_b = (b["Demanda Solicitada (m³/s)"] > 1e-9)
fal_sis = int(((c["Falha"] == "Sim") & (b["Falha"] == "Sim")).sum())
# meses em que a demanda conjunta foi atendida integralmente pelo responsável
at_conj = (c["Demanda Atendida (m³/s)"] + b["Demanda Atendida (m³/s)"]) * K
sol_conj = 0.160 * K
trocas = int((resp_b.astype(int).diff().abs() > 0).sum())
# períodos de responsabilidade do Batalhão
per = []; i = 0; rb = resp_b.values
while i < len(rb):
    if rb[i]:
        j = i
        while j < len(rb) and rb[j]: j += 1
        per.append((c.Data[i], c.Data[j-1], j-i)); i = j
    else: i += 1
out["carnaubal"] = dict(
    carn=indicadores(c, r1["capacidade"]), bat=indicadores(b, r2["capacidade"]),
    cap_c=r1["capacidade"], cap_b=r2["capacidade"],
    meses_bat=int(resp_b.sum()), trocas=trocas, periodos_bat=per, n_periodos=len(per),
    falhas_sistemicas=fal_sis, atend_conj_pct=float(100*at_conj.sum()/(sol_conj*len(c))),
    meses_falha_conj=int((at_conj < sol_conj - 1e-6).sum()),
    deficit_conj=float((sol_conj - at_conj).clip(lower=0).sum()),
    res=float(max(residuo(c).abs().max(), residuo(b).abs().max())),
    meses_carn_abaixo_gatilho=int((c["Armazenamento Inicial"] < 0.1*r1["capacidade"]).sum()))
# --- Fogareiro–Quixeramobim: Série, regras consolidadas do PGPS como dado de entrada
f = res(119, 100, 272); q = res(16, 100, 342, gatilho=30)
dd = simular([f, q], modo="Série", conjunta_lps=500, ini=("JAN", 1911), fim=("DEZ", 2019), niveis_meta=True,
             cenario=main.FOGAREIRO_QUIXERAMOBIM_CENARIO_1_ID)
fo, qx = dd["Fogareiro"], dd["Quixeramobim"]
fo.to_csv("fogareiro.csv", index=False); qx.to_csv("quixeramobim.csv", index=False)
trf = qx["Transferência Recebida (m³/s)"]
out["fq"] = dict(fog=indicadores(fo, f["capacidade"]), qx=indicadores(qx, q["capacidade"]),
    cap_f=f["capacidade"], cap_q=q["capacidade"],
    meses_transf=int((trf > 1e-12).sum()), vol_transf_hm3=float(trf.sum()*K),
    transf_por_estado=qx.loc[trf > 1e-12].groupby(fo.loc[trf > 1e-12, "Modo Operação"])["Transferência Recebida (m³/s)"].agg(["count","mean"]).to_dict(),
    meses_gatilho=int((trf > 1e-12).sum()),
    ativ_por_ano=qx.loc[trf > 1e-12, "Ano"].value_counts().sort_index().to_dict(),
    estados_fog=fo["Modo Operação"].value_counts().to_dict(),
    falhas_sistemicas=int(((fo["Falha"] == "Sim") & (qx["Falha"] == "Sim")).sum()),
    res=float(max(residuo(fo).abs().max(), residuo(qx).abs().max())),
    transf_enviada_eq=float((fo["Transferência Enviada (m³/s)"] - trf).abs().max()))
json.dump(out, open("apps.json", "w"), indent=1, default=str, ensure_ascii=False)
print(json.dumps(out, indent=1, default=str, ensure_ascii=False)[:6000])
