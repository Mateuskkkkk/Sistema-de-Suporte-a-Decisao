from simlib import *
import json
out = {}
# --- Flor do Campo: Individual com a Regra 2 da 3ª oficina do hidrossistema (curvas do Cenário 1, 60 L/s)
VM1 = [36.6, 34.6, 33.3, 32.5, 32.0, 30.9, 59.7, 56.2, 52.3, 48.3, 43.9, 39.9]
VM2 = [22.2, 20.7, 19.7, 19.2, 19.0, 18.3, 39.9, 37.2, 34.1, 31.0, 27.6, 24.6]
VM3 = [7.9, 7.0, 6.5, 6.3, 6.3, 5.9, 18.3, 16.6, 14.8, 12.9, 11.0, 9.2]
RAC = (8.33, 33.33, 55.00)  # Regra 2: 60 L/s -> 55 / 40 / 27 L/s no açude
COD, Q, CAP = 172, 60, 105.0
def episodios(serie_bool, datas):
    eps = []; i = 0; v = serie_bool.values
    while i < len(v):
        if v[i]:
            j = i
            while j < len(v) and v[j]: j += 1
            eps.append((datas[i], datas[j-1], j-i)); i = j
        else: i += 1
    return sorted(eps, key=lambda e: -e[2])
d = simular([res(COD, 50, Q, plano=faixas(VM1, VM2, VM3, rac=RAC))], niveis_meta=True)["Flor do Campo"]
d.to_csv("flor_curvas.csv", index=False)
s = simular([res(COD, 50, Q)])["Flor do Campo"]; s.to_csv("flor_sem.csv", index=False)
for nome, df in [("curvas", d), ("sem", s)]:
    ind = indicadores(df, CAP); ind["res"] = float(residuo(df).abs().max())
    ind["data_vmin"] = df.loc[df["Armazenamento Final"].idxmin(), "Data"]
    ind["meses_rac"] = int((df["Racionamento (%)"] > 0).sum())
    ind["vazoes"] = (df["Demanda Solicitada (m³/s)"]*(1-df["Racionamento (%)"]/100)).round(5).value_counts().to_dict()
    ind["meses_vert"] = int((df["Vertimento (hm³)"] > 0).sum())
    ind["meses_vazio"] = int((df["Armazenamento Final"] < 1e-6).sum())
    falha = df["Falha"] == "Sim"
    ind["episodios_falha_top"] = episodios(falha, df.Data)[:6]; ind["n_episodios_falha"] = len(episodios(falha, df.Data))
    ind["falhas_por_estado"] = df.loc[falha, "Modo Operação"].value_counts().to_dict()
    ind["episodios_ss_top"] = episodios(df["Modo Operação"] == "Seca Severa", df.Data)[:5]
    ind["afl_hm3"] = float(df["Afluências (hm³/mês)"].sum())
    ind["ret_hm3"] = float((df["Demanda Atendida (m³/s)"]*K).sum())
    ind["deficit_nominal_hm3"] = float(((df["Demanda Solicitada (m³/s)"]-df["Demanda Atendida (m³/s)"]).clip(lower=0)*K).sum())
    ind["nao_fornecido_rac_hm3"] = float((df["Demanda Solicitada (m³/s)"]*df["Racionamento (%)"]/100*K).sum())
    # falhas por década
    ind["falhas_decada"] = df.loc[falha, "Ano"].floordiv(10).mul(10).value_counts().sort_index().to_dict()
    out["flor_" + nome] = ind
# distribuição de estados por mês do ano
d["mes"] = d["Ordem_Mês"] if "Ordem_Mês" in d else d.index % 12 + 1
tab = pd.crosstab(d["Modo Operação"], d["mes"], normalize="columns").mul(100).round(1)
out["estados_mes"] = tab.to_dict()

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
