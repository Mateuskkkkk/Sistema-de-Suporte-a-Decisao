"""Métricas de Hashimoto, Stedinger e Loucks (1982) para as três aplicações.

Confiabilidade: fração dos meses sem falha.
Resiliência: probabilidade de o mês seguinte a uma falha ser satisfatório
(transições falha -> sucesso / meses com falha).
Vulnerabilidade: severidade das falhas, expressa pelo déficit médio por mês de
falha e pelo maior déficit mensal, em hm³; acrescenta-se a maior duração de
uma sequência contínua de falhas.
O déficit é medido em relação à demanda aplicada no mês (após racionamento),
o mesmo critério que define a falha.
"""
from simlib import *
import json

def metricas(falha, deficit):
    falha = np.asarray(falha, dtype=bool); deficit = np.asarray(deficit, dtype=float)
    n, nf = len(falha), int(falha.sum())
    trans = int(np.sum(falha[:-1] & ~falha[1:]))
    dur, run = 0, 0
    for x in falha:
        run = run + 1 if x else 0; dur = max(dur, run)
    return dict(meses=n, falhas=nf, confiabilidade=100 * (1 - nf / n),
                resiliencia=(trans / nf if nf else None), episodios=int(np.sum(falha & ~np.r_[False, falha[:-1]])),
                maior_duracao=dur, deficit_medio=(float(deficit[falha].mean()) if nf else 0.0),
                deficit_max=float(deficit.max()), deficit_total=float(deficit.sum()))

def deficit_aplicado(df):
    apl = df["Demanda Solicitada (m³/s)"] * (1 - df["Racionamento (%)"] / 100) * K
    return (apl - df["Demanda Atendida (m³/s)"] * K).clip(lower=0)

out = {}
VM1 = [36.6, 34.6, 33.3, 32.5, 32.0, 30.9, 59.7, 56.2, 52.3, 48.3, 43.9, 39.9]
VM2 = [22.2, 20.7, 19.7, 19.2, 19.0, 18.3, 39.9, 37.2, 34.1, 31.0, 27.6, 24.6]
VM3 = [7.9, 7.0, 6.5, 6.3, 6.3, 5.9, 18.3, 16.6, 14.8, 12.9, 11.0, 9.2]
d = simular([res(172, 50, 60, plano=faixas(VM1, VM2, VM3, rac=(8.33, 33.33, 55.00)))], niveis_meta=True)["Flor do Campo"]
s = simular([res(172, 50, 60)])["Flor do Campo"]
for nome, df in [("flor_curvas", d), ("flor_sem", s)]:
    out[nome] = metricas(df["Falha"] == "Sim", deficit_aplicado(df))

r1 = res(53, 50, 0, gatilho=10); r2 = res(202, 50, 0, gatilho=0)
dd = simular([r1, r2], modo="Paralelo", conjunta_lps=160)
c, b = dd["Carnaubal"], dd["Barragem do Batalhão"]
sol = 0.160 * K
at = (c["Demanda Atendida (m³/s)"] + b["Demanda Atendida (m³/s)"]) * K
out["carnaubal_conjunta"] = metricas(at < sol - 1e-6, (sol - at).clip(lower=0))

f = res(119, 100, 272); q = res(16, 100, 342, gatilho=30)
dd = simular([f, q], modo="Série", conjunta_lps=500, ini=("JAN", 1911), fim=("DEZ", 2019), niveis_meta=True,
             cenario=main.FOGAREIRO_QUIXERAMOBIM_CENARIO_1_ID)
for nome, df in [("fogareiro", dd["Fogareiro"]), ("quixeramobim", dd["Quixeramobim"])]:
    out[nome] = metricas(df["Falha"] == "Sim", deficit_aplicado(df))

json.dump(out, open("metricas_hashimoto.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(out, indent=1, ensure_ascii=False))
