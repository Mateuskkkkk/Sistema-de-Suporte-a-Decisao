import json, pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6, "axes.edgecolor": "#8a8984",
                     "axes.labelcolor": "#2b2b2a", "xtick.color": "#52514e", "ytick.color": "#52514e", "legend.frameon": False,
                     "savefig.dpi": 220})
B, O, A, Y = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
EST = {"Normal": "#cfe3f7", "Alerta": "#f6dfa5", "Seca": "#f2b28c", "Seca Severa": "#e57a6a"}
def dt(df): return pd.to_datetime(df["Data"] + "-01")
# ---- Mundaú
m = pd.read_csv("mundau.csv"); t = dt(m); cap = 21.3
vm1=[55,52,50,51,54,79,78,75,71,67,63,59]; vm2=[35,32,30,31,35,58,57,54,50,47,43,39]; vm3=[21,18,16,17,21,43,42,39,36,32,28,25]
mi = m["Ordem_Mês"].values - 1
fig, ax = plt.subplots(2, 1, figsize=(10, 5.4), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
for nome, cor in EST.items():
    sel = (m["Modo Operação"] == nome).values
    ax[0].fill_between(t, 0, 100, where=sel, color=cor, step="post", linewidth=0, label=nome)
ax[0].plot(t, m["Armazenamento Final"] / cap * 100, color="#1f3b63", lw=0.9, label="Volume final (%)")
ax[0].set_ylabel("Volume (% da capacidade)"); ax[0].set_ylim(0, 100); ax[0].legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.13), fontsize=8)
ax[1].plot(t, m["Demanda Atendida (m³/s)"] * 1000, color=B, lw=0.8, drawstyle="steps-post")
ax[1].set_ylabel("Atendida (L/s)"); ax[1].set_ylim(0, 280); ax[1].set_yticks([75, 125, 250])
ax[1].xaxis.set_major_locator(mdates.YearLocator(10)); ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
fig.tight_layout(); fig.savefig("fig/mundau-volume-estados.png"); plt.close(fig)
# Mundaú: estados por mês do ano (permanência sazonal)
tab = pd.crosstab(m["Ordem_Mês"], m["Modo Operação"]).reindex(columns=list(EST)).fillna(0)
tab = tab.div(tab.sum(1), axis=0) * 100
fig, ax = plt.subplots(figsize=(8, 3.2)); base = np.zeros(12)
for nome in EST:
    ax.bar(range(12), tab[nome].values, bottom=base, color=EST[nome], edgecolor="white", linewidth=1.2, label=nome); base += tab[nome].values
ax.set_xticks(range(12)); ax.set_xticklabels(["JAN","FEV","MAR","ABR","MAI","JUN","JUL","AGO","SET","OUT","NOV","DEZ"])
ax.set_ylabel("Meses no estado (%)"); ax.set_ylim(0, 100); ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.16)); ax.grid(axis="x", visible=False)
fig.tight_layout(); fig.savefig("fig/mundau-estados-mes.png"); plt.close(fig)
# ---- Carnaubal–Batalhão
c = pd.read_csv("carnaubal.csv"); b = pd.read_csv("batalhao.csv"); t = dt(c)
fig, ax = plt.subplots(2, 1, figsize=(10, 5.2), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
ax[0].plot(t, c["Armazenamento Final"] / 46.621 * 100, color=B, lw=0.9, label="Carnaubal")
ax[0].plot(t, b["Armazenamento Final"] / 1.6388 * 100, color=O, lw=0.8, label="Barragem do Batalhão")
ax[0].axhline(10, color="#52514e", lw=0.8, ls="--"); ax[0].text(t.iloc[5], 12, "gatilho de Carnaubal (10%)", fontsize=8, color="#52514e")
ax[0].set_ylabel("Volume (% da capacidade)"); ax[0].set_ylim(0, 102); ax[0].legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.12))
resp_b = (b["Demanda Solicitada (m³/s)"] > 1e-9).astype(int)
falha = (c["Falha"] == "Sim").astype(int)
ax[1].fill_between(t, 0, resp_b, step="post", color=O, lw=0, label="Batalhão responsável")
ax[1].fill_between(t, 0, -falha, step="post", color="#e34948", lw=0, label="Falha em Carnaubal")
ax[1].set_ylim(-1.2, 1.2); ax[1].set_yticks([]); ax[1].legend(ncol=2, loc="upper left", fontsize=8)
ax[1].xaxis.set_major_locator(mdates.YearLocator(10)); ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
fig.tight_layout(); fig.savefig("fig/carnaubal-batalhao.png"); plt.close(fig)
# ---- Fogareiro–Quixeramobim
f = pd.read_csv("fogareiro.csv"); q = pd.read_csv("quixeramobim.csv"); t = dt(f)
fig, ax = plt.subplots(2, 1, figsize=(10, 5.2), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]})
ax[0].plot(t, f["Armazenamento Final"] / 118 * 100, color=B, lw=0.9, label="Fogareiro (controlador)")
ax[0].plot(t, q["Armazenamento Final"] / 7.88 * 100, color=O, lw=0.7, label="Quixeramobim (receptor)")
ax[0].axhline(30, color="#52514e", lw=0.8, ls="--"); ax[0].text(t.iloc[5], 32, "gatilho de Quixeramobim (30%)", fontsize=8, color="#52514e")
ax[0].set_ylabel("Volume (% da capacidade)"); ax[0].set_ylim(0, 102); ax[0].legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.12))
ax[1].bar(t, q["Transferência Recebida (m³/s)"] * 1000, width=28, color=A)
ax[1].set_ylabel("Transferência (L/s)"); ax[1].set_yticks([85, 300, 400, 500])
ax[1].xaxis.set_major_locator(mdates.YearLocator(10)); ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
fig.tight_layout(); fig.savefig("fig/fogareiro-quixeramobim.png"); plt.close(fig)
# ---- Sensibilidade
S = json.load(open("sens.json"))
fig, axs = plt.subplots(2, 3, figsize=(10.5, 5.6))
def painel(ax, dados, xl, titulo, chave="atend", cor=B, y2=None):
    x = [d["x"] for d in dados]; y = [d[chave] for d in dados]
    ax.plot(x, y, color=cor, lw=2, marker="o", ms=5); ax.set_xlabel(xl); ax.set_title(titulo, fontsize=9, loc="left")
painel(axs[0,0], S["v0"], "Volume inicial (% cap.)", "(a) Mundaú: falhas × volume inicial", "falhas")
painel(axs[0,1], S["dem"], "Demanda (L/s)", "(b) Mundaú: falhas × demanda", "falhas")
painel(axs[0,2], S["evap"], "Fator multiplicador da evaporação", "(c) Mundaú: falhas × evaporação", "falhas")
painel(axs[1,0], S["gat"], "Gatilho de transferência (% cap.)", "(d) Quixeramobim: vol. mínimo (%)", "vmin", O)
painel(axs[1,1], S["trf"], "Vazão transferida no estado Normal (L/s)", "(e) Quixeramobim: falhas", "falhas", O)
painel(axs[1,2], S["trf"], "Vazão transferida no estado Normal (L/s)", "(f) Quixeramobim: meses com transferência", "meses_transf", A)
for a in axs.flat: a.set_ylim(bottom=0)
fig.tight_layout(); fig.savefig("fig/sensibilidade.png"); plt.close(fig)
print("ok")
