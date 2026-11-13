import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
S = json.load(open("sens.json"))
AZ, LA, VE = "#2b74d8", "#e8622c", "#1aaa77"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
fig, ax = plt.subplots(2, 3, figsize=(11.55, 6.16), dpi=200)
def p(a, serie, y, cor, titulo, xl, x=lambda r: r["x"]):
    xs = [x(r) for r in S[serie]]; ys = [r[y] for r in S[serie]]
    a.plot(xs, ys, "-o", color=cor, lw=2.2, ms=6)
    a.set_title(titulo, loc="left", fontsize=10.3); a.set_xlabel(xl)
    a.grid(True, color="#e3e3e3", lw=0.8); a.set_axisbelow(True)
    for s in ("top", "right"): a.spines[s].set_visible(False)
    for s in ("left", "bottom"): a.spines[s].set_color("#777")
    a.tick_params(colors="#444"); a.set_ylim(0, max(ys) * 1.08 if max(ys) > 0 else 1)
p(ax[0,0], "v0", "falhas", AZ, "(a) Flor do Campo: falhas × volume inicial", "Volume inicial (% cap.)")
p(ax[0,1], "dem", "falhas", AZ, "(b) Flor do Campo: falhas × demanda", "Demanda (L/s)")
p(ax[0,2], "evap", "falhas", AZ, "(c) Flor do Campo: falhas × evaporação", "Fator multiplicador da evaporação")
p(ax[1,0], "gat", "vmin", LA, "(d) Quixeramobim: vol. mínimo (%)", "Gatilho de transferência (% cap.)")
p(ax[1,1], "trf", "falhas", LA, "(e) Quixeramobim: falhas", "Vazão transferida no estado Normal (L/s)")
p(ax[1,2], "trf", "meses_transf", VE, "(f) Quixeramobim: meses com transferência", "Vazão transferida no estado Normal (L/s)")
fig.tight_layout(h_pad=2.2)
fig.savefig("sensibilidade.png")
