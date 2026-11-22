# Gráfico da validação cruzada (5 dobras) do RF multiclasse, a partir do cv_ajuste.csv gravado pelo notebook 05m.
from pathlib import Path
import pandas as pd, numpy as np, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter
AZUL, LARANJA = "#2471a3", "#e67e22"
plt.rcParams.update({"savefig.dpi": 200, "axes.spines.top": False, "axes.spines.right": False})
def pt(x, c=0): return f"{x:,.{c}f}".replace(",", "X").replace(".", ",").replace("X", ".")

def grafico(PASTA, SAIDA, modelo):
    cv = pd.read_csv(Path(PASTA) / "cv_ajuste.csv"); SAIDA = Path(SAIDA)
    n = len(cv); fig = plt.figure(figsize=(12.5, 0.75*n + 2.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.6, 1.15, 1.15], wspace=0.08)
    a0, a1, a2 = (fig.add_subplot(gs[0, i]) for i in range(3))
    for i, r in enumerate(cv.itertuples()):
        y = n-1-i
        for j in range(n):
            a0.add_patch(Rectangle((j, y-.32), .96, .64, color=LARANJA if j == i else AZUL, lw=0))
        if hasattr(r, "n_treino"):
            info = f"dobra {r.dobra}\ntreino {pt(r.n_treino)} | avaliação {pt(r.n_avaliacao)}\n{r.tempo_treino_s:.1f} s de treino"
        else:  # SVM: o csv não traz os tamanhos; mostra tempo e vetores de suporte
            info = f"dobra {r.dobra}\n{r.tempo_treino_s:.0f} s de treino\n{pt(r.n_sv)} vetores de suporte"
        a0.text(n+.15, y, info, va="center", fontsize=8.5)
    a0.set_xlim(0, n+3.2); a0.set_ylim(-.6, n-.4); a0.axis("off")
    fig.suptitle(f"Validação cruzada no ajuste ({n} dobras), {modelo} multiclasse: cada linha é um treino diferente", x=0.01, ha="left", fontsize=11, fontweight="bold", y=1.04)
    a0.add_patch(Rectangle((0,-1.0),.3,.3,color=AZUL,lw=0,clip_on=False)); a0.text(.4,-.85,"treino (4 partes do ajuste)",fontsize=8.5,va="center")
    a0.add_patch(Rectangle((0,-1.5),.3,.3,color=LARANJA,lw=0,clip_on=False)); a0.text(.4,-1.35,"avaliação (1 parte, que o modelo não viu)",fontsize=8.5,va="center")
    for ax, col, nome in ((a1,"mcc","MCC"),(a2,"f1_macro","F1 macro")):
        v = cv[col].to_numpy(float); ys = np.arange(n)[::-1]
        ax.scatter(v, ys, color="black", s=26, zorder=3); ax.axvline(v.mean(), color=LARANJA, lw=1.4, ls="--")
        for x_, y_ in zip(v, ys): ax.text(x_, y_+.28, pt(x_,4), ha="center", fontsize=7.5)
        m = (v.max()-v.min()) or .01; ax.set_xlim(v.min()-m*.8, v.max()+m*.8); ax.set_ylim(-.6, n-.4)
        ax.set_yticks([]); ax.spines["left"].set_visible(False)
        ax.set_title(f"{nome}\ntraço laranja = média {pt(v.mean(),4)}", fontsize=9.5); ax.grid(axis="x", alpha=.25)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_: pt(x,4)))
    fig.savefig(SAIDA, bbox_inches="tight"); plt.close(fig); print("salvo em", SAIDA.resolve())


# Uso: python grafico_cv_rf_multi.py PASTA_RF SAIDA_RF.png [PASTA_SVM SAIDA_SVM.png]
# PASTA = pasta da execução que contém cv_ajuste.csv (RF: notebook 05m; SVM: notebook 11m)
grafico(sys.argv[1], sys.argv[2], "RF")
if len(sys.argv) >= 5:
    grafico(sys.argv[3], sys.argv[4], "SVM")
