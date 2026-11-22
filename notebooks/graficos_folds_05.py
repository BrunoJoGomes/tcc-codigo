# Gráficos da validação cruzada (5 dobras no ajuste) e das 5 partições da validação, lidos dos arquivos gravados pelo notebook 05.
# Não precisa do kernel anterior nem do ClickHouse: pode rodar em um kernel novo (ou como script), só precisa de pandas e matplotlib.
from pathlib import Path
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter

# Pasta do experimento: usa a variável PASTA se existir (rodando dentro do notebook 05); senão, a execução mais recente em artefatos_modelos/rf_binario.
if "PASTA" not in globals():
    raiz = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
    PASTA = sorted((raiz / "artefatos_modelos" / "rf_binario").glob("rf_bin_*"))[-1]
PASTA = Path(PASTA); print("Lendo de:", PASTA)
cv, part = pd.read_csv(PASTA / "cv_ajuste.csv"), pd.read_csv(PASTA / "dispersao_particoes_validacao.csv")

AZUL, LARANJA, CINZA = "#2471a3", "#e67e22", "#d5d8dc"
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "axes.spines.top": False, "axes.spines.right": False})

def pt(x, casas=0): return f"{x:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")

def figura(tab, k_col, titulo, rotulo_ajuste, rotulo_aval, texto_lateral, arquivo, tem_treino):
    n = len(tab); fig = plt.figure(figsize=(12.5, 0.75 * n + 2.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.6, 1.15, 1.15], wspace=0.08)
    a0, a1, a2 = (fig.add_subplot(gs[0, i]) for i in range(3))
    for i, r in enumerate(tab.itertuples()):
        y = n - 1 - i
        for j in range(n):                                  # n blocos iguais; o bloco i é o avaliado
            avaliado = (j == i)
            cor = LARANJA if avaliado else (AZUL if tem_treino else CINZA)
            a0.add_patch(Rectangle((j, y - .32), .96, .64, color=cor, lw=0))
        a0.text(n + .15, y, texto_lateral(r), va="center", fontsize=8.5)
    a0.set_xlim(0, n + 3.2); a0.set_ylim(-.6, n - .4); a0.axis("off")
    fig.suptitle(titulo, x=0.01, ha="left", fontsize=11, fontweight="bold", y=1.04)
    a0.add_patch(Rectangle((0, -1.0), .3, .3, color=AZUL if tem_treino else CINZA, lw=0, clip_on=False)); a0.text(.4, -.85, rotulo_ajuste, fontsize=8.5, va="center")
    a0.add_patch(Rectangle((0, -1.5), .3, .3, color=LARANJA, lw=0, clip_on=False)); a0.text(.4, -1.35, rotulo_aval, fontsize=8.5, va="center")
    for ax, col, nome, esc in ((a1, "recall", "Recall (ataques)", 100), (a2, "taxa_falso_alarme", "Falso alarme (%)", 100)):
        v = tab[col].to_numpy(float) * esc; ys = np.arange(n)[::-1]
        ax.scatter(v, ys, color="black", s=26, zorder=3); ax.axvline(v.mean(), color=LARANJA, lw=1.4, ls="--")
        for x_, y_ in zip(v, ys): ax.text(x_, y_ + .28, pt(x_, 3), ha="center", fontsize=7.5)
        m = (v.max() - v.min()) or .01; ax.set_xlim(max(v.min() - m * .8, -m * .1) if col == "taxa_falso_alarme" else v.min() - m * .8, min(v.max() + m * .8, 100 + m * .1) if col == "recall" else v.max() + m * .8); ax.set_ylim(-.6, n - .4)
        ax.set_yticks([]); ax.spines["left"].set_visible(False); ax.set_title(f"{nome}\ntraço laranja = média {pt(v.mean(), 3)}", fontsize=9.5); ax.grid(axis="x", alpha=.25); ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: pt(x, 2)))
    fig.savefig(PASTA / arquivo, bbox_inches="tight"); plt.show()

figura(cv, "dobra", f"Validação cruzada no ajuste ({len(cv)} dobras): cada linha é um treino diferente",
       "treino (4 partes do ajuste)", "avaliação (1 parte, que o modelo não viu)",
       lambda r: f"dobra {r.dobra}\ntreino {pt(r.n_treino)} | avaliação {pt(r.n_avaliacao)}\n{r.tempo_treino_s:.1f} s de treino",
       "fig_cv_ajuste_5dobras.png", True)
figura(part, "particao", f"Dispersão na validação ({len(part)} partições): o modelo é o mesmo, só muda o pedaço avaliado",
       "demais partições (não entram nessa medida)", "partição avaliada (modelo treinado no ajuste, sem retreino)",
       lambda r: f"partição {r.particao}\n{pt(r.n)} registros ({pt(r.benign)} BENIGN)",
       "fig_particoes_validacao.png", False)
print("✔ Figuras gravadas em", PASTA)
