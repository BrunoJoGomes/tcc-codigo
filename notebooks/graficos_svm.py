# Figuras do SVM binário equivalentes às do RF (notebook 05), lidas dos arquivos gravados pelo notebook 11. Não precisa do ClickHouse.
import json, math
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter

# Pasta do experimento: usa a variável PASTA se existir; senão, a execução mais recente em artefatos_modelos/svm_binario_busca
# (roda de dentro de notebooks/ ou da raiz do projeto).
if "PASTA" not in globals():
    raiz = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
    PASTA = sorted((raiz / "artefatos_modelos" / "svm_binario_busca").glob("svm_busca_*"))[-1]
PASTA = Path(PASTA); print("Lendo de:", PASTA)
SAIDA = PASTA  # as figuras são gravadas na própria pasta da execução, como as do RF
lj = lambda n: json.loads((PASTA / n).read_text(encoding="utf-8"))
met, lim, custo, cfg = lj("metricas_validacao.json"), lj("limiares_validacao.json"), lj("custo_computacional.json"), lj("config_treino.json")
cv, part = pd.read_csv(PASTA / "cv_ajuste.csv"), pd.read_csv(PASTA / "dispersao_particoes_validacao.csv")
sem, real, LIMIAR = met["sem_pesos"], met["proporcao_real"], met["limiar"]
N = cfg["dados"]
cv["n_avaliacao"] = [16246, 16246, 16246, 16245, 16245]; cv["n_treino"] = N["ajuste"] - cv["n_avaliacao"]  # dobras de tamanho 16.245/16.246 (divisão estratificada de 81.228)

AZUL, LARANJA, VERM, CINZA = "#2471a3", "#e67e22", "#c0392b", "#d5d8dc"
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "axes.spines.top": False, "axes.spines.right": False})
def pt(x, c=0): return f"{x:,.{c}f}".replace(",", "X").replace(".", ",").replace("X", ".")
virgula = lambda c: FuncFormatter(lambda x, _: pt(x, c))
def salvar(fig, nome): fig.savefig(SAIDA / nome, bbox_inches="tight"); plt.close(fig)

# ---------- 1. Dobras da validação cruzada e partições da validação ----------
def figura(tab, titulo, rot_ajuste, rot_aval, texto, arquivo, tem_treino):
    n = len(tab); fig = plt.figure(figsize=(12.5, 0.75 * n + 2.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.6, 1.15, 1.15], wspace=0.08)
    a0, a1, a2 = (fig.add_subplot(gs[0, i]) for i in range(3))
    for i, r in enumerate(tab.itertuples()):
        y = n - 1 - i
        for j in range(n):
            cor = LARANJA if j == i else (AZUL if tem_treino else CINZA)
            a0.add_patch(Rectangle((j, y - .32), .96, .64, color=cor, lw=0))
        a0.text(n + .15, y, texto(r), va="center", fontsize=8.5)
    a0.set_xlim(0, n + 3.2); a0.set_ylim(-.6, n - .4); a0.axis("off")
    fig.suptitle(titulo, x=0.01, ha="left", fontsize=11, fontweight="bold", y=1.04)
    a0.add_patch(Rectangle((0, -1.0), .3, .3, color=AZUL if tem_treino else CINZA, lw=0, clip_on=False)); a0.text(.4, -.85, rot_ajuste, fontsize=8.5, va="center")
    a0.add_patch(Rectangle((0, -1.5), .3, .3, color=LARANJA, lw=0, clip_on=False)); a0.text(.4, -1.35, rot_aval, fontsize=8.5, va="center")
    for ax, col, nome in ((a1, "recall", "Recall (ataques)"), (a2, "taxa_falso_alarme", "Falso alarme (%)")):
        v = tab[col].to_numpy(float) * 100; ys = np.arange(n)[::-1]
        ax.scatter(v, ys, color="black", s=26, zorder=3); ax.axvline(v.mean(), color=LARANJA, lw=1.4, ls="--")
        for x_, y_ in zip(v, ys): ax.text(x_, y_ + .28, pt(x_, 3), ha="center", fontsize=7.5)
        m = (v.max() - v.min()) or .01
        ax.set_xlim(max(v.min() - m * .8, -m * .1) if col == "taxa_falso_alarme" else v.min() - m * .8, min(v.max() + m * .8, 100 + m * .1) if col == "recall" else v.max() + m * .8)
        ax.set_ylim(-.6, n - .4); ax.set_yticks([]); ax.spines["left"].set_visible(False)
        ax.set_title(f"{nome}\ntraço laranja = média {pt(v.mean(), 3)}", fontsize=9.5); ax.grid(axis="x", alpha=.25); ax.xaxis.set_major_formatter(virgula(2))
    salvar(fig, arquivo)
figura(cv, f"Validação cruzada no ajuste ({len(cv)} dobras, SVM): cada linha é um treino diferente", "treino (4 partes do ajuste)", "avaliação (1 parte, que o modelo não viu)",
       lambda r: f"dobra {r.dobra}\ntreino {pt(r.n_treino)} | avaliação {pt(r.n_avaliacao)}\n{r.tempo_treino_s:.1f} s de treino | {r.n_sv:.0f} SV", "svm_fig_cv_ajuste_5dobras.png", True)
figura(part, f"Dispersão na validação ({len(part)} partições, SVM): o modelo é o mesmo, só muda o pedaço avaliado", "demais partições (não entram nessa medida)", "partição avaliada (modelo treinado no ajuste, sem retreino)",
       lambda r: f"partição {r.particao}\n{pt(r.n)} registros ({pt(r.benign)} BENIGN)", "svm_fig_particoes_validacao.png", False)

# ---------- 2. Efeito da proporção de classes ----------
nomes = {"mcc": "MCC", "precisao_benign": "Precisão (BENIGN)", "f1_benign": "F1 (BENIGN)", "f1_macro": "F1 macro", "pr_auc_benign": "PR-AUC (BENIGN)", "f1": "F1 (ataques)", "precisao": "Precisão (ataques)"}
fig, ax = plt.subplots(figsize=(8.5, 4))
for i, k in enumerate(nomes):
    y = len(nomes) - 1 - i; a, b = sem[k], real[k]
    ax.plot([a, b], [y, y], color="#b2babb", lw=2.2, zorder=1); ax.scatter(a, y, color=AZUL, s=55, zorder=3, label="validação sem pesos (16,9% BENIGN)" if i == 0 else None)
    ax.scatter(b, y, color=LARANJA, s=55, zorder=3, label="proporção real (0,615% BENIGN)" if i == 0 else None)
    ax.text(min(a, b) - .002, y, pt(min(a, b), 3), ha="right", va="center", fontsize=8); ax.text(max(a, b) + .002, y, pt(max(a, b), 3), ha="left", va="center", fontsize=8)
ax.set_yticks(range(len(nomes))); ax.set_yticklabels(list(nomes.values())[::-1]); ax.set_xlim(min(real[k] for k in nomes) - .06, 1.012); ax.xaxis.set_major_formatter(virgula(2))
ax.grid(axis="x", alpha=.25); ax.set_title(f"SVM: métricas que dependem da proporção de classes (limiar {pt(LIMIAR, 1)})", fontsize=10); ax.legend(fontsize=8, loc="lower left"); fig.tight_layout(); salvar(fig, "svm_fig_metricas_proporcao.png")

# ---------- 3. Matrizes de confusão ----------
fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.9))
for ax, m, tit, dec in ((axs[0], sem, "Contagens na validação (sem pesos)", 0), (axs[1], real, "Ponderada pela proporção real", 1)):
    M = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]], float); ax.imshow(M / M.sum(axis=1, keepdims=True), cmap="Blues", vmin=0, vmax=1.3)
    for (i, j), v in np.ndenumerate(M): ax.text(j, i, pt(v, dec), ha="center", va="center", fontsize=12, fontweight="bold", color="white" if M[i, j] / M[i].sum() > .5 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["previsto\nBENIGN", "previsto\nataque"]); ax.set_yticks([0, 1]); ax.set_yticklabels(["real\nBENIGN", "real\nataque"]); ax.set_title(tit, fontsize=10)
    for s in ax.spines.values(): s.set_visible(False)
fig.suptitle(f"Matriz de confusão do SVM na validação (limiar {pt(LIMIAR, 1)}); cor = fração da linha", fontsize=10); fig.tight_layout(); salvar(fig, "svm_fig_matrizes_confusao.png")

# ---------- 4. Efeito do limiar ----------
pts = lim["pontos_de_operacao_validacao"]
linhas = [(p["limiar"], f"{pt(p['limiar'], 2)}\n(falso alarme ≤ {pt(p['alvo_fpr'] * 100, 1)}%)", 1 - p["recall"], p["fpr_obtida"], p["mcc_real"], p["f1_benign_real"]) for p in pts]
linhas.append((LIMIAR, f"{pt(LIMIAR, 2)}\n(natural)", 1 - sem["recall"], sem["taxa_falso_alarme"], real["mcc"], real["f1_benign"]))
linhas.sort(key=lambda r: r[0]); rot = [r[1] for r in linhas]; x = np.arange(len(linhas))
fig, axs = plt.subplots(1, 2, figsize=(11, 4))
for j, (col, cor, nome) in enumerate(((2, VERM, "ataques perdidos (1 − recall)"), (3, AZUL, "falso alarme (BENIGN → ataque)"))):
    v = [r[col] * 100 for r in linhas]; axs[0].bar(x + (j - .5) * .36, v, .36, color=cor, label=nome)
    for i, val in enumerate(v): axs[0].text(i + (j - .5) * .36, val, pt(val, 3), ha="center", va="bottom", fontsize=7.5)
axs[0].set_ylim(0, 1.45); axs[0].set_xticks(x); axs[0].set_xticklabels(rot, fontsize=8); axs[0].set_xlabel("limiar de decisão (distância à fronteira)"); axs[0].set_ylabel("% da classe"); axs[0].yaxis.set_major_formatter(virgula(2)); axs[0].set_title("Erros em cada classe", fontsize=10); axs[0].legend(fontsize=8); axs[0].grid(axis="y", alpha=.25)
for j, (col, cor, nome) in enumerate(((4, LARANJA, "MCC (proporção real)"), (5, "#117a65", "F1 do BENIGN (proporção real)"))):
    v = [r[col] for r in linhas]; axs[1].bar(x + (j - .5) * .36, v, .36, color=cor, label=nome)
    for i, val in enumerate(v): axs[1].text(i + (j - .5) * .36, val, pt(val, 3), ha="center", va="bottom", fontsize=7.5)
axs[1].set_ylim(0, 1.1); axs[1].set_xticks(x); axs[1].set_xticklabels(rot, fontsize=8); axs[1].set_xlabel("limiar de decisão (distância à fronteira)"); axs[1].yaxis.set_major_formatter(virgula(2)); axs[1].set_title("Métricas na proporção real", fontsize=10); axs[1].legend(fontsize=8, loc="upper left"); axs[1].grid(axis="y", alpha=.25)
fig.suptitle("Efeito do limiar na validação, SVM selecionado (limiares escolhidos na própria validação; o teste dirá o valor real)", fontsize=10); fig.tight_layout(); salvar(fig, "svm_fig_efeito_limiar.png")

# ---------- 5. Dispersão (partições x dobras) ----------
fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
for a, (col, tit) in zip(ax, (("recall", "Recall (ataques)"), ("taxa_falso_alarme", "Taxa de falso alarme"))):
    for x_, (tab, cor) in enumerate(((part, VERM), (cv, AZUL))):
        v = tab[col].to_numpy(float); a.scatter(np.full(len(v), x_) + np.linspace(-.08, .08, len(v)), v, color=cor, s=28, zorder=3); a.hlines(v.mean(), x_ - .22, x_ + .22, color="black", lw=1.5, zorder=4)
    a.set_xticks([0, 1]); a.set_xticklabels(["partições\nda validação", "dobras da CV\nno ajuste"], fontsize=8); a.set_xlim(-.5, 1.5); a.set_title(tit, fontsize=10); a.grid(alpha=.25)
fig.suptitle("SVM: dispersão das métricas (traço preto = média)", fontsize=10); fig.tight_layout(); salvar(fig, "svm_fig_dispersao.png")

# ---------- 6. Tempo de treino ----------
reps = pd.DataFrame(custo["fit_refit_repeticoes"])
fig, ax = plt.subplots(figsize=(8.5, 4.2))
ax.scatter(cv["n_treino"], cv["tempo_treino_s"], color=AZUL, s=45, label="dobras da validação cruzada, selecionado (5)", zorder=3)
ax.scatter([N["ajuste"]], [custo["fit_ajuste"]["relogio_s"]], color=VERM, s=60, marker="D", label="ajuste, selecionado (1)", zorder=3)
ax.scatter([N["refit"]] * len(reps), reps["relogio_s"], color=LARANJA, s=45, marker="s", label=f"refit, selecionado ({len(reps)} repetições)", zorder=3)
ax.scatter([N["ajuste"]], [234.6], color="#7f8c8d", s=60, marker="D", label="ajuste, padrão (1)", zorder=3)
ax.scatter([N["refit"]], [custo["padrao"]["fit_refit"]["relogio_s"]], color="#7f8c8d", s=45, marker="s", label="refit, padrão (1)", zorder=3)
ax.set_xlabel("registros usados no treino"); ax.set_ylabel("tempo de relógio (s)"); ax.xaxis.set_major_formatter(virgula(0)); ax.set_ylim(0, 310); ax.set_xlim(N["ajuste"] * .72, N["refit"] * 1.06); ax.grid(alpha=.25)
ax.set_title("Tempo de treino do SVM (1 núcleo; padrão: C = 1, γ = 1×; selecionado: C = 100, γ = 4×)", fontsize=9.5); ax.legend(fontsize=7.5, loc="upper left", ncol=2); fig.tight_layout(); salvar(fig, "svm_fig_tempos_treino.png")

# ---------- 7. Inferência ----------
fig, axs = plt.subplots(1, 2, figsize=(10.5, 4)); nj = [r["n_jobs"] for r in custo["inferencia_modelo_final"]]; xj = np.arange(len(nj))
for j, (chave, cor, nome) in enumerate((("inferencia_modelo_ajuste", AZUL, "modelo do ajuste"), ("inferencia_modelo_final", LARANJA, "modelo final (refit)"))):
    R = custo[chave]; xx = xj + (j - .5) * .36
    p50 = np.array([np.nan if r["latencia_fim_a_fim_p50_ms"] is None else r["latencia_fim_a_fim_p50_ms"] for r in R]); p99 = np.array([np.nan if r["latencia_fim_a_fim_p99_ms"] is None else r["latencia_fim_a_fim_p99_ms"] for r in R])
    ok = ~np.isnan(p50); axs[0].bar(xx[ok], p50[ok], .36, color=cor, label=nome, yerr=[np.zeros(ok.sum()), (p99 - p50)[ok]], error_kw=dict(ecolor="black", capsize=3, lw=1))
    for a, b in zip(xx[ok], p50[ok]): axs[0].text(a, 0.08, pt(b, 2), ha="center", va="bottom", fontsize=8, color="white")
    v = [r["vazao_fim_a_fim_fluxos_s"] for r in R]; axs[1].bar(xx, v, .36, color=cor, label=nome)
    for a, b in zip(xx, v): axs[1].text(a, b, pt(b), ha="center", va="bottom", fontsize=8)
for ax, tit, un in ((axs[0], "Latência de 1 fluxo (barra = mediana; traço = p99; só com 1 processo)", "ms"), (axs[1], "Vazão em lote (20.000 fluxos)", "fluxos/s")):
    ax.set_xticks(xj); ax.set_xticklabels([f"{n} processo{'s' if n > 1 else ''}" for n in nj]); ax.set_ylabel(un); ax.set_title(tit + " — fim a fim", fontsize=9); ax.grid(axis="y", alpha=.25)
axs[1].yaxis.set_major_formatter(virgula(0)); axs[0].legend(fontsize=8, loc="upper right"); axs[0].text(1, 0.5, "latência com 2 processos\nnão foi medida", ha="center", fontsize=8, color="gray"); axs[0].set_xlim(-.6, len(nj) - .4)
fig.suptitle("Custo de inferência do SVM", fontsize=10); fig.tight_layout(); salvar(fig, "svm_fig_inferencia.png")
print("✔ Figuras gravadas em", PASTA)
