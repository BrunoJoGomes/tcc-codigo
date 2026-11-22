# Gráficos dos RESULTADOS do notebook 05, lidos dos arquivos gravados na pasta do experimento (não precisa do kernel anterior nem do ClickHouse).
import json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

if "PASTA" not in globals():
    raiz = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
    PASTA = sorted((raiz / "artefatos_modelos" / "rf_binario").glob("rf_bin_*"))[-1]
PASTA = Path(PASTA); print("Lendo de:", PASTA)
lj = lambda nome: json.loads((PASTA / nome).read_text(encoding="utf-8"))
met, lim, custo, cfg = lj("metricas_validacao.json"), lj("limiares_validacao.json"), lj("custo_computacional.json"), lj("config_treino.json")
cv, refit = pd.read_csv(PASTA / "cv_ajuste.csv"), pd.read_csv(PASTA / "refit_ids.csv")
sem, real, LIMIAR = met["sem_pesos"], met["proporcao_real"], met["limiar"]

AZUL, LARANJA, VERM, CINZA = "#2471a3", "#e67e22", "#c0392b", "#7f8c8d"
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25})
def pt(x, c=0): return f"{x:,.{c}f}".replace(",", "X").replace(".", ",").replace("X", ".")
def salvar(fig, nome): fig.savefig(PASTA / nome, bbox_inches="tight"); plt.show()
virgula = lambda c: FuncFormatter(lambda x, _: pt(x, c))

# ---- 1. Efeito da proporção de classes nas métricas (sem pesos x proporção real) ----
nomes = {"mcc": "MCC", "precisao_benign": "Precisão (BENIGN)", "f1_benign": "F1 (BENIGN)", "f1_macro": "F1 macro", "pr_auc_benign": "PR-AUC (BENIGN)", "f1": "F1 (ataques)", "precisao": "Precisão (ataques)"}
fig, ax = plt.subplots(figsize=(8.5, 4))
for i, k in enumerate(nomes):
    y = len(nomes) - 1 - i; a, b = sem[k], real[k]
    ax.plot([a, b], [y, y], color="#b2babb", lw=2.2, zorder=1); ax.scatter(a, y, color=AZUL, s=55, zorder=3, label="validação sem pesos (16,9% BENIGN)" if i == 0 else None)
    ax.scatter(b, y, color=LARANJA, s=55, zorder=3, label="proporção real (0,615% BENIGN)" if i == 0 else None)
    ax.text(min(a, b) - .002, y, pt(min(a, b), 3), ha="right", va="center", fontsize=8); ax.text(max(a, b) + .002, y, pt(max(a, b), 3), ha="left", va="center", fontsize=8)
ax.set_yticks(range(len(nomes))); ax.set_yticklabels(list(nomes.values())[::-1]); ax.set_xlim(min(real[k] for k in nomes) - .03, 1.012); ax.xaxis.set_major_formatter(virgula(2)); ax.grid(axis="y", alpha=0)
ax.set_title(f"Métricas que dependem da proporção de classes (limiar {pt(LIMIAR, 1)})", fontsize=10); ax.legend(fontsize=8, loc="lower left"); fig.tight_layout(); salvar(fig, "fig_metricas_proporcao.png")

# ---- 2. Matrizes de confusão: contagens reais e ponderadas pela proporção real ----
fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.9))
for ax, m, tit, dec in ((axs[0], sem, "Contagens na validação (sem pesos)", 0), (axs[1], real, "Ponderada pela proporção real", 1)):
    M = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]], float); ax.imshow(M / M.sum(axis=1, keepdims=True), cmap="Blues", vmin=0, vmax=1.3)
    for (i, j), v in np.ndenumerate(M): ax.text(j, i, pt(v, dec), ha="center", va="center", fontsize=12, fontweight="bold", color="white" if M[i, j] / M[i].sum() > .5 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["previsto\nBENIGN", "previsto\nataque"]); ax.set_yticks([0, 1]); ax.set_yticklabels(["real\nBENIGN", "real\nataque"]); ax.grid(False); ax.set_title(tit, fontsize=10)
    for s in ax.spines.values(): s.set_visible(False)
fig.suptitle(f"Matriz de confusão do RF na validação (limiar {pt(LIMIAR, 1)}); cor = fração da linha", fontsize=10); fig.tight_layout(); salvar(fig, "fig_matrizes_confusao.png")

# ---- 3. Efeito do limiar de decisão ----
pts = lim["pontos_de_operacao_validacao"]
linhas = [(f"{pt(p['limiar'], 2)}\n(falso alarme ≤ {pt(p['alvo_fpr'] * 100, 1)}%)", 1 - p["recall"], p["fpr_obtida"], p["mcc_real"], p["f1_benign_real"]) for p in pts]
linhas.append((f"{pt(LIMIAR, 2)}\n(padrão)", 1 - sem["recall"], sem["taxa_falso_alarme"], real["mcc"], real["f1_benign"]))
linhas.sort(key=lambda r: float(r[0].split("\n")[0].replace(",", ".")))
rot = [r[0] for r in linhas]; x = np.arange(len(linhas))
fig, axs = plt.subplots(1, 2, figsize=(11, 4))
for j, (col, cor, nome) in enumerate(((1, VERM, "ataques perdidos (1 − recall)"), (2, AZUL, "falso alarme (BENIGN → ataque)"))):
    v = [r[col] * 100 for r in linhas]; axs[0].bar(x + (j - .5) * .36, v, .36, color=cor, label=nome)
    for i, val in enumerate(v): axs[0].text(i + (j - .5) * .36, val, pt(val, 3), ha="center", va="bottom", fontsize=7.5)
axs[0].set_xticks(x); axs[0].set_xticklabels(rot, fontsize=8); axs[0].set_xlabel("limiar de decisão"); axs[0].set_ylabel("% da classe"); axs[0].yaxis.set_major_formatter(virgula(2)); axs[0].set_title("Erros em cada classe", fontsize=10); axs[0].legend(fontsize=8)
for j, (col, cor, nome) in enumerate(((3, LARANJA, "MCC (proporção real)"), (4, "#117a65", "F1 do BENIGN (proporção real)"))):
    v = [r[col] for r in linhas]; axs[1].bar(x + (j - .5) * .36, v, .36, color=cor, label=nome)
    for i, val in enumerate(v): axs[1].text(i + (j - .5) * .36, val, pt(val, 3), ha="center", va="bottom", fontsize=7.5)
axs[1].set_ylim(.9, 1.03); axs[1].set_xticks(x); axs[1].set_xticklabels(rot, fontsize=8); axs[1].set_xlabel("limiar de decisão"); axs[1].yaxis.set_major_formatter(virgula(2)); axs[1].set_title("Métricas na proporção real", fontsize=10); axs[1].legend(fontsize=8, loc="upper right")
fig.suptitle("Efeito do limiar na validação (limiares escolhidos na própria validação; o teste dirá o valor real)", fontsize=10); fig.tight_layout(); salvar(fig, "fig_efeito_limiar.png")

# ---- 4. Composição do refit por tipo (de onde vem cada registro do treino final) ----
tab = refit.groupby(["tipo", "origem"]).size().unstack(fill_value=0)
for c in ("ajuste", "validacao_benign", "validacao_ataque_extra"):
    if c not in tab: tab[c] = 0
tab["total"] = tab.sum(axis=1); tab = tab.sort_values("total")
fig, ax = plt.subplots(figsize=(8.5, 4.6)); esq = np.zeros(len(tab))
for c, cor, nome in (("ajuste", AZUL, "já estava no ajuste"), ("validacao_benign", "#117a65", "BENIGN vindos da validação"), ("validacao_ataque_extra", LARANJA, "ataques vindos da validação")):
    ax.barh(tab.index, tab[c], left=esq, color=cor, label=nome)
    for i, (v, e) in enumerate(zip(tab[c], esq)):
        if v > 2500: ax.text(e + v / 2, i, pt(v), ha="center", va="center", fontsize=7.5, color="white")
    esq += tab[c].to_numpy()
for i, v in enumerate(tab["total"]): ax.text(v + 60, i, pt(v), va="center", fontsize=8)
ax.set_xlabel("registros no treino final (refit)"); ax.xaxis.set_major_formatter(virgula(0)); ax.grid(axis="y", alpha=0); ax.legend(fontsize=8, loc="lower right"); ax.set_xlim(0, tab["total"].max() * 1.12)
ax.set_title(f"Composição do refit por tipo ({pt(len(refit))} registros)", fontsize=10); fig.tight_layout(); salvar(fig, "fig_composicao_refit.png")

# ---- 5. Tempo de treino: ajuste, dobras da validação cruzada e repetições do refit ----
reps = pd.DataFrame(custo["fit_refit_repeticoes"]); N = cfg["dados"]
fig, ax = plt.subplots(figsize=(8, 4))
ax.scatter(cv["n_treino"], cv["tempo_treino_s"], color=AZUL, s=45, label="dobras da validação cruzada (5)", zorder=3)
ax.scatter([N["ajuste"]], [custo["fit_ajuste"]["relogio_s"]], color=VERM, s=60, marker="D", label="treino no ajuste (1)", zorder=3)
ax.scatter([N["refit"]] * len(reps), reps["relogio_s"], color=LARANJA, s=45, marker="s", label=f"refit ({len(reps)} repetições)", zorder=3)
ax.set_xlabel("registros usados no treino"); ax.set_ylabel("tempo de relógio (s)"); ax.xaxis.set_major_formatter(virgula(0)); ax.set_ylim(0, max(cv["tempo_treino_s"].max(), reps["relogio_s"].max()) * 1.25)
ax.set_xlim(N["ajuste"] * .72, N["refit"] * 1.06); ax.set_title("Tempo de treino do RF (100 árvores, todos os núcleos)", fontsize=10); ax.legend(fontsize=8, loc="upper center", ncol=3); fig.tight_layout(); salvar(fig, "fig_tempos_treino.png")

# ---- 6. Inferência: latência de 1 fluxo e vazão em lote, modelo do ajuste x modelo final ----
fig, axs = plt.subplots(1, 2, figsize=(10.5, 4)); nj = [r["n_jobs"] for r in custo["inferencia_modelo_final"]]; xj = np.arange(len(nj))
for j, (chave, cor, nome) in enumerate((("inferencia_modelo_ajuste", AZUL, "modelo do ajuste"), ("inferencia_modelo_final", LARANJA, "modelo final (refit)"))):
    R = custo[chave]; p50 = np.array([r["latencia_fim_a_fim_p50_ms"] for r in R]); p99 = np.array([r["latencia_fim_a_fim_p99_ms"] for r in R]); xx = xj + (j - .5) * .36
    axs[0].bar(xx, p50, .36, color=cor, label=nome, yerr=[np.zeros(len(R)), p99 - p50], error_kw=dict(ecolor="black", capsize=3, lw=1))
    for a, b in zip(xx, p50): axs[0].text(a, 0.4, pt(b, 1), ha="center", va="bottom", fontsize=8, color="white")
    v = [r["vazao_fim_a_fim_fluxos_s"] for r in R]; axs[1].bar(xx, v, .36, color=cor, label=nome)
    for a, b in zip(xx, v): axs[1].text(a, b, pt(b), ha="center", va="bottom", fontsize=8)
for ax, tit, un in ((axs[0], "Latência de 1 fluxo (barra = mediana; traço = p99)", "ms"), (axs[1], "Vazão em lote (validação inteira)", "fluxos/s")):
    ax.set_xticks(xj); ax.set_xticklabels([f"n_jobs = {n}" for n in nj]); ax.set_ylabel(un); ax.set_title(tit + " — fim a fim", fontsize=9.5); ax.grid(axis="x", alpha=0)
axs[1].yaxis.set_major_formatter(virgula(0)); axs[0].legend(fontsize=8, loc="upper left")
fig.suptitle("Custo de inferência (diferenças de ±10% entre medições estão dentro do ruído)", fontsize=10); fig.tight_layout(); salvar(fig, "fig_inferencia.png")
print("✔ Figuras gravadas em", PASTA)
