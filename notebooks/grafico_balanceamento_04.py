# Gráficos do balanceamento (notebook 04). Os valores são os da Tabela 9 / Tabela 8 (saídas conferidas do notebook 04); não precisa de ClickHouse.
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from pathlib import Path

SAIDA = Path(globals().get("SAIDA", "."))          # pasta onde gravar os PNG
AZUL, LARANJA, VERM = "#2471a3", "#e67e22", "#c0392b"
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25})
def pt(x, c=0): return f"{x:,.{c}f}".replace(",", "X").replace(".", ",").replace("X", ".")

tipos = ["TFTP", "NTP", "UDP", "SSDP", "MSSQL", "Syn", "SNMP", "DNS", "UDPLag", "LDAP", "NetBIOS", "WebDDoS"]
treino = [4396919, 1112936, 1075337, 891221, 207003, 155515, 115316, 113259, 89066, 30099, 19490, 414]
ajuste = [3663, 3662, 3662, 3662, 3662, 3662, 3662, 3662, 3662, 3662, 3662, 331]
valid = [4538, 4538, 4538, 4538, 4538, 4538, 4538, 4538, 4538, 4538, 4537, 83]
assert sum(ajuste) == 40614 and sum(valid) == 50000 and sum(treino) == 8206575      # fecham com as Tabelas 8 e 9

# ---- Figura A: proporção de BENIGN em cada conjunto ----
nomes = ["Treino\ndeduplicado", "Ajuste", "Validação\n(sem pesos)", "Validação\n(ponderada)", "Teste sem\nsobreposição"]
perc = [0.61, 50.0, 16.9, 0.615, 0.78]; cores = [VERM, AZUL, AZUL, "#85c1e9", VERM]
fig, ax = plt.subplots(figsize=(8.5, 3.9)); b = ax.bar(nomes, perc, color=cores)
for r, v in zip(b, perc): ax.text(r.get_x() + r.get_width() / 2, v, pt(v, 2 if v < 1 else 1) + "%", ha="center", va="bottom", fontsize=9)
ax.set_yscale("log"); ax.set_ylim(.3, 100); ax.set_ylabel("% de BENIGN (escala log)"); ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: pt(x, 1 if x < 1 else 0))); ax.grid(axis="x", alpha=0)
ax.set_title("Proporção de BENIGN: o ajuste é balanceado; validação e teste mantêm (ou recompõem) a proporção real", fontsize=9.5)
fig.tight_layout(); fig.savefig(SAIDA / "fig_balanceamento_proporcao.png", bbox_inches="tight"); plt.show()

# ---- Figura B: ataques por tipo antes e depois da subamostragem estratificada ----
y = np.arange(len(tipos))[::-1]; h = .27
fig, ax = plt.subplots(figsize=(9.5, 5.4))
ax.barh(y + h, treino, h, color=VERM, label="treino deduplicado (antes)"); ax.barh(y, ajuste, h, color=AZUL, label="ajuste"); ax.barh(y - h, valid, h, color=LARANJA, label="validação")
for yy, a, b_, c in zip(y, treino, ajuste, valid):
    ax.text(a * 1.15, yy + h, pt(a), va="center", fontsize=7.5); ax.text(b_ * 1.15, yy, pt(b_), va="center", fontsize=7.5); ax.text(c * 1.15, yy - h, pt(c), va="center", fontsize=7.5)
ax.set_xscale("log"); ax.set_xlim(50, 3e7); ax.set_yticks(y); ax.set_yticklabels(tipos); ax.set_xlabel("registros de ataque (escala log)"); ax.grid(axis="y", alpha=0)
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: pt(x))); ax.legend(fontsize=8, loc="lower right")
ax.set_title("Ataques por tipo: de 4,4 milhões (TFTP) a 414 (WebDDoS) no treino; quantidades iguais no ajuste", fontsize=9.5)
fig.tight_layout(); fig.savefig(SAIDA / "fig_balanceamento_tipos.png", bbox_inches="tight"); plt.show()
