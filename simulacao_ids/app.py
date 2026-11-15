"""Simulação de IDS para detecção de DDoS (TCC: Random Forest x SVM RBF no CICDDoS2019).

Rodar (na pasta simulacao_ids):   streamlit run app.py

Os modelos são carregados dos arquivos salvos (nenhum treino) e classificam, em lotes, uma amostra CONGELADA de fluxos
do conjunto de teste (outro dia de captura). O rótulo real nunca entra como atributo: serve só para calcular as métricas.
"""
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import nucleo as nu

st.set_page_config(page_title="Simulação IDS — DDoS", page_icon="🛡️", layout="wide")

# ----------------------------------------------------------------------------- tema
BG, CARD, BORDA, TEXTO, MUDO = "#0b1220", "#111a2e", "#1f2d4a", "#e6edf7", "#8a99b5"
CIANO, VERMELHO, VERDE, AMBAR, ROXO = "#22d3ee", "#f43f5e", "#34d399", "#fbbf24", "#a78bfa"
CORES_CLASSE = {"BENIGN": VERDE, "LDAP": VERMELHO, "MSSQL": AMBAR, "NetBIOS": ROXO, "Syn": "#fb7185", "UDP": CIANO, "UDPLag": "#f97316"}
CORES_MODELO = [CIANO, AMBAR]

st.markdown(f"""
<style>
.stApp {{ background: radial-gradient(1200px 500px at 10% -10%, #14213d 0%, {BG} 55%); color: {TEXTO}; }}
[data-testid="stSidebar"] {{ background: #0a101d; border-right: 1px solid {BORDA}; }}
[data-testid="stHeader"] {{ background: transparent; }}
h1, h2, h3, h4, h5, p, label, span, div {{ color: {TEXTO}; }}
.topo {{ display:flex; align-items:center; justify-content:space-between; padding: 6px 2px 14px 2px; }}
.topo .tit {{ font-size: 1.55rem; font-weight: 700; letter-spacing: .3px; }}
.topo .sub {{ color:{MUDO}; font-size:.85rem; margin-top:2px; }}
.pill {{ padding: 5px 14px; border-radius: 999px; font-size:.78rem; font-weight:700; letter-spacing:.8px; border:1px solid; }}
.pill.on  {{ color:{VERDE}; border-color:{VERDE}; background:rgba(52,211,153,.10); }}
.pill.off {{ color:{AMBAR}; border-color:{AMBAR}; background:rgba(251,191,36,.10); }}
.pill.fim {{ color:{CIANO}; border-color:{CIANO}; background:rgba(34,211,238,.10); }}
.faseb {{ display:inline-block; padding:4px 12px; border-radius:8px; background:{CARD}; border:1px solid {BORDA}; color:{MUDO}; font-size:.82rem; }}
.faseb b {{ color:{TEXTO}; }}
.kpi {{ background:{CARD}; border:1px solid {BORDA}; border-radius:14px; padding:14px 16px; height:100%; position:relative; overflow:hidden; }}
.kpi::before {{ content:""; position:absolute; left:0; top:0; bottom:0; width:4px; background:var(--c); }}
.kpi .t {{ color:{MUDO}; font-size:.74rem; text-transform:uppercase; letter-spacing:.9px; }}
.kpi .v {{ font-size:1.75rem; font-weight:700; line-height:1.25; color:{TEXTO}; }}
.kpi .s {{ color:{MUDO}; font-size:.76rem; }}
.secao {{ font-size:.95rem; font-weight:600; color:{TEXTO}; margin: 14px 0 4px 2px; letter-spacing:.3px; }}
.aviso {{ background:rgba(251,191,36,.07); border:1px solid rgba(251,191,36,.35); color:#f3deb0; padding:10px 14px; border-radius:12px; font-size:.82rem; }}
.aviso b {{ color:{AMBAR}; }}
div[data-testid="stTabs"] button {{ font-weight:600; }}
.stButton > button {{ border-radius:10px; border:1px solid {BORDA}; background:{CARD}; color:{TEXTO}; font-weight:600; }}
.stButton > button:hover {{ border-color:{CIANO}; color:{CIANO}; }}
</style>
""", unsafe_allow_html=True)


def card(titulo, valor, sub="", cor=CIANO):
    return f'<div class="kpi" style="--c:{cor}"><div class="t">{titulo}</div><div class="v">{valor}</div><div class="s">{sub or "&nbsp;"}</div></div>'


def layout(fig, altura=330, titulo=None, legenda=True):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=altura,
                      margin=dict(l=8, r=8, t=40 if titulo else 12, b=78 if legenda else 8), showlegend=legenda,
                      title=dict(text=titulo, font=dict(size=14, color=TEXTO), x=0.01) if titulo else None,
                      legend=dict(orientation="h", yanchor="top", y=-0.32, xanchor="left", x=0, font=dict(color=MUDO)), font=dict(color=MUDO))
    fig.update_xaxes(gridcolor="#18243d", zeroline=False)
    fig.update_yaxes(gridcolor="#18243d", zeroline=False)
    return fig


# ----------------------------------------------------------------------------- recursos em cache
@st.cache_resource(show_spinner="Carregando o catálogo de modelos...")
def _registro():
    return nu.carregar_registro()


@st.cache_resource(show_spinner="Carregando a amostra congelada...")
def _amostra():
    return nu.carregar_amostra()


@st.cache_resource(show_spinner="Carregando o modelo (só leitura do arquivo salvo)...")
def _modelo(mid: str):
    return nu.carregar_modelo(_registro()["modelos"][mid])


@st.cache_data(show_spinner=False)
def _cenario(nome: str):
    return nu.montar_cenario(_amostra()["meta"], nome)


try:
    REG = _registro()["modelos"]
    AM = _amostra()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()


# ----------------------------------------------------------------------------- barra lateral
st.sidebar.markdown("### 🛡️ Configuração")
cenario = st.sidebar.radio(
    "Cenário", ["Fases de ataque", "Balanceado"],
    help="**Fases de ataque**: tráfego normal → uma fase de cada ataque → tráfego normal (linha do tempo construída). "
         "**Balanceado**: mesma quantidade de cada classe, embaralhadas.")
variante = st.sidebar.radio("Hiperparâmetros", ["ajustado", "padrão"],
                            help="Ajustado = escolhido pela busca; padrão = configuração-base do sklearn.")
familia = st.sidebar.radio("Modelo principal", ["rf", "svm"], format_func=lambda f: "Random Forest" if f == "rf" else "SVM RBF")
comparar = st.sidebar.checkbox("Comparar RF × SVM no mesmo fluxo", value=True,
                               help="Roda também o outro modelo (mesma variante) nos mesmos fluxos. O SVM multiclasse é mais lento.")
lote = st.sidebar.slider("Tamanho do lote (fluxos)", 100, 500, 100, step=50)
vel = st.sidebar.slider("Velocidade simulada (fluxos/s)", 50, 2000, 200, step=50,
                        help="Só controla o ritmo da simulação. A velocidade REAL do modelo é medida e mostrada à parte.")

assinatura = (cenario, variante, familia, comparar, lote)
if st.session_state.get("assinatura") != assinatura:     # mudou a configuração: recomeça (sem apagar nada em disco)
    st.session_state["assinatura"] = assinatura
    for k in ("bin", "multi"):
        st.session_state.pop(f"est_{k}", None)

meta_cen = _cenario(cenario)
st.sidebar.caption(f"{len(meta_cen):,} fluxos no cenário · amostra de {len(AM['meta']):,} fluxos")


def modelos_ativos(tipo: str) -> list[str]:
    fam = [familia] + ([("svm" if familia == "rf" else "rf")] if comparar else [])
    sufixo = "ajustado" if variante == "ajustado" else "padrao"
    return [f"{tipo}_{f}_{sufixo}" for f in fam]


def novo_estado(tipo: str) -> dict:
    ativos = modelos_ativos(tipo)
    return {"cursor": 0, "rodando": False, "ativos": ativos, "preds": {m: [] for m in ativos},
            "tempos": {m: 0.0 for m in ativos}, "ultimos": pd.DataFrame(), "t_ini": None,
            "hist": [], "lat": {m: [] for m in ativos}}


def estado(tipo: str) -> dict:
    return st.session_state.setdefault(f"est_{tipo}", novo_estado(tipo))


# ----------------------------------------------------------------------------- processamento
def passo(tipo: str, est: dict):
    """Classifica o próximo lote com cada modelo ativo."""
    i0, i1 = est["cursor"], min(est["cursor"] + lote, len(meta_cen))
    if i0 >= i1:
        return
    linhas = meta_cen.iloc[i0:i1]
    Xfull = AM["X_bin"] if tipo == "bin" else AM["X_multi"]
    X = Xfull.iloc[linhas["pos"].to_numpy()].reset_index(drop=True)
    if est["t_ini"] is None:
        est["t_ini"] = time.perf_counter()
    por_modelo = {}
    for mid in est["ativos"]:
        pred, seg = nu.classificar(_modelo(mid), REG[mid], X)
        est["preds"][mid].append(pred)
        est["tempos"][mid] += seg
        est["lat"][mid].append(seg / len(X) * 1000)
        por_modelo[mid] = (pred, seg / len(X))
    principal = est["ativos"][0]
    pred, ms = por_modelo[principal]
    real = linhas["ataque_real"].to_numpy() if tipo == "bin" else linhas["classe_real"].to_numpy()
    nome = (lambda v: "ATAQUE" if v == 1 else "BENIGNO") if tipo == "bin" else (lambda v: str(v))
    tab = pd.DataFrame({"fluxo (id)": linhas["id_linha"].to_numpy(), "fase": linhas["fase"].to_numpy(),
                        "real": [nome(v) for v in real], "previsto": [nome(v) for v in pred],
                        "resultado": np.where(pred == real, "✔ certo", "✘ erro"),
                        "ms/fluxo (lote)": round(ms * 1000, 3)})
    est["ultimos"] = pd.concat([tab.iloc[::-1], est["ultimos"]]).head(15)
    # histórico por lote (alimenta a linha do tempo)
    if tipo == "bin":
        atq_prev = int((pred == 1).sum())
        reg = {"lote": len(est["hist"]) + 1, "fluxos": i1, "BENIGNO": len(pred) - atq_prev, "ATAQUE": atq_prev,
               "ataque_real": int((real == 1).sum())}
    else:
        reg = {"lote": len(est["hist"]) + 1, "fluxos": i1, **{c: int((pred == c).sum()) for c in AM["classes"]}}
    est["hist"].append(reg)
    est["cursor"] = i1


def vetores(tipo: str, est: dict, mid: str):
    n = sum(len(p) for p in est["preds"][mid])
    pred = np.concatenate(est["preds"][mid]) if n else np.array([])
    linhas = meta_cen.iloc[:n]
    real = linhas["ataque_real"].to_numpy() if tipo == "bin" else linhas["classe_real"].to_numpy()
    return real, pred


# ----------------------------------------------------------------------------- visuais
def kpis(tipo: str, est: dict):
    mid = est["ativos"][0]
    real, pred = vetores(tipo, est, mid)
    n = len(pred)
    ataques = int((pred == 1).sum()) if tipo == "bin" else int((pred != nu.BENIGNO).sum())
    ms = est["tempos"][mid] / n * 1000 if n else float("nan")
    fps = n / est["tempos"][mid] if est["tempos"][mid] > 0 else float("nan")
    c = st.columns(6)
    c[0].markdown(card("Fluxos processados", f"{n:,}", f"de {len(meta_cen):,}", CIANO), unsafe_allow_html=True)
    c[1].markdown(card("Ataques detectados", f"{ataques:,}", REG[mid]["rotulo"], VERMELHO), unsafe_allow_html=True)
    c[2].markdown(card("Benignos", f"{n - ataques:,}", "classificados como normais", VERDE), unsafe_allow_html=True)
    c[3].markdown(card("% de ataques", "—" if n == 0 else f"{100 * ataques / n:.1f}%", "entre os processados", AMBAR), unsafe_allow_html=True)
    c[4].markdown(card("Tempo médio de inferência", "—" if n == 0 else f"{ms:.3f} ms", "por fluxo (só o predict)", ROXO), unsafe_allow_html=True)
    c[5].markdown(card("Velocidade real", "—" if n == 0 else f"{fps:,.0f}/s", "fluxos por segundo", CIANO), unsafe_allow_html=True)


def donut(tipo: str, est: dict):
    mid = est["ativos"][0]
    _, pred = vetores(tipo, est, mid)
    if len(pred) == 0:
        return layout(go.Figure(), 320, "Classificações (previsto)")
    if tipo == "bin":
        nomes, vals = ["BENIGNO", "ATAQUE"], [int((pred == 0).sum()), int((pred == 1).sum())]
        cores = [VERDE, VERMELHO]
    else:
        nomes = [c for c in AM["classes"] if (pred == c).any()]
        vals = [int((pred == c).sum()) for c in nomes]
        cores = [CORES_CLASSE.get(c, MUDO) for c in nomes]
    fig = go.Figure(go.Pie(labels=nomes, values=vals, hole=0.62, marker=dict(colors=cores, line=dict(color=BG, width=2)),
                           textinfo="percent", sort=False))
    fig.add_annotation(text=f"<b>{sum(vals):,}</b><br><span style='font-size:11px'>fluxos</span>", showarrow=False,
                       font=dict(size=20, color=TEXTO))
    return layout(fig, 320, "Classificações (previsto)")


def barras_real_previsto(tipo: str, est: dict):
    mid = est["ativos"][0]
    real, pred = vetores(tipo, est, mid)
    if len(pred) == 0:
        return layout(go.Figure(), 320, "Real × previsto")
    if tipo == "bin":
        ordem, rot = ["BENIGNO", "ATAQUE"], {0: "BENIGNO", 1: "ATAQUE"}
        r, p = [rot[v] for v in real], [rot[v] for v in pred]
    else:
        ordem, r, p = list(AM["classes"]), list(real), list(pred)
    cr = pd.Series(r).value_counts().reindex(ordem).fillna(0)
    cp = pd.Series(p).value_counts().reindex(ordem).fillna(0)
    fig = go.Figure([go.Bar(x=ordem, y=cr.values, name="real", marker_color="#4b5b7a"),
                     go.Bar(x=ordem, y=cp.values, name="previsto", marker_color=CIANO)])
    fig.update_layout(barmode="group", bargap=0.25)
    return layout(fig, 320, "Real × previsto, por classe")


def linha_do_tempo(tipo: str, est: dict):
    h = pd.DataFrame(est["hist"])
    if h.empty:
        return layout(go.Figure(), 320, "Linha do tempo das detecções (por lote)")
    fig = go.Figure()
    if tipo == "bin":
        fig.add_trace(go.Scatter(x=h["fluxos"], y=h["BENIGNO"], name="benigno (previsto)", stackgroup="a", line=dict(width=0.5, color=VERDE), fillcolor="rgba(52,211,153,.45)"))
        fig.add_trace(go.Scatter(x=h["fluxos"], y=h["ATAQUE"], name="ataque (previsto)", stackgroup="a", line=dict(width=0.5, color=VERMELHO), fillcolor="rgba(244,63,94,.55)"))
        fig.add_trace(go.Scatter(x=h["fluxos"], y=h["ataque_real"], name="ataque (real)", mode="lines", line=dict(color=AMBAR, dash="dot", width=2)))
    else:
        for c in AM["classes"]:
            if c in h and h[c].sum() > 0:
                fig.add_trace(go.Bar(x=h["fluxos"], y=h[c], name=c, marker_color=CORES_CLASSE.get(c, MUDO)))
        fig.update_layout(barmode="stack", bargap=0.05)
    fig.update_xaxes(title="fluxos processados")
    fig.update_yaxes(title="fluxos no lote")
    return layout(fig, 320, "Linha do tempo das detecções (por lote)")


def latencia(est: dict):
    fig = go.Figure()
    for k, mid in enumerate(est["ativos"]):
        y = est["lat"][mid]
        if y:
            fig.add_trace(go.Scatter(x=list(range(1, len(y) + 1)), y=y, name=REG[mid]["rotulo"], mode="lines",
                                     line=dict(color=CORES_MODELO[k % 2], width=2.2, shape="spline")))
    fig.update_xaxes(title="lote")
    fig.update_yaxes(title="ms por fluxo")
    return layout(fig, 280, "Custo de inferência por lote")


def medidor(valor, titulo, cor):
    v = None if valor is None or valor != valor else float(valor)
    fig = go.Figure(go.Indicator(mode="gauge+number", value=0 if v is None else 100 * v, number=dict(suffix="%", font=dict(size=26, color=TEXTO)),
                                 title=dict(text=titulo, font=dict(size=13, color=MUDO)),
                                 gauge=dict(axis=dict(range=[0, 100], tickcolor=MUDO), bar=dict(color=cor, thickness=0.28), bgcolor="#0d1628",
                                            borderwidth=0, steps=[dict(range=[0, 100], color="#101b30")])))
    fig.update_layout(height=190, margin=dict(l=10, r=10, t=40, b=0), paper_bgcolor="rgba(0,0,0,0)", font=dict(color=MUDO))
    return fig


def matriz(cm, rotulos, titulo):
    z = np.asarray(cm)
    pct = z / np.maximum(z.sum(axis=1, keepdims=True), 1)
    fig = go.Figure(go.Heatmap(z=pct, x=rotulos, y=rotulos, colorscale=[[0, "#0e1a30"], [0.5, "#155e75"], [1, CIANO]], zmin=0, zmax=1,
                               showscale=False, text=z, texttemplate="%{text}", textfont=dict(color=TEXTO, size=13),
                               hovertemplate="real %{y} → previsto %{x}: %{text}<extra></extra>", xgap=2, ygap=2))
    fig.update_layout(xaxis_title="previsto", yaxis_title="real", yaxis_autorange="reversed")
    return layout(fig, 360, titulo, legenda=False)


def barras_por_classe(tabela: pd.DataFrame):
    fig = go.Figure()
    for nome, cor in (("precisão", CIANO), ("recall", VERDE), ("F1", AMBAR)):
        fig.add_trace(go.Bar(x=tabela["classe"], y=tabela[nome], name=nome, marker_color=cor))
    fig.update_layout(barmode="group", bargap=0.2)
    fig.update_yaxes(range=[0, 1.02])
    return layout(fig, 280, "Precisão, recall e F1 por classe")


def metricas(tipo: str, est: dict):
    if sum(len(p) for p in est["preds"][est["ativos"][0]]) == 0:
        return
    st.markdown('<div class="secao">Desempenho do modelo (rótulo real × previsto)</div>', unsafe_allow_html=True)
    resumo = {}
    cols = st.columns(len(est["ativos"]))
    for k, (col, mid) in enumerate(zip(cols, est["ativos"])):
        real, pred = vetores(tipo, est, mid)
        n = len(pred)
        seg = est["tempos"][mid]
        with col:
            st.markdown(f"**{REG[mid]['rotulo']}**")
            if tipo == "bin":
                m = nu.metricas_binarias(real, pred)
                resumo[REG[mid]["rotulo"]] = {"Precisão": m["precisao"], "Recall": m["recall"], "F1": m["f1"], "Acurácia": m["acuracia"]}
                g1, g2 = st.columns(2)
                g1.plotly_chart(medidor(m["recall"], "Recall (ataques pegos)", VERDE), key=f"g1_{tipo}_{mid}")
                g2.plotly_chart(medidor(m["precisao"], "Precisão (alertas certos)", CIANO), key=f"g2_{tipo}_{mid}")
                a = st.columns(4)
                a[0].markdown(card("F1", f"{m['f1']:.4f}", "", AMBAR), unsafe_allow_html=True)
                a[1].markdown(card("MCC", "—" if m["mcc"] != m["mcc"] else f"{m['mcc']:.4f}", "", ROXO), unsafe_allow_html=True)
                a[2].markdown(card("Falsos positivos", f"{m['fp']:,}", "benigno → ataque", AMBAR), unsafe_allow_html=True)
                a[3].markdown(card("Falsos negativos", f"{m['fn']:,}", "ataque → benigno", VERMELHO), unsafe_allow_html=True)
                st.plotly_chart(matriz(m["cm"], ["BENIGNO", "ATAQUE"], "Matriz de confusão"), key=f"cm_{tipo}_{mid}")
            else:
                m = nu.metricas_multi(real, pred, list(AM["classes"]))
                resumo[REG[mid]["rotulo"]] = {"Acurácia": m["acuracia"], "F1 macro": m["f1_macro"]}
                g1, g2 = st.columns(2)
                g1.plotly_chart(medidor(m["acuracia"], "Acurácia", VERDE), key=f"g1_{tipo}_{mid}")
                g2.plotly_chart(medidor(m["f1_macro"], "F1 macro*", CIANO), key=f"g2_{tipo}_{mid}")
                a = st.columns(3)
                a[0].markdown(card("MCC", "—" if m["mcc"] != m["mcc"] else f"{m['mcc']:.4f}", "", ROXO), unsafe_allow_html=True)
                a[1].markdown(card("Falsos positivos", f"{m['fp']:,}", "benigno → ataque", AMBAR), unsafe_allow_html=True)
                a[2].markdown(card("Falsos negativos", f"{m['fn']:,}", "ataque → benigno", VERMELHO), unsafe_allow_html=True)
                st.plotly_chart(matriz(m["cm"], list(AM["classes"]), "Matriz de confusão"), key=f"cm_{tipo}_{mid}")
                st.plotly_chart(barras_por_classe(m["por_classe"]), key=f"pc_{tipo}_{mid}")
                with st.expander("Tabela por classe"):
                    st.dataframe(m["por_classe"].round(4), hide_index=True)
                st.caption("*média só das classes que já apareceram no cenário até agora.")
            st.caption(f"Inferência: {seg / n * 1000:.3f} ms/fluxo · {n / seg:,.0f} fluxos/s (tempo medido só no `predict`).")
    if len(resumo) == 2:                       # comparação RF × SVM nos mesmos fluxos
        df = pd.DataFrame(resumo)
        fig = go.Figure([go.Bar(x=df.index, y=df[c], name=c, marker_color=CORES_MODELO[i]) for i, c in enumerate(df.columns)])
        fig.update_layout(barmode="group", bargap=0.25)
        fig.update_yaxes(range=[0, 1.02])
        st.plotly_chart(layout(fig, 280, "RF × SVM nos mesmos fluxos"), key=f"cmp_{tipo}")


def feed(e):
    if not len(e["ultimos"]):
        st.info("Aguardando o início da simulação.")
        return

    def cor(v):
        return f"color:{VERDE}; font-weight:600" if "certo" in str(v) else f"color:{VERMELHO}; font-weight:600"

    def cor_prev(v):
        return f"color:{VERMELHO}" if v not in ("BENIGNO", "BENIGN") else f"color:{VERDE}"

    sty = e["ultimos"].style.map(cor, subset=["resultado"]).map(cor_prev, subset=["previsto"])
    st.dataframe(sty, hide_index=True, height=330)


# ----------------------------------------------------------------------------- aba
def aba(tipo: str):
    est = estado(tipo)
    total = len(meta_cen)
    intervalo = max(0.5, lote / vel)

    c1, c2, c3, c4 = st.columns([1, 1, 1, 4])
    if c1.button("▶ Iniciar", key=f"ini_{tipo}", disabled=est["rodando"] or est["cursor"] >= total):
        est["rodando"] = True
        st.rerun()
    if c2.button("⏸ Pausar", key=f"pau_{tipo}", disabled=not est["rodando"]):
        est["rodando"] = False
        st.rerun()
    if c3.button("↺ Reiniciar", key=f"rei_{tipo}"):
        st.session_state[f"est_{tipo}"] = novo_estado(tipo)
        st.rerun()
    c4.caption(f"Cenário **{cenario}** · lotes de {lote} fluxos a cada {intervalo:.1f}s (simulado: {vel} fluxos/s) · "
               f"modelos: {', '.join(REG[m]['rotulo'] for m in est['ativos'])}")

    @st.fragment(run_every=intervalo if est["rodando"] else None)
    def painel():
        e = estado(tipo)
        if e["rodando"]:
            passo(tipo, e)
            if e["cursor"] >= total:
                e["rodando"] = False
                st.rerun()
        status = ("<span class='pill on'>● EM EXECUÇÃO</span>" if e["rodando"] else
                  "<span class='pill fim'>✔ CONCLUÍDO</span>" if e["cursor"] >= total else "<span class='pill off'>❚❚ PAUSADO</span>")
        fase = meta_cen["fase"].iloc[e["cursor"] - 1] if e["cursor"] else "—"
        st.markdown(f"<div style='display:flex;gap:10px;align-items:center;margin:8px 0'>{status}"
                    f"<span class='faseb'>Fase atual: <b>{fase}</b></span></div>", unsafe_allow_html=True)
        st.progress(e["cursor"] / total, text=f"{e['cursor']:,} / {total:,} fluxos")
        kpis(tipo, e)
        g1, g2, g3 = st.columns([2, 3, 3])
        g1.plotly_chart(donut(tipo, e), key=f"do_{tipo}")
        g2.plotly_chart(linha_do_tempo(tipo, e), key=f"tl_{tipo}")
        g3.plotly_chart(barras_real_previsto(tipo, e), key=f"br_{tipo}")
        h1, h2 = st.columns([3, 2])
        with h1:
            st.markdown('<div class="secao">Últimos fluxos classificados</div>', unsafe_allow_html=True)
            feed(e)
        with h2:
            st.plotly_chart(latencia(e), key=f"la_{tipo}")
        metricas(tipo, e)

    painel()


st.markdown(f"""<div class="topo"><div><div class="tit">🛡️ Simulação de IDS — detecção de DDoS</div>
<div class="sub">Random Forest × SVM RBF · CICDDoS2019 · modelos treinados carregados do disco, sem retreino</div></div></div>""",
            unsafe_allow_html=True)
st.markdown(
    "<div class='aviso'><b>Como ler os números.</b> Simulação sobre uma amostra congelada do dia de teste "
    f"({len(AM['meta']):,} fluxos, cenário construído). As métricas daqui <b>não substituem nem são comparáveis</b> às do TCC, calculadas nos "
    "~7,2 milhões de fluxos do teste completo. As views não têm horário: a ordem dos fluxos é construída pelo cenário. "
    "Limiares naturais dos modelos (RF 0,5; SVM 0), sem ajuste no teste.</div>", unsafe_allow_html=True)
st.write("")
t_bin, t_multi = st.tabs(["Binário (BENIGNO × ATAQUE)", "Multiclasse (BENIGN + 6 tipos de ataque)"])
with t_bin:
    aba("bin")
with t_multi:
    aba("multi")
