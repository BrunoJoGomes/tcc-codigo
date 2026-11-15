"""Núcleo da simulação IDS (sem Streamlit): carregar modelos, montar cenários, classificar lotes e calcular métricas.

Tudo aqui é leitura: os modelos são só carregados (nenhum .fit) e a amostra é um arquivo congelado.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (confusion_matrix, matthews_corrcoef,
                             precision_recall_fscore_support)

PASTA = Path(__file__).resolve().parent
ARQ_REGISTRO = PASTA / "modelos.json"
ARQ_AMOSTRA = PASTA / "amostra_congelada.joblib"
BENIGNO = "BENIGN"

# --------------------------------------------------------------------------------------- registro e modelos


def carregar_registro() -> dict:
    if not ARQ_REGISTRO.exists():
        raise FileNotFoundError("modelos.json não existe: rode `python exportar_modelos.py` primeiro.")
    return json.loads(ARQ_REGISTRO.read_text(encoding="utf-8"))


def carregar_modelo(info: dict):
    caminho = Path(info["caminho_abs"])
    if not caminho.exists():
        caminho = (PASTA.parent / info["caminho_rel"]).resolve()
    return joblib.load(caminho)


def carregar_amostra() -> dict:
    if not ARQ_AMOSTRA.exists():
        raise FileNotFoundError("amostra_congelada.joblib não existe: rode `python preparar_demo.py` primeiro.")
    return joblib.load(ARQ_AMOSTRA)


# --------------------------------------------------------------------------------------- cenários


def cenario_balanceado(meta: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    """Mesma quantidade de cada classe (a menor), embaralhadas com semente fixa. Sem repetir fluxo."""
    n = int(meta.groupby("classe_real").size().min())
    partes = [g.sample(n=n, random_state=seed) for _, g in meta.groupby("classe_real")]
    out = pd.concat(partes).sample(frac=1.0, random_state=seed)
    out = out.assign(fase="mistura balanceada")
    return out.reset_index(drop=True)


def cenario_fases(meta: pd.DataFrame, seed: int = 7, ordem=("Syn", "UDP", "LDAP", "MSSQL", "NetBIOS", "UDPLag"),
                  por_fase: int = 400, benigno_na_fase: float = 0.25) -> pd.DataFrame:
    """Linha do tempo CONSTRUÍDA (as views não têm horário): tráfego normal -> uma fase de cada ataque (com um pouco de
    tráfego benigno misturado) -> tráfego normal. Cada fluxo aparece no máximo uma vez."""
    rng = np.random.default_rng(seed)
    benignos = meta[meta.classe_real == BENIGNO].sample(frac=1.0, random_state=seed)
    ataques = {c: meta[meta.classe_real == c].sample(frac=1.0, random_state=seed) for c in ordem}
    ataques = {c: g for c, g in ataques.items() if len(g)}
    n_ben_fase = int(round(por_fase * benigno_na_fase))
    reserva = max(len(benignos) - n_ben_fase * len(ataques), 0)
    n_aquec = reserva // 2
    n_final = reserva - n_aquec
    pos = 0

    def tomar(k):
        nonlocal pos
        b = benignos.iloc[pos:pos + k]
        pos += len(b)
        return b

    blocos = [tomar(n_aquec).assign(fase="1. tráfego normal")]
    for i, (c, g) in enumerate(ataques.items(), start=2):
        mix = pd.concat([g.iloc[:por_fase], tomar(n_ben_fase)]).sample(frac=1.0, random_state=int(rng.integers(1 << 30)))
        blocos.append(mix.assign(fase=f"{i}. ataque {c}"))
    blocos.append(tomar(n_final).assign(fase=f"{len(ataques) + 2}. tráfego normal"))
    return pd.concat([b for b in blocos if len(b)]).reset_index(drop=True)


def montar_cenario(meta: pd.DataFrame, nome: str, seed: int = 7) -> pd.DataFrame:
    """Devolve o `meta` na ordem de exibição. A coluna `pos` aponta a linha original em X (preservada em `pos`)."""
    meta = meta.reset_index(drop=True).assign(pos=lambda d: np.arange(len(d)))
    if nome == "Balanceado":
        return cenario_balanceado(meta, seed)
    if nome == "Fases de ataque":
        return cenario_fases(meta, seed)
    raise ValueError(nome)


# --------------------------------------------------------------------------------------- inferência


def _escolher(X: pd.DataFrame, colunas: list[str]) -> np.ndarray:
    faltam = [c for c in colunas if c not in X.columns]
    if faltam:
        raise KeyError(f"colunas ausentes na amostra: {faltam[:5]}...")
    return X[colunas].to_numpy(dtype=np.float64)


def classificar(modelo, info: dict, X: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Classifica um lote. Devolve (rótulos previstos, segundos gastos só na inferência).

    Binário: 1 = ataque, 0 = benigno (limiar natural do modelo: RF prob >= 0,5; SVM decision_function > 0).
    Multiclasse: RF = maior probabilidade; SVM = maior pontuação um-contra-resto (ou predict, se o formato for outro).
    """
    A = _escolher(X, info["colunas"])
    classes = np.array(info["classes"])
    t0 = time.perf_counter()
    if info["tipo"] == "bin":
        if info["familia"] == "rf":
            p = modelo.predict_proba(A)[:, list(modelo.classes_).index(info["classe_positiva"])]
            pred = (p >= info["limiar"]).astype(int)
        else:
            s = np.ravel(modelo.decision_function(A))
            pred = (s > info["limiar"]).astype(int)
    else:
        if info["familia"] == "rf":
            pred = classes[np.argmax(modelo.predict_proba(A), axis=1)]
        else:
            s = modelo.decision_function(A)
            pred = classes[np.argmax(s, axis=1)] if np.ndim(s) == 2 and s.shape[1] == len(classes) else classes[np.asarray(modelo.predict(A), dtype=int)]
    return np.asarray(pred), time.perf_counter() - t0


# --------------------------------------------------------------------------------------- métricas


def metricas_binarias(y, p) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    if len(y) == 0:
        return {}
    cm = confusion_matrix(y, p, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, labels=[1], zero_division=0)
    mcc = float(matthews_corrcoef(y, p)) if len(np.unique(np.r_[y, p])) > 1 else float("nan")
    return {"cm": cm, "tn": tn, "fp": fp, "fn": fn, "tp": tp, "precisao": float(pr[0]), "recall": float(rc[0]),
            "f1": float(f1[0]), "mcc": mcc, "acuracia": (tp + tn) / len(y),
            "taxa_fp": fp / (fp + tn) if (fp + tn) else float("nan"),
            "taxa_fn": fn / (fn + tp) if (fn + tp) else float("nan")}


def metricas_multi(y, p, classes) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    if len(y) == 0:
        return {}
    cm = confusion_matrix(y, p, labels=classes)
    pr, rc, f1, sup = precision_recall_fscore_support(y, p, labels=classes, zero_division=0)
    por_classe = pd.DataFrame({"classe": classes, "precisão": pr, "recall": rc, "F1": f1, "suporte": sup})
    presentes = sup > 0
    mcc = float(matthews_corrcoef(y, p)) if len(np.unique(np.r_[y, p])) > 1 else float("nan")
    ben = classes.index(BENIGNO) if BENIGNO in classes else None
    fp = fn = None
    if ben is not None:
        yb, pb = (y != BENIGNO).astype(int), (p != BENIGNO).astype(int)
        fp = int(((yb == 0) & (pb == 1)).sum())   # benigno tratado como ataque
        fn = int(((yb == 1) & (pb == 0)).sum())   # ataque tratado como benigno
    return {"cm": cm, "por_classe": por_classe, "f1_macro": float(f1[presentes].mean()) if presentes.any() else float("nan"),
            "mcc": mcc, "acuracia": float((y == p).mean()), "fp": fp, "fn": fn}
