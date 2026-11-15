"""Gera UMA VEZ a amostra congelada da simulação (`amostra_congelada.joblib`), a partir das views de teste do ClickHouse.

- SOMENTE LEITURA no servidor (só SELECT; qualquer outro comando é bloqueado aqui).
- Não usa o .env por texto na tela: a senha nunca é impressa.
- Não cria nem altera nada no servidor e não sobrescreve a amostra se ela já existir (use --forcar).
- Cada fluxo aparece uma única vez (sem duplicar nem gerar amostras artificiais). O rótulo real fica só em `meta`;
  as matrizes de atributos (`X_bin`, `X_multi`) não têm rótulo.

Uso (na pasta simulacao_ids, depois de `python exportar_modelos.py`):
    python preparar_demo.py                       # 400 fluxos por ataque + 1200 benignos (~3.600 fluxos)
    python preparar_demo.py --por-classe 150 --benignos 450     # versão menor, para testar rápido
"""
import argparse
import http.client
import json
import os
import re
import sys
import time
import urllib.parse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PASTA = Path(__file__).resolve().parent
REPO = PASTA.parent
VIEW_BIN = "prep_vw_rf_bin_teste_sem_sobrep"
VIEW_MULTI = "prep_vw_rf_multi_teste_sem_sobrep"
TAB_TESTE = "prep_final_teste"
COL_ID, COL_ROTULO = "id_linha", "label_multi"
BENIGNO = "BENIGN"


# ------------------------------------------------------------------ conexão (igual à dos notebooks, só leitura)
def carregar_env(caminho: Path) -> dict:
    v = {}
    for linha in caminho.read_text(encoding="utf-8").splitlines():
        linha = linha.strip()
        if linha and not linha.startswith("#") and "=" in linha:
            k, _, val = linha.partition("=")
            v[k.strip()] = val.strip().strip('"').strip("'")
    return v


env_path = REPO / ".env"
if not env_path.exists():
    sys.exit(f"Arquivo .env não encontrado em {env_path}")
ENV = {**carregar_env(env_path), **os.environ}
CH_HOST, CH_PORT = ENV["CLICKHOUSE_HOST"], int(ENV["CLICKHOUSE_PORT"])
CH_USER, CH_PASSWORD, CH_DB = ENV["CLICKHOUSE_USER"], ENV["CLICKHOUSE_PASSWORD"], ENV["CLICKHOUSE_DATABASE"]
CH_SCHEME = ENV.get("CLICKHOUSE_SCHEME", "https")
ESPERAS = (0, 5, 15, 45, 90)
ESPERAS_MEMORIA = (30, 60, 120, 240)
_conn = None


def _fechar():
    global _conn
    try:
        if _conn is not None:
            _conn.close()
    except Exception:
        pass
    _conn = None


def consultar(sql: str) -> pd.DataFrame:
    """Só SELECT/SHOW/DESCRIBE. Repete em queda de conexão e espera quando o servidor fica sem memória (Code 241)."""
    global _conn
    if not re.match(r"\s*(SELECT|SHOW|DESCRIBE)\b", sql, re.I):
        raise PermissionError("preparar_demo.py só executa leituras.")
    caminho = "/?" + urllib.parse.urlencode({"database": CH_DB, "default_format": "JSONCompact", "wait_end_of_query": 1})
    cab = {"X-ClickHouse-User": CH_USER, "X-ClickHouse-Key": CH_PASSWORD, "Connection": "keep-alive"}
    ultimo, mem = None, 0
    tentativas = 0
    while tentativas < len(ESPERAS):
        espera = ESPERAS[tentativas]
        if espera:
            print(f"  conexão instável ({str(ultimo)[:120]}); nova tentativa em {espera}s...", flush=True)
            time.sleep(espera)
        tentativas += 1
        try:
            if _conn is None:
                cls = http.client.HTTPSConnection if CH_SCHEME == "https" else http.client.HTTPConnection
                _conn = cls(CH_HOST, CH_PORT, timeout=1800)
            _conn.request("POST", caminho, body=sql.encode("utf-8"), headers=cab)
            r = _conn.getresponse()
            corpo = r.read()
        except (OSError, http.client.HTTPException) as e:
            ultimo = e
            _fechar()
            continue
        if r.status in (502, 503, 504):
            ultimo = Exception(f"HTTP {r.status}")
            _fechar()
            continue
        if r.status != 200:
            texto = corpo.decode("utf-8", "replace")
            if "241" in texto[:40] or "memory limit" in texto.lower():
                if mem < len(ESPERAS_MEMORIA):
                    print(f"  ⚠ servidor sem memória; esperando {ESPERAS_MEMORIA[mem]}s e tentando de novo...", flush=True)
                    time.sleep(ESPERAS_MEMORIA[mem])
                    mem += 1
                    tentativas -= 1
                    continue
            raise RuntimeError(texto[:1500])
        d = json.loads(corpo)
        return pd.DataFrame(d.get("data", []), columns=[m["name"] for m in d.get("meta", [])])
    raise RuntimeError(f"Sem conexão com o ClickHouse: {ultimo!r}")


def lit(s):
    return "'" + str(s).replace("\\", "\\\\").replace("'", "\\'") + "'"


def q(c):
    return "`" + str(c).replace("`", "``") + "`"


def colunas_de(tabela):
    return consultar(f"SELECT name FROM system.columns WHERE database = currentDatabase() AND `table` = {lit(tabela)} ORDER BY position")["name"].tolist()


def ler_por_ids(view, colunas, ids, lote=400):
    """Lê as linhas da view para os ids pedidos (em lotes). id_linha vem como texto para não perder precisão."""
    partes = []
    sel = ", ".join(["toString(id_linha) AS id_linha"] + [q(c) for c in colunas])
    for i in range(0, len(ids), lote):
        lista = ",".join(ids[i:i + lote])
        partes.append(consultar(f"SELECT {sel} FROM {view} WHERE id_linha IN ({lista})"))
    df = pd.concat(partes, ignore_index=True) if partes else pd.DataFrame(columns=["id_linha"] + colunas)
    for c in colunas:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--por-classe", type=int, default=400, help="fluxos por tipo de ataque (padrão 400)")
    ap.add_argument("--benignos", type=int, default=1200, help="fluxos benignos (padrão 1200)")
    ap.add_argument("--folga", type=float, default=1.3, help="sobra pedida ao servidor para o cruzamento binário x multiclasse")
    ap.add_argument("--forcar", action="store_true", help="refaz a amostra se já existir")
    a = ap.parse_args()

    saida = PASTA / "amostra_congelada.joblib"
    if saida.exists() and not a.forcar:
        sys.exit(f"{saida.name} já existe (nada foi alterado). Use --forcar para refazer.")
    reg_path = PASTA / "modelos.json"
    if not reg_path.exists():
        sys.exit("Rode primeiro: python exportar_modelos.py")
    modelos = json.loads(reg_path.read_text(encoding="utf-8"))["modelos"]
    col_bin = sorted({c for m in modelos.values() if m["tipo"] == "bin" for c in m["colunas"]})
    col_multi = sorted({c for m in modelos.values() if m["tipo"] == "multi" for c in m["colunas"]})
    classes = list(next(m for m in modelos.values() if m["tipo"] == "multi")["classes"])
    ataques = [c for c in classes if c != BENIGNO]

    print(f"Servidor: {CH_HOST}:{CH_PORT}/{CH_DB} (somente leitura)")
    # 1) conferir o que existe antes de gastar tempo
    for tab, esperado in [(VIEW_BIN, [COL_ID] + col_bin), (VIEW_MULTI, [COL_ID] + col_multi), (TAB_TESTE, [COL_ID, COL_ROTULO])]:
        cols = colunas_de(tab)
        faltam = [c for c in esperado if c not in cols]
        if faltam:
            print(f"\nPARE: em {tab} faltam as colunas {faltam[:8]}{'...' if len(faltam) > 8 else ''}")
            print(f"Colunas existentes ({len(cols)}): {cols[:60]}")
            print("Me envie esta mensagem que eu ajusto o script.")
            sys.exit(1)
    print("Colunas conferidas nas 2 views e na tabela de rótulos.")

    # 2) escolher ids por classe EM RODADAS (ordem estável pelo hash do id) e ler as duas views até completar a cota.
    #    Só entram fluxos que existem nas duas views (sem sobreposição com o treino); os dois modelos veem o MESMO fluxo.
    cols_teste = colunas_de(TAB_TESTE)
    filtro = " AND sobrepoe_treino = 0" if "sobrepoe_treino" in cols_teste else ""
    alvo = {BENIGNO: a.benignos, **{c: a.por_classe for c in ataques}}
    manter, Xm_partes, Xb_partes = [], [], []
    for c, n in alvo.items():
        pagina, deslocamento, esgotou, rodada, ids_ok = max(int(np.ceil(n * a.folga)), 200), 0, False, 0, []
        while len(ids_ok) < n and not esgotou and rodada < 40:
            rodada += 1
            ids = consultar(f"SELECT toString({COL_ID}) AS id FROM {TAB_TESTE} WHERE {COL_ROTULO} = {lit(c)}{filtro} "
                            f"ORDER BY cityHash64({COL_ID}) LIMIT {pagina} OFFSET {deslocamento}")["id"].tolist()
            deslocamento += len(ids)
            esgotou = len(ids) < pagina
            if not ids:
                break
            xm = ler_por_ids(VIEW_MULTI, col_multi, ids).drop_duplicates("id_linha")
            xb = ler_por_ids(VIEW_BIN, col_bin, ids).drop_duplicates("id_linha")
            comuns = set(xm["id_linha"]) & set(xb["id_linha"])
            novos = [i for i in ids if i in comuns][: n - len(ids_ok)]
            ids_ok += novos
            Xm_partes.append(xm[xm["id_linha"].isin(novos)])
            Xb_partes.append(xb[xb["id_linha"].isin(novos)])
        manter.append(pd.DataFrame({"id_linha": ids_ok, "classe_real": c}))
        print(f"  {c:8s} nas duas views: {len(ids_ok):>5,} (alvo {n:,}; {rodada} rodada(s) de leitura)", flush=True)
    Xm = pd.concat(Xm_partes, ignore_index=True)
    Xb = pd.concat(Xb_partes, ignore_index=True)
    meta = pd.concat(manter, ignore_index=True)
    meta["ataque_real"] = (meta["classe_real"] != BENIGNO).astype(int)
    Xm = meta[["id_linha"]].merge(Xm.drop_duplicates("id_linha"), on="id_linha", how="left")
    Xb = meta[["id_linha"]].merge(Xb.drop_duplicates("id_linha"), on="id_linha", how="left")
    X_multi, X_bin = Xm.drop(columns="id_linha"), Xb.drop(columns="id_linha")
    assert len(meta) == len(X_multi) == len(X_bin)
    assert not X_multi.isna().any().any() and not X_bin.isna().any().any(), "há valores vazios nos atributos (não esperado nas views)"

    pequenas = [c for c in alvo if (meta.classe_real == c).sum() < alvo[c]]
    joblib.dump({"meta": meta, "X_bin": X_bin, "X_multi": X_multi, "classes": classes,
                 "origem": {"views": [VIEW_BIN, VIEW_MULTI], "tabela_rotulos": TAB_TESTE, "alvo_por_classe": alvo,
                            "criado_em": time.strftime("%Y-%m-%d %H:%M:%S"), "classes_com_menos_que_o_alvo": pequenas}}, saida)
    meta.drop(columns=[]).groupby("classe_real").size().rename("fluxos").to_csv(PASTA / "amostra_resumo.csv", encoding="utf-8")
    print(f"\nCriada: {saida.name}  ({len(meta):,} fluxos; {int(meta.ataque_real.sum()):,} ataques, "
          f"{int((1 - meta.ataque_real).sum()):,} benignos). Resumo em amostra_resumo.csv")
    if pequenas:
        print(f"Aviso: classes com menos fluxos que o alvo: {pequenas}")


if __name__ == "__main__":
    main()
