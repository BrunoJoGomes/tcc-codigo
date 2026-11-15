"""Gera `modelos.json`: o catálogo dos 8 modelos finais (RF/SVM x ajustado/padrão x binário/multiclasse).

SOMENTE LEITURA nos modelos: cada .joblib é carregado e inspecionado (classes, colunas, versão do sklearn), nunca
copiado nem alterado. O único arquivo criado é `modelos.json` (se já existir, o script para; use --forcar para refazer).

Uso (na pasta simulacao_ids):  python exportar_modelos.py
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import sklearn

PASTA = Path(__file__).resolve().parent
REPO = PASTA.parent
ARTEFATOS = REPO / "artefatos_modelos"

# (id, tipo, familia, variante, pasta do run, arquivo do modelo)
CATALOGO = [
    ("bin_rf_ajustado", "bin", "rf", "ajustado", "rf_binario_busca/rf_busca_20261024_175105", "modelo_final_refit.joblib"),
    ("bin_rf_padrao", "bin", "rf", "padrão", "rf_binario/rf_bin_20261021_221311", "modelo_final_refit.joblib"),
    ("bin_svm_ajustado", "bin", "svm", "ajustado", "svm_binario_busca/svm_busca_20261025_004828", "modelo_final_refit.joblib"),
    ("bin_svm_padrao", "bin", "svm", "padrão", "svm_binario_busca/svm_busca_20261025_004828", "modelo_padrao_refit.joblib"),
    ("multi_rf_ajustado", "multi", "rf", "ajustado", "rf_multiclasse_busca/rfm_busca_20261102_001711", "modelo_final_refit.joblib"),
    ("multi_rf_padrao", "multi", "rf", "padrão", "rf_multiclasse/rfm_20261101_210334", "modelo_final_refit.joblib"),
    ("multi_svm_ajustado", "multi", "svm", "ajustado", "svm_multiclasse_busca/svm_busca_20261107_013148", "modelo_final_refit.joblib"),
    ("multi_svm_padrao", "multi", "svm", "padrão", "svm_multiclasse_busca/svm_busca_20261107_013148", "modelo_padrao_refit.joblib"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forcar", action="store_true", help="refaz o modelos.json se já existir")
    a = ap.parse_args()
    saida = PASTA / "modelos.json"
    if saida.exists() and not a.forcar:
        sys.exit(f"{saida.name} já existe (nada foi alterado). Use --forcar para refazer.")

    modelos, problemas = {}, []
    for mid, tipo, fam, var, pasta, arq in CATALOGO:
        caminho = ARTEFATOS / pasta / arq
        prep = ARTEFATOS / pasta / "preprocessamento.json"
        if not caminho.exists() or not prep.exists():
            problemas.append(f"{mid}: não achei {caminho if not caminho.exists() else prep}")
            continue
        m = joblib.load(caminho)
        colunas = json.loads(prep.read_text(encoding="utf-8")).get("colunas_modelo")
        if not colunas and hasattr(m, "feature_names_in_"):
            colunas = list(m.feature_names_in_)
        if not colunas:
            problemas.append(f"{mid}: não consegui descobrir as colunas do modelo")
            continue
        classes = [c.item() if hasattr(c, "item") else c for c in m.classes_]
        codigos_modelo = classes
        if tipo == "multi":      # o modelo guarda códigos 0..6; os nomes vêm do classes.json da mesma pasta
            cj = ARTEFATOS / pasta / "classes.json"
            if not cj.exists():
                problemas.append(f"{mid}: não achei {cj}")
                continue
            dados = json.loads(cj.read_text(encoding="utf-8"))
            nomes = dados["classes"]
            if [dados["codigos"][n] for n in nomes] != classes:
                problemas.append(f"{mid}: códigos do classes.json não batem com model.classes_ {classes}")
                continue
            classes = nomes
        if tipo == "bin" and sorted(classes) != [0, 1]:
            problemas.append(f"{mid}: classes binárias inesperadas {classes} (esperado [0, 1])")
            continue
        n_in = getattr(m, "n_features_in_", None)
        if n_in is not None and n_in != len(colunas):
            problemas.append(f"{mid}: modelo espera {n_in} colunas mas o preprocessamento.json lista {len(colunas)}")
            continue
        modelos[mid] = {
            "tipo": tipo, "familia": fam, "variante": var,
            "rotulo": f"{'Random Forest' if fam == 'rf' else 'SVM RBF'} ({var})",
            "caminho_rel": str(Path("artefatos_modelos") / pasta / arq).replace("\\", "/"),
            "caminho_abs": str(caminho),
            "colunas": colunas, "classes": classes, "codigos_modelo": codigos_modelo,
            "classe_positiva": 1 if tipo == "bin" else None,
            # limiares naturais do modelo (nenhum ajuste feito no teste): RF prob >= 0,5; SVM decision_function > 0
            "limiar": 0.5 if fam == "rf" else 0.0,
        }
        print(f"OK  {mid:20s} {len(colunas)} colunas | classes: {classes}")

    if problemas:
        print("\nPROBLEMAS (o modelos.json NÃO foi criado):")
        for p in problemas:
            print("  -", p)
        sys.exit(1)

    saida.write_text(json.dumps({"sklearn_versao_exportacao": sklearn.__version__, "modelos": modelos},
                                indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nCriado: {saida}  (sklearn {sklearn.__version__}; os modelos continuam onde estavam, intactos)")


if __name__ == "__main__":
    main()
