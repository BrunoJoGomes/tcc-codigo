# Simulação de IDS (RF × SVM) — dashboard Streamlit

Pasta: `railway-ch/simulacao_ids/`. Tudo é **somente leitura**: nenhum modelo é treinado ou alterado, o ClickHouse só recebe `SELECT`.

## Passo a passo (uma vez)
```
pip install -r requirements_simulacao.txt
python exportar_modelos.py          # cria modelos.json (confere os 8 modelos: colunas, classes, versão)
python preparar_demo.py             # cria amostra_congelada.joblib (~3.600 fluxos; usa o .env da pasta acima)
streamlit run app.py
```
Depois da primeira vez, só `streamlit run app.py`: a amostra é congelada, então toda execução/vídeo usa os mesmos fluxos.
`python preparar_demo.py --por-classe 150 --benignos 450` gera uma versão menor para testar rápido.

## O que é cada coisa
- `nucleo.py`: carrega modelos, monta os cenários, classifica lotes, calcula métricas.
- `app.py`: dashboard (duas abas: binário e multiclasse; Iniciar/Pausar/Reiniciar; KPIs; distribuição; últimos fluxos; matriz de confusão, precisão/recall/F1, MCC, FP/FN, ms/fluxo e fluxos/s).
- Cenários: **Fases de ataque** (tráfego normal → uma fase de cada ataque → normal; linha do tempo construída) e **Balanceado** (mesma quantidade de cada classe).
- Modelos: ajustados (busca) ou padrão, RF ou SVM, com opção de comparar os dois no mesmo fluxo.

## Cuidados para o texto do TCC
- As views não têm horário: a ordem dos fluxos é construída pelo cenário (diga isso).
- Métricas da simulação (amostra de poucos milhares) **não** são comparáveis às do teste completo (~7,2 milhões).
- Limiares naturais (RF 0,5; SVM 0), sem ajuste no teste. `Portmap` (classe só do teste) não entra nesta versão.
- A velocidade simulada (fluxos/s do controle) é só o ritmo da tela; a velocidade real do modelo é medida à parte.
