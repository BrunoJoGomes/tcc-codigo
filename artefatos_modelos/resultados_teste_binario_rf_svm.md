# Teste binário (BENIGN × ataque) no CICDDoS2019: RF × SVM (RBF)

Registro escrito em 25/10/2026, a partir das saídas dos notebooks 08 (RF) e 12 (SVM) e das buscas dos notebooks 07 e 11.
Os números são **descritivos**: o teste foi usado uma vez por modelo e não decidiu nenhuma escolha.
Tabela completa das métricas: `comparacao_metricas_teste_rf_svm.csv`. Tempos e custo: `comparacao_tempos_teste_rf_svm.csv`.

## 1. O que foi comparado

| | RF | SVM |
|---|---|---|
| Padrão | 100 árvores, profundidade livre, folha 1 | C = 1, gamma = scale |
| Ajustado (escolhido pela validação cruzada) | 300 árvores, profundidade 15, folha 1 | C = 100, gamma = 4× o de referência (0,09521) |
| Escore e limiar natural | probabilidade, 0,5 | `decision_function`, 0 |
| Pasta | `artefatos_modelos\rf_binario_busca\rf_busca_20261024_175105` | `artefatos_modelos\svm_binario_busca\svm_busca_20261025_004828` |

Teste: 7.219.189 linhas (56.216 BENIGN = 0,779%), sem sobreposição com o treino. Tipos no teste: LDAP, MSSQL, NetBIOS, Portmap, Syn, UDP, UDPLag e BENIGN. O **Portmap** (2.702 linhas) **não existe no treino**. O **Syn** é 4.887.060 linhas (68% dos ataques).
Regra de escolha (fixada antes dos resultados): validação cruzada de 5 dobras (as mesmas para os dois, impressão digital `2defef26`), ROC-AUC e MCC na proporção real, 1 erro-padrão, desempate pelo modelo mais barato.

## 2. Resultados no teste (modelos ajustados)

| | RF ajustado | SVM ajustado |
|---|---|---|
| ROC-AUC | 0,99136 | **0,99531** |
| PR-AUC do BENIGN (linha de base 0,00779) | **0,97473** | 0,96823 |
| **Limiar natural**: recall / falso alarme | 0,9526 / 1,76% | **0,9964 / 1,16%** |
| MCC | 0,361 | **0,821** |
| ataques perdidos (FN) / alarmes falsos (FP) | 339.595 / 991 | **25.657 / 652** |
| **Limiar da validação p/ 1%**: recall / falso alarme | 1,0000 / 26,4% | 0,9997 / **3,2%** |
| MCC | 0,857 | **0,965** |
| **Limiar da validação p/ 0,1%**: recall / falso alarme | **0,9997** / 3,0% | 0,5933 / **0,57%** |
| MCC | **0,967** | 0,105 |

Modelos padrão: ROC-AUC RF 0,98993 e SVM 0,99504. No limiar natural, o MCC do RF padrão é 0,355 e o do SVM padrão é 0,148 (recall 0,746; só 62,8% dos `Syn`). A diferença de ROC-AUC entre ajustado e padrão é +0,00143 no RF e +0,00026 no SVM.

## 3. Recall por tipo de ataque (limiar natural; BENIGN = especificidade)

| Tipo | n no teste | RF ajustado | RF padrão | SVM ajustado | SVM padrão |
|---|---|---|---|---|---|
| BENIGN | 56.216 | 0,9824 | 0,9790 | 0,9884 | 0,9909 |
| LDAP | 112.491 | 0,9992 | 0,9992 | 0,9994 | 0,9992 |
| MSSQL | 854.242 | 0,9998 | 0,9997 | 0,9997 | 0,9997 |
| NetBIOS | 91.175 | 0,9993 | 0,9993 | 0,9994 | 0,9992 |
| Portmap (fora do treino) | 2.702 | 0,9304 | 0,9289 | 0,9567 | 0,9341 |
| **Syn** | 4.887.060 | **0,9307** | 0,9285 | **0,9949** | 0,6278 |
| UDP | 1.214.746 | 0,9999 | 0,9999 | 1,0000 | 0,9999 |
| UDPLag | 557 | 0,8995 | 0,8977 | 0,8815 | 0,8115 |

O RF ajustado perde cerca de 338,7 mil `Syn` (99,8% dos seus 339,6 mil erros). O SVM ajustado perde 24,9 mil `Syn` (97% dos seus 25,7 mil erros).

## 4. Transferência da validação para o teste

| | RF ajustado | SVM ajustado |
|---|---|---|
| Validação (limiar natural): recall / falso alarme | 0,9992 / 0,079% | 0,9962 / 0,227% |
| Teste (limiar natural): recall / falso alarme | 0,9526 / 1,76% (22×) | 0,9964 / 1,16% (5×) |
| ROC-AUC validação → teste | 0,99999 → 0,99136 | 0,99948 → 0,99531 |
| Falso alarme no teste ÷ alvo, limiar de 1% | 26× | 3,2× |
| Falso alarme no teste ÷ alvo, limiar de 0,1% | 30× | 5,7× |

Os limiares da validação (mesmo dia do treino) não se transferem para o teste em nenhum dos dois modelos: no RF o falso alarme explode, e no SVM o limiar de 0,1% derruba o recall para 59%.

## 5. O que a validação previa (outro ranking)

| | RF ajustado | SVM ajustado |
|---|---|---|
| Validação cruzada: ROC-AUC / MCC na proporção real | 0,99999 / 0,928 | 0,99917 / 0,701 |
| Validação separada: ROC-AUC / MCC na proporção real | 0,99999 / 0,939 | 0,99948 / 0,785 |
| Validação separada: alarmes falsos / ataques perdidos | 8 / 41 | 23 / 189 |

Na validação o RF era claramente melhor; no teste, o ROC-AUC e o MCC no limiar natural favorecem o SVM.

## 6. Custo

| | RF ajustado | SVM ajustado |
|---|---|---|
| Busca de hiperparâmetros | 1 h 08 min (27 configurações, 162 florestas) | 43 min 05 s (9 configurações, 54 SVMs) |
| Refit (101.536 linhas) | 20,9 s | 126,8 s |
| Modelo em disco | 2,5 MiB | 0,37 MiB (2.661 vetores de suporte) |
| Previsão dos 7,2 milhões de fluxos | 52,3 s (138 mil fluxos/s) | 1.227,6 s (5,9 mil fluxos/s, 2 processos) |
| Tempo total do teste (dois modelos) | 3 min 39 s | 44 min 03 s |
| Latência de 1 fluxo, p50 fim a fim (1 processo) | 29,7 ms | 1,41 ms |
| Vazão em lote do modelo (1 processo) | 59 mil fluxos/s | 2,6 mil fluxos/s |

O tempo de previsão do SVM é proporcional ao número de vetores de suporte (padrão 2.539: 1.174,5 s; ajustado 2.661: 1.227,6 s). O RF é cerca de 23× mais rápido em lote; o SVM é cerca de 21× mais rápido para um fluxo isolado.

## 7. Leitura

1. Não há vencedor único: depende do ponto de operação. O SVM é melhor em ROC-AUC e no limiar natural; o RF é melhor no PR-AUC do BENIGN e no ponto de 0,1%. Nos melhores pontos os dois empatam (MCC 0,967 do RF em 0,1% e 0,965 do SVM em 1%).
2. A diferença no limiar natural vem do `Syn`, que é 68% dos ataques. No RF, 7% dos `Syn` ficam abaixo do limiar de 0,5 (mediana do escore 0,62, perto do limiar); no SVM, só 0,5% ficam abaixo de 0 (mediana 0,81).
3. O ajuste do SVM mudou o limiar natural, não o ranking: o ROC-AUC do padrão (0,99504) e do ajustado (0,99531) são quase iguais, mas o recall do padrão no limiar 0 é só 0,746. Um C alto reposicionou a fronteira.
4. Custo: o RF é muito mais rápido em lote; o SVM é mais rápido por fluxo e gera um modelo menor.

## 8. Limitações (declarar no texto)

- O teste é uma única partição, com cerca de 89% de linhas repetidas: as diferenças são descritivas, sem teste de significância.
- A validação é do mesmo dia do treino e superestima os dois modelos. Por isso previu um ranking diferente do teste.
- O RF depende da semente: no estudo de sementes (notebook 09), o MCC no limiar 0,5 do RF padrão variou de 0,34 a 0,91, e o do ajustado ficou entre 0,357 e 0,362. O SVM é determinístico (uma execução).
- Limiar natural do RF (0,5) e do SVM (0) não são comparáveis como probabilidades; o ROC-AUC é a métrica sem limiar.
- O vencedor da busca do SVM está no canto da grade (maior C e maior gamma); o ótimo pode estar fora dela. A grade 3 × 3 foi mantida por decisão de comparabilidade com as 3 variações por hiperparâmetro do RF.
- O SVM tem recall baixo no `WebDDoS` (62% na validação cruzada e 67% na validação; classe com 331 linhas no ajuste), que não aparece no teste.
- Os tempos de relógio do teste vêm das saídas coladas (notebooks antigos), e a busca do RF está arredondada ao minuto.

## 9. Onde estão os números originais

Em cada pasta de teste: `metricas_teste.json`, `metricas_teste.csv`, `comparacao_padrao_vs_ajustado_teste.csv`, `recall_por_tipo_teste.csv`, `custo_inferencia_teste.json`, `curvas_teste.png`, `recall_por_tipo_teste.png` e `predicoes_teste.npz` (escore de cada linha do teste: permite recalcular qualquer métrica sem o servidor). O registro de todas as execuções está em `registro_testes.csv`, na pasta de cima.
