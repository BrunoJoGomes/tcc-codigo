# railway-ch

Ingestão do dataset [CICDDoS2019](https://www.unb.ca/cic/datasets/ddos-2019.html)
numa instância ClickHouse hospedada na Railway, mais um treino de classificação
em PySpark que lê os dados direto do banco.

## Estado

18 tabelas raw, **70.427.637 linhas**, ~2,7 GiB comprimidos (ZSTD nível 3).
A contagem de cada tabela confere exatamente com o CSV de origem.

| Grupo | Tabelas |
|---|---|
| `data_*` | `drdos_dns`, `drdos_ldap`, `drdos_mssql`, `drdos_netbios`, `drdos_ntp`, `drdos_snmp`, `drdos_ssdp`, `drdos_udp`, `syn`, `tftp`, `udplag` |
| `data2_*` | `ldap`, `mssql`, `netbios`, `portmap`, `syn`, `udp`, `udplag` |

Os CSVs de origem (~30 GB) **não** estão versionados — veja `.gitignore`.

## Configuração

Copie `.env.example` para `.env` e preencha a senha:

```bash
cp .env.example .env
```

A Railway expõe o ClickHouse apenas por **HTTPS na porta 443**; a 8123 não é
publicada. O `.env` é ignorado pelo Git.

## Ingestão

```bash
bash scripts/import_raw_csvs.sh
```

Espera os CSVs em `data/` e `data2/`. O script é **idempotente e retomável**:

- Cria as tabelas com `CREATE TABLE IF NOT EXISTS`, derivando as 88 colunas do
  cabeçalho do CSV (todas `String CODEC(ZSTD(3))` — carga fiel, sem tipagem).
- Compara `count()` com as linhas do arquivo e pula o que já está completo.
- Numa tabela parcial, retoma exatamente da próxima linha não carregada.
- Envia em blocos de 100.000 linhas, comprimidos com gzip.
- Ao final de cada tabela, falha se a contagem não bater com a origem.

### Sobre os 502 da Railway

O proxy da Railway devolve `502 Application failed to respond` esporadicamente.
Os dois scripts tratam isso com até 6 tentativas e backoff progressivo.

No upload de blocos há um cuidado a mais: um 502 pode significar tanto que o
ClickHouse *não* recebeu o INSERT quanto que ele **gravou e a resposta se
perdeu**. Reenviar às cegas duplicaria linhas, então
`upload_clickhouse_chunk.sh` compara `count()` antes e depois de cada falha:

- `depois == antes + linhas_do_bloco` → já gravou, segue adiante;
- `depois == antes` → nada entrou, reenvia;
- qualquer outro valor → aborta com erro de inserção parcial.

> **Não rode duas instâncias do script ao mesmo tempo.** O ponto de retomada vem
> de `count()`, então processos concorrentes resumem da mesma linha e duplicam
> faixas inteiras.

## Treino

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/jupyter notebook notebooks/pyspark_clickhouse_classification.ipynb
```

Classificação binária **BENIGN vs ataque** com regressão logística. Os dados vêm
do ClickHouse por JDBC — nenhum CSV local é lido. A função `merge()` do
ClickHouse trata as 18 tabelas como uma só, e a amostragem, o cast para
`Float64` e o tratamento de `Infinity` são empurrados para o servidor.

As classes são fortemente desbalanceadas — 113.828 linhas BENIGN em 70,4 M,
ou 0,16 % do total. A query pega 100 k linhas benignas e uma amostra
estratificada de ataque com `LIMIT n BY label`, garantindo presença dos 18
tipos e um split final de ~50/50.

### Resultados

Regressão logística sobre 198.312 linhas (80/20, seed 42):

| Métrica | |
|---|---|
| AUC-ROC | 0,9968 |
| Acurácia | 0,9919 |
| F1 | 0,9919 |

O recall por tipo de ataque mostra o que a média esconde: 16 das 18 famílias
ficam em 1,00, mas **WebDDoS fica em 0,00** — são 439 linhas no dataset inteiro,
e o modelo simplesmente não aprende a classe. Um baseline linear resolve o caso
fácil; distinguir as famílias raras exige tratar o desbalanceamento (pesos de
classe ou reamostragem) e provavelmente um modelo não linear.

## Estrutura

```
scripts/import_raw_csvs.sh          orquestra a ingestão, retomável
scripts/upload_clickhouse_chunk.sh  envia um bloco, com retry seguro
notebooks/                          treino PySpark
```
