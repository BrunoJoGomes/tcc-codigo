#!/usr/bin/env bash

set -Eeuo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$script_dir/.." && pwd)
env_file="$repo_dir/.env"

if [[ ! -f "$env_file" ]]; then
  printf 'Missing environment file: %s\n' "$env_file" >&2
  exit 2
fi

set -a
# shellcheck disable=SC1090
source "$env_file"
set +a

: "${CLICKHOUSE_PROTOCOL:?CLICKHOUSE_PROTOCOL is required}"
: "${CLICKHOUSE_HOST:?CLICKHOUSE_HOST is required}"
: "${CLICKHOUSE_PORT:?CLICKHOUSE_PORT is required}"
: "${CLICKHOUSE_USER:?CLICKHOUSE_USER is required}"
: "${CLICKHOUSE_PASSWORD:?CLICKHOUSE_PASSWORD is required}"
: "${CLICKHOUSE_DATABASE:?CLICKHOUSE_DATABASE is required}"

if [[ ! "$CLICKHOUSE_DATABASE" =~ ^[A-Za-z0-9_]+$ ]]; then
  printf 'Invalid database name: %s\n' "$CLICKHOUSE_DATABASE" >&2
  exit 2
fi

endpoint="${CLICKHOUSE_PROTOCOL}://${CLICKHOUSE_HOST}:${CLICKHOUSE_PORT}/?database=${CLICKHOUSE_DATABASE}"

max_attempts=${CLICKHOUSE_MAX_ATTEMPTS:-6}

# Every query here is idempotent (reads plus CREATE TABLE IF NOT EXISTS), so a
# transient 502 from the Railway proxy can be retried without side effects.
ch_query() {
  local query=${1:?"query is required"}
  local attempt response
  for ((attempt = 1; attempt <= max_attempts; attempt++)); do
    if response=$(curl \
      -sS \
      --fail-with-body \
      --connect-timeout 20 \
      --max-time 300 \
      -H 'Expect:' \
      --user "${CLICKHOUSE_USER}:${CLICKHOUSE_PASSWORD}" \
      --data-binary "$query" \
      "$endpoint" 2>&1); then
      printf '%s' "$response"
      return 0
    fi
    printf 'Query failed (attempt %d/%d), retrying in %ds: %s\n' \
      "$attempt" "$max_attempts" "$((attempt * 5))" "$response" >&2
    sleep $((attempt * 5))
  done
  printf 'Query permanently failed after %d attempts: %.120s\n' "$max_attempts" "$query" >&2
  return 1
}

csv_files=(
  "data2/Portmap.csv:data2_portmap"
  "data/UDPLag.csv:data_udplag"
  "data2/UDPLag.csv:data2_udplag"
  "data/Syn.csv:data_syn"
  "data/DrDoS_NTP.csv:data_drdos_ntp"
  "data2/LDAP.csv:data2_ldap"
  "data/DrDoS_LDAP.csv:data_drdos_ldap"
  "data/DrDoS_SSDP.csv:data_drdos_ssdp"
  "data/DrDoS_UDP.csv:data_drdos_udp"
  "data2/NetBIOS.csv:data2_netbios"
  "data2/UDP.csv:data2_udp"
  "data/DrDoS_NetBIOS.csv:data_drdos_netbios"
  "data2/Syn.csv:data2_syn"
  "data/DrDoS_MSSQL.csv:data_drdos_mssql"
  "data/DrDoS_DNS.csv:data_drdos_dns"
  "data/DrDoS_SNMP.csv:data_drdos_snmp"
  "data2/MSSQL.csv:data2_mssql"
  "data/TFTP.csv:data_tftp"
)

declare -A expected_rows_by_table=()

first_csv="$repo_dir/${csv_files[0]%%:*}"
schema=$(head -n 1 "$first_csv" | LC_ALL=C awk -F',' '
  {
    for (i = 1; i <= NF; i++) {
      name = tolower($i)
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", name)
      gsub(/[^a-z0-9]+/, "_", name)
      gsub(/^_+|_+$/, "", name)
      if (name ~ /^[0-9]/) {
        name = "c_" name
      }
      if (seen[name]++) {
        name = name "_" seen[name]
      }
      printf "%s`%s` String CODEC(ZSTD(3))", (i == 1 ? "" : ", "), name
    }
  }
')

printf 'Creating and validating %d raw tables in database %s...\n' "${#csv_files[@]}" "$CLICKHOUSE_DATABASE"

for entry in "${csv_files[@]}"; do
  relative_path=${entry%%:*}
  table_name=${entry##*:}
  csv_path="$repo_dir/$relative_path"

  if [[ ! -f "$csv_path" ]]; then
    printf 'Missing CSV: %s\n' "$csv_path" >&2
    exit 2
  fi

  create_sql="CREATE TABLE IF NOT EXISTS \`${table_name}\` (${schema}) ENGINE = MergeTree ORDER BY tuple()"
  ch_query "$create_sql"

  current_rows=$(ch_query "SELECT count() FROM \`${table_name}\` FORMAT TSV")
  expected_rows=$(( $(wc -l < "$csv_path") - 1 ))
  expected_rows_by_table["$table_name"]=$expected_rows

  if (( current_rows == expected_rows )); then
    printf '%s already complete (%d rows); skipping.\n' "$table_name" "$current_rows"
  elif (( current_rows > expected_rows )); then
    printf '%s contains more rows than the source: loaded=%d expected=%d.\n' \
      "$table_name" "$current_rows" "$expected_rows" >&2
    exit 3
  elif (( current_rows != 0 )); then
    printf '%s partially loaded (%d of %d rows); will resume at source row %d.\n' \
      "$table_name" "$current_rows" "$expected_rows" "$((current_rows + 1))"
  else
    printf '%s ready (%d rows expected).\n' "$table_name" "$expected_rows"
  fi
done

printf 'Starting raw CSV uploads in 100,000-line chunks...\n'

for entry in "${csv_files[@]}"; do
  relative_path=${entry%%:*}
  table_name=${entry##*:}
  csv_path="$repo_dir/$relative_path"
  expected_rows=${expected_rows_by_table["$table_name"]}
  current_rows=$(ch_query "SELECT count() FROM \`${table_name}\` FORMAT TSV")

  if (( current_rows == expected_rows )); then
    continue
  fi

  printf 'Loading %s from %s (%d of %d rows remaining)...\n' \
    "$table_name" "$relative_path" "$((expected_rows - current_rows))" "$expected_rows"

  # Inserts are performed sequentially and each HTTP insert is atomic. Skip the
  # header plus the already committed prefix so rerunning this script is safe.
  tail -n "+$((current_rows + 2))" "$csv_path" | split \
    --lines=100000 \
    --numeric-suffixes=0 \
    --suffix-length=6 \
    --filter="bash \"$script_dir/upload_clickhouse_chunk.sh\" \"\$FILE\" \"$table_name\"" \
    - "/tmp/clickhouse_${table_name}_"

  loaded_rows=$(ch_query "SELECT count() FROM \`${table_name}\` FORMAT TSV")
  if (( loaded_rows != expected_rows )); then
    printf '%s count mismatch: loaded=%d expected=%d\n' "$table_name" "$loaded_rows" "$expected_rows" >&2
    exit 4
  fi

  table_size=$(ch_query "SELECT formatReadableSize(total_bytes) FROM system.tables WHERE database = currentDatabase() AND name = '${table_name}' FORMAT TSV")
  free_space=$(ch_query "SELECT formatReadableSize(free_space) FROM system.disks WHERE name = 'default' FORMAT TSV")
  printf '%s complete: %d rows, %s stored, %s free.\n' "$table_name" "$loaded_rows" "$table_size" "$free_space"
done

printf 'All raw CSV imports completed successfully.\n'
