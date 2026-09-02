#!/usr/bin/env bash

set -Eeuo pipefail

chunk_id=${1:?"chunk id is required"}
table_name=${2:?"table name is required"}

: "${CLICKHOUSE_PROTOCOL:?CLICKHOUSE_PROTOCOL is required}"
: "${CLICKHOUSE_HOST:?CLICKHOUSE_HOST is required}"
: "${CLICKHOUSE_PORT:?CLICKHOUSE_PORT is required}"
: "${CLICKHOUSE_USER:?CLICKHOUSE_USER is required}"
: "${CLICKHOUSE_PASSWORD:?CLICKHOUSE_PASSWORD is required}"
: "${CLICKHOUSE_DATABASE:?CLICKHOUSE_DATABASE is required}"

if [[ ! "$table_name" =~ ^[a-z0-9_]+$ ]]; then
  printf 'Invalid table name: %s\n' "$table_name" >&2
  exit 2
fi

max_attempts=${CLICKHOUSE_MAX_ATTEMPTS:-6}

work_dir=$(mktemp -d)
trap 'rm -rf "$work_dir"' EXIT
chunk_file="$work_dir/chunk.csv"
response_file="$work_dir/response"

# Buffer the chunk so a failed upload can be replayed; split streams it on stdin.
cat >"$chunk_file"
chunk_rows=$(wc -l <"$chunk_file")

base_url="${CLICKHOUSE_PROTOCOL}://${CLICKHOUSE_HOST}:${CLICKHOUSE_PORT}/?database=${CLICKHOUSE_DATABASE}"
insert_url="${base_url}&query=INSERT%20INTO%20%60${table_name}%60%20FORMAT%20CSV"

# Row count is the only way to tell a lost response from a lost insert: the
# Railway proxy returns 502 both when ClickHouse never saw the request and when
# it committed but the reply was dropped.
row_count() {
  local attempt value
  for ((attempt = 1; attempt <= max_attempts; attempt++)); do
    if value=$(curl \
      -sS \
      --fail-with-body \
      --connect-timeout 20 \
      --max-time 120 \
      -H 'Expect:' \
      --user "${CLICKHOUSE_USER}:${CLICKHOUSE_PASSWORD}" \
      --data-binary "SELECT count() FROM \`${table_name}\` FORMAT TSV" \
      "$base_url" 2>/dev/null); then
      printf '%s' "$value"
      return 0
    fi
    sleep $((attempt * 5))
  done
  return 1
}

if ! rows_before=$(row_count); then
  printf 'Could not read row count for %s before chunk %s\n' "$table_name" "$chunk_id" >&2
  exit 1
fi

for ((attempt = 1; attempt <= max_attempts; attempt++)); do
  if gzip -1 -c "$chunk_file" | curl \
    -sS \
    --fail-with-body \
    --connect-timeout 20 \
    -H 'Expect:' \
    -H 'Content-Encoding: gzip' \
    --user "${CLICKHOUSE_USER}:${CLICKHOUSE_PASSWORD}" \
    --data-binary @- \
    "$insert_url" \
    >"$response_file" 2>"$response_file.err"; then
    printf '  %s: chunk %s loaded (%d rows)\n' "$table_name" "${chunk_id##*_}" "$chunk_rows"
    exit 0
  fi

  if ! rows_after=$(row_count); then
    printf 'Could not read row count for %s after failed chunk %s\n' "$table_name" "$chunk_id" >&2
    exit 1
  fi

  if (( rows_after == rows_before + chunk_rows )); then
    printf '  %s: chunk %s committed despite a dropped response (%d rows)\n' \
      "$table_name" "${chunk_id##*_}" "$chunk_rows"
    exit 0
  fi

  if (( rows_after != rows_before )); then
    printf '%s chunk %s landed partially: before=%d after=%d expected=%d\n' \
      "$table_name" "$chunk_id" "$rows_before" "$rows_after" "$((rows_before + chunk_rows))" >&2
    exit 1
  fi

  printf '  %s: chunk %s failed (attempt %d/%d), retrying in %ds\n' \
    "$table_name" "${chunk_id##*_}" "$attempt" "$max_attempts" "$((attempt * 5))" >&2
  sed -n '1,10p' "$response_file" >&2
  sed -n '1,10p' "$response_file.err" >&2
  sleep $((attempt * 5))
done

printf 'Upload permanently failed for %s, chunk %s after %d attempts\n' \
  "$table_name" "$chunk_id" "$max_attempts" >&2
exit 1
