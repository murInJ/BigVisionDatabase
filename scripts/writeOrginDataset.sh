#!/usr/bin/env bash
set -euo pipefail

# 默认值（可通过环境变量覆盖）
DB_ROOT_DEFAULT=""
DUCKDB_PATH_DEFAULT=""
MAX_WORKERS_DEFAULT="8"
THREADS_DEFAULT="0"
DRY_RUN_DEFAULT="0"
NO_PROGRESS_DEFAULT="0"
DEMO_DEFAULT="0"

usage() {
  cat <<'EOF'
Usage: scripts/writeOriginDataset.sh [options]

Options:
  -d <path>    Database root (overrides $DB_ROOT)
  -p           Disable progress bar (sets --no-progress)
  -w <int>     Max workers (default 8; overrides $MAX_WORKERS)
  -t <int>     DuckDB threads PRAGMA (default 0; overrides $THREADS)
  -n           Dry-run (no writes)
  -m           Demo single-sample mode (instead of registry ingestion)
  -h           Show this help

Env overrides:
  DB_ROOT=/data/mydb
  DUCKDB_PATH=/data/mydb/db/catalog.duckdb
  MAX_WORKERS=16
  THREADS=8
  DRY_RUN=1
  NO_PROGRESS=1
  DEMO=1
EOF
}

DB_ROOT="${DB_ROOT:-$DB_ROOT_DEFAULT}"
DUCKDB_PATH="${DUCKDB_PATH:-$DUCKDB_PATH_DEFAULT}"
MAX_WORKERS="${MAX_WORKERS:-$MAX_WORKERS_DEFAULT}"
THREADS="${THREADS:-$THREADS_DEFAULT}"
DRY_RUN="${DRY_RUN:-$DRY_RUN_DEFAULT}"
NO_PROGRESS="${NO_PROGRESS:-$NO_PROGRESS_DEFAULT}"
DEMO="${DEMO:-$DEMO_DEFAULT}"

# Parse flags
while getopts ":d:w:t:npmh" opt; do
  case $opt in
    d) DB_ROOT="$OPTARG" ;;
    w) MAX_WORKERS="$OPTARG" ;;
    t) THREADS="$OPTARG" ;;
    n) DRY_RUN="1" ;;
    p) NO_PROGRESS="1" ;;
    m) DEMO="1" ;;
    h) usage; exit 0 ;;
    \?) echo "Invalid option: -$OPTARG" >&2; usage; exit 2 ;;
    :)  echo "Option -$OPTARG requires an argument." >&2; usage; exit 2 ;;
  esac
done

ARGS=()

# Mode
if [[ "${DEMO}" == "1" ]]; then
  ARGS+=( "--demo-sample" )
else
  ARGS+=( "--registry" )
fi

# Paths
if [[ -n "${DB_ROOT}" ]]; then
  ARGS+=( "-d" "${DB_ROOT}" )
fi
if [[ -n "${DUCKDB_PATH}" ]]; then
  ARGS+=( "--duckdb-path" "${DUCKDB_PATH}" )
fi

# Concurrency
ARGS+=( "-w" "${MAX_WORKERS}" "-t" "${THREADS}" )

# Flags
if [[ "${DRY_RUN}" == "1" ]]; then
  ARGS+=( "-n" )
fi
if [[ "${NO_PROGRESS}" == "1" ]]; then
  ARGS+=( "--no-progress" )
fi

# Run
exec python -m Database.db "${ARGS[@]}"