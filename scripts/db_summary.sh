#!/usr/bin/env bash
set -euo pipefail

# 默认从 Config.setting.GetDatabaseConfig() 取 database_root，
# 失败则回退到 ./bigvision_db。
# 可用环境变量覆盖：
#   DB_ROOT=/data/mydb
#   DUCKDB_PATH=/data/mydb/db/catalog.duckdb
#   THREADS=8

DB_ROOT="${DB_ROOT:-}"
DUCKDB_PATH="${DUCKDB_PATH:-}"
THREADS="${THREADS:-0}"

usage() {
  cat <<'EOF'
Usage: scripts/db_summary.sh [-d <db_root>] [-u <duckdb_path>] [-t <threads>]

Print database summary (totals + per-protocol datasets & relation counts).

Options:
  -d <path>   Database root (overrides $DB_ROOT)
  -u <path>   DuckDB path   (overrides $DUCKDB_PATH)
  -t <int>    DuckDB threads PRAGMA (overrides $THREADS; default 0=auto)
  -h          Show this help

Env overrides:
  DB_ROOT=/data/mydb
  DUCKDB_PATH=/data/mydb/db/catalog.duckdb
  THREADS=8
EOF
}

while getopts ":d:u:t:h" opt; do
  case $opt in
    d) DB_ROOT="$OPTARG" ;;
    u) DUCKDB_PATH="$OPTARG" ;;
    t) THREADS="$OPTARG" ;;
    h) usage; exit 0 ;;
    \?) echo "Invalid option: -$OPTARG" >&2; usage; exit 2 ;;
    :)  echo "Option -$OPTARG requires an argument." >&2; usage; exit 2 ;;
  esac
done

python - <<'PY'
import os, json, sys
# Resolve db_root
db_root = os.environ.get("DB_ROOT")
if not db_root:
    try:
        from Config.setting import GetDatabaseConfig  # type: ignore
        cfg = GetDatabaseConfig()
        db_root = cfg["database_root"]
        print(f"[INFO] database_root from Config.setting: {db_root}", file=sys.stderr)
    except Exception:
        import os as _os
        db_root = _os.path.abspath("./bigvision_db")
        print(f"[INFO] Using fallback database_root: {db_root}", file=sys.stderr)

duckdb_path = os.environ.get("DUCKDB_PATH") or None
threads = int(os.environ.get("THREADS","0") or "0")

try:
    from Database.db import BigVisionDatabase
except Exception:
    from db import BigVisionDatabase  # type: ignore

db = BigVisionDatabase(database_root=db_root, duckdb_path=duckdb_path, threads=threads)
try:
    summary = db.get_db_summary()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
finally:
    db.close()
PY
