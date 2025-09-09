#!/usr/bin/env bash
set -euo pipefail

# -------- Defaults (can be overridden by env or flags) --------
DB_ROOT="${DB_ROOT:-}"
DUCKDB_PATH="${DUCKDB_PATH:-}"
THREADS="${THREADS:-0}"                # 0 = let DuckDB decide
REMOVE_ORPHANS="${REMOVE_ORPHANS:-0}"  # 1 = actually delete orphan .npy files
CHECK_DB_MISSING="${CHECK_DB_MISSING:-1}" # 1 = report DB rows pointing to missing files
REPORT_LIMIT="${REPORT_LIMIT:-20}"     # max samples to show in report
VERBOSE="${VERBOSE:-0}"                # 1 = print progress logs

usage() {
  cat <<'EOF'
Usage: scripts/db_gc.sh [options]

Garbage-collect orphan .npy files under <database_root>/images, and report DB rows
that reference missing files. Uses Database.db.BigVisionDatabase.garbage_collect().

Options:
  -d <path>   Database root (overrides $DB_ROOT). If not set, tries Config.setting.GetDatabaseConfig();
              if that fails, falls back to ./bigvision_db.
  -u <path>   DuckDB file path (overrides $DUCKDB_PATH). Default: <database_root>/db/catalog.duckdb
  -t <int>    DuckDB PRAGMA threads (overrides $THREADS). Default: 0 (auto)
  -x          Remove orphan files (sets $REMOVE_ORPHANS=1). Default: only report
  -c          Do NOT check DB-missing files (sets $CHECK_DB_MISSING=0)
  -l <int>    Report limit (overrides $REPORT_LIMIT). Default: 20
  -v          Verbose (sets $VERBOSE=1)
  -h          Show this help

Env overrides:
  DB_ROOT=/data/mydb
  DUCKDB_PATH=/data/mydb/db/catalog.duckdb
  THREADS=8
  REMOVE_ORPHANS=1
  CHECK_DB_MISSING=0
  REPORT_LIMIT=50
  VERBOSE=1
EOF
}

# -------- Parse flags --------
while getopts ":d:u:t:l:xcvh" opt; do
  case $opt in
    d) DB_ROOT="$OPTARG" ;;
    u) DUCKDB_PATH="$OPTARG" ;;
    t) THREADS="$OPTARG" ;;
    l) REPORT_LIMIT="$OPTARG" ;;
    x) REMOVE_ORPHANS="1" ;;
    c) CHECK_DB_MISSING="0" ;;
    v) VERBOSE="1" ;;
    h) usage; exit 0 ;;
    \?) echo "Invalid option: -$OPTARG" >&2; usage; exit 2 ;;
    :)  echo "Option -$OPTARG requires an argument." >&2; usage; exit 2 ;;
  esac
done

# -------- Run GC via inline Python --------
# We prefer a here-doc to avoid shell-escaping issues.
python - <<'PY'
import os, json, sys
from pathlib import Path

# Resolve database_root
db_root = os.environ.get("DB_ROOT")
if not db_root:
    try:
        from Config.setting import GetDatabaseConfig  # type: ignore
        cfg = GetDatabaseConfig()
        db_root = cfg["database_root"]
        print(f"[INFO] database_root from Config.setting: {db_root}", file=sys.stderr)
    except Exception:
        db_root = os.path.abspath("./bigvision_db")
        print(f"[INFO] Using fallback database_root: {db_root}", file=sys.stderr)

duckdb_path = os.environ.get("DUCKDB_PATH") or None
threads = int(os.environ.get("THREADS","0") or "0")
remove_orphans = os.environ.get("REMOVE_ORPHANS","0") == "1"
check_db_missing = os.environ.get("CHECK_DB_MISSING","1") == "1"
report_limit = int(os.environ.get("REPORT_LIMIT","20") or "20")
verbose = os.environ.get("VERBOSE","0") == "1"

try:
    from Database.db import BigVisionDatabase
except Exception:
    # fallback import layout
    from db import BigVisionDatabase  # type: ignore

db = BigVisionDatabase(
    database_root=db_root,
    duckdb_path=duckdb_path,
    threads=threads,
)

try:
    summary = db.garbage_collect(
        remove_orphan_files=remove_orphans,
        check_db_missing_files=check_db_missing,
        report_limit=report_limit,
        verbose=verbose,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
finally:
    db.close()
PY
