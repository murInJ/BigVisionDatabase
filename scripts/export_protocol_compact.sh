#!/usr/bin/env bash
set -euo pipefail

# scripts/export_protocol_compact.sh
# 导出指定 protocol 为 Compact 训练就绪数据集；未提供 -o 时自动导出到 <db>/tmp/compact/...
#
# 用法：
#   scripts/export_protocol_compact.sh -p <protocol_name> [-o <out_path>] [-z|--no-zip] [-f] [-d <db_root>]
#
# 示例：
#   scripts/export_protocol_compact.sh -p train_v1
#   scripts/export_protocol_compact.sh -p train_v1 -o /data/exports/train_v1_compact.zip -f

usage() {
  cat <<'USAGE'
Usage:
  scripts/export_protocol_compact.sh -p <protocol_name> [options]

Required:
  -p, --protocol <name>       Protocol name to export

Options:
  -o, --out <path>            Output path; if omitted, will default to <db_root>/tmp/compact/<proto>_compact_<ts>.zip
  -z, --zip                   Zip output (default: enabled)
      --no-zip                Do not zip; write folder instead
  -f, --overwrite             Overwrite if target exists (default: false)
  -d, --db-root <path>        Override database_root
  -h, --help                  Show this help

Env:
  DB_ROOT                     Used when -d not provided and Config.setting is unavailable.

Examples:
  scripts/export_protocol_compact.sh -p train_v1
  scripts/export_protocol_compact.sh -p train_v1 -o ./out_dir --no-zip -f
USAGE
}

# ---- defaults ----
PROTOCOL=""
OUT_PATH=""
ZIP="1"
OVERWRITE="0"
DB_ROOT_ARG=""

# parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    -p|--protocol)
      PROTOCOL="${2:-}"; shift 2;;
    -o|--out)
      OUT_PATH="${2:-}"; shift 2;;
    -z|--zip)
      ZIP="1"; shift 1;;
    --no-zip)
      ZIP="0"; shift 1;;
    -f|--overwrite)
      OVERWRITE="1"; shift 1;;
    -d|--db-root)
      DB_ROOT_ARG="${2:-}"; shift 2;;
    -h|--help)
      usage; exit 0;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      usage; exit 2;;
  esac
done

if [[ -z "$PROTOCOL" ]]; then
  echo "[ERROR] --protocol is required." >&2
  usage; exit 2
fi

# repo root = parent of scripts/
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# propagate args to python via env (avoid quoting issues)
export _BVD_PROTO="$PROTOCOL"
# only export out path when provided
if [[ -n "${OUT_PATH}" ]]; then
  export _BVD_OUT="$OUT_PATH"
else
  unset _BVD_OUT || true
fi
export _BVD_ZIP="$ZIP"
export _BVD_OVERWRITE="$OVERWRITE"

# db-root preference: CLI > env(DB_ROOT) > Config.setting > ./bigvision_db
if [[ -n "$DB_ROOT_ARG" ]]; then
  export DB_ROOT="$DB_ROOT_ARG"
fi

python - <<'PYCODE'
import os, sys, json, traceback
from pathlib import Path

try:
    from Database.db import BigVisionDatabase
except Exception as e:
    print(f"[ERROR] Cannot import Database.db.BigVisionDatabase: {e}", file=sys.stderr)
    traceback.print_exc()
    sys.exit(3)

proto       = os.environ.get("_BVD_PROTO")
out_path    = os.environ.get("_BVD_OUT")  # may be None
zip_output  = (os.environ.get("_BVD_ZIP", "1") == "1")
overwrite   = (os.environ.get("_BVD_OVERWRITE", "0") == "1")

# resolve database_root
database_root = os.environ.get("DB_ROOT")
if not database_root:
    try:
        from Config.setting import GetDatabaseConfig  # type: ignore
        database_root = GetDatabaseConfig()["database_root"]
        print(f"[INFO] database_root from Config.setting: {database_root}")
    except Exception:
        database_root = os.path.abspath("./bigvision_db")
        print(f"[INFO] Using fallback database_root: {database_root}")

db = None
try:
    workers = max(8, (os.cpu_count() or 8))
    db = BigVisionDatabase(
        database_root=database_root,
        duckdb_path=None,
        max_workers=workers,
        threads=0,
    )

    res = db.export_protocol_compact_trainset(
        protocol_name=proto,
        out_path=out_path,      # None -> auto tmp/<proto>_compact_<ts>[.zip]
        zip_output=zip_output,
        overwrite=overwrite,
    )
    print(json.dumps(res, ensure_ascii=False, indent=2))
except Exception as e:
    print(json.dumps({"ok": False, "error": str(e)} , ensure_ascii=False, indent=2))
    sys.exit(1)
finally:
    if db is not None:
        db.close()
PYCODE
