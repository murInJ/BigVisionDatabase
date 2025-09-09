#!/usr/bin/env bash
set -euo pipefail

# Merge multiple protocols into a new protocol (union/intersect)
# DB_ROOT 可用环境变量覆盖数据库根目录；否则脚本内的 Python 会尝试从 Config.setting 中读取，
# 再不行则回落到 ./bigvision_db

usage() {
  cat <<'USAGE'
Usage:
  protocol_merge.sh -o NEW_PROTOCOL [-m union|intersect] [-R] [-n] SRC_PROTOCOL [SRC_PROTOCOL2 ...]
Options:
  -o NEW_PROTOCOL   目标新 protocol 名（必填）
  -m MODE           合并模式：union(默认) 或 intersect
  -R                不覆盖已有同名 protocol（默认会 replace）
  -n                dry-run，仅打印将要执行的结果，不真正写库
Env:
  DB_ROOT           可选，指定 database_root；不设置则由 Python 端自行解析
Examples:
  ./protocol_merge.sh -o merged_train union_train_a union_train_b
  ./protocol_merge.sh -o inter_p -m intersect protoA protoB protoC
  DB_ROOT=/data/bigvision ./protocol_merge.sh -o merged -n p1 p2
USAGE
}

NEW_PROTOCOL=""
MODE="union"
REPLACE=1
DRYRUN=0

while getopts ":o:m:Rn" opt; do
  case "$opt" in
    o) NEW_PROTOCOL="$OPTARG" ;;
    m) MODE="$OPTARG" ;;
    R) REPLACE=0 ;;   # do NOT replace
    n) DRYRUN=1 ;;
    *) usage; exit 2 ;;
  esac
done
shift $((OPTIND-1))

if [[ -z "${NEW_PROTOCOL}" ]]; then
  echo "[ERROR] -o NEW_PROTOCOL is required" >&2
  usage
  exit 2
fi

if [[ $# -lt 2 ]]; then
  echo "[ERROR] need at least two source protocols to merge" >&2
  usage
  exit 2
fi

SOURCES=("$@")  # remaining args

python3 - <<'PY'
import os, json, sys
from pathlib import Path

# 解析入参（通过环境和 heredoc 注入）
NEW_PROTOCOL = os.environ.get("NEW_PROTOCOL")
MODE = os.environ.get("MODE","union")
REPLACE = os.environ.get("REPLACE","1") == "1"
DRYRUN = os.environ.get("DRYRUN","0") == "1"
SOURCES = json.loads(os.environ.get("SOURCES_JSON","[]"))

# database_root 解析：优先 DB_ROOT，其次 Config.setting，最后 ./bigvision_db
db_root = os.environ.get("DB_ROOT")
if not db_root:
    try:
        from Config.setting import GetDatabaseConfig  # type: ignore
        cfg = GetDatabaseConfig()
        db_root = cfg["database_root"]
        print(f"[INFO] database_root from Config.setting: {db_root}")
    except Exception:
        db_root = os.path.abspath("./bigvision_db")
        print(f"[INFO] Using fallback database_root: {db_root}")

try:
    from Database.db import BigVisionDatabase  # type: ignore
except Exception:
    from db import BigVisionDatabase  # type: ignore

db = None
try:
    db = BigVisionDatabase(database_root=db_root, duckdb_path=None, max_workers=max(8, os.cpu_count() or 8), threads=0)
    res = db.merge_protocols(
        new_protocol=NEW_PROTOCOL,
        sources=SOURCES,
        mode=MODE,
        replace=REPLACE,
        dry_run=DRYRUN,
    )
    print(json.dumps(res, ensure_ascii=False, indent=2))
finally:
    if db is not None:
        db.close()
PY
