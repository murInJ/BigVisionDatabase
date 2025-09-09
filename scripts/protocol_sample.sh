#!/usr/bin/env bash
set -euo pipefail

# Sample k relations from a protocol; optionally write to a new protocol
# DB_ROOT 可用环境变量覆盖数据库根目录；否则脚本内的 Python 会尝试从 Config.setting 中读取，
# 再不行则回落到 ./bigvision_db

usage() {
  cat <<'USAGE'
Usage:
  protocol_sample.sh -p SRC_PROTOCOL -k K [-d DST_PROTOCOL] [-s SEED] [-R] [-n]
Options:
  -p SRC_PROTOCOL   源 protocol 名（必填）
  -k K              采样条数（必填、正整数）
  -d DST_PROTOCOL   目标新 protocol 名（可选；不提供则仅打印采样到的 relation_id 列表）
  -s SEED           采样随机种子（可选；不提供则使用 DuckDB ORDER BY random()）
  -R                不覆盖目标同名 protocol（默认会 replace）
  -n                dry-run，仅打印将要执行的结果，不真正写库
Env:
  DB_ROOT           可选，指定 database_root；不设置则由 Python 端自行解析
Examples:
  ./protocol_sample.sh -p train -k 500
  ./protocol_sample.sh -p train -k 1000 -d train_sample_1k -s 42
  DB_ROOT=/data/bigvision ./protocol_sample.sh -p eval -k 200 -d eval_sub -n
USAGE
}

SRC_PROTOCOL=""
K=""
DST_PROTOCOL=""
SEED=""
REPLACE=1
DRYRUN=0

while getopts ":p:k:d:s:Rn" opt; do
  case "$opt" in
    p) SRC_PROTOCOL="$OPTARG" ;;
    k) K="$OPTARG" ;;
    d) DST_PROTOCOL="$OPTARG" ;;
    s) SEED="$OPTARG" ;;
    R) REPLACE=0 ;;
    n) DRYRUN=1 ;;
    *) usage; exit 2 ;;
  esac
done
shift $((OPTIND-1))

if [[ -z "${SRC_PROTOCOL}" ]]; then
  echo "[ERROR] -p SRC_PROTOCOL is required" >&2
  usage
  exit 2
fi
if [[ -z "${K}" || ! "${K}" =~ ^[0-9]+$ || "${K}" -le 0 ]]; then
  echo "[ERROR] -k K must be a positive integer" >&2
  usage
  exit 2
fi

export SRC_PROTOCOL K DST_PROTOCOL SEED
export REPLACE DRYRUN

python3 - <<'PY'
import os, json, sys
from pathlib import Path

SRC_PROTOCOL = os.environ.get("SRC_PROTOCOL","")
K = int(os.environ.get("K","0"))
DST_PROTOCOL = os.environ.get("DST_PROTOCOL") or None
SEED_ENV = os.environ.get("SEED")
SEED = int(SEED_ENV) if (SEED_ENV not in (None, "")) else None
REPLACE = os.environ.get("REPLACE","1") == "1"
DRYRUN = os.environ.get("DRYRUN","0") == "1"

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
    res = db.sample_protocol(
        src_protocol=SRC_PROTOCOL,
        k=K,
        seed=SEED,
        dst_protocol=DST_PROTOCOL,
        replace=REPLACE,
        dry_run=DRYRUN,
    )
    print(json.dumps(res, ensure_ascii=False, indent=2))
finally:
    if db is not None:
        db.close()
PY
