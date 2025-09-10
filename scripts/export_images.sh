#!/usr/bin/env bash
set -euo pipefail

# ========== Defaults (env overridable) ==========
DB_ROOT="${DB_ROOT:-}"                 # 如不设，将尝试 Config.setting.GetDatabaseConfig()；再失败回退 ./bigvision_db
DUCKDB_PATH="${DUCKDB_PATH:-}"         # 默认 <db_root>/db/catalog.duckdb
THREADS="${THREADS:-0}"                # DuckDB PRAGMA threads；0=auto

OUT_DIR="${OUT_DIR:-}"                 # 导出目录；空则用 <db_root>/tmp/exports/<ts>-<rand>/
FORMAT="${FORMAT:-png}"                # png | npy | both
NORMALIZE="${NORMALIZE:-1}"            # 1=归一化到8bit (png)；0=不归一化
ZIP_OUTPUT="${ZIP_OUTPUT:-0}"          # 1=打包zip
ZIP_PATH="${ZIP_PATH:-}"               # zip 输出路径；空则默认 out_dir 同名
OVERWRITE="${OVERWRITE:-1}"            # 1=覆盖同名文件；0=不覆盖
COLOR_ORDER="${COLOR_ORDER:-bgr}"      # bgr(默认) | rgb
SAMPLE_LIMIT="${SAMPLE_LIMIT:-20}"     # 结果里列出示例文件的条数

IDS_STRING="${IDS_STRING:-}"           # 通过字符串传入 ids（逗号/空格分隔）
IDS_FILE="${IDS_FILE:-}"               # 从文件读 ids（一行一个，或逗号/空格分隔）

usage() {
  cat <<'EOF'
Usage: scripts/export_images.shell [options] [image_id ...]
Export images by image_id via BigVisionDatabase.export_images_by_ids().

ID sources (combined):
  1) positional args: image_id ...
  2) -s "<ids>"     or IDS_STRING="..."    (comma/space separated)
  3) -i <file>      or IDS_FILE=<file>     (one per line or comma/space)

Options:
  -d <path>     Database root (overrides $DB_ROOT)
  -u <path>     DuckDB file path (overrides $DUCKDB_PATH)
  -t <int>      DuckDB threads PRAGMA (overrides $THREADS; default 0=auto)

  -o <dir>      Output directory (overrides $OUT_DIR)
  -F <fmt>      Output format: png | npy | both (default png)
  -N            Disable normalization for PNG (sets NORMALIZE=0)
  -z            Zip output (sets ZIP_OUTPUT=1)
  -Z <path>     Zip path (overrides $ZIP_PATH)
  -r            Do NOT overwrite existing files (sets OVERWRITE=0)
  -c <order>    Color order for 3-channel: bgr | rgb (default bgr)
  -l <int>      Sample limit in result (default 20)

  -h            Show this help

Env overrides also supported:
  DB_ROOT, DUCKDB_PATH, THREADS, OUT_DIR, FORMAT, NORMALIZE, ZIP_OUTPUT,
  ZIP_PATH, OVERWRITE, COLOR_ORDER, SAMPLE_LIMIT, IDS_STRING, IDS_FILE

Examples:
  scripts/export_images.shell -d /data/mydb 123 456 789
  scripts/export_images.shell -d /data/mydb -s "a1,b2,c3" -z
  scripts/export_images.shell -d /data/mydb -i ids.txt -F both -c rgb
EOF
}

# -------- Parse flags --------
while getopts ":d:u:t:o:F:Z:c:l:i:s:Nrzh" opt; do
  case $opt in
    d) DB_ROOT="$OPTARG" ;;
    u) DUCKDB_PATH="$OPTARG" ;;
    t) THREADS="$OPTARG" ;;
    o) OUT_DIR="$OPTARG" ;;
    F) FORMAT="$OPTARG" ;;
    Z) ZIP_PATH="$OPTARG" ;;
    c) COLOR_ORDER="$OPTARG" ;;
    l) SAMPLE_LIMIT="$OPTARG" ;;
    i) IDS_FILE="$OPTARG" ;;
    s) IDS_STRING="$OPTARG" ;;
    N) NORMALIZE="0" ;;
    z) ZIP_OUTPUT="1" ;;
    r) OVERWRITE="0" ;;
    h) usage; exit 0 ;;
    \?) echo "Invalid option: -$OPTARG" >&2; usage; exit 2 ;;
    :)  echo "Option -$OPTARG requires an argument." >&2; usage; exit 2 ;;
  esac
done
shift $((OPTIND-1))

# -------- Collect image IDs (positional + IDS_STRING + IDS_FILE) --------
IDS=()

# From positional args
if [[ "$#" -gt 0 ]]; then
  for id in "$@"; do
    [[ -n "$id" ]] && IDS+=("$id")
  done
fi

# From IDS_STRING (comma/space separated)
if [[ -n "${IDS_STRING}" ]]; then
  # Replace commas/semicolons/newlines with spaces, then split
  read -r -a _arr <<<"$(printf "%s" "$IDS_STRING" | tr ',;\n' '   ')"
  for id in "${_arr[@]}"; do
    [[ -n "$id" ]] && IDS+=("$id")
  done
fi

# From IDS_FILE
if [[ -n "${IDS_FILE}" ]]; then
  if [[ "${IDS_FILE}" == "-" ]]; then
    # read from stdin
    mapfile -t lines < <(cat -)
  else
    if [[ ! -f "${IDS_FILE}" ]]; then
      echo "IDs file not found: ${IDS_FILE}" >&2
      exit 2
    fi
    mapfile -t lines < "${IDS_FILE}"
  fi
  for line in "${lines[@]}"; do
    read -r -a _arr <<<"$(printf "%s" "$line" | tr ',;\n' '   ')"
    for id in "${_arr[@]}"; do
      [[ -n "$id" ]] && IDS+=("$id")
    done
  done
fi

if [[ ${#IDS[@]} -eq 0 ]]; then
  echo "No image_id provided. Use positional args, -s, or -i." >&2
  usage
  exit 2
fi

# Make unique while keeping order
declare -A seen
UNIQ_IDS=()
for id in "${IDS[@]}"; do
  if [[ -z "${seen[$id]+x}" ]]; then
    UNIQ_IDS+=("$id"); seen[$id]=1
  fi
done

# Build CSV for Python env
IDS_CSV="$(printf "%s," "${UNIQ_IDS[@]}")"
IDS_CSV="${IDS_CSV%,}"

# -------- Export via inline Python --------
# Pass all controls via env to avoid quoting pitfalls
export DB_ROOT DUCKDB_PATH THREADS OUT_DIR FORMAT NORMALIZE ZIP_OUTPUT ZIP_PATH OVERWRITE COLOR_ORDER SAMPLE_LIMIT IDS_CSV

python - <<'PY'
import os, json, sys
from pathlib import Path

# Resolve db_root
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

out_dir = os.environ.get("OUT_DIR") or None
fmt = os.environ.get("FORMAT","png").lower()
normalize = os.environ.get("NORMALIZE","1") == "1"
zip_output = os.environ.get("ZIP_OUTPUT","0") == "1"
zip_path = os.environ.get("ZIP_PATH") or None
overwrite = os.environ.get("OVERWRITE","1") == "1"
color_order = os.environ.get("COLOR_ORDER","bgr").lower()
sample_limit = int(os.environ.get("SAMPLE_LIMIT","20") or "20")

ids_csv = os.environ.get("IDS_CSV","").strip()
if not ids_csv:
    print("[ERROR] IDS_CSV empty.", file=sys.stderr)
    sys.exit(2)
image_ids = [tok.strip() for tok in ids_csv.split(",") if tok.strip()]
if not image_ids:
    print("[ERROR] No valid image_id parsed.", file=sys.stderr)
    sys.exit(2)

try:
    from Database.db import BigVisionDatabase
except Exception:
    from db import BigVisionDatabase  # type: ignore

db = BigVisionDatabase(database_root=db_root, duckdb_path=duckdb_path, threads=threads)
try:
    res = db.export_images_by_ids(
        image_ids=image_ids,
        out_dir=out_dir,
        output=fmt,
        normalize=normalize,
        zip_output=zip_output,
        zip_path=zip_path,
        overwrite=overwrite,
        sample_limit=sample_limit,
        color_order=color_order,
    )
    print(json.dumps(res, ensure_ascii=False, indent=2))
finally:
    db.close()
PY
