#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/export_protocol_bundle.sh [-p <name[,name2,...]>] [-o <out_path_or_dir>] [-m copy|hardlink|symlink|manifest-only] [-t] [-c rgb|bgr] [-Z] [-f]

Options:
  -p  protocol_name(们)             省略时默认导出**全部**；也可逗号分隔多个
  -o  out_path_or_dir               多协议时当作目录；单协议时若以 .zip 结尾则输出 zip，否则输出目录
  -m  copy_mode                     {copy|hardlink|symlink|manifest-only}，默认 copy
  -t                                包含缩略图（thumbnails）
  -c  color_order                   {rgb|bgr} 生成缩略图时的通道顺序，默认 bgr
  -Z                                不打包 zip（导出为目录）
  -f                                覆盖已存在的目标（overwrite）

Env:
  DB_ROOT                           若未提供，将在脚本内 Python 中尝试从 Config.setting 读取；失败时回落到 ./bigvision_db
USAGE
}

protocol_name=""
out_path=""
copy_mode="copy"
include_thumbs="0"
color_order="bgr"
zip_output="1"
overwrite="0"

while getopts ":p:o:m:tc:Zf" opt; do
  case $opt in
    p) protocol_name="$OPTARG" ;;
    o) out_path="$OPTARG" ;;
    m) copy_mode="$OPTARG" ;;
    t) include_thumbs="1" ;;
    c) color_order="$OPTARG" ;;
    Z) zip_output="0" ;;
    f) overwrite="1" ;;
    *) usage; exit 2 ;;
  esac
done

python3 - <<'PY' "$protocol_name" "${out_path}" "${copy_mode}" "${include_thumbs}" "${color_order}" "${zip_output}" "${overwrite}"
import os, sys, json, pathlib

argv = [a for a in sys.argv[1:] if a != '--']
if len(argv) != 7:
    raise SystemExit(f"Expected 7 args, got {len(argv)}: {argv}")

protocol_name, out_path, copy_mode, include_thumbs, color_order, zip_output, overwrite = argv

# 解析 DB Root
database_root = os.environ.get("DB_ROOT")
if not database_root:
    try:
        from Config.setting import GetDatabaseConfig  # type: ignore
        cfg = GetDatabaseConfig()
        database_root = cfg["database_root"]
        print(f"[INFO] database_root from Config.setting: {database_root}")
    except Exception:
        database_root = os.path.abspath("./bigvision_db")
        print(f"[INFO] Fallback database_root: {database_root}")

include_thumbs = (include_thumbs == "1")
zip_output = (zip_output == "1")
overwrite = (overwrite == "1")

from Database.db import BigVisionDatabase  # type: ignore

db = BigVisionDatabase(database_root=database_root, duckdb_path=None, max_workers=max(8, (os.cpu_count() or 8)), threads=0)
try:
    # 允许 protocol_name 为空串：函数内部会默认全导出；允许逗号分隔：函数内部会拆分
    name_arg = (None if not protocol_name.strip() else protocol_name.strip())
    res = db.export_protocol_bundle(
        protocol_name=name_arg,
        out_path=(None if not out_path.strip() else out_path.strip()),
        copy_mode=copy_mode,
        include_thumbnails=include_thumbs,
        color_order=color_order,
        zip_output=zip_output,
        overwrite=overwrite,
    )
    print(json.dumps(res, ensure_ascii=False, indent=2))
finally:
    db.close()
PY
