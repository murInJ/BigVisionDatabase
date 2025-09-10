#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/load_bundle.sh -b <bundle_path> [-m <mode>] [-c <copy_mode>] [-n] [-k] [-B <batch_size>] [-q]

Options:
  -b  bundle_path    (必填) Bundle 路径：.zip 或目录（*.bvbundle）
  -m  mode           导入模式：strict | overwrite | skip-existing（默认 strict）
  -c  copy_mode      文件导入方式：copy | hardlink | symlink（ZIP 会自动退化为 copy；默认 copy）
  -n                 不进行 verify（默认开启 verify）
  -k                 verify 时跳过校验和（默认校验）
  -B  batch_size     批大小（默认 2000）
  -q                 安静模式（关闭进度条/额外输出）

Env:
  DB_ROOT            指定数据库根目录；若未设置，将在 Python 端尝试从
                     Config.setting.GetDatabaseConfig() 读取，失败时回落到 ./bigvision_db
USAGE
}

bundle_path=""
mode="strict"
copy_mode="copy"
do_verify="1"
verify_checksums="1"
batch_size="2000"
verbose="1"

while getopts ":b:m:c:nkB:q" opt; do
  case $opt in
    b) bundle_path="$OPTARG" ;;
    m) mode="$OPTARG" ;;
    c) copy_mode="$OPTARG" ;;
    n) do_verify="0" ;;
    k) verify_checksums="0" ;;
    B) batch_size="$OPTARG" ;;
    q) verbose="0" ;;
    *) usage; exit 2 ;;
  esac
done

if [[ -z "${bundle_path}" ]]; then
  usage; exit 2
fi

if [[ "${bundle_path}" != /* ]]; then
  bundle_path="$(pwd)/${bundle_path}"
fi

python3 - <<'PY' "$bundle_path" "$mode" "$copy_mode" "$do_verify" "$verify_checksums" "$batch_size" "$verbose"
import os, sys, json

argv = [a for a in sys.argv[1:] if a != "--"]
if len(argv) != 7:
    raise SystemExit(f"Expected 7 args, got {len(argv)}: {argv}")

bundle_path, mode, copy_mode, do_verify, verify_checksums, batch_size, verbose = argv
do_verify = (do_verify == "1")
verify_checksums = (verify_checksums == "1")
batch_size = int(batch_size)
verbose = (verbose == "1")

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

from Database.db import BigVisionDatabase  # type: ignore

db = BigVisionDatabase(database_root=database_root, duckdb_path=None, max_workers=max(8, (os.cpu_count() or 8)), threads=0)
try:
    res = db.load_bundle(
        bundle_path=bundle_path,
        mode=mode,
        copy_mode=copy_mode,
        verify=do_verify,
        verify_checksums=verify_checksums,
        batch_size=batch_size,
        verbose=verbose,
    )
    print(json.dumps({"ok": True, "result": res}, ensure_ascii=False, indent=2))
except Exception as e:
    print(json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False, indent=2))
    sys.exit(1)
finally:
    db.close()
PY
