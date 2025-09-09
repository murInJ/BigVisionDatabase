#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/verify_bundle.sh -b <bundle_path> [-l] [-k] [-s <sample_limit>]

Options:
  -b  bundle_path            (必填) Bundle 路径：.zip 或目录（*.bvbundle）
  -l                         宽松模式（strict=False）；默认 strict=True
  -k                         跳过 SHA256 校验（check_sha256=False）；默认校验
  -s  sample_limit           输出报告中的示例上限，默认 20

Env:
  DB_ROOT                    指定数据库根目录；若未设置，将在 Python 端尝试从
                             Config.setting.GetDatabaseConfig() 读取，失败时回落到 ./bigvision_db
USAGE
}

bundle_path=""
strict="1"
check_sha256="1"
sample_limit="20"

while getopts ":b:lks:" opt; do
  case $opt in
    b) bundle_path="$OPTARG" ;;
    l) strict="0" ;;
    k) check_sha256="0" ;;
    s) sample_limit="$OPTARG" ;;
    *) usage; exit 2 ;;
  esac
done

if [[ -z "${bundle_path}" ]]; then
  usage; exit 2
fi

# 将 bundle_path 规范为绝对路径（便于从其他工作目录调用）
if [[ "${bundle_path}" != /* ]]; then
  bundle_path="$(pwd)/${bundle_path}"
fi

python3 - <<'PY' "$bundle_path" "$strict" "$check_sha256" "$sample_limit"
import os, sys, json

# 过滤可能混入的 "--"
argv = [a for a in sys.argv[1:] if a != "--"]
if len(argv) != 4:
    raise SystemExit(f"Expected 4 args, got {len(argv)}: {argv}")

bundle_path, strict, check_sha256, sample_limit = argv
strict = (strict == "1")
check_sha256 = (check_sha256 == "1")
sample_limit = int(sample_limit)

# 解析 database_root
database_root = os.environ.get("DB_ROOT")
if not database_root:
    try:
        from Config.setting import GetDatabaseConfig  # type: ignore
        cfg = GetDatabaseConfig()
        database_root = cfg["database_root"]
        print(f"[INFO] database_root from Config.setting: {database_root}")
    except Exception:
        import os as _os
        database_root = _os.path.abspath("./bigvision_db")
        print(f"[INFO] Fallback database_root: {database_root}")

from Database.db import BigVisionDatabase  # type: ignore

db = BigVisionDatabase(database_root=database_root, duckdb_path=None, max_workers=max(8, (os.cpu_count() or 8)), threads=0)
try:
    try:
        res = db.verify_bundle(
            bundle_path=bundle_path,
            strict=strict,
            check_sha256=check_sha256,
            sample_limit=sample_limit,
        )
        print(json.dumps({"ok": True, "result": res}, ensure_ascii=False, indent=2))
    except Exception as e:
        # 保持非零退出码，便于 CI/脚本联动
        print(json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False, indent=2))
        sys.exit(1)
finally:
    db.close()
PY
