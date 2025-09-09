#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------
# scripts/writeOriginDataset.sh
# 作用：不运行 Database/db.py 的 __main__。
#      通过内联 Python 直接 import 并调用 writer 流程（ingest_from_registry）。
#
# 使用：
#   直接运行： ./scripts/writeOriginDataset.sh
#   环境覆盖（可选）：
#     DB_ROOT=/data/mydb        # 覆盖 Config.setting 的 database_root
#     DRY_RUN=1                 # 仅模拟，不真正写入（默认 0）
#     NO_PROGRESS=1             # 关闭进度条（默认 0）
# ---------------------------------------------

: "${DRY_RUN:=0}"
: "${NO_PROGRESS:=0}"

export PYTHONUNBUFFERED=1

python - <<'PY'
import os, sys, json

# 1) 解析 database_root：优先环境变量 DB_ROOT，其次 Config.setting；不再回落到 ./bigvision_db
db_root_env = os.environ.get("DB_ROOT", "").strip()
database_root = None
duckdb_path = None
threads = 0
max_workers = None

if db_root_env:
    database_root = db_root_env
else:
    try:
        from Config.setting import GetDatabaseConfig  # type: ignore
        cfg = GetDatabaseConfig() or {}
        database_root = cfg.get("database_root")
        duckdb_path   = cfg.get("duckdb_path")  # 可选
        threads       = int(cfg.get("threads", 0) or 0)
        max_workers   = cfg.get("max_workers")
        if max_workers is not None:
            max_workers = int(max_workers)
    except Exception as e:
        print(json.dumps({"ok": False, "error": f"Failed to read Config.setting.GetDatabaseConfig(): {e}"},
                         ensure_ascii=False))
        sys.exit(2)

if not database_root:
    print(json.dumps({"ok": False, "error": "database_root is not set. Provide DB_ROOT env or ensure Config.setting.GetDatabaseConfig()['database_root'] exists."},
                     ensure_ascii=False))
    sys.exit(2)

# 并发设置：优先 config 的 max_workers；否则用 max(8, CPU)
if max_workers is None:
    max_workers = max(8, (os.cpu_count() or 8))

dry_run = (os.environ.get("DRY_RUN", "0") == "1")
show_progress = not (os.environ.get("NO_PROGRESS", "0") == "1")

print(f"[INFO] database_root: {database_root}")
if duckdb_path:
    print(f"[INFO] duckdb_path from config: {duckdb_path}")
print(f"[INFO] max_workers={max_workers} threads={threads} dry_run={dry_run} show_progress={show_progress}")

try:
    from Database.db import BigVisionDatabase
except Exception as e:
    print(json.dumps({"ok": False, "error": f"Import BigVisionDatabase failed: {e}"}, ensure_ascii=False))
    sys.exit(1)

db = None
try:
    db = BigVisionDatabase(
        database_root=database_root,
        duckdb_path=duckdb_path,  # None 则用默认 <db_root>/db/catalog.duckdb
        max_workers=max_workers,
        threads=threads,
    )
    # 通过 facade 调用 writer 的写入流程（内部调 DatasetWriter.write_from_registry）
    db.ingest_from_registry(dry_run=dry_run, show_progress=show_progress)
    print(json.dumps({"ok": True, "database_root": database_root, "dry_run": dry_run}, ensure_ascii=False))
except Exception as e:
    print(json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False))
    sys.exit(1)
finally:
    if db is not None:
        try:
            db.close()
        except Exception:
            pass
PY
