#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/delete_protocol.sh -p <protocol_name> [-y] [--purge-orphans] [--dry-run]

Options:
  -p  protocol_name     (必填) 要删除的 protocol 名
  -y                    不交互确认，直接执行
  --purge-orphans       额外清理“孤立的 relations”（即在 protocol 表里不再被引用的关系）
  --dry-run             只做统计与展示，不做实际删除

Env:
  DB_ROOT               指定数据库根目录；若未设置，将在 Python 端尝试从
                        Config.setting.GetDatabaseConfig() 读取，失败时回落到 ./bigvision_db

说明:
  - 本脚本仅删除 protocol 映射；images 不受影响。
  - 如需同时清除已不再被任何 protocol 使用的 relations，请加 --purge-orphans。
USAGE
}

protocol_name=""
assume_yes="0"
purge_orphans="0"
dry_run="0"

# 解析参数
while [[ $# -gt 0 ]]; do
  case "$1" in
    -p)
      protocol_name="${2:-}"; shift 2;;
    -y)
      assume_yes="1"; shift;;
    --purge-orphans)
      purge_orphans="1"; shift;;
    --dry-run)
      dry_run="1"; shift;;
    -*)
      usage; exit 2;;
    *)
      shift;;
  esac
done

if [[ -z "${protocol_name}" ]]; then
  usage; exit 2
fi

if [[ "${dry_run}" != "1" && "${assume_yes}" != "1" ]]; then
  read -r -p "确认删除 protocol '${protocol_name}' ? (y/N) " ans
  case "${ans}" in
    y|Y|yes|YES) ;;
    *) echo "已取消。"; exit 0;;
  esac
fi

python3 - <<'PY' "$protocol_name" "$purge_orphans" "$dry_run"
import os, sys, json

argv = [a for a in sys.argv[1:] if a != "--"]
if len(argv) != 3:
    raise SystemExit(f"Expected 3 args, got {len(argv)}: {argv}")

protocol_name, purge_orphans, dry_run = argv
purge_orphans = (purge_orphans == "1")
dry_run = (dry_run == "1")

# 解析 database_root
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
    # 先统计将要删除的 protocol 行数与样例
    cnt = db.conn.execute(
        "SELECT COUNT(*) FROM protocol WHERE protocol_name = ?",
        [protocol_name],
    ).fetchone()[0]
    sample_rel = [r[0] for r in db.conn.execute(
        "SELECT relation_id FROM protocol WHERE protocol_name = ? LIMIT 10",
        [protocol_name],
    ).fetchall()]

    result = {
        "protocol_name": protocol_name,
        "protocol_rows": int(cnt),
        "sample_relation_ids": sample_rel,
        "purge_orphans": purge_orphans,
        "dry_run": dry_run,
    }

    if dry_run:
        print(json.dumps({"ok": True, "result": result}, ensure_ascii=False, indent=2))
        sys.exit(0)

    db.conn.execute("BEGIN;")
    try:
        # 删除该 protocol 的所有映射
        db.conn.execute("DELETE FROM protocol WHERE protocol_name = ?", [protocol_name])
        deleted_protocol_rows = int(cnt)  # DuckDB 无 changes()，使用删除前的统计值

        orphan_relations_found = 0
        orphan_relations_deleted = 0

        if purge_orphans:
            # 找出 protocol 中不再引用的 relations
            rows = db.conn.execute("""
                SELECT r.relation_id
                FROM relations r
                LEFT JOIN protocol p ON r.relation_id = p.relation_id
                WHERE p.relation_id IS NULL
            """).fetchall()
            orphan_ids = [r[0] for r in rows]
            orphan_relations_found = len(orphan_ids)
            if orphan_ids:
                ph = ",".join(["?"] * len(orphan_ids))
                db.conn.execute(f"DELETE FROM relations WHERE relation_id IN ({ph})", orphan_ids)
                orphan_relations_deleted = len(orphan_ids)  # 同理，用我们自己统计的数量

        db.conn.execute("COMMIT;")

        result.update({
            "deleted_protocol_rows": deleted_protocol_rows,
            "orphan_relations_found": orphan_relations_found,
            "orphan_relations_deleted": orphan_relations_deleted,
        })
        print(json.dumps({"ok": True, "result": result}, ensure_ascii=False, indent=2))
    except Exception:
        db.conn.execute("ROLLBACK;")
        raise

except Exception as e:
    print(json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False, indent=2))
    sys.exit(1)
finally:
    db.close()
PY
