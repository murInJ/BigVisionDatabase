# Database/db.py
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    # 项目内导入（推荐）
    from Database.utils import require_duckdb, init_duckdb_schema
except Exception:  # pragma: no cover
    # 兼容简单结构
    from utils import require_duckdb, init_duckdb_schema  # type: ignore

try:
    from OriginDataset.writer import DatasetWriter
except Exception:  # pragma: no cover
    from writer import DatasetWriter  # type: ignore


class BigVisionDatabase:
    """
    统一的数据库外观类：
      - 维护唯一的 DuckDB 连接（单进程读写约束）
      - 初始化 schema
      - 提供写入入口（单样本 / 单 adaptor / registry）
      - 提供垃圾回收：清理磁盘孤儿 .npy 文件，并报告 DB 指向缺失文件的条目

    目录约定：
      <database_root>/images/<dataset_name>/<UUID>.npy
      <database_root>/db/catalog.duckdb
    """

    def __init__(
        self,
        database_root: str,
        duckdb_path: Optional[str] = None,
        max_workers: int = 8,
        threads: Optional[int] = None,
    ) -> None:
        """
        Args:
            database_root: DB 根目录
            duckdb_path:   自定义 DuckDB 文件路径；默认 <database_root>/db/catalog.duckdb
            max_workers:   写图并发
            threads:       DuckDB PRAGMA threads（None/0 表示让 DuckDB 自行决定）
        """
        self.database_root = database_root
        self.duckdb_path = duckdb_path or f"{database_root}/db/catalog.duckdb"

        # 管理连接（单处持有）
        self.conn = require_duckdb(
            duckdb_path=self.duckdb_path,
            threads=(threads or 0),
        )
        # 初始化 schema（集中在 DB 层）
        init_duckdb_schema(self.conn)

        # 注入连接到 writer（writer 不再管理连接与 schema）
        self.writer = DatasetWriter(
            conn=self.conn,
            database_root=database_root,
            max_workers=max_workers,
        )

    # ---------------- 写入封装 ----------------

    def add_sample(
        self,
        *,
        images: Dict[str, Any],
        relations: Dict[str, Dict[str, Any]],
        protocols: Dict[str, List[str]],
        dry_run: bool = False,
    ) -> Dict[str, int]:
        """
        写入单条样本（一个 adaptor item）。
        返回计数字典：{'imgs': X, 'rels': Y, 'proto': Z}
        """
        return self.writer.write_dataset(
            images=images,
            relations=relations,
            protocols=protocols,
            dry_run=dry_run,
        )

    def ingest_from_adaptor(
        self,
        adaptor: Any,
        *,
        dry_run: bool = False,
    ) -> Dict[str, int]:
        """
        从一个 adaptor 实例顺序写入（不显示进度；可由上层包 tqdm）。
        返回聚合计数 {'imgs': X, 'rels': Y, 'proto': Z, 'err': E}
        """
        agg = {"imgs": 0, "rels": 0, "proto": 0, "err": 0}
        for _idx, data in enumerate(adaptor):
            try:
                s = self.add_sample(
                    images=data["images"],
                    relations=data["relation"],
                    protocols=data["protocol"],
                    dry_run=dry_run,
                )
                agg["imgs"] += s.get("imgs", 0)
                agg["rels"] += s.get("rels", 0)
                agg["proto"] += s.get("proto", 0)
            except Exception:
                agg["err"] += 1
        return agg

    def ingest_from_registry(
        self,
        *,
        dry_run: bool = False,
        show_progress: bool = True,
    ) -> None:
        """
        扫描注册器批量写入（每个 adaptor 一个进度条，逻辑在 writer 内）。
        """
        self.writer.write_from_registry(dry_run=dry_run, show_progress=show_progress)

    # ---------------- 垃圾回收 ----------------

    def garbage_collect(
        self,
        *,
        remove_orphan_files: bool = False,
        check_db_missing_files: bool = True,
        report_limit: int = 20,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        清理/报告不一致数据（幂等、只针对“单机本地路径”）：

        1) **孤儿文件**（Orphan Files）：
           - 条件：磁盘上的 .npy 文件，其 image_id（文件名 stem）不在 DB images.image_id 中，
             或者虽在 DB 中但其相对路径与 DB 的 uri 不一致（视为“路径不一致的副本”）。
           - 行为：默认仅报告；若 remove_orphan_files=True，则实际删除这些孤儿文件。

        2) **DB 缺失文件**（Missing File Rows）：
           - 条件：DB images 表中的某条记录，其 uri 指向的文件在磁盘上不存在。
           - 行为：默认仅报告（不删除 DB 行，以免破坏 relations.payload.image_ids 的一致性）。

        Args:
            remove_orphan_files: True 则删除孤儿 .npy 文件；默认 False 只做报告。
            check_db_missing_files: 是否扫描 DB 行并报告缺失文件；默认 True。
            report_limit: 报告中示例文件/行的最大展示数量。
            verbose: 打印更详细的过程日志。

        Returns:
            {
              "files_scanned": int,
              "db_images": int,
              "orphan_files": int,
              "orphan_removed": int,
              "orphan_samples": [<相对路径> ...],
              "missing_file_rows": int,
              "missing_samples": [(image_id, uri), ...],
            }
        """
        root = Path(self.database_root)
        images_root = root / "images"

        # --- 构建 DB 快照（image_id -> uri） ---
        db_rows = self.conn.execute("SELECT image_id, uri FROM images").fetchall()
        db_images_count = len(db_rows)
        id2uri: Dict[str, str] = {r[0]: r[1] for r in db_rows}
        db_ids: Set[str] = set(id2uri.keys())

        # --- 扫描磁盘 ---
        files_scanned = 0
        orphan_paths: List[Path] = []
        for p in images_root.rglob("*.npy"):
            files_scanned += 1
            stem = p.stem  # UUID 作为 image_id
            rel_uri = p.relative_to(root).as_posix()  # "images/<dataset>/<uuid>.npy"

            if stem not in db_ids:
                # 完全没有 DB 记录 -> 孤儿
                orphan_paths.append(p)
            else:
                # 有 DB 记录，但路径不一致（多半来自异常中断后的重试）
                if id2uri.get(stem) != rel_uri:
                    orphan_paths.append(p)

        # --- 可选：删除孤儿文件 ---
        orphan_removed = 0
        orphan_samples: List[str] = []
        if orphan_paths:
            if verbose:
                print(f"[GC] Found {len(orphan_paths)} orphan files.")
            # 收集样例
            orphan_samples = [op.relative_to(root).as_posix() for op in orphan_paths[:report_limit]]
            if remove_orphan_files:
                for p in orphan_paths:
                    try:
                        p.unlink(missing_ok=True)  # py>=3.8
                        orphan_removed += 1
                    except TypeError:
                        try:
                            if p.exists():
                                p.unlink()
                                orphan_removed += 1
                        except Exception:
                            # 忽略个别删除失败
                            pass

        # --- 可选：检查 DB 行对应的文件是否缺失 ---
        missing_file_rows = 0
        missing_samples: List[Tuple[str, str]] = []
        if check_db_missing_files:
            for image_id, uri in id2uri.items():
                p = root / uri
                if not p.exists():
                    missing_file_rows += 1
                    if len(missing_samples) < report_limit:
                        missing_samples.append((image_id, uri))

            if verbose and missing_file_rows:
                print(f"[GC] Found {missing_file_rows} DB rows pointing to missing files.")

        # --- 汇总 ---
        summary: Dict[str, Any] = {
            "files_scanned": files_scanned,
            "db_images": db_images_count,
            "orphan_files": len(orphan_paths),
            "orphan_removed": orphan_removed,
            "orphan_samples": orphan_samples,
            "missing_file_rows": missing_file_rows,
            "missing_samples": missing_samples,
        }
        if verbose:
            print(f"[GC] Summary: {summary}")
        return summary

    # ---------------- 生命周期 ----------------

    def close(self) -> None:
        """关闭连接（可重复调用）"""
        try:
            self.conn.close()
        except Exception:
            pass

    # 上下文管理（可选）
    def __enter__(self) -> "BigVisionDatabase":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


# ---------------- 直接可运行：无需传参 ----------------
if __name__ == "__main__":
    # 1) database_root 优先用 Config.setting.GetDatabaseConfig()
    try:
        from Config.setting import GetDatabaseConfig  # type: ignore
        cfg = GetDatabaseConfig()
        database_root = cfg["database_root"]
        print(f"[INFO] database_root from Config.setting: {database_root}")
    except Exception:
        # 2) 失败则回落到环境变量 DB_ROOT，否则使用 ./bigvision_db
        database_root = os.environ.get("DB_ROOT", os.path.abspath("./bigvision_db"))
        print(f"[INFO] Using fallback database_root: {database_root}")

    # 并发默认：max(8, CPU)
    workers = max(8, (os.cpu_count() or 8))

    db = None
    try:
        db = BigVisionDatabase(
            database_root=database_root,
            duckdb_path=None,        # 默认 <database_root>/db/catalog.duckdb
            max_workers=workers,
            threads=0,               # 让 DuckDB 自己决定
        )
        # 默认执行：从注册器写入
        db.ingest_from_registry(dry_run=False, show_progress=True)
        print("[OK] registry ingestion finished.")

        # 可选：立刻做一次 GC（只报告，不删除），你可以根据需要删掉下面两行
        gc_summary = db.garbage_collect(remove_orphan_files=False, check_db_missing_files=True, verbose=True)
        print(f"[OK] GC summary: {gc_summary}")

    except Exception as e:
        print(f"[ERROR] ingestion failed: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
    finally:
        if db is not None:
            db.close()
