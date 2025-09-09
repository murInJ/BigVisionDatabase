# Database/db.py
from __future__ import annotations

import json
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

    # -------------- DB summary ---------------

    def get_db_summary(self) -> Dict[str, Any]:
        """
        返回数据库概览汇总：

        {
          "totals": {
            "images": <int>,             # images 表总行数
            "relations": <int>,          # relations 表总行数
            "protocol_rows": <int>,      # protocol 表总行数
            "protocols": <int>           # 不同 protocol_name 的数量
          },
          "protocols": [
            {
              "protocol_name": <str>,
              "n_relations": <int>,      # 该 protocol_name 下关联的去重 relation_id 数
              "datasets": [<str>, ...],  # 该 protocol 覆盖到的 images.dataset_name 去重集合（按字母序）
            },
            ...
          ]
        }
        """
        # 1) 顶部 totals
        totals = {}
        totals["images"] = self.conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
        totals["relations"] = self.conn.execute("SELECT COUNT(*) FROM relations").fetchone()[0]
        totals["protocol_rows"] = self.conn.execute("SELECT COUNT(*) FROM protocol").fetchone()[0]
        totals["protocols"] = self.conn.execute("SELECT COUNT(DISTINCT protocol_name) FROM protocol").fetchone()[0]

        # 2) 若没有任何 protocol 行，直接返回
        proto_rows = self.conn.execute("SELECT protocol_name, relation_id FROM protocol").fetchall()
        if not proto_rows:
            return {"totals": totals, "protocols": []}

        # 3) protocol_name -> set(relation_id)
        from collections import defaultdict
        proto_to_relids: Dict[str, set] = defaultdict(set)
        for pname, rid in proto_rows:
            proto_to_relids[str(pname)].add(str(rid))

        # 4) 拉取 relations.payload 中的 image_ids（一次或分块）
        def _chunks(seq, n=1000):
            it = list(seq)
            for i in range(0, len(it), n):
                yield it[i:i+n]

        all_relids = set().union(*proto_to_relids.values())
        relid_to_imgids: Dict[str, List[str]] = {}
        if all_relids:
            for chunk in _chunks(list(all_relids), 1000):
                placeholders = ",".join(["?"] * len(chunk))
                rows = self.conn.execute(
                    f"SELECT relation_id, payload FROM relations WHERE relation_id IN ({placeholders})",
                    chunk,
                ).fetchall()
                import json as _json
                for rid, payload in rows:
                    try:
                        js = _json.loads(payload)
                        img_ids = js.get("image_ids", []) or []
                        relid_to_imgids[str(rid)] = [str(x) for x in img_ids]
                    except Exception:
                        relid_to_imgids[str(rid)] = []

        # 5) 汇总所有 image_id 并查询其 dataset_name
        all_img_ids = set()
        for rids in proto_to_relids.values():
            for rid in rids:
                all_img_ids.update(relid_to_imgids.get(rid, []))

        imgid_to_ds: Dict[str, str] = {}
        if all_img_ids:
            for chunk in _chunks(list(all_img_ids), 1000):
                placeholders = ",".join(["?"] * len(chunk))
                rows = self.conn.execute(
                    f"SELECT image_id, dataset_name FROM images WHERE image_id IN ({placeholders})",
                    chunk,
                ).fetchall()
                for iid, ds in rows:
                    imgid_to_ds[str(iid)] = str(ds)

        # 6) 逐 protocol 汇总：n_relations & datasets
        out_list: List[Dict[str, Any]] = []
        for pname in sorted(proto_to_relids.keys()):
            relids = proto_to_relids[pname]
            used_ds = set()
            for rid in relids:
                for iid in relid_to_imgids.get(rid, []):
                    ds = imgid_to_ds.get(iid)
                    if ds:
                        used_ds.add(ds)
            out_list.append({
                "protocol_name": pname,
                "n_relations": len(relids),
                "datasets": sorted(used_ds),
            })

        return {"totals": totals, "protocols": out_list}

    # ------------- images methods -------------

    def export_images_by_ids(
            self,
            image_ids: List[str],
            out_dir: Optional[str] = None,
            *,
            output: str = "png",  # 'png' | 'npy' | 'both'
            normalize: bool = True,  # PNG 导出是否 0-255 归一化
            zip_output: bool = False,  # 是否 zip 打包
            zip_path: Optional[str] = None,  # 自定义 zip 输出路径
            overwrite: bool = True,
            sample_limit: int = 20,
            color_order: str = "bgr",  # 默认强制按 BGR 处理三通道（避免“蓝脸”）
    ) -> Dict[str, Any]:
        """
        根据一组 image_id 导出图像（PNG/NPY），并可选 zip 打包。
        默认将三通道图像视为 **BGR 来源**（OpenCV 常见读法），从而避免“把 BGR 当 RGB 写”的蓝脸问题。
        如你的数据确认为 RGB，可显式传 color_order="rgb"。
        """
        import time
        import uuid as _uuid
        import shutil
        from pathlib import Path
        import numpy as _np

        if output not in ("png", "npy", "both"):
            raise ValueError("output must be one of: 'png' | 'npy' | 'both'")
        if color_order not in ("rgb", "bgr"):
            raise ValueError("color_order must be 'rgb' | 'bgr'")

        root = Path(self.database_root)
        if out_dir is None:
            tmp_dir = root / "tmp" / "exports"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            out_dir_path = tmp_dir / (time.strftime("%Y%m%d-%H%M%S") + "-" + _uuid.uuid4().hex[:6])
        else:
            out_dir_path = Path(out_dir)
        out_dir_path.mkdir(parents=True, exist_ok=True)

        # 查询 images 表
        def _fetch_rows(ids: List[str]) -> Dict[str, Dict[str, Any]]:
            if not ids:
                return {}
            rows: Dict[str, Dict[str, Any]] = {}
            chunk_size = 1000
            for i in range(0, len(ids), chunk_size):
                chunk = ids[i:i + chunk_size]
                placeholders = ",".join(["?"] * len(chunk))
                q = f"""
                SELECT image_id, uri, alias, dataset_name, modality
                FROM images
                WHERE image_id IN ({placeholders})
                """
                for image_id, uri, alias, dataset_name, modality in self.conn.execute(q, chunk).fetchall():
                    rows[str(image_id)] = {
                        "uri": str(uri),
                        "alias": None if alias is None else str(alias),
                        "dataset_name": str(dataset_name),
                        "modality": None if modality is None else str(modality),
                    }
            return rows

        id2row = _fetch_rows(image_ids)
        missing = [iid for iid in image_ids if iid not in id2row]

        # 数组 -> 适合写 PNG 的 uint8（保持 HxW 或 HxWxC）
        def _prep_uint8(arr: _np.ndarray) -> _np.ndarray:
            a = _np.asarray(arr)
            # 尝试把 (C,H,W) 变为 (H,W,C)
            if a.ndim == 3 and a.shape[0] in (1, 3) and a.shape[2] not in (1, 3):
                a = _np.moveaxis(a, 0, 2)
            while a.ndim > 3:
                a = a.squeeze(axis=0)
            if a.dtype == _np.uint8 and not normalize:
                pass
            else:
                a = a.astype(_np.float32)
                amin, amax = _np.nanmin(a), _np.nanmax(a)
                if amax > amin:
                    a = (a - amin) / (amax - amin)
                else:
                    a = _np.zeros_like(a, dtype=_np.float32)
                a = (a * 255.0).round().clip(0, 255).astype(_np.uint8)
            # 统一成 HxW 或 HxWx3
            if a.ndim == 2:
                return a
            if a.ndim == 3:
                if a.shape[2] == 1:
                    return a[:, :, 0]
                if a.shape[2] >= 3:
                    return a[:, :, :3]
            # 其他奇怪形状 → 灰度回退
            return a.reshape(a.shape[0], a.shape[1]).astype(_np.uint8)

        # 写 PNG：优先 imageio，回退 cv2
        try:
            import imageio.v3 as iio  # type: ignore
            def _save_png(dst: Path, img: _np.ndarray) -> bool:
                iio.imwrite(str(dst), img)
                return True

            writer_backend = "imageio"  # 期望 RGB
        except Exception:
            try:
                import cv2  # type: ignore
                def _save_png(dst: Path, img: _np.ndarray) -> bool:
                    return bool(cv2.imwrite(str(dst), img))

                writer_backend = "cv2"  # 期望 BGR
            except Exception:
                raise RuntimeError("Neither imageio nor OpenCV is available to write PNG files.")

        exported_files: List[Path] = []
        seen_names: set[str] = set()

        def _safe_name(base: str, ext: str) -> str:
            name = "".join(ch for ch in base if ch.isalnum() or ch in ("-", "_", ".", "+"))
            if not name:
                name = "image"
            candidate = f"{name}.{ext}"
            if overwrite or candidate not in seen_names:
                seen_names.add(candidate)
                return candidate
            k = 1
            while True:
                candidate = f"{name}_{k}.{ext}"
                if candidate not in seen_names:
                    seen_names.add(candidate)
                    return candidate
                k += 1

        # 主循环（保持输入顺序）
        for idx, iid in enumerate(image_ids):
            row = id2row.get(iid)
            if row is None:
                continue

            src_path = (root / row["uri"]).resolve()
            alias = row["alias"] or row["modality"] or iid
            base = f"{idx:04d}_{alias}"

            # PNG
            if output in ("png", "both"):
                try:
                    arr = _np.load(src_path, allow_pickle=False)
                    img = _prep_uint8(arr)

                    # 若是三通道：默认按 BGR 来源修正到目标写入后端的期望
                    if isinstance(img, _np.ndarray) and img.ndim == 3 and img.shape[2] == 3:
                        if writer_backend == "imageio" and color_order == "bgr":
                            # imageio 期望 RGB，我们的来源是 BGR -> 交换
                            img = img[:, :, ::-1]
                        elif writer_backend == "cv2" and color_order == "rgb":
                            # cv2 期望 BGR，我们的来源是 RGB -> 交换
                            img = img[:, :, ::-1]

                    dst_name = _safe_name(base, "png")
                    dst_path = out_dir_path / dst_name
                    if dst_path.exists() and not overwrite:
                        pass
                    else:
                        _save_png(dst_path, img)
                        exported_files.append(dst_path)
                except Exception as e:
                    print(f"[WARN] PNG export failed for image_id={iid}: {e}", file=sys.stderr)

            # NPY
            if output in ("npy", "both"):
                try:
                    dst_name = _safe_name(base, "npy")
                    dst_path = out_dir_path / dst_name
                    if dst_path.exists() and not overwrite:
                        pass
                    else:
                        shutil.copy2(src_path, dst_path)
                        exported_files.append(dst_path)
                except Exception as e:
                    print(f"[WARN] NPY copy failed for image_id={iid}: {e}", file=sys.stderr)

        # ZIP（可选）
        out_zip: Optional[Path] = None
        if zip_output:
            import shutil
            if zip_path:
                zp = Path(zip_path)
                if zp.suffix.lower() != ".zip":
                    zp.parent.mkdir(parents=True, exist_ok=True)
                    base, _ = os.path.splitext(str(zp))
                    out_zip = Path(shutil.make_archive(base, "zip", root_dir=str(out_dir_path)))
                else:
                    tmp_base = out_dir_path.with_suffix("")
                    tmp_zip = Path(shutil.make_archive(str(tmp_base), "zip", root_dir=str(out_dir_path)))
                    zp.parent.mkdir(parents=True, exist_ok=True)
                    out_zip = zp
                    if out_zip.exists():
                        if overwrite:
                            out_zip.unlink()
                        else:
                            k = 1
                            while True:
                                cand = out_zip.with_name(out_zip.stem + f"_{k}").with_suffix(".zip")
                                if not cand.exists():
                                    out_zip = cand
                                    break
                    tmp_zip.replace(out_zip)
            else:
                base = str(out_dir_path)
                out_zip = Path(shutil.make_archive(base, "zip", root_dir=str(out_dir_path)))

        files_rel = [p.relative_to(out_dir_path).as_posix() for p in exported_files[:sample_limit]]
        return {
            "out_dir": str(out_dir_path),
            "exported": len(exported_files),
            "unique_images": len(image_ids) - len(missing),
            "missing": missing,
            "files": files_rel,
            "zip_path": (str(out_zip) if out_zip else None),
        }


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
        # db.ingest_from_registry(dry_run=False, show_progress=True)
        # print("[OK] registry ingestion finished.")

        # 可选：立刻做一次 GC（只报告，不删除），你可以根据需要删掉下面两行
        # gc_summary = db.garbage_collect(remove_orphan_files=False, check_db_missing_files=True, verbose=True)
        # print(f"[OK] GC summary: {gc_summary}")

        summary = db.get_db_summary()
        print("[OK] DB Summary:")
        print(json.dumps(summary, ensure_ascii=False, indent=2))

        sample_n = 16
        # 从 DB 随机抽样 image_id（若 DuckDB 版本不支持 TABLESAMPLE，可用 ORDER BY random()）
        rows = db.conn.execute(
            "SELECT image_id FROM images ORDER BY random() LIMIT ?;",
            [sample_n],
        ).fetchall()
        sample_ids = [r[0] for r in rows]

        if not sample_ids:
            print("[INFO] No images found in DB; skip export test.")
        else:
            export_res = db.export_images_by_ids(
                sample_ids,
                out_dir=None,  # 自动生成 tmp/exports/<ts>-<rand>/
                output="png",  # 导出 PNG 供直观看图
                normalize=True,  # 归一化到 8-bit
                zip_output=True,  # 额外打包 zip
                zip_path=None,  # 默认 zip 到导出目录同名
                overwrite=True,
                sample_limit=10,
            )
            print("[OK] Export test result:")
            print(json.dumps(export_res, ensure_ascii=False, indent=2))

    except Exception as e:
        print(f"[ERROR] ingestion failed: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
    finally:
        if db is not None:
            db.close()
