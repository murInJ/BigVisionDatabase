# Database/db.py
from __future__ import annotations

import json
import os
import sys
import traceback
import uuid
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
      - 提供 DB 概览统计
      - 提供按 image_id 导出图像（PNG/NPY，并可 zip 打包）
      - 提供 relations 的增删查改（CRUD）

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
        1) 孤儿文件（磁盘存在但 DB 无引用或路径不一致）
        2) DB 缺失文件（DB 行指向的文件在磁盘上不存在）
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
                orphan_paths.append(p)
            else:
                if id2uri.get(stem) != rel_uri:
                    orphan_paths.append(p)

        # --- 可选：删除孤儿文件 ---
        orphan_removed = 0
        orphan_samples: List[str] = []
        if orphan_paths:
            if verbose:
                print(f"[GC] Found {len(orphan_paths)} orphan files.")
            orphan_samples = [op.relative_to(root).as_posix() for op in orphan_paths[:report_limit]]
            if remove_orphan_files:
                for p in orphan_paths:
                    try:
                        p.unlink(missing_ok=True)
                        orphan_removed += 1
                    except TypeError:
                        try:
                            if p.exists():
                                p.unlink()
                                orphan_removed += 1
                        except Exception:
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

    # ---------------- 概览统计 ----------------

    def get_db_summary(self) -> Dict[str, Any]:
        """
        返回数据库概览汇总（总数 + 按 protocol 汇总数据集覆盖）。
        """
        totals = {}
        totals["images"] = self.conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
        totals["relations"] = self.conn.execute("SELECT COUNT(*) FROM relations").fetchone()[0]
        totals["protocol_rows"] = self.conn.execute("SELECT COUNT(*) FROM protocol").fetchone()[0]
        totals["protocols"] = self.conn.execute("SELECT COUNT(DISTINCT protocol_name) FROM protocol").fetchone()[0]

        proto_rows = self.conn.execute("SELECT protocol_name, relation_id FROM protocol").fetchall()
        if not proto_rows:
            return {"totals": totals, "protocols": []}

        from collections import defaultdict
        proto_to_relids: Dict[str, set] = defaultdict(set)
        for pname, rid in proto_rows:
            proto_to_relids[str(pname)].add(str(rid))

        # 批量拉取 relations.payload
        def _chunks(seq, n=1000):
            it = list(seq)
            for i in range(0, len(it), n):
                yield it[i:i + n]

        all_relids = set().union(*proto_to_relids.values())
        relid_to_imgids: Dict[str, List[str]] = {}
        if all_relids:
            for chunk in _chunks(list(all_relids), 1000):
                placeholders = ",".join(["?"] * len(chunk))
                rows = self.conn.execute(
                    f"SELECT relation_id, payload FROM relations WHERE relation_id IN ({placeholders})",
                    chunk,
                ).fetchall()
                for rid, payload in rows:
                    try:
                        js = json.loads(payload)
                        img_ids = js.get("image_ids", []) or []
                        relid_to_imgids[str(rid)] = [str(x) for x in img_ids]
                    except Exception:
                        relid_to_imgids[str(rid)] = []

        # 查询 images.dataset_name
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

    # ------------- Relations CRUD -------------

    def add_relation(
        self,
        *,
        payload: Dict[str, Any],
        protocols: Optional[List[str]] = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        新增一条 relation。payload 必须包含 image_ids(list)。
        协议列表 protocols（可为空）会将该 relation 归入多个 protocol（relation_set=protocol_name）。
        """
        if not isinstance(payload, dict):
            raise ValueError("payload must be a dict")
        image_ids = payload.get("image_ids") or []
        if not isinstance(image_ids, list):
            raise ValueError("payload['image_ids'] must be a list")

        # 校验 image_ids 是否存在（仅报告）
        missing = []
        if image_ids:
            placeholders = ",".join(["?"] * len(image_ids))
            q = f"SELECT image_id FROM images WHERE image_id IN ({placeholders})"
            found = {r[0] for r in self.conn.execute(q, image_ids).fetchall()}
            missing = [iid for iid in image_ids if iid not in found]

        rid = uuid.uuid4().hex
        payload_json = json.dumps(payload, ensure_ascii=False, sort_keys=True)

        if dry_run:
            return {
                "relation_id": rid,
                "insert_relations": 1,
                "insert_protocol": len(protocols or []),
                "missing_image_ids": missing,
                "dry_run": True,
            }

        self.conn.execute("BEGIN;")
        try:
            self.conn.execute(
                "INSERT INTO relations (relation_id, payload) VALUES (?, ?)",
                [rid, payload_json],
            )
            if protocols:
                proto_rows = [(p, rid, p) for p in protocols]
                self.conn.register("proto_df_tmp", self._rows_to_df(proto_rows, ["protocol_name", "relation_id", "relation_set"]))
                self.conn.execute("INSERT INTO protocol SELECT * FROM proto_df_tmp;")
            self.conn.execute("COMMIT;")
        except Exception:
            self.conn.execute("ROLLBACK;")
            raise

        return {
            "relation_id": rid,
            "insert_relations": 1,
            "insert_protocol": len(protocols or []),
            "missing_image_ids": missing,
            "dry_run": False,
        }

    def get_relation(self, relation_id: str) -> Optional[Dict[str, Any]]:
        """
        读取单条 relation：返回 {relation_id, payload(dict), protocols:[...]} 或 None
        """
        row = self.conn.execute(
            "SELECT payload FROM relations WHERE relation_id = ?",
            [relation_id],
        ).fetchone()
        if row is None:
            return None
        payload = json.loads(row[0])
        prots = [r[0] for r in self.conn.execute(
            "SELECT protocol_name FROM protocol WHERE relation_id = ? ORDER BY protocol_name",
            [relation_id],
        ).fetchall()]
        return {"relation_id": relation_id, "payload": payload, "protocols": prots}

    def update_relation(
        self,
        relation_id: str,
        *,
        payload: Optional[Dict[str, Any]] = None,           # 若提供则全量替换 payload
        add_protocols: Optional[List[str]] = None,          # 需要新增归属的协议名
        remove_protocols: Optional[List[str]] = None,       # 需要移除归属的协议名
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        更新 relation：可替换 payload、增删协议归属。
        """
        # 是否存在
        exists = self.conn.execute(
            "SELECT COUNT(*) FROM relations WHERE relation_id = ?",
            [relation_id],
        ).fetchone()[0]
        if not exists:
            return {"relation_id": relation_id, "updated": False, "error": "relation not found"}

        # 检查 payload 合法性（如包含 image_ids 则校验）
        missing = []
        if payload is not None:
            img_ids = payload.get("image_ids") if isinstance(payload, dict) else None
            if img_ids is not None and not isinstance(img_ids, list):
                raise ValueError("payload['image_ids'] must be a list when provided")
            if img_ids:
                placeholders = ",".join(["?"] * len(img_ids))
                q = f"SELECT image_id FROM images WHERE image_id IN ({placeholders})"
                found = {r[0] for r in self.conn.execute(q, img_ids).fetchall()}
                missing = [iid for iid in img_ids if iid not in found]

        add_protocols = add_protocols or []
        remove_protocols = remove_protocols or []

        if dry_run:
            return {
                "relation_id": relation_id,
                "replace_payload": payload is not None,
                "add_protocols": add_protocols,
                "remove_protocols": remove_protocols,
                "missing_image_ids": missing,
                "dry_run": True,
            }

        self.conn.execute("BEGIN;")
        try:
            if payload is not None:
                payload_json = json.dumps(payload, ensure_ascii=False, sort_keys=True)
                self.conn.execute(
                    "UPDATE relations SET payload = ? WHERE relation_id = ?",
                    [payload_json, relation_id],
                )
            # 删除协议
            if remove_protocols:
                placeholders = ",".join(["?"] * len(remove_protocols))
                self.conn.execute(
                    f"DELETE FROM protocol WHERE relation_id = ? AND protocol_name IN ({placeholders})",
                    [relation_id, *remove_protocols],
                )
            # 新增协议（去重）
            if add_protocols:
                # 先查已有
                existing = {r[0] for r in self.conn.execute(
                    "SELECT protocol_name FROM protocol WHERE relation_id = ?",
                    [relation_id],
                ).fetchall()}
                to_add = [p for p in add_protocols if p not in existing]
                if to_add:
                    rows = [(p, relation_id, p) for p in to_add]
                    self.conn.register("proto_df_tmp2", self._rows_to_df(rows, ["protocol_name", "relation_id", "relation_set"]))
                    self.conn.execute("INSERT INTO protocol SELECT * FROM proto_df_tmp2;")

            self.conn.execute("COMMIT;")
        except Exception:
            self.conn.execute("ROLLBACK;")
            raise

        return {
            "relation_id": relation_id,
            "replace_payload": payload is not None,
            "add_protocols": add_protocols,
            "remove_protocols": remove_protocols,
            "missing_image_ids": missing,
            "dry_run": False,
        }

    def delete_relation(
        self,
        relation_id: str,
        *,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        删除 relation（及其在 protocol 中的引用）。不删除 images。
        """
        exists = self.conn.execute(
            "SELECT COUNT(*) FROM relations WHERE relation_id = ?",
            [relation_id],
        ).fetchone()[0]
        if not exists:
            return {"relation_id": relation_id, "deleted": False, "error": "relation not found"}

        if dry_run:
            ref_cnt = self.conn.execute(
                "SELECT COUNT(*) FROM protocol WHERE relation_id = ?",
                [relation_id],
            ).fetchone()[0]
            return {"relation_id": relation_id, "deleted": True, "protocol_refs": int(ref_cnt), "dry_run": True}

        self.conn.execute("BEGIN;")
        try:
            self.conn.execute("DELETE FROM protocol WHERE relation_id = ?", [relation_id])
            self.conn.execute("DELETE FROM relations WHERE relation_id = ?", [relation_id])
            self.conn.execute("COMMIT;")
        except Exception:
            self.conn.execute("ROLLBACK;")
            raise
        return {"relation_id": relation_id, "deleted": True, "dry_run": False}

    def list_relations_by_protocol(
        self,
        protocol_name: str,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        按 protocol_name 列出 relations（带 relation_id + payload）。
        """
        rows = self.conn.execute(
            """
            SELECT r.relation_id, r.payload
            FROM protocol p
            JOIN relations r ON p.relation_id = r.relation_id
            WHERE p.protocol_name = ?
            ORDER BY r.relation_id
            LIMIT ? OFFSET ?
            """,
            [protocol_name, limit, offset],
        ).fetchall()
        out: List[Dict[str, Any]] = []
        for rid, payload in rows:
            try:
                js = json.loads(payload)
            except Exception:
                js = {"_raw": payload}
            out.append({"relation_id": str(rid), "payload": js})
        return out

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
        默认将三通道图像视为 BGR 来源（OpenCV 常见读法）。如你的数据确认为 RGB，可显式传 color_order="rgb"。
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
            # (C,H,W) -> (H,W,C)
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

            src_path = (Path(self.database_root) / row["uri"]).resolve()
            alias = row["alias"] or row["modality"] or iid
            base = f"{idx:04d}_{alias}"

            # PNG
            if output in ("png", "both"):
                try:
                    import numpy as _np2
                    arr = _np2.load(src_path, allow_pickle=False)
                    img = _prep_uint8(arr)
                    if isinstance(img, _np2.ndarray) and img.ndim == 3 and img.shape[2] == 3:
                        if writer_backend == "imageio" and color_order == "bgr":
                            img = img[:, :, ::-1]  # BGR -> RGB
                        elif writer_backend == "cv2" and color_order == "rgb":
                            img = img[:, :, ::-1]  # RGB -> BGR
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
                    import shutil as _sh
                    dst_name = _safe_name(base, "npy")
                    dst_path = out_dir_path / dst_name
                    if dst_path.exists() and not overwrite:
                        pass
                    else:
                        _sh.copy2(src_path, dst_path)
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
                        out_zip.unlink()
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

    # ---------------- helpers ----------------

    @staticmethod
    def _rows_to_df(rows: List[Tuple[Any, ...]], columns: List[str]):
        import pandas as pd
        return pd.DataFrame(rows, columns=columns)

    # ---------------- 生命周期 ----------------

    def close(self) -> None:
        """关闭连接（可重复调用）"""
        try:
            self.conn.close()
        except Exception:
            pass

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

        # ------- DB 概览 -------
        summary = db.get_db_summary()
        print("[OK] DB Summary:")
        print(json.dumps(summary, ensure_ascii=False, indent=2))

        # ------- 随机采样导出（示例） -------
        sample_n = 16
        rows = db.conn.execute(
            "SELECT image_id FROM images ORDER BY random() LIMIT ?;",
            [sample_n],
        ).fetchall()
        sample_ids = [r[0] for r in rows]
        # if sample_ids:
        #     export_res = db.export_images_by_ids(
        #         sample_ids,
        #         out_dir=None,
        #         output="png",
        #         normalize=True,
        #         zip_output=True,
        #         zip_path=None,
        #         overwrite=True,
        #         sample_limit=10,
        #     )
        #     print("[OK] Export test result:")
        #     print(json.dumps(export_res, ensure_ascii=False, indent=2))
        # else:
        #     print("[INFO] No images found in DB; skip export test.")

        # ------- Relations CRUD（dry_run 测试，不实际修改） -------
        # 1) 构造一个新 relation 的 payload（使用随机采样的 image_ids）
        if not sample_ids:
            # 如果没有图片，跳过 CRUD 测试
            print("[INFO] Skip relation CRUD dry-run tests (no images).")
        else:
            new_payload = {
                "task_type": "demo",
                "annotation": {"note": "dry-run create test"},
                "image_ids": sample_ids[: min(3, len(sample_ids))],  # 取前 3 张
            }
            print("\n[TEST] add_relation (dry_run)")
            add_res = db.add_relation(payload=new_payload, protocols=["demo_proto", "debug_view"], dry_run=True)
            print(json.dumps(add_res, ensure_ascii=False, indent=2))

            # 2) 挑选一条已存在的 relation 做 query/update/delete（dry_run）
            exist_row = db.conn.execute("SELECT relation_id, payload FROM relations LIMIT 1").fetchone()
            if exist_row:
                exist_rid = exist_row[0]
                print("\n[TEST] get_relation")
                got = db.get_relation(exist_rid)
                print(json.dumps(got, ensure_ascii=False, indent=2))

                print("\n[TEST] update_relation (dry_run)")
                # 替换 payload：在原 payload 基础上添加一个字段（不去真正读出再写，纯演示）
                upd_payload = {
                    "task_type": "updated_demo",
                    "annotation": {"note": "dry-run update"},
                    "image_ids": sample_ids[: min(2, len(sample_ids))],  # 换一组 id 测试校验
                }
                upd_res = db.update_relation(
                    exist_rid,
                    payload=upd_payload,
                    add_protocols=["extra_proto"],
                    remove_protocols=["debug_view"],  # 即便不存在也不报错
                    dry_run=True,
                )
                print(json.dumps(upd_res, ensure_ascii=False, indent=2))

                print("\n[TEST] delete_relation (dry_run)")
                del_res = db.delete_relation(exist_rid, dry_run=True)
                print(json.dumps(del_res, ensure_ascii=False, indent=2))
            else:
                print("[INFO] No existing relations to test update/delete.")

            # 3) 列出某个 protocol 下的 relations
            proto_row = db.conn.execute("SELECT DISTINCT protocol_name FROM protocol LIMIT 1").fetchone()
            if proto_row:
                pname = proto_row[0]
                print(f"\n[TEST] list_relations_by_protocol('{pname}')")
                lst = db.list_relations_by_protocol(pname, limit=5, offset=0)
                print(json.dumps(lst, ensure_ascii=False, indent=2))
            else:
                print("[INFO] No protocol found to list relations.")

    except Exception as e:
        print(f"[ERROR] run failed: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
    finally:
        if db is not None:
            db.close()
