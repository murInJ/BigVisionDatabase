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
      - 提供 protocol 的增删查改、拷贝/合并/重命名、采样与基于 relations 构建等
    目录：
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
        self.database_root = database_root
        self.duckdb_path = duckdb_path or f"{database_root}/db/catalog.duckdb"

        # 单处持有连接
        self.conn = require_duckdb(
            duckdb_path=self.duckdb_path,
            threads=(threads or 0),
        )
        # DB 层统一建表
        init_duckdb_schema(self.conn)

        # writer 专注写入
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
        root = Path(self.database_root)
        images_root = root / "images"

        db_rows = self.conn.execute("SELECT image_id, uri FROM images").fetchall()
        db_images_count = len(db_rows)
        id2uri: Dict[str, str] = {r[0]: r[1] for r in db_rows}
        db_ids: Set[str] = set(id2uri.keys())

        files_scanned = 0
        orphan_paths: List[Path] = []
        for p in images_root.rglob("*.npy"):
            files_scanned += 1
            stem = p.stem
            rel_uri = p.relative_to(root).as_posix()
            if stem not in db_ids or id2uri.get(stem) != rel_uri:
                orphan_paths.append(p)

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
        if not isinstance(payload, dict):
            raise ValueError("payload must be a dict")
        image_ids = payload.get("image_ids") or []
        if not isinstance(image_ids, list):
            raise ValueError("payload['image_ids'] must be a list")

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
        payload: Optional[Dict[str, Any]] = None,
        add_protocols: Optional[List[str]] = None,
        remove_protocols: Optional[List[str]] = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        exists = self.conn.execute(
            "SELECT COUNT(*) FROM relations WHERE relation_id = ?",
            [relation_id],
        ).fetchone()[0]
        if not exists:
            return {"relation_id": relation_id, "updated": False, "error": "relation not found"}

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
            if remove_protocols:
                placeholders = ",".join(["?"] * len(remove_protocols))
                self.conn.execute(
                    f"DELETE FROM protocol WHERE relation_id = ? AND protocol_name IN ({placeholders})",
                    [relation_id, *remove_protocols],
                )
            if add_protocols:
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

    # ------------- Protocols CRUD/组织/采样 -------------

    def list_protocols(self) -> List[Dict[str, Any]]:
        """
        列出所有 protocol 及其 relation 数。
        """
        rows = self.conn.execute(
            "SELECT protocol_name, COUNT(DISTINCT relation_id) AS n FROM protocol GROUP BY protocol_name ORDER BY protocol_name"
        ).fetchall()
        return [{"protocol_name": str(r[0]), "n_relations": int(r[1])} for r in rows]

    def get_protocol_relations(
        self,
        protocol_name: str,
        *,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """
        返回某 protocol 的 relation_id 列表（分页）及总数。
        """
        n = self.conn.execute(
            "SELECT COUNT(DISTINCT relation_id) FROM protocol WHERE protocol_name = ?",
            [protocol_name],
        ).fetchone()[0]
        if n == 0:
            return {"protocol_name": protocol_name, "total": 0, "relation_ids": []}
        if limit is None:
            q = "SELECT DISTINCT relation_id FROM protocol WHERE protocol_name = ? ORDER BY relation_id"
            rels = [r[0] for r in self.conn.execute(q, [protocol_name]).fetchall()]
        else:
            q = "SELECT DISTINCT relation_id FROM protocol WHERE protocol_name = ? ORDER BY relation_id LIMIT ? OFFSET ?"
            rels = [r[0] for r in self.conn.execute(q, [protocol_name, limit, offset]).fetchall()]
        return {"protocol_name": protocol_name, "total": int(n), "relation_ids": [str(x) for x in rels]}

    def create_protocol(
        self,
        protocol_name: str,
        relation_ids: List[str],
        *,
        replace: bool = False,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        用给定 relation_ids 构建/重建一个 protocol。
        replace=True 时会先清空同名 protocol。
        """
        relation_ids = list(dict.fromkeys(relation_ids))  # 去重保序
        # 校验 relation 是否存在
        missing, found = self._check_relations_exist(relation_ids)

        if dry_run:
            return {
                "protocol_name": protocol_name,
                "replace": replace,
                "insert_rows": len(found),
                "missing_relations": missing,
                "dry_run": True,
            }

        self.conn.execute("BEGIN;")
        try:
            if replace:
                self.conn.execute("DELETE FROM protocol WHERE protocol_name = ?", [protocol_name])
            if found:
                rows = [(protocol_name, rid, protocol_name) for rid in found]
                self.conn.register("proto_df_new", self._rows_to_df(rows, ["protocol_name", "relation_id", "relation_set"]))
                self.conn.execute("INSERT INTO protocol SELECT * FROM proto_df_new;")
            self.conn.execute("COMMIT;")
        except Exception:
            self.conn.execute("ROLLBACK;")
            raise

        return {
            "protocol_name": protocol_name,
            "replace": replace,
            "insert_rows": len(found),
            "missing_relations": missing,
            "dry_run": False,
        }

    def add_relations_to_protocol(
        self,
        protocol_name: str,
        relation_ids: List[str],
        *,
        deduplicate: bool = True,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        往现有 protocol 追加 relations（自动去重）。
        """
        relation_ids = list(dict.fromkeys(relation_ids))
        missing, found = self._check_relations_exist(relation_ids)

        # 过滤已存在映射
        existing = {
            r[0] for r in self.conn.execute(
                "SELECT DISTINCT relation_id FROM protocol WHERE protocol_name = ?",
                [protocol_name],
            ).fetchall()
        }
        to_add = [rid for rid in found if (rid not in existing) or not deduplicate]

        if dry_run:
            return {
                "protocol_name": protocol_name,
                "to_add": len(to_add),
                "already_exists": len(found) - len(to_add),
                "missing_relations": missing,
                "dry_run": True,
            }

        self.conn.execute("BEGIN;")
        try:
            if to_add:
                rows = [(protocol_name, rid, protocol_name) for rid in to_add]
                self.conn.register("proto_df_add", self._rows_to_df(rows, ["protocol_name", "relation_id", "relation_set"]))
                self.conn.execute("INSERT INTO protocol SELECT * FROM proto_df_add;")
            self.conn.execute("COMMIT;")
        except Exception:
            self.conn.execute("ROLLBACK;")
            raise

        return {
            "protocol_name": protocol_name,
            "added": len(to_add),
            "skipped": len(found) - len(to_add),
            "missing_relations": missing,
            "dry_run": False,
        }

    def remove_relations_from_protocol(
        self,
        protocol_name: str,
        relation_ids: List[str],
        *,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        从 protocol 中移除指定 relations 的映射。
        """
        relation_ids = list(dict.fromkeys(relation_ids))
        if dry_run:
            n = self.conn.execute(
                f"SELECT COUNT(*) FROM protocol WHERE protocol_name = ? AND relation_id IN ({','.join(['?']*len(relation_ids))})",
                [protocol_name, *relation_ids],
            ).fetchone()[0] if relation_ids else 0
            return {"protocol_name": protocol_name, "would_delete": int(n), "dry_run": True}

        self.conn.execute("BEGIN;")
        try:
            if relation_ids:
                self.conn.execute(
                    f"DELETE FROM protocol WHERE protocol_name = ? AND relation_id IN ({','.join(['?']*len(relation_ids))})",
                    [protocol_name, *relation_ids],
                )
            self.conn.execute("COMMIT;")
        except Exception:
            self.conn.execute("ROLLBACK;")
            raise
        return {"protocol_name": protocol_name, "deleted": len(relation_ids), "dry_run": False}

    def delete_protocol(
        self,
        protocol_name: str,
        *,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        删除整个 protocol（不影响 relations 表）。
        """
        cnt = self.conn.execute(
            "SELECT COUNT(*) FROM protocol WHERE protocol_name = ?",
            [protocol_name],
        ).fetchone()[0]
        if dry_run:
            return {"protocol_name": protocol_name, "rows": int(cnt), "dry_run": True}

        self.conn.execute("BEGIN;")
        try:
            self.conn.execute("DELETE FROM protocol WHERE protocol_name = ?", [protocol_name])
            self.conn.execute("COMMIT;")
        except Exception:
            self.conn.execute("ROLLBACK;")
            raise
        return {"protocol_name": protocol_name, "rows": int(cnt), "dry_run": False}

    def rename_protocol(
        self,
        old_name: str,
        new_name: str,
        *,
        overwrite: bool = False,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        重命名 protocol（同时更新 relation_set = new_name）。
        overwrite=True 时，会先删除目标同名 protocol 后再改名。
        """
        old_cnt = self.conn.execute(
            "SELECT COUNT(*) FROM protocol WHERE protocol_name = ?",
            [old_name],
        ).fetchone()[0]
        new_cnt = self.conn.execute(
            "SELECT COUNT(*) FROM protocol WHERE protocol_name = ?",
            [new_name],
        ).fetchone()[0]

        if not old_cnt:
            return {"ok": False, "error": "source protocol not found", "old_name": old_name, "new_name": new_name}

        if dry_run:
            return {
                "ok": True,
                "old_rows": int(old_cnt),
                "target_exists": bool(new_cnt),
                "overwrite": overwrite,
                "dry_run": True,
            }

        self.conn.execute("BEGIN;")
        try:
            if new_cnt and overwrite:
                self.conn.execute("DELETE FROM protocol WHERE protocol_name = ?", [new_name])
            # 同时更新两列
            self.conn.execute(
                "UPDATE protocol SET protocol_name = ?, relation_set = ? WHERE protocol_name = ?",
                [new_name, new_name, old_name],
            )
            self.conn.execute("COMMIT;")
        except Exception:
            self.conn.execute("ROLLBACK;")
            raise

        return {"ok": True, "old_rows": int(old_cnt), "dry_run": False}

    def copy_protocol(
        self,
        src_protocol: str,
        dst_protocol: str,
        *,
        overwrite: bool = False,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        拷贝一个 protocol 到新名字（去重）。overwrite=True 时先清空目标。
        """
        rels = [r[0] for r in self.conn.execute(
            "SELECT DISTINCT relation_id FROM protocol WHERE protocol_name = ?",
            [src_protocol],
        ).fetchall()]
        if not rels:
            return {"ok": False, "error": "source protocol empty or not found", "src": src_protocol, "dst": dst_protocol}

        if dry_run:
            dst_cnt = self.conn.execute(
                "SELECT COUNT(*) FROM protocol WHERE protocol_name = ?",
                [dst_protocol],
            ).fetchone()[0]
            return {
                "ok": True,
                "src_relations": len(rels),
                "dst_exists": bool(dst_cnt),
                "overwrite": overwrite,
                "dry_run": True,
            }

        return self.create_protocol(dst_protocol, rels, replace=overwrite, dry_run=False)

    def merge_protocols(
        self,
        new_protocol: str,
        sources: List[str],
        *,
        mode: str = "union",   # 'union' | 'intersect'
        replace: bool = True,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        将多个 protocol 合并为新 protocol（并集/交集）。
        """
        sources = [s for s in sources if s]
        if not sources:
            return {"ok": False, "error": "no source protocols"}
        sets: List[Set[str]] = []
        for s in sources:
            rels = {r[0] for r in self.conn.execute(
                "SELECT DISTINCT relation_id FROM protocol WHERE protocol_name = ?",
                [s],
            ).fetchall()}
            if not rels:
                sets.append(set())
            else:
                sets.append(rels)

        if mode == "union":
            merged = set().union(*sets) if sets else set()
        elif mode == "intersect":
            merged = set.intersection(*sets) if sets else set()
        else:
            return {"ok": False, "error": "mode must be 'union' or 'intersect'"}

        rel_ids = sorted(merged)
        if dry_run:
            return {"ok": True, "new_protocol": new_protocol, "n_relations": len(rel_ids), "mode": mode, "replace": replace, "dry_run": True}

        return self.create_protocol(new_protocol, rel_ids, replace=replace, dry_run=False)

    def build_protocol_from_relations(
        self,
        protocol_name: str,
        relation_ids: List[str],
        *,
        replace: bool = False,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        直接用一个 relation_id 列表构建/重建 protocol（与 create_protocol 等价，命名更直观）。
        """
        return self.create_protocol(protocol_name, relation_ids, replace=replace, dry_run=dry_run)

    def sample_protocol(
        self,
        src_protocol: str,
        k: int,
        *,
        seed: Optional[int] = None,
        dst_protocol: Optional[str] = None,
        replace: bool = True,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        从指定 protocol 随机采样 k 条 relation_id；若提供 dst_protocol，则将采样子集写为一个新 protocol。
        """
        if k <= 0:
            return {"ok": False, "error": "k must be > 0"}

        # DuckDB 采样：ORDER BY random()
        if seed is None:
            rows = self.conn.execute(
                "SELECT DISTINCT relation_id FROM protocol WHERE protocol_name = ? ORDER BY random() LIMIT ?",
                [src_protocol, k],
            ).fetchall()
        else:
            # 使用固定种子：通过内置 random() 不直接支持 seed，这里用简化策略：
            # 取所有 relation_id 后在 Python 端做可复现实验（避免大表OOM时可分块改造）
            all_rows = self.conn.execute(
                "SELECT DISTINCT relation_id FROM protocol WHERE protocol_name = ?",
                [src_protocol],
            ).fetchall()
            import random as _random
            _random.seed(int(seed))
            rows = [(_rid,) for _rid in _random.sample([r[0] for r in all_rows], k=min(k, len(all_rows)))]

        sample_rel_ids = [r[0] for r in rows]
        if not dst_protocol:
            return {"ok": True, "sample_size": len(sample_rel_ids), "relation_ids": sample_rel_ids}

        # 写入为新 protocol
        return self.create_protocol(dst_protocol, sample_rel_ids, replace=replace, dry_run=dry_run)

    # ------------- images 导出 -------------

    def export_images_by_ids(
        self,
        image_ids: List[str],
        out_dir: Optional[str] = None,
        *,
        output: str = "png",
        normalize: bool = True,
        zip_output: bool = False,
        zip_path: Optional[str] = None,
        overwrite: bool = True,
        sample_limit: int = 20,
        color_order: str = "bgr",
    ) -> Dict[str, Any]:
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

        def _prep_uint8(arr: _np.ndarray) -> _np.ndarray:
            a = _np.asarray(arr)
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
            if a.ndim == 2:
                return a
            if a.ndim == 3:
                if a.shape[2] == 1:
                    return a[:, :, 0]
                if a.shape[2] >= 3:
                    return a[:, :, :3]
            return a.reshape(a.shape[0], a.shape[1]).astype(_np.uint8)

        try:
            import imageio.v3 as iio  # type: ignore
            def _save_png(dst: Path, img: _np.ndarray) -> bool:
                iio.imwrite(str(dst), img)
                return True
            writer_backend = "imageio"
        except Exception:
            try:
                import cv2  # type: ignore
                def _save_png(dst: Path, img: _np.ndarray) -> bool:
                    return bool(cv2.imwrite(str(dst), img))
                writer_backend = "cv2"
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

        for idx, iid in enumerate(image_ids):
            row = id2row.get(iid)
            if row is None:
                continue
            src_path = (Path(self.database_root) / row["uri"]).resolve()
            alias = row["alias"] or row["modality"] or iid
            base = f"{idx:04d}_{alias}"

            if output in ("png", "both"):
                try:
                    import numpy as _np2
                    arr = _np2.load(src_path, allow_pickle=False)
                    img = _prep_uint8(arr)
                    if isinstance(img, _np2.ndarray) and img.ndim == 3 and img.shape[2] == 3:
                        if writer_backend == "imageio" and color_order == "bgr":
                            img = img[:, :, ::-1]
                        elif writer_backend == "cv2" and color_order == "rgb":
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

    def _check_relations_exist(self, relation_ids: List[str]) -> Tuple[List[str], List[str]]:
        """
        返回 (missing, found) 列表（均保序）。
        """
        if not relation_ids:
            return ([], [])
        placeholders = ",".join(["?"] * len(relation_ids))
        rows = self.conn.execute(
            f"SELECT relation_id FROM relations WHERE relation_id IN ({placeholders})",
            relation_ids,
        ).fetchall()
        found_set = {r[0] for r in rows}
        found = [rid for rid in relation_ids if rid in found_set]
        missing = [rid for rid in relation_ids if rid not in found_set]
        return (missing, found)

    # ---------------- 生命周期 ----------------

    def close(self) -> None:
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
        database_root = os.environ.get("DB_ROOT", os.path.abspath("./bigvision_db"))
        print(f"[INFO] Using fallback database_root: {database_root}")

    workers = max(8, (os.cpu_count() or 8))

    db = None
    try:
        db = BigVisionDatabase(
            database_root=database_root,
            duckdb_path=None,
            max_workers=workers,
            threads=0,
        )

        # ------- DB 概览 -------
        summary = db.get_db_summary()
        print("[OK] DB Summary:")
        print(json.dumps(summary, ensure_ascii=False, indent=2))

        # ------- 随机采样导出（示例：注释掉即可） -------
        sample_n = 8
        rows = db.conn.execute(
            "SELECT image_id FROM images ORDER BY random() LIMIT ?;",
            [sample_n],
        ).fetchall()
        sample_ids = [r[0] for r in rows]
        if sample_ids:
            print(f"[INFO] sampled {len(sample_ids)} image_ids")
        else:
            print("[INFO] No images found in DB; skip export test.")

        # ------- Relations CRUD（dry_run） -------
        # if sample_ids:
        #     new_payload = {
        #         "task_type": "demo",
        #         "annotation": {"note": "dry-run create test"},
        #         "image_ids": sample_ids[: min(3, len(sample_ids))],
        #     }
        #     print("\n[TEST] add_relation (dry_run)")
        #     print(json.dumps(db.add_relation(payload=new_payload, protocols=["demo_proto", "debug_view"], dry_run=True),
        #                      ensure_ascii=False, indent=2))
        #
        # exist_row = db.conn.execute("SELECT relation_id FROM relations LIMIT 1").fetchone()
        # if exist_row:
        #     exist_rid = exist_row[0]
        #     print("\n[TEST] get_relation")
        #     print(json.dumps(db.get_relation(exist_rid), ensure_ascii=False, indent=2))
        #
        #     print("\n[TEST] update_relation (dry_run)")
        #     upd_payload = {
        #         "task_type": "updated_demo",
        #         "annotation": {"note": "dry-run update"},
        #         "image_ids": sample_ids[: min(2, len(sample_ids))] if sample_ids else [],
        #     }
        #     print(json.dumps(db.update_relation(
        #         exist_rid,
        #         payload=upd_payload,
        #         add_protocols=["extra_proto"],
        #         remove_protocols=["debug_view"],
        #         dry_run=True,
        #     ), ensure_ascii=False, indent=2))
        #
        #     print("\n[TEST] delete_relation (dry_run)")
        #     print(json.dumps(db.delete_relation(exist_rid, dry_run=True), ensure_ascii=False, indent=2))
        # else:
        #     print("[INFO] No existing relations to test update/delete.")

        # ------- Protocols CRUD/组织/采样（全部 dry_run） -------
        print("\n[TEST] list_protocols()")
        print(json.dumps(db.list_protocols(), ensure_ascii=False, indent=2))

        # 选择一个已有 protocol 做采样
        proto_row = db.conn.execute("SELECT DISTINCT protocol_name FROM protocol LIMIT 1").fetchone()
        if proto_row:
            p0 = proto_row[0]
            print(f"\n[TEST] get_protocol_relations('{p0}', limit=5)")
            print(json.dumps(db.get_protocol_relations(p0, limit=5), ensure_ascii=False, indent=2))

            # 采样并生成新 protocol（dry_run）
            new_pname = f"{p0}_sample_demo"
            print(f"\n[TEST] sample_protocol('{p0}', k=5, dst_protocol='{new_pname}') (dry_run)")
            print(json.dumps(db.sample_protocol(p0, 5, seed=42, dst_protocol=new_pname, replace=True, dry_run=True),
                             ensure_ascii=False, indent=2))

            # 拷贝到新 protocol（dry_run）
            copy_name = f"{p0}_copy_demo"
            print(f"\n[TEST] copy_protocol('{p0}' -> '{copy_name}') (dry_run)")
            print(json.dumps(db.copy_protocol(p0, copy_name, overwrite=True, dry_run=True), ensure_ascii=False, indent=2))

            # 往 protocol 追加/移除 relations（dry_run）
            rels5 = [r[0] for r in db.conn.execute(
                "SELECT relation_id FROM protocol WHERE protocol_name = ? LIMIT 5", [p0]
            ).fetchall()]
            print(f"\n[TEST] add_relations_to_protocol('{p0}', {len(rels5)} ids) (dry_run)")
            print(json.dumps(db.add_relations_to_protocol(p0, rels5, deduplicate=True, dry_run=True),
                             ensure_ascii=False, indent=2))

            print(f"\n[TEST] remove_relations_from_protocol('{p0}', {len(rels5)} ids) (dry_run)")
            print(json.dumps(db.remove_relations_from_protocol(p0, rels5, dry_run=True),
                             ensure_ascii=False, indent=2))

        # 合并两个 protocol（若存在至少两个）
        proto_two = db.conn.execute("SELECT DISTINCT protocol_name FROM protocol LIMIT 2").fetchall()
        if len(proto_two) >= 2:
            pA, pB = proto_two[0][0], proto_two[1][0]
            merged_name = f"merge_{pA}_{pB}_union_demo"
            print(f"\n[TEST] merge_protocols(['{pA}','{pB}'], mode='union' -> '{merged_name}') (dry_run)")
            print(json.dumps(db.merge_protocols(merged_name, [pA, pB], mode="union", replace=True, dry_run=True),
                             ensure_ascii=False, indent=2))
        else:
            print("[INFO] Not enough protocols to test merge.")

        # 重命名 protocol（dry_run）
        if proto_row:
            p0 = proto_row[0]
            renamed = f"{p0}_renamed_demo"
            print(f"\n[TEST] rename_protocol('{p0}' -> '{renamed}') (dry_run)")
            print(json.dumps(db.rename_protocol(p0, renamed, overwrite=True, dry_run=True),
                             ensure_ascii=False, indent=2))

        # 删除 protocol（dry_run）
        if proto_row:
            del_name = f"{proto_row[0]}_to_delete_demo"
            print(f"\n[TEST] delete_protocol('{del_name}') (dry_run)")
            print(json.dumps(db.delete_protocol(del_name, dry_run=True), ensure_ascii=False, indent=2))

    except Exception as e:
        print(f"[ERROR] run failed: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
    finally:
        if db is not None:
            db.close()
