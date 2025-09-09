import os
import uuid
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm  # 单行进度条 + 汇总信息（postfix）

try:
    # 支持包内/脚本两种布局
    from .utils import ensure_dir, require_duckdb  # type: ignore
except Exception:  # pragma: no cover
    from utils import ensure_dir, require_duckdb  # type: ignore

from Config.setting import GetDatabaseConfig  # 仅 __main__ 使用
from OriginDataset.base import BaseAdaptor
from registry import discover_and_register, REGISTRY


class DatasetWriter:
    """极简写入器（无导入批次概念；全部以 adaptor 提供为准）

    磁盘:
      images/<dataset_name>/<UUID>.npy           # dataset_name 为每张图像自身来源

    DuckDB 表:
      - images(image_id, uri, modality, dataset_name, alias, extra)
      - relations(relation_id, payload JSON)     # payload 内会注入有序 image_ids（按 relation.image_names）
      - protocol(protocol_name, relation_id, relation_set)
    """

    def __init__(
        self,
        database_root: str,
        duckdb_path: str | None = None,
        max_workers: int = 8,
    ) -> None:
        self.root = Path(database_root)
        self.images_root = self.root / "images"
        self.db_root = self.root / "db"
        self.max_workers = max(1, int(max_workers))

        for p in [self.root, self.images_root, self.db_root]:
            ensure_dir(p)

        threads = max(1, os.cpu_count() or 4)
        db_path = duckdb_path or str(self.db_root / "catalog.duckdb")
        self.conn = require_duckdb(db_path, threads)
        self._init_duckdb_schema()

    # -----------------------
    # Schema
    # -----------------------
    def _init_duckdb_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS images (
                image_id     TEXT PRIMARY KEY,
                uri          TEXT NOT NULL,     -- 'images/<dataset_name>/<uuid>.npy'
                modality     TEXT,
                dataset_name TEXT NOT NULL,     -- 每张图本身的来源数据集
                alias        TEXT,              -- 导出时的别名（来自 val['alias'] 或回退为键名）
                extra        JSON
            );
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS relations (
                relation_id  TEXT PRIMARY KEY,
                payload      JSON NOT NULL      -- 含 annotation 等 + image_ids（按 relation.image_names 顺序）
            );
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS protocol (
                protocol_name TEXT NOT NULL,    -- 协议/集合名（由 adaptor 决定）
                relation_id   TEXT NOT NULL,    -- 指向 relations.relation_id
                relation_set  TEXT NOT NULL     -- 由 adaptor 决定（可与 protocol_name 相同）
            );
            """
        )
        # 简单索引
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_images_dataset ON images(dataset_name);")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_protocol_set   ON protocol(relation_set);")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_protocol_name  ON protocol(protocol_name);")

    # -----------------------
    # Public（单条样本写入：静默；返回计数给外层进度条展示）
    # -----------------------
    def write_dataset(
        self,
        images: Dict[str, Any],               # {key|alias: {'image':ndarray/torch, 'dataset_name':str, 'modality':str?, 'alias':str?, 'extra':dict?}}
        relations: Dict[str, Dict[str, Any]], # {relation_name: {'image_names':[...], ...}}（image_names 可选；缺省=全部）
        protocols: Dict[str, List[str]],      # {relation_set(=协议名/集合名): [relation_name, ...]}
        *,
        dry_run: bool = False,
    ) -> Dict[str, int]:
        """写入单个 adaptor item；返回 {'imgs':X,'rels':Y,'proto':Z} 用于进度条后缀展示"""
        # 0) 预创建各来源目录（静默）
        keys_order = list(images.keys())
        source_dirs = set(self._per_image_dataset_name(images[k]) for k in keys_order)
        if not dry_run:
            for ds in source_dirs:
                ensure_dir(self.images_root / ds)

        # 1) 写图片（并发，静默）
        if dry_run:
            key_meta: Dict[str, Dict[str, Any]] = {}
            for k in keys_order:
                img_id = uuid.uuid4().hex
                ds = self._per_image_dataset_name(images[k])
                key_meta[k] = {
                    "image_id":     img_id,
                    "uri":          f"images/{ds}/{img_id}.npy",
                    "modality":     self._modality_from(k, images[k]),
                    "dataset_name": ds,
                    "alias":        self._alias_from(k, images[k]),
                    "extra":        self._extra_from(images[k]),
                }
        else:
            key_meta = self._write_images_concurrently(images)

        key_to_id   = {k: m["image_id"] for k, m in key_meta.items()}
        alias_to_id = {m.get("alias", k): m["image_id"] for k, m in key_meta.items()}

        # 2) 组装 DF（静默）
        images_df = self._build_images_rows(key_meta)
        rel_df    = self._build_relation_rows(relations, keys_order, key_to_id, alias_to_id)
        proto_df  = self._build_protocol_rows(protocols, rel_df)

        # 3) 持久化（静默）
        if not dry_run:
            self._persist_metadata(images_df, rel_df, proto_df)

        # 返回计数给外层进度条
        return {"imgs": len(images_df), "rels": len(rel_df), "proto": len(proto_df)}

    # -----------------------
    # 批量写入（注册表）：每个 adaptor 一个进度条 + 汇总信息（postfix）
    # -----------------------
    def write_from_registry(self, *, dry_run: bool = False, show_progress: bool = True) -> None:
        discover_and_register(
            root_package="OriginDataset",
            target_filename="adaptor.py",
            class_name="Adaptor",
            base_class=BaseAdaptor,
        )

        for name, adaptor_cls in REGISTRY.items():
            adaptor = adaptor_cls()
            plugin_label = getattr(adaptor_cls, "__plugin_name__", name)

            total = self._estimate_total(adaptor) if show_progress else None
            pbar = tqdm(
                total=total,
                desc=f"{plugin_label}",
                unit="item",
                dynamic_ncols=True,
                leave=True,
                disable=not show_progress,
            )

            # 汇总计数：显示到 postfix
            agg_imgs = 0
            agg_rels = 0
            agg_proto = 0
            errors = 0

            try:
                for _idx, data in enumerate(adaptor):
                    try:
                        summary = self.write_dataset(
                            images=data["images"],
                            relations=data["relation"],
                            protocols=data["protocol"],
                            dry_run=dry_run,
                        )
                        agg_imgs  += summary.get("imgs", 0)
                        agg_rels  += summary.get("rels", 0)
                        agg_proto += summary.get("proto", 0)
                    except Exception:
                        errors += 1
                    finally:
                        pbar.update(1)
                        # 只更新一行 postfix，不刷屏
                        pbar.set_postfix(
                            imgs=agg_imgs,
                            rels=agg_rels,
                            proto=agg_proto,
                            err=errors,
                            refresh=False
                        )
            finally:
                pbar.close()

    # -----------------------
    # Helpers
    # -----------------------
    @staticmethod
    def _estimate_total(adaptor: Any) -> Optional[int]:
        """尽量估算 adaptor 的样本总数，便于显示百分比；无法估算则返回 None。"""
        try:
            return len(adaptor)  # 优先 __len__
        except Exception:
            pass
        ds = getattr(adaptor, "dataset", None)
        if ds is not None:
            try:
                return len(ds)
            except Exception:
                pass
        return None

    @staticmethod
    def _as_ndarray(val: Any) -> np.ndarray:
        if not isinstance(val, dict) or "image" not in val:
            raise ValueError("Each images[...] value must be a dict with an 'image' field.")
        img = val["image"]
        try:
            import torch  # type: ignore
            if hasattr(img, "detach") and hasattr(img, "cpu") and hasattr(img, "numpy"):
                img = img.detach().cpu().numpy()
        except Exception:
            pass
        return np.asarray(img)

    @staticmethod
    def _per_image_dataset_name(val: Any) -> str:
        if not isinstance(val, dict) or not val.get("dataset_name"):
            raise ValueError("Each images[...] value must provide 'dataset_name'.")
        return str(val["dataset_name"])

    @staticmethod
    def _modality_from(key: str, val: Any) -> str:
        if isinstance(val, dict) and val.get("modality"):
            return str(val["modality"])
        return str(key)

    @staticmethod
    def _alias_from(key: str, val: Any) -> str:
        if isinstance(val, dict) and val.get("alias"):
            return str(val["alias"])
        return str(key)

    @staticmethod
    def _extra_from(val: Any) -> str:
        extra_obj: Dict[str, Any] = {}
        if isinstance(val, dict) and isinstance(val.get("extra"), dict):
            extra_obj.update(val["extra"])
        return json.dumps(extra_obj, ensure_ascii=False, sort_keys=True)

    def _write_images_concurrently(
        self,
        images: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        """写入 .npy 到 images/<dataset_name>/ 并返回元信息（静默，无局部进度）"""
        written_files: List[Path] = []

        def _write_one(item):
            key, val = item
            arr = self._as_ndarray(val)
            ds = self._per_image_dataset_name(val)
            out_dir = self.images_root / ds
            ensure_dir(out_dir)

            image_uuid = uuid.uuid4().hex
            out_path = out_dir / f"{image_uuid}.npy"
            np.save(out_path, arr)
            written_files.append(out_path)

            uri = f"images/{ds}/{image_uuid}.npy"
            return key, {
                "image_id":     image_uuid,
                "uri":          uri,
                "modality":     self._modality_from(key, val),
                "dataset_name": ds,
                "alias":        self._alias_from(key, val),
                "extra":        self._extra_from(val),
            }

        key_meta: Dict[str, Dict[str, Any]] = {}
        errors: List[str] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = [ex.submit(_write_one, kv) for kv in images.items()]
            for fut in as_completed(futures):
                try:
                    key, row = fut.result()
                    key_meta[key] = row
                except Exception as e:
                    errors.append(str(e))

        if errors:
            # best-effort cleanup
            for p in written_files:
                try:
                    if p.exists():
                        p.unlink()
                except Exception:
                    pass
            raise RuntimeError("One or more image writes failed:\n" + "\n".join(errors))

        return key_meta

    @staticmethod
    def _build_images_rows(key_meta: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        rows = []
        for _k, m in key_meta.items():
            rows.append({
                "image_id":     m["image_id"],
                "uri":          m["uri"],
                "modality":     m.get("modality"),
                "dataset_name": m.get("dataset_name"),
                "alias":        m.get("alias"),
                "extra":        m.get("extra", json.dumps({}, ensure_ascii=False, sort_keys=True)),
            })
        return pd.DataFrame(rows, columns=["image_id", "uri", "modality", "dataset_name", "alias", "extra"])

    @staticmethod
    def _map_names_to_ids(
        names: List[str],
        key_to_id: Dict[str, str],
        alias_to_id: Dict[str, str],
        fallback_all_ids: List[str],
    ) -> List[str]:
        """将 relation.image_names（可能是键名或别名）映射为 image_ids；若未提供 names 则返回全部."""
        if not names:
            return list(fallback_all_ids)
        ids: List[str] = []
        for n in names:
            if n in key_to_id:
                ids.append(key_to_id[n])
            elif n in alias_to_id:
                ids.append(alias_to_id[n])
            else:
                raise ValueError(f"relation references unknown image name/alias: '{n}'")
        return ids

    @staticmethod
    def _build_relation_rows(
        relations: Dict[str, Dict[str, Any]],
        keys_order: List[str],
        key_to_id: Dict[str, str],
        alias_to_id: Dict[str, str],
    ) -> pd.DataFrame:
        """relations 仅保存 payload；payload 会加入按 image_names 顺序映射出的 image_ids。
           若 relation 缺少 image_names，则默认使用全部 images（按 keys_order 顺序）。
        """
        all_ids_in_order = [key_to_id[k] for k in keys_order]
        rows = []
        for rel_name, rel in relations.items():
            rid = uuid.uuid4().hex
            body = dict(rel)  # 复制，避免修改调用方
            names = body.pop("image_names", None)
            names = list(names) if names is not None else []
            image_ids = DatasetWriter._map_names_to_ids(names, key_to_id, alias_to_id, all_ids_in_order)
            body["image_ids"] = image_ids

            payload = json.dumps(body, ensure_ascii=False, sort_keys=True)
            rows.append({"relation_id": rid, "payload": payload, "__rel_name": rel_name})
        return pd.DataFrame(rows, columns=["relation_id", "payload", "__rel_name"])

    @staticmethod
    def _build_protocol_rows(
        protocols: Dict[str, List[str]],
        rel_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """protocol_name 与 relation_set 均由 adaptor 的字典键提供；二者可相同或不同。"""
        name_to_id = {row["__rel_name"]: row["relation_id"] for _, row in rel_df.iterrows()}
        rows = []
        for relation_set, rel_names in protocols.items():
            for rn in rel_names:
                rid = name_to_id[rn]
                rows.append({
                    "protocol_name": relation_set,
                    "relation_id":   rid,
                    "relation_set":  relation_set,
                })
        return pd.DataFrame(rows, columns=["protocol_name", "relation_id", "relation_set"])

    def _persist_metadata(
        self,
        images_df: pd.DataFrame,
        rel_df: pd.DataFrame,
        proto_df: pd.DataFrame,
    ) -> None:
        rel_insert_df = rel_df.drop(columns=["__rel_name"]) if "__rel_name" in rel_df.columns else rel_df

        self.conn.execute("BEGIN;")
        try:
            if not images_df.empty:
                self.conn.register("images_df", images_df)
                self.conn.execute("INSERT INTO images SELECT * FROM images_df;")

            self.conn.register("rel_df", rel_insert_df)
            self.conn.execute("INSERT INTO relations SELECT * FROM rel_df;")

            if not proto_df.empty:
                self.conn.register("proto_df", proto_df)
                self.conn.execute("INSERT INTO protocol SELECT * FROM proto_df;")

            self.conn.execute("COMMIT;")
        except Exception:
            self.conn.execute("ROLLBACK;")
            raise


# -----------------------
# Backwards-compatible entry point
# -----------------------
def write(config: dict, *, dry_run: bool = False, max_workers: int = 8, show_progress: bool = True) -> None:
    writer = DatasetWriter(
        database_root=config["database_root"],
        max_workers=max_workers,
    )
    writer.write_from_registry(dry_run=dry_run, show_progress=show_progress)


if __name__ == "__main__":
    cfg = GetDatabaseConfig()
    write(cfg, dry_run=False, max_workers=max(8, (os.cpu_count() or 8)), show_progress=True)
