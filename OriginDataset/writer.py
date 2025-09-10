# OriginDataset/writer.py
from __future__ import annotations

import json
import uuid
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

import numpy as np
import pandas as pd

try:
    from OriginDataset.utils import ensure_dir
except Exception:  # pragma: no cover
    from utils import ensure_dir  # type: ignore

# 注册器

from OriginDataset.base import BaseAdaptor
from OriginDataset.registry import discover_and_register, REGISTRY



class DatasetWriter:
    """
    仅负责“写入”；不创建连接/不初始化 schema。
    由调用方传入 DuckDB 连接（BigVisionDatabase 管理）。
    """

    def __init__(
        self,
        *,
        conn,  # duckdb.DuckDBPyConnection
        database_root: str,
        max_workers: int = 8,
    ) -> None:
        self.conn = conn
        self.root = Path(database_root)
        self.images_root = self.root / "images"
        ensure_dir(self.images_root)
        self.max_workers = max(1, int(max_workers))

    # ---------------- 公共 API ----------------

    def write_dataset(
        self,
        images: Dict[str, Any],               # {key|alias: {'image': ndarray/tensor, 'dataset_name': str, 'modality'?: str, 'alias'?: str, 'extra'?: dict}}
        relations: Dict[str, Dict[str, Any]], # {relation_name: {'image_names'?: [...], 其它业务键...}}
        protocols: Dict[str, List[str]],      # {protocol_name(or relation_set): [relation_name, ...]}
        *,
        dry_run: bool = False,
    ) -> Dict[str, int]:
        """
        写入单个 adaptor item；返回 {'imgs': X, 'rels': Y, 'proto': Z}
        """
        # 0) 预创建目录
        keys_order = list(images.keys())
        for k in keys_order:
            ds = _per_image_dataset_name(images[k])
            ensure_dir(self.images_root / ds)

        # 1) 写图片
        key_meta = self._write_images_concurrently(images) if not dry_run else _fake_meta(images)

        key_to_id   = {k: m["image_id"] for k, m in key_meta.items()}
        alias_to_id = {m.get("alias", k): m["image_id"] for k, m in key_meta.items()}

        # 2) 组装 DF
        images_df = _build_images_rows(key_meta)
        rel_df    = _build_relation_rows(relations, keys_order, key_to_id, alias_to_id)
        proto_df  = _build_protocol_rows(protocols, rel_df)

        # 3) 事务持久化
        if not dry_run:
            self._persist_metadata(images_df, rel_df, proto_df)

        return {
            "imgs": len(images_df),
            "rels": len(rel_df),
            "proto": len(proto_df),
        }

    def write_from_registry(self, *, dry_run: bool = False, show_progress: bool = True) -> None:
        """
        扫描并运行各 adaptor，将数据写入。
        进度显示（每 adaptor 一条 tqdm）在此方法内部处理，避免刷屏。
        """
        try:
            from tqdm import tqdm  # 局部依赖，可选安装
        except Exception:
            tqdm = None  # type: ignore

        discover_and_register(
            root_package="OriginDataset",
            target_filename="adaptor.py",
            class_name="Adaptor",
            base_class=BaseAdaptor,
        )

        for name, adaptor_cls in REGISTRY.items():
            adaptor = adaptor_cls()
            label = getattr(adaptor_cls, "__plugin_name__", name)
            total = _estimate_total(adaptor) if show_progress else None

            pbar = None
            if tqdm and show_progress:
                pbar = tqdm(total=total, desc=label, unit="item", dynamic_ncols=True, leave=True)

            agg = {"imgs": 0, "rels": 0, "proto": 0, "err": 0}
            try:
                for _idx, data in enumerate(adaptor):
                    try:
                        s = self.write_dataset(
                            images=data["images"],
                            relations=data["relation"],
                            protocols=data["protocol"],
                            dry_run=dry_run,
                        )
                        agg["imgs"]  += s.get("imgs", 0)
                        agg["rels"]  += s.get("rels", 0)
                        agg["proto"] += s.get("proto", 0)
                    except Exception:
                        agg["err"] += 1
                    finally:
                        if pbar:
                            pbar.update(1)
                            pbar.set_postfix(imgs=agg["imgs"], rels=agg["rels"], proto=agg["proto"], err=agg["err"], refresh=False)
            finally:
                if pbar:
                    pbar.close()

    # ---------------- 内部：写图片 ----------------

    def _write_images_concurrently(
        self,
        images: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        written_files: List[Path] = []

        def _write_one(item):
            key, val = item
            arr = _as_ndarray(val)
            ds = _per_image_dataset_name(val)
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
                "modality":     _modality_from(key, val),
                "dataset_name": ds,
                "alias":        _alias_from(key, val),
                "extra":        _extra_from(val),
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
            # best-effort 清理
            for p in written_files:
                try:
                    if p.exists():
                        p.unlink()
                except Exception:
                    pass
            raise RuntimeError("One or more image writes failed:\n" + "\n".join(errors))
        return key_meta

    # ---------------- 内部：持久化 ----------------

    def _persist_metadata(
        self,
        images_df: pd.DataFrame,
        rel_df: pd.DataFrame,
        proto_df: pd.DataFrame,
    ) -> None:
        # relations DF 含 __rel_name（用于构造 protocol），落库前移除
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


# ---------------- 辅助方法（与 DB 解耦） ----------------

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

def _per_image_dataset_name(val: Any) -> str:
    if not isinstance(val, dict) or not val.get("dataset_name"):
        raise ValueError("Each images[...] value must provide 'dataset_name'.")
    return str(val["dataset_name"])

def _modality_from(key: str, val: Any) -> str:
    if isinstance(val, dict) and val.get("modality"):
        return str(val["modality"])
    return str(key)

def _alias_from(key: str, val: Any) -> str:
    if isinstance(val, dict) and val.get("alias"):
        return str(val["alias"])
    return str(key)

def _extra_from(val: Any) -> str:
    extra_obj: Dict[str, Any] = {}
    if isinstance(val, dict) and isinstance(val.get("extra"), dict):
        extra_obj.update(val["extra"])
    return json.dumps(extra_obj, ensure_ascii=False, sort_keys=True)

def _fake_meta(images: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    dry_run 模式下的元信息构造：不落盘，仅模拟 image_id/uri 等字段。
    """
    key_meta: Dict[str, Dict[str, Any]] = {}
    for key, val in images.items():
        ds = _per_image_dataset_name(val)
        img_id = uuid.uuid4().hex
        key_meta[key] = {
            "image_id":     img_id,
            "uri":          f"images/{ds}/{img_id}.npy",
            "modality":     _modality_from(key, val),
            "dataset_name": ds,
            "alias":        _alias_from(key, val),
            "extra":        _extra_from(val),
        }
    return key_meta

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

def _map_names_to_ids(
    names: List[str],
    key_to_id: Dict[str, str],
    alias_to_id: Dict[str, str],
    fallback_all_ids: List[str],
) -> List[str]:
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

def _build_relation_rows(
    relations: Dict[str, Dict[str, Any]],
    keys_order: List[str],
    key_to_id: Dict[str, str],
    alias_to_id: Dict[str, str],
) -> pd.DataFrame:
    """将 relation.image_names（可为键名或 alias）映射为 image_ids；未提供则用全部 images（按 keys_order 保序）"""
    all_ids_in_order = [key_to_id[k] for k in keys_order]
    rows = []
    for rel_name, rel in relations.items():
        rid = uuid.uuid4().hex
        body = dict(rel)
        names = body.get("image_names", None)
        names = list(names) if names is not None else []
        image_ids = _map_names_to_ids(names, key_to_id, alias_to_id, all_ids_in_order)
        body["image_ids"] = image_ids

        payload = json.dumps(body, ensure_ascii=False, sort_keys=True)
        rows.append({"relation_id": rid, "payload": payload, "__rel_name": rel_name})
    return pd.DataFrame(rows, columns=["relation_id", "payload", "__rel_name"])

def _build_protocol_rows(
    protocols: Dict[str, List[str]],
    rel_df: pd.DataFrame,
) -> pd.DataFrame:
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

def _estimate_total(adaptor: Any) -> int | None:
    try:
        return len(adaptor)
    except Exception:
        pass
    ds = getattr(adaptor, "dataset", None)
    if ds is not None:
        try:
            return len(ds)
        except Exception:
            pass
    return None
