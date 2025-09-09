# misc/loader.py
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


def _is_zip_path(p: str | Path) -> bool:
    p = str(p).lower()
    return p.endswith(".zip")


def _cache_dir_for_zip(zip_path: Path) -> Path:
    zp = zip_path.resolve()
    stat = zp.stat()
    key = f"{str(zp)}::{int(stat.st_mtime)}::{stat.st_size}"
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    base = Path(os.environ.get("BVB_COMPACT_CACHE", Path.home() / ".cache" / "bvb_compact"))
    return base / (zp.stem + "_" + h)


def _unzip_if_needed(zip_path: Path) -> Path:
    cache_root = _cache_dir_for_zip(zip_path)
    top_marker = cache_root / ".ok"
    if top_marker.exists():
        subdirs = [p for p in cache_root.iterdir() if p.is_dir()]
        if len(subdirs) == 1:
            return subdirs[0]
        return cache_root

    cache_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(str(zip_path), "r") as zf:
        zf.extractall(str(cache_root))
    top_marker.write_text("ok")

    subdirs = [p for p in cache_root.iterdir() if p.is_dir()]
    if len(subdirs) == 1:
        return subdirs[0]
    return cache_root


def _resolve_dataset_dir(path: str | Path) -> Path:
    p = Path(path)
    if p.is_file() and _is_zip_path(p):
        return _unzip_if_needed(p)
    if p.is_dir():
        return p
    raise FileNotFoundError(f"Invalid dataset path (not a folder or zip): {p}")


def _load_parquet_shards(samples_dir: Path) -> List[Path]:
    return sorted(samples_dir.glob("samples-*.parquet"))


def _safe_json_loads(s: Any) -> dict:
    if s is None:
        return {}
    if isinstance(s, float) and math.isnan(s):
        return {}
    if isinstance(s, dict):
        return s
    if isinstance(s, (bytes, bytearray)):
        try:
            s = s.decode("utf-8", "ignore")
        except Exception:
            return {}
    if isinstance(s, str):
        s = s.strip()
        if not s:
            return {}
        try:
            return json.loads(s)
        except Exception:
            return {}
    # 其他类型直接空字典
    return {}


def _ensure_list_str(x: Any) -> List[str]:
    """把 x 规范成 List[str]。支持 list/np.ndarray/pyarrow/标量/NaN/JSON 字符串。"""
    if x is None:
        return []
    if isinstance(x, float) and math.isnan(x):
        return []
    if isinstance(x, list):
        return [str(v) for v in x]
    if isinstance(x, np.ndarray):
        # 可能是一维数组
        try:
            return [str(v) for v in x.tolist()]
        except Exception:
            return [str(v) for v in x.reshape(-1).tolist()]
    # 如果意外是 JSON 字符串（很少见）
    if isinstance(x, (bytes, bytearray)):
        try:
            x = x.decode("utf-8", "ignore")
        except Exception:
            return []
    if isinstance(x, str):
        xs = x.strip()
        if xs.startswith("[") and xs.endswith("]"):
            try:
                arr = json.loads(xs)
                if isinstance(arr, list):
                    return [str(v) for v in arr]
            except Exception:
                pass
        # 其他情况当单元素
        return [xs] if xs else []
    # 单标量
    return [str(x)]


def verify_compact_dataset(dataset_dir_or_zip: str | Path) -> Dict[str, Any]:
    ds_dir = _resolve_dataset_dir(dataset_dir_or_zip)
    samples_dir = ds_dir / "samples"
    images_dir = ds_dir / "images"
    manifest = ds_dir / "manifest.json"

    shards = _load_parquet_shards(samples_dir)
    if not shards:
        return {"ok": False, "shards_found": [], "checksums_ok": None}

    checksums_ok: Optional[bool] = None
    if manifest.exists():
        try:
            manifest_obj = json.loads(manifest.read_text(encoding="utf-8"))
            chks = manifest_obj.get("checksums", {})
            ok_all = True
            for sh in shards:
                rel = sh.relative_to(ds_dir).as_posix()
                want = chks.get(rel)
                if not want or not want.startswith("sha256:"):
                    continue
                want_hex = want.split("sha256:", 1)[1]
                h = hashlib.sha256()
                with sh.open("rb") as f:
                    for chunk in iter(lambda: f.read(1024 * 1024), b""):
                        h.update(chunk)
                if h.hexdigest().lower() != want_hex.lower():
                    ok_all = False
                    break
            checksums_ok = ok_all
        except Exception:
            checksums_ok = None

    return {
        "ok": True,
        "shards_found": [p.name for p in shards],
        "images_dir_exists": images_dir.exists(),
        "checksums_ok": checksums_ok,
        "dataset_dir": str(ds_dir),
    }


class CompactDataset(Dataset):
    """
    读取 Compact 导出的训练就绪数据集：
      - images/ 下为原样 .npy（H×W×C 或 H×W）
      - samples/ 下为分片 Parquet，包含 image_paths、image_aliases、task_type、annotation、extra
    每个 __getitem__ 返回：
      {
        "images": { alias: np.ndarray, ... },
        "task_type": str | None,
        "annotation": dict,
        "extra": dict,
        "relation_id": str
      }
    """
    def __init__(self, dataset_dir_or_zip: str | Path, shard_limit: Optional[int] = None):
        self.root_dir = _resolve_dataset_dir(dataset_dir_or_zip)
        self.samples_dir = self.root_dir / "samples"
        self.images_root = self.root_dir / "images"

        self.shards = _load_parquet_shards(self.samples_dir)
        if not self.shards:
            raise FileNotFoundError(f"No shard files found at {self.samples_dir}/samples-*.parquet")
        if shard_limit is not None:
            self.shards = self.shards[: max(1, int(shard_limit))]

        # 预读全部分片（对象列保留为 Python 对象，以便统一规整）
        dfs = []
        for sh in self.shards:
            df = pd.read_parquet(sh)
            dfs.append(df)
        if dfs:
            self.df = pd.concat(dfs, ignore_index=True)
        else:
            self.df = pd.DataFrame(columns=[
                "protocol_name","relation_id","image_paths","image_aliases","task_type","annotation","extra"
            ])

        # 列规整：把 image_paths / image_aliases 统一成 Python List[str]
        if "image_paths" in self.df.columns:
            self.df["image_paths"] = self.df["image_paths"].apply(_ensure_list_str)
        else:
            self.df["image_paths"] = [[] for _ in range(len(self.df))]
        if "image_aliases" in self.df.columns:
            self.df["image_aliases"] = self.df["image_aliases"].apply(_ensure_list_str)
        else:
            self.df["image_aliases"] = [[] for _ in range(len(self.df))]

    def __len__(self) -> int:
        return len(self.df)

    def _load_npy(self, rel_path: str) -> np.ndarray:
        p = (self.root_dir / rel_path).resolve()
        return np.load(p, allow_pickle=False)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]

        paths = row["image_paths"]
        aliases = row["image_aliases"]
        if not isinstance(paths, list):
            paths = _ensure_list_str(paths)
        if not isinstance(aliases, list):
            aliases = _ensure_list_str(aliases)
        if len(aliases) != len(paths):
            aliases = [f"img{i}" for i in range(len(paths))]

        imgs: Dict[str, np.ndarray] = {}
        for a, relp in zip(aliases, paths):
            imgs[str(a)] = self._load_npy(str(relp))

        return {
            "images": imgs,
            "task_type": row.get("task_type"),
            "annotation": _safe_json_loads(row.get("annotation")),
            "extra": _safe_json_loads(row.get("extra")),
            "relation_id": row.get("relation_id"),
        }


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Test loader for BigVision Compact dataset (dir or zip)")
    ap.add_argument("dataset_path", type=str, help="Path to <protocol>.compact/ or .zip")
    ap.add_argument("-b", "--batch-size", type=int, default=4)
    ap.add_argument("-n", "--num-workers", type=int, default=0)
    ap.add_argument("--limit-shards", type=int, default=None, help="Only load first N shards")
    ap.add_argument("--no-verify", action="store_true")
    return ap


if __name__ == "__main__":
    args = _build_argparser().parse_args()

    if not args.no_verify:
        v = verify_compact_dataset(args.dataset_path)
        print("[verify]", json.dumps(v, ensure_ascii=False))
        if not v.get("ok", False):
            print("[verify] failed; continue anyway...", file=sys.stderr)

    ds = CompactDataset(
        dataset_dir_or_zip=args.dataset_path,
        shard_limit=args.limit_shards,
    )
    print(f"[dataset] len={len(ds)} shards={len(ds.shards)}")

    dl = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, collate_fn=lambda x: x)
    for i, batch in enumerate(dl):
        if i >= 2:
            break
        first = batch[0] if batch else {}
        print(f"[batch {i}] size={len(batch)} keys={list(first.keys()) if isinstance(first, dict) else type(first)}")
