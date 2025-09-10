#!/usr/bin/env python3
"""
Dataset/loader.py — Load samples directly from the BigVisionDatabase without exporting

This module provides a PyTorch-compatible Dataset that reads image samples
straight from the database/NPY store using the project conventions:

<db_root>/
  images/<dataset_name>/<image_id>.npy
  db/catalog.duckdb

Each sample corresponds to a row in `relations` that is linked to a protocol via
`protocol(protocol_name, relation_id, relation_set)`.

The relation payload (JSON) MUST contain `image_ids: List[str]`. Optional fields
like `task_type`, `annotation`, `extra`, `image_aliases`, `image_names` are
propagated into the returned sample metadata.

Key design notes:
- The DuckDB connection is used only during index construction; it is closed
  afterwards so the dataset can be safely used with multiple DataLoader workers
  (read-only path lookups, no DB writes).
- Images are loaded lazily from NPY on __getitem__. Optional memmap is used to
  reduce memory if desired.
- Color handling mirrors project defaults: NPY is stored as-is; if arrays are
  3-channel and `color_order='bgr'`, we convert BGR→RGB on read to avoid "blue faces".
- Transform hooks allow per-image and per-sample processing; dtype/normalization
  are configurable.

Dependencies: duckdb, numpy, (optional) torch
"""
from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import duckdb
import numpy as np

# ---------------------------------
# Config resolver helpers (module-level)
# ---------------------------------

def _find_and_import_GetDatabaseConfig():
    """Try multiple strategies to import Config.setting.GetDatabaseConfig.
    Returns the callable on success; raises on failure.
    """
    try:
        from Config.setting import GetDatabaseConfig  # type: ignore
        return GetDatabaseConfig
    except Exception:
        pass
    try:
        from BigVisionDatabase.Config.setting import GetDatabaseConfig  # type: ignore
        return GetDatabaseConfig
    except Exception:
        pass
    import sys, os as _os
    here = _os.path.abspath(_os.path.dirname(__file__))
    candidates = [
        here,
        _os.path.dirname(here),
        _os.path.dirname(_os.path.dirname(here)),
        _os.getcwd(),
    ]
    for root in candidates:
        cfg_py = _os.path.join(root, 'Config', 'setting.py')
        if _os.path.exists(cfg_py):
            if root not in sys.path:
                sys.path.insert(0, root)
            try:
                from Config.setting import GetDatabaseConfig  # type: ignore
                return GetDatabaseConfig
            except Exception:
                continue
    raise ImportError("Config.setting.GetDatabaseConfig not found nearby; ensure your project root (containing 'Config/') is on PYTHONPATH, or pass db_root explicitly.")

try:
    import torch
    from torch.utils.data import Dataset
    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    class Dataset:  # minimal shim to keep type hints readable
        pass
    TORCH_AVAILABLE = False


# -----------------------------
# Data structures & utilities
# -----------------------------

@dataclass
class ImageMeta:
    image_id: str
    dataset_name: str
    alias: Optional[str] = None
    modality: Optional[str] = None
    uri: Optional[str] = None
    extra: Optional[str] = None  # raw JSON string from DB

    def npy_path(self, db_root: str) -> str:
        return os.path.join(db_root, 'images', self.dataset_name, f'{self.image_id}.npy')


@dataclass
class SampleIndex:
    relation_id: str
    image_ids: List[str]
    image_aliases: Optional[List[str]] = None
    image_names: Optional[List[str]] = None
    payload: Dict[str, Any] = None  # parsed JSON


def _chunked(seq: Sequence[Any], n: int) -> Iterable[Sequence[Any]]:
    for i in range(0, len(seq), n):
        yield seq[i:i+n]


# -----------------------------
# Main Dataset
# -----------------------------

class DBProtocolDataset(Dataset):
    """Read samples from DB by protocol name.

    Args:
        db_root: Database root directory.
        protocol_name: Name of the protocol to load relations from.
        duckdb_path: Explicit path to DuckDB file. Defaults to <db_root>/db/catalog.duckdb.
        limit: Optional cap on number of relations (for quick smoke tests).
        offset: Start offset when reading relations for the protocol.
        color_order: One of {"bgr", "rgb", None}. If "bgr", 3-channel arrays will be
            converted to RGB for output. If "rgb", arrays are left as-is. If None,
            never reorder channels.
        normalize: If True and dtype is integer-like, converts to float32/float_dtype
            and scales to [0, 1].
        to_tensor: If True and torch is available, outputs torch.Tensor per image.
        float_dtype: torch dtype when normalizing (default: torch.float32).
        mmap: Use numpy memmap on load (reduces peak memory, disables in-memory caching).
        strict: If True, missing files raise an error; otherwise, logs warning and
            substitutes zeros of the right shape when possible.
        image_transform: Optional callable(img: np.ndarray | torch.Tensor) -> same-type
        sample_transform: Optional callable(sample_dict) -> sample_dict
        cache_images: If True, cache decoded arrays in-process (ignored if mmap=True).
        cache_size: Max number of decoded images to cache when cache_images=True.

    Returns from __getitem__ (dict):
        {
            'relation_id': str,
            'images': List[np.ndarray | torch.Tensor],
            'image_ids': List[str],
            'aliases': List[str | None],
            'modalities': List[str | None],
            'payload': Dict[str, Any],  # full parsed payload
        }
    """

    def __init__(
        self,
        protocol_name: str,
        db_root: Optional[str] = None,
        duckdb_path: Optional[str] = None,
        limit: Optional[int] = None,
        offset: int = 0,
        color_order: Optional[str] = 'bgr',
        normalize: bool = False,
        to_tensor: bool = True,
        float_dtype: Optional['torch.dtype'] = None,
        mmap: bool = True,
        strict: bool = True,
        image_transform: Optional[Callable[[Any], Any]] = None,
        sample_transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        cache_images: bool = False,
        cache_size: int = 128,
    ) -> None:
        # Resolve db_root/duckdb_path: default to Config, allow override/env fallback
        if db_root is None:
            try:
                GetDatabaseConfig = _find_and_import_GetDatabaseConfig()
                cfg = GetDatabaseConfig()
                root = (
                    (cfg.get('database_root') if hasattr(cfg, 'get') else None)
                    or getattr(cfg, 'database_root', None)
                    or (cfg.get('paths', {}).get('database_root') if hasattr(cfg, 'get') else None)
                )
                if not root:
                    raise ValueError('database_root not found in config')
            except Exception:
                env_root = os.getenv('DB_ROOT')
                if not env_root:
                    raise
                root = env_root
        else:
            root = db_root
        self.db_root = os.path.abspath(root)
        self.duckdb_path = duckdb_path or os.path.join(self.db_root, 'db', 'catalog.duckdb')
        self.protocol_name = protocol_name
        self.limit = limit
        self.offset = offset
        self.color_order = color_order
        self.normalize = normalize
        self.to_tensor = to_tensor and TORCH_AVAILABLE
        self.float_dtype = float_dtype or (torch.float32 if TORCH_AVAILABLE else None)
        self.mmap = mmap
        self.strict = strict
        self.image_transform = image_transform
        self.sample_transform = sample_transform
        self.cache_images = cache_images and not mmap
        self.cache_size = max(1, int(cache_size))

        if self.to_tensor and not TORCH_AVAILABLE:
            warnings.warn("torch not available; falling back to numpy outputs")
            self.to_tensor = False

        # Build index and image metadata map, then close DB connection.
        self._samples: List[SampleIndex] = []
        self._imgmeta: Dict[str, ImageMeta] = {}
        self._build_index()

        # Setup a tiny LRU cache if requested
        self._cache: Dict[str, Any] = {}
        self._cache_order: List[str] = []

    # -----------------------------
    # Index construction
    # -----------------------------
    def _build_index(self) -> None:
        if not os.path.exists(self.duckdb_path):
            raise FileNotFoundError(f"DuckDB not found: {self.duckdb_path}")
        con = duckdb.connect(self.duckdb_path, read_only=True)
        try:
            # Pull relations for the protocol
            q = (
                "SELECT r.relation_id, r.payload "
                "FROM protocol p JOIN relations r ON p.relation_id = r.relation_id "
                "WHERE p.protocol_name = ? "
                "ORDER BY r.relation_id"
            )
            if self.limit is not None:
                q += " LIMIT ? OFFSET ?"
                rows = con.execute(q, [self.protocol_name, self.limit, self.offset]).fetchall()
            else:
                rows = con.execute(q, [self.protocol_name]).fetchall()

            if not rows:
                warnings.warn(f"No relations found for protocol '{self.protocol_name}'.")

            # Parse payloads and collect all image_ids
            all_image_ids: List[str] = []
            for relation_id, payload_json in rows:
                try:
                    payload = json.loads(payload_json)
                except Exception as e:
                    raise ValueError(f"Invalid payload JSON for relation {relation_id}: {e}")
                image_ids = payload.get('image_ids')
                if not isinstance(image_ids, list) or not all(isinstance(x, str) for x in image_ids):
                    raise ValueError(
                        f"relation {relation_id} payload must contain 'image_ids: List[str]'"
                    )
                si = SampleIndex(
                    relation_id=relation_id,
                    image_ids=list(image_ids),
                    image_aliases=payload.get('image_aliases'),
                    image_names=payload.get('image_names'),
                    payload=payload,
                )
                self._samples.append(si)
                all_image_ids.extend(image_ids)

            uniq_ids = sorted(set(all_image_ids))

            # Fetch image metadata in chunks to avoid oversized IN lists
            if uniq_ids:
                for chunk in _chunked(uniq_ids, 1000):
                    placeholders = ','.join(['?'] * len(chunk))
                    q_img = (
                        f"SELECT image_id, dataset_name, alias, modality, uri, extra "
                        f"FROM images WHERE image_id IN ({placeholders})"
                    )
                    for row in con.execute(q_img, list(chunk)).fetchall():
                        image_id, dataset_name, alias, modality, uri, extra = row
                        self._imgmeta[image_id] = ImageMeta(
                            image_id=image_id,
                            dataset_name=dataset_name,
                            alias=alias,
                            modality=modality,
                            uri=uri,
                            extra=extra,
                        )

            # Sanity checks for missing image rows
            missing_meta = [iid for iid in uniq_ids if iid not in self._imgmeta]
            if missing_meta:
                msg = f"{len(missing_meta)} image_id(s) referenced in relations but missing in images table. Example: {missing_meta[:5]}"
                if self.strict:
                    raise KeyError(msg)
                else:
                    warnings.warn(msg)
        finally:
            try:
                con.close()
            except Exception:
                pass

    # -----------------------------
    # Dataset protocol
    # -----------------------------
    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self._samples[idx]
        imgs: List[Any] = []
        aliases: List[Optional[str]] = []
        modalities: List[Optional[str]] = []

        for i, iid in enumerate(s.image_ids):
            meta = self._imgmeta.get(iid)
            if meta is None:
                if self.strict:
                    raise KeyError(f"image_id not found in images table: {iid}")
                else:
                    warnings.warn(f"Missing image row for {iid}; substituting zeros")
                    # Try best-effort zeros with unknown shape
                    imgs.append(self._zeros_like_unknown())
                    aliases.append(None)
                    modalities.append(None)
                    continue

            path = meta.npy_path(self.db_root)
            arr = self._load_npy(path)

            # Color handling
            arr = self._apply_color(arr)  # may flip channels; guarantees contiguous if flipped

            # Normalize / dtype
            arr = self._maybe_normalize(arr)

            # to tensor if requested
            if self.to_tensor:
                arr = self._to_tensor(arr)

            # per-image transform
            if self.image_transform is not None:
                arr = self.image_transform(arr)

            imgs.append(arr)

            # resolve alias priority for observation only (export rules are for Compact)
            alias = None
            if s.image_aliases and i < len(s.image_aliases):
                alias = s.image_aliases[i]
            elif s.image_names and i < len(s.image_names):
                alias = s.image_names[i]
            elif meta.alias:
                alias = meta.alias
            elif meta.modality:
                alias = meta.modality
            else:
                alias = f"img{i}"
            aliases.append(alias)
            modalities.append(meta.modality)

        sample = {
            'relation_id': s.relation_id,
            'images': imgs,
            'image_ids': list(s.image_ids),
            'aliases': aliases,
            'modalities': modalities,
            'payload': s.payload,
        }

        if self.sample_transform is not None:
            sample = self.sample_transform(sample)
        return sample

    # -----------------------------
    # Loading / transforms
    # -----------------------------
    def _load_npy(self, path: str) -> np.ndarray:
        if self.cache_images:
            cached = self._cache_get(path)
            if cached is not None:
                return cached
        try:
            arr = np.load(path, mmap_mode='r' if self.mmap else None)
            if self.mmap:
                # If memmap, keep as-is. If caching, copy into RAM so it's independent.
                out = arr
            else:
                out = np.array(arr)  # ensure regular ndarray
            if self.cache_images:
                self._cache_put(path, out if isinstance(out, np.ndarray) else np.array(out))
            return out
        except FileNotFoundError as e:
            if self.strict:
                raise
            warnings.warn(f"File missing: {path}; substituting zeros")
            return self._zeros_like_unknown()

    def _apply_color(self, arr: np.ndarray) -> np.ndarray:
        if self.color_order == 'bgr' and arr.ndim >= 3 and arr.shape[-1] == 3:
            # BGR → RGB
            arr = arr[..., ::-1].copy()  # ensure positive strides / contiguous
        # If 'rgb' or None: leave as-is
        return arr

    def _maybe_normalize(self, arr: np.ndarray) -> np.ndarray:
        if not self.normalize:
            return arr
        if np.issubdtype(arr.dtype, np.floating):
            return arr.astype(np.float32, copy=False)
        # assume integer-like, map to [0,1]
        info = np.iinfo(arr.dtype) if np.issubdtype(arr.dtype, np.integer) else None
        denom = (info.max if info else 255)
        return (arr.astype(np.float32) / float(denom))

    def _to_tensor(self, arr: np.ndarray):
        assert TORCH_AVAILABLE
        t = torch.from_numpy(np.ascontiguousarray(arr))
        if self.normalize:
            t = t.to(self.float_dtype or torch.float32)
        return t

    def _zeros_like_unknown(self) -> np.ndarray:
        # Fallback shape (1,1) if we truly don't know. Users should set strict=True in prod.
        return np.zeros((1, 1), dtype=np.uint8)

    # -----------------------------
    # Tiny LRU cache for decoded NPY
    # -----------------------------
    def _cache_get(self, key: str):
        if key in self._cache:
            # move to end
            self._cache_order.remove(key)
            self._cache_order.append(key)
            return self._cache[key]
        return None

    def _cache_put(self, key: str, value: Any) -> None:
        if key in self._cache:
            try:
                self._cache_order.remove(key)
            except ValueError:
                pass
        self._cache[key] = value
        self._cache_order.append(key)
        while len(self._cache_order) > self.cache_size:
            old = self._cache_order.pop(0)
            self._cache.pop(old, None)

    # -----------------------------
    # Collate helpers
    # -----------------------------
    @staticmethod
    def collate_batch(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """A flexible collate_fn that tolerates variable #images per relation.

        - If all samples have the same #images AND same image shapes, images are
          stacked along a new batch dimension (B, ...).
        - Otherwise, images remain a list per sample.
        - Metadata fields are batched as lists.
        """
        if not samples:
            return {}

        # detect consistent shapes
        def img_shapes(s):
            return [tuple(getattr(img, 'shape', None) or getattr(img, 'size', None)) for img in s['images']]

        same_count = len({len(s['images']) for s in samples}) == 1
        can_stack = False
        if same_count:
            shapes0 = img_shapes(samples[0])
            can_stack = all(img_shapes(s) == shapes0 for s in samples)

        batch: Dict[str, Any] = {
            'relation_id': [s['relation_id'] for s in samples],
            'image_ids': [s['image_ids'] for s in samples],
            'aliases': [s['aliases'] for s in samples],
            'modalities': [s['modalities'] for s in samples],
            'payload': [s['payload'] for s in samples],
        }

        if can_stack and TORCH_AVAILABLE and isinstance(samples[0]['images'][0], torch.Tensor):
            # stack per position, then stack across batch -> List[Tensor] of length K
            k = len(samples[0]['images'])
            images_per_pos: List[List[torch.Tensor]] = [[] for _ in range(k)]
            for s in samples:
                for i, img in enumerate(s['images']):
                    images_per_pos[i].append(img)
            batch['images'] = [torch.stack(images_per_pos[i], dim=0) for i in range(k)]
        else:
            batch['images'] = [s['images'] for s in samples]

        return batch


# -----------------------------
# Minimal CLI self-check
# -----------------------------
if __name__ == '__main__':  # pragma: no cover
    import argparse
    import sys
    import os

    def _find_and_import_GetDatabaseConfig():
        """Try multiple strategies to import Config.setting.GetDatabaseConfig.
        Search common roots relative to this file and CWD if direct import fails.
        Returns the imported callable on success; raises on failure.
        """
        # 0) direct import as-is
        try:
            from Config.setting import GetDatabaseConfig  # type: ignore
            return GetDatabaseConfig
        except Exception:
            pass
        # 1) sometimes the package lives under BigVisionDatabase.Config
        try:
            from BigVisionDatabase.Config.setting import GetDatabaseConfig  # type: ignore
            return GetDatabaseConfig
        except Exception:
            pass
        # 2) walk up and add candidates to sys.path
        here = os.path.abspath(os.path.dirname(__file__))
        candidates = [
            here,
            os.path.dirname(here),                # e.g. .../src
            os.path.dirname(os.path.dirname(here)),  # e.g. project root
            os.getcwd(),                          # current working dir
        ]
        for root in candidates:
            cfg_py = os.path.join(root, 'Config', 'setting.py')
            if os.path.exists(cfg_py):
                if root not in sys.path:
                    sys.path.insert(0, root)
                try:
                    from Config.setting import GetDatabaseConfig  # type: ignore
                    return GetDatabaseConfig
                except Exception:
                    continue
        raise ImportError("Config.setting.GetDatabaseConfig not found on sys.path or nearby; ensure your project root (containing 'Config/') is on PYTHONPATH, or provide --db-root.")

    def _resolve_db_paths(cli_db_root: Optional[str], cli_duck: Optional[str]) -> Tuple[str, str]:
        """Resolve <db_root> and duckdb_path with the following priority:
        1) CLI --db-root / --duckdb-path
        2) ENV DB_ROOT
        3) Config.setting.GetDatabaseConfig().database_root
        """
        # 1) CLI
        if cli_db_root:
            root = cli_db_root
        else:
            # 2) ENV
            env_root = os.getenv('DB_ROOT')
            if env_root:
                root = env_root
            else:
                # 3) Config (robust import)
                GetDatabaseConfig = _find_and_import_GetDatabaseConfig()
                cfg = GetDatabaseConfig()
                # allow either dict-like or object-like access
                root = (
                    (cfg.get('database_root') if hasattr(cfg, 'get') else None)
                    or getattr(cfg, 'database_root', None)
                )
                if not root:
                    # try nested structure if any
                    root = (
                        cfg.get('paths', {}).get('database_root')
                        if hasattr(cfg, 'get') else None
                    )
                if not root:
                    raise ValueError('database_root not found in config')
        duck = cli_duck or os.path.join(root, 'db', 'catalog.duckdb')
        return root, duck

    parser = argparse.ArgumentParser(description='DB loader smoke test')
    parser.add_argument('--db-root', required=False, help='Path to <database_root>. If omitted, falls back to DB_ROOT env or Config.setting.GetDatabaseConfig().')
    parser.add_argument('--duckdb-path', required=False, help='Optional explicit path to DuckDB file; defaults to <db_root>/db/catalog.duckdb')
    parser.add_argument('--protocol', required=True, help='Protocol name to read')
    parser.add_argument('--limit', type=int, default=4)
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--mmap', action='store_true', default=False)
    parser.add_argument('--no-tensor', action='store_true', default=False)
    parser.add_argument('--normalize', action='store_true', default=False)
    parser.add_argument('--color-order', default='bgr', choices=['bgr','rgb','none'], help="Interpretation of 3-channel NPY ('bgr' converts to RGB on read; 'none' disables reordering)")
    parser.add_argument('--strict', dest='strict', action='store_true', default=True)
    parser.add_argument('--no-strict', dest='strict', action='store_false')
    args = parser.parse_args()

    db_root, duckdb_path = _resolve_db_paths(args.db_root, args.duckdb_path)

    ds = DBProtocolDataset(
        db_root=db_root,
        protocol_name=args.protocol,
        duckdb_path=duckdb_path,
        limit=args.limit,
        offset=args.offset,
        mmap=args.mmap,
        to_tensor=not args.no_tensor,
        normalize=args.normalize,
        color_order=(None if args.color_order == 'none' else args.color_order),
        strict=args.strict,
    )
    print(f"DB root: {db_root}\nDuckDB:   {duckdb_path}")
    print(f"Loaded {len(ds)} relations from protocol '{args.protocol}' (limit={args.limit}, offset={args.offset})")
    for i in range(min(len(ds), args.limit)):
        s = ds[i]
        imgs = s['images']
        if TORCH_AVAILABLE and isinstance(imgs[0], torch.Tensor):
            shapes = [tuple(img.shape) for img in imgs]
            dtypes = [str(img.dtype) for img in imgs]
        else:
            shapes = [tuple(img.shape) for img in imgs]
            dtypes = [str(img.dtype) for img in imgs]
        print(f"#{i}: relation={s['relation_id']} n_imgs={len(imgs)} shapes={shapes} dtypes={dtypes}")
