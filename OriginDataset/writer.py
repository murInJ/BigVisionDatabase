import os
import uuid
import json
import traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Tuple, List

import cv2
import pandas as pd

try:
    # Support both package and flat script layouts
    from .utils_duckdb import ensure_dir, rm_tree, move_to_quarantine, atomic_replace_dir, require_duckdb  # type: ignore
except Exception:  # pragma: no cover
    from utils_duckdb import ensure_dir, rm_tree, move_to_quarantine, atomic_replace_dir, require_duckdb  # type: ignore

from Config.setting import GetDatabaseConfig
from OriginDataset.base import BaseAdaptor
from registry import discover_and_register, REGISTRY


class DatasetWriter:
    """High-performance dataset writer storing *all metadata in DuckDB*.

    Storage layout under `database_root`:
      images/<dataset>/UUID.jpg
      db/catalog.duckdb  (relations & protocol_entries are stored here)
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
        self.trash_root = self.root / ".trash"
        self.max_workers = max_workers

        for p in [self.root, self.images_root, self.db_root, self.trash_root]:
            ensure_dir(p)

        threads = max(1, os.cpu_count() or 4)
        db_path = duckdb_path or str(self.db_root / "catalog.duckdb")
        self.conn = require_duckdb(db_path, threads)
        self._init_duckdb_schema()

    # -----------------------
    # Schema & DB helpers
    # -----------------------

    def _init_duckdb_schema(self) -> None:
        # Relations table
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS relations (
                relation_id TEXT PRIMARY KEY,
                dataset_name TEXT NOT NULL,
                payload JSON NOT NULL
            );
            """
        )
        # Protocol mapping table
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS protocol_entries (
                protocol_name TEXT NOT NULL,
                relation_id TEXT NOT NULL,
                relation_set TEXT NOT NULL
            );
            """
        )
        # Helpful indexes
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_relations_dataset ON relations(dataset_name);")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_protocols_set ON protocol_entries(relation_set);")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_protocols_name ON protocol_entries(protocol_name);")

    # -----------------------
    # Public API
    # -----------------------

    def write_dataset(
        self,
        dataset_name: str,
        images: Dict[str, "cv2.Mat"],
        relations: Dict[str, Dict],
        protocols: Dict[str, List[str]],
        *,
        force_rewrite: bool = False,
        dry_run: bool = False,
    ) -> None:
        """Write one dataset (images + relations + protocols). Always uses DuckDB.

        If `force_rewrite` is True, safely remove any existing data for this dataset before writing.
        """
        # Short-circuit if dataset already exists and not forcing rewrite
        if not force_rewrite and self._dataset_exists(dataset_name):
            msg = (
                f"[DRY-RUN] Would skip existing dataset '{dataset_name}'. Use force_rewrite=True to overwrite."
                if dry_run
                else f"[SKIP] Dataset '{dataset_name}' already exists. Use force_rewrite=True to overwrite."
            )
            print(msg)
            return

        dataset_images_final = self.images_root / dataset_name
        dataset_images_tmp = self.images_root / f"{dataset_name}__tmp-{uuid.uuid4().hex[:8]}"

        quarantine_dir = None

        # 1) If forcing rewrite, quarantine existing directory and purge DB rows
        if force_rewrite:
            if dry_run:
                print(f"[DRY-RUN] Would quarantine & delete dataset '{dataset_name}'.")
            else:
                quarantine_dir = move_to_quarantine(dataset_images_final, self.trash_root)
                self._delete_dataset_rows(dataset_name)

        # 2) Write images concurrently into a temp dir
        if dry_run:
            print(f"[DRY-RUN] Would write {len(images)} images for dataset '{dataset_name}'.")
            image_name_uuid_map = {k: uuid.uuid4().hex for k in images.keys()}
        else:
            image_name_uuid_map = self._write_images_concurrently(dataset_images_tmp, images)

        # 3) Materialize DataFrames for bulk insert
        rel_df, relation_name_uuid_map = self._build_relation_rows(dataset_name, relations, image_name_uuid_map)
        proto_df = self._build_protocol_rows(dataset_name, protocols, relation_name_uuid_map)

        # 4) Persist metadata transactionally into DuckDB
        if dry_run:
            print(f"[DRY-RUN] Would insert {len(rel_df)} relations and {len(proto_df)} protocol rows.")
        else:
            self._persist_metadata(rel_df, proto_df)

        # 5) Atomically swap image directory into place
        if not dry_run:
            ensure_dir(dataset_images_tmp)
            atomic_replace_dir(dataset_images_tmp, dataset_images_final)
            if quarantine_dir and Path(quarantine_dir).exists():
                rm_tree(Path(quarantine_dir))

    def write_from_registry(self, *, force_rewrite: bool = False, dry_run: bool = False) -> None:
        discover_and_register(
            root_package="OriginDataset",
            target_filename="adaptor.py",
            class_name="Adaptor",
            base_class=BaseAdaptor,
        )
        for name, adaptor_cls in REGISTRY.items():
            adaptor = adaptor_cls()
            for idx, data in enumerate(adaptor):
                try:
                    images: Dict[str, "cv2.Mat"] = data["images"]
                    relations: Dict[str, Dict] = data["relation"]
                    protocols: Dict[str, List[str]] = data["protocol"]

                    self.write_dataset(
                        dataset_name=name,
                        images=images,
                        relations=relations,
                        protocols=protocols,
                        force_rewrite=force_rewrite,
                        dry_run=dry_run,
                    )
                except Exception as e:
                    print(f"[ERROR] Failed to write dataset '{name}' (item #{idx}): {e}\n{traceback.format_exc()}")

    # -----------------------
    # Internal helpers
    # -----------------------

    def _dataset_exists(self, dataset_name: str) -> bool:
        """Return True if the dataset appears to have been written already.

        We consider it existing if either:
          - DuckDB has rows in `relations` for this dataset, OR
          - images/<dataset>/ contains any files (to catch orphaned image dirs)
        """
        count = self.conn.execute(
            "SELECT count(*) FROM relations WHERE dataset_name = ?;",
            [dataset_name],
        ).fetchone()[0]
        if count and int(count) > 0:
            return True
        img_dir = self.images_root / dataset_name
        try:
            has_images = img_dir.exists() and any(img_dir.iterdir())
        except Exception:
            has_images = img_dir.exists()
        return has_images

    def _delete_dataset_rows(self, dataset_name: str) -> None:
        self.conn.execute("BEGIN TRANSACTION;")
        try:
            self.conn.execute("DELETE FROM protocol_entries WHERE relation_set = ?;", [dataset_name])
            self.conn.execute("DELETE FROM relations WHERE dataset_name = ?;", [dataset_name])
            self.conn.execute("COMMIT;")
        except Exception:
            self.conn.execute("ROLLBACK;")
            raise

    def _write_images_concurrently(self, tmp_dir: Path, images: Dict[str, "cv2.Mat"]) -> Dict[str, str]:
        ensure_dir(tmp_dir)

        def _write_one(item):
            name, img = item
            img_uuid = uuid.uuid4().hex
            out = tmp_dir / f"{img_uuid}.jpg"
            ok = cv2.imwrite(str(out), img)
            if not ok:
                raise RuntimeError(f"cv2.imwrite failed for {out}")
            return name, img_uuid

        name_uuid_map: Dict[str, str] = {}
        errors: List[str] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = [ex.submit(_write_one, kv) for kv in images.items()]
            for fut in as_completed(futures):
                try:
                    name, img_uuid = fut.result()
                    name_uuid_map[name] = img_uuid
                except Exception as e:
                    errors.append(str(e))
        if errors:
            rm_tree(tmp_dir)
            raise RuntimeError("One or more image writes failed: \n" + "\n".join(errors))
        return name_uuid_map

    def _build_relation_rows(
        self,
        dataset_name: str,
        relations: Dict[str, Dict],
        image_name_uuid_map: Dict[str, str],
    ) -> Tuple[pd.DataFrame, Dict[str, str]]:
        rows = []
        rel_name_to_uuid: Dict[str, str] = {}
        for rel_name, rel in relations.items():
            rid = uuid.uuid4().hex
            rel = dict(rel)  # avoid mutating caller data
            if "image_names" in rel:
                rel["image_ids"] = [image_name_uuid_map[name] for name in rel["image_names"]]
                rel.pop("image_names", None)
            payload = json.dumps(rel, ensure_ascii=False, sort_keys=True)
            rows.append({"relation_id": rid, "dataset_name": dataset_name, "payload": payload})
            rel_name_to_uuid[rel_name] = rid
        df = pd.DataFrame(rows, columns=["relation_id", "dataset_name", "payload"])
        return df, rel_name_to_uuid

    def _build_protocol_rows(
        self,
        dataset_name: str,
        protocols: Dict[str, List[str]],
        relation_name_uuid_map: Dict[str, str],
    ) -> pd.DataFrame:
        rows = []
        for proto_name, rel_names in protocols.items():
            for rn in rel_names:
                rid = relation_name_uuid_map[rn]
                rows.append({
                    "protocol_name": proto_name,
                    "relation_id": rid,
                    "relation_set": dataset_name,
                })
        return pd.DataFrame(rows, columns=["protocol_name", "relation_id", "relation_set"])

    def _persist_metadata(self, rel_df: pd.DataFrame, proto_df: pd.DataFrame) -> None:
        self.conn.execute("BEGIN TRANSACTION;")
        try:
            self.conn.register("rel_df", rel_df)
            self.conn.register("proto_df", proto_df)
            self.conn.execute("INSERT INTO relations SELECT * FROM rel_df;")
            if not proto_df.empty:
                self.conn.execute("INSERT INTO protocol_entries SELECT * FROM proto_df;")
            self.conn.execute("COMMIT;")
        except Exception:
            self.conn.execute("ROLLBACK;")
            raise


# --------------------------------------
# Backwards-compatible entry point
# --------------------------------------

def write(config: dict, *, force_rewrite: bool = False, dry_run: bool = False, max_workers: int = 8) -> None:
    writer = DatasetWriter(
        database_root=config["database_root"],
        max_workers=max_workers,
    )
    writer.write_from_registry(force_rewrite=force_rewrite, dry_run=dry_run)


if __name__ == "__main__":
    cfg = GetDatabaseConfig()
    write(cfg, force_rewrite=True, dry_run=False, max_workers=max(8, (os.cpu_count() or 8)))
