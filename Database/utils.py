# Database/utils.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Dict

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def require_duckdb(duckdb_path: str, threads: int = 0):
    """
    返回 DuckDB 连接（并确保父目录存在）。
    threads=0 表示由 DuckDB 自行决定（通常=CPU核心数）。
    """
    try:
        import duckdb  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("DuckDB is required. Install with `pip install duckdb`.") from e

    db_path = Path(duckdb_path)
    ensure_dir(db_path.parent)
    conn = duckdb.connect(str(db_path))
    if threads and threads > 0:
        conn.execute(f"PRAGMA threads={int(threads)}")
    else:
        # 让 DuckDB 自己决定；也可按需设定成 max(1, os.cpu_count() or 1)
        pass
    return conn

def init_duckdb_schema(conn) -> None:
    """
    在给定连接上初始化 schema（幂等）。
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS images (
            image_id     TEXT PRIMARY KEY,
            uri          TEXT NOT NULL,     -- 'images/<dataset_name>/<uuid>.npy'
            modality     TEXT,
            dataset_name TEXT NOT NULL,
            alias        TEXT,
            extra        JSON
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS relations (
            relation_id  TEXT PRIMARY KEY,
            payload      JSON NOT NULL      -- 含 image_ids + 任意业务键
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS protocol (
            protocol_name TEXT NOT NULL,
            relation_id   TEXT NOT NULL,
            relation_set  TEXT NOT NULL
        );
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_images_dataset ON images(dataset_name);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_protocol_set  ON protocol(relation_set);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_protocol_name ON protocol(protocol_name);")

# --------- 可选保留：Parquet 兼容工具（若项目仍有使用） ---------
import pandas as pd

def ensure_parquet_with_schema(
    path: str | Path,
    columns: tuple[str, ...],
    dtypes: dict[str, str] | None = None,
    check_dtypes: bool = False,
) -> None:
    """
    兼容保留：确保指定 Parquet 文件存在且列集合与期望一致；若不存在则创建空文件。
    """
    p = Path(path)
    ensure_dir(p.parent)

    if dtypes is None:
        dtypes = {c: "string" for c in columns}

    if not p.exists():
        empty_df = pd.DataFrame({c: pd.Series(dtype=dtypes.get(c, "string")) for c in columns})
        empty_df.to_parquet(p, index=False)
        return

    try:
        df = pd.read_parquet(p)
    except Exception as e:
        raise ValueError(f"无法读取已存在的 Parquet 文件：{p}，错误：{e}")

    expected = set(columns)
    actual = set(df.columns)
    if actual != expected:
        raise ValueError(f"目标文件列不匹配，期望 {expected}，实际 {actual}。")

    if check_dtypes and dtypes:
        def _norm_dtype(x: str) -> str:
            return str(pd.Series(dtype=x).dtype)
        expected_dtypes = {c: _norm_dtype(t) for c, t in dtypes.items() if c in df.columns}
        for c in expected_dtypes:
            if str(df[c].dtype) != expected_dtypes[c]:
                raise ValueError(f"列 {c} 的 dtype 不匹配：期望 {expected_dtypes[c]}，实际 {df[c].dtype}")
