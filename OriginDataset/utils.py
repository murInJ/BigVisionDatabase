import os
from pathlib import Path
import pandas as pd
import shutil
from datetime import datetime
from typing import Optional

def ensure_parquet_with_schema(
    path: str | Path,
    columns: tuple[str, ...],
    dtypes: dict[str, str] | None = None,
    check_dtypes: bool = False,
) -> None:
    """
    确保指定 Parquet 文件存在且列集合与期望一致；若不存在则创建 0 行的空文件。
    - columns: 期望的列名集合（顺序不敏感，只校验集合）
    - dtypes: 可选，创建新文件时使用的 pandas dtype（如 "string"、"int64"）
    - check_dtypes: True 时会在文件已存在时校验每列 dtype（可能因不同引擎略有差异，默认关闭）

    依赖：pandas + pyarrow
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    # 默认所有列用 pandas "string" dtype
    if dtypes is None:
        dtypes = {c: "string" for c in columns}

    if not p.exists():
        # 创建 0 行、具备指定列与 dtype 的 DataFrame
        empty_df = pd.DataFrame({c: pd.Series(dtype=dtypes.get(c, "string")) for c in columns})
        empty_df.to_parquet(p, index=False)
        return

    # 已存在：读取并校验列集合
    try:
        df = pd.read_parquet(p)
    except Exception as e:
        raise ValueError(f"无法读取已存在的 Parquet 文件：{p}，错误：{e}")

    expected = set(columns)
    actual = set(df.columns)
    if actual != expected:
        raise ValueError(f"目标文件列不匹配，期望 {expected}，实际 {actual}。")

    if check_dtypes and dtypes:
        # 注意：不同写入/读取引擎可能导致 dtype 轻微差异，必要时再打开此严格校验
        def _norm_dtype(x: str) -> str:
            return str(pd.Series(dtype=x).dtype)
        expected_dtypes = {c: _norm_dtype(t) for c, t in dtypes.items() if c in df.columns}
        for c in expected_dtypes:
            if str(df[c].dtype) != expected_dtypes[c]:
                raise ValueError(f"列 {c} 的 dtype 不匹配：期望 {expected_dtypes[c]}，实际 {df[c].dtype}")

import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def rm_tree(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    elif path.exists():
        try:
            path.unlink(missing_ok=True)  # type: ignore[arg-type]
        except TypeError:
            try:
                path.unlink()
            except FileNotFoundError:
                pass


def move_to_quarantine(path: Path, trash_root: Path) -> Optional[Path]:
    """Move a path to a timestamped quarantine folder under `.trash/`.

    Returns the new quarantine path or None if the path didn't exist.
    """
    if not path.exists():
        return None
    ensure_dir(trash_root)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    quarantine = trash_root / f"{path.name}-{ts}.quarantine"
    if quarantine.exists():
        rm_tree(quarantine)
    shutil.move(str(path), str(quarantine))
    return quarantine


def atomic_replace_dir(temp_dir: Path, final_dir: Path) -> None:
    """Best-effort atomic replacement by renaming a prepared temp dir into place."""
    if final_dir.exists():
        rm_tree(final_dir)
    temp_dir.rename(final_dir)


def require_duckdb(duckdb_path: str, threads: int):
    """Return an initialized DuckDB connection; raise if DuckDB is unavailable.

    This project now *requires* DuckDB for metadata storage (no Parquet fallback).
    """
    try:
        import duckdb  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "DuckDB is required but not available. Install with `pip install duckdb`."
        ) from e

    db_path = Path(duckdb_path)
    ensure_dir(db_path.parent)
    conn = duckdb.connect(str(db_path))
    conn.execute("PRAGMA threads = {}".format(max(1, threads)))
    return conn
