# OriginDataset/utils.py
from __future__ import annotations

from pathlib import Path

def ensure_dir(path: Path) -> None:
    """
    数据集侧的通用目录创建工具。
    （与 Database/utils.ensure_dir 职责分离，避免跨层依赖）
    """
    path.mkdir(parents=True, exist_ok=True)
