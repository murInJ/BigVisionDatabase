# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any

import yaml

# py3.9+ 统一用 importlib.resources.files 读取打包资源
try:
    from importlib.resources import files as ir_files
except Exception:  # py<3.9 可不走这一分支
    ir_files = None  # type: ignore


def _read_yaml_text(text: str) -> Dict[str, Any]:
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError("YAML content must be a mapping (dict).")
    return data


def _load_yaml_file(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return _read_yaml_text(f.read())


def _search_yaml_in_package(pkg: str, filename: str) -> Optional[Dict[str, Any]]:
    """
    从打包资源读取（zip/whl）。只有当 yaml 和本 setting.py 同包时才会命中。
    """
    if ir_files is None:
        return None
    try:
        res = ir_files(pkg).joinpath(filename)
        # Traversable 都有 read_text
        if hasattr(res, "read_text"):
            return _read_yaml_text(res.read_text(encoding="utf-8"))
    except Exception:
        pass
    return None


def _iter_candidate_paths(filename: str):
    """
    生成一系列候选文件路径，按优先级从高到低。
    """
    # 1) 模块同目录（当前文件所在包内）
    here = Path(__file__).resolve()
    pkg_dir = here.parent  # .../BigVisionDatabase/Config
    yield pkg_dir / filename

    # 2) 包上一层/两层的 Config 目录（更健壮，防止目录布局变化）
    for up in [pkg_dir.parent, pkg_dir.parent.parent]:
        if up.exists():
            yield up / "Config" / filename

    # 3) 当前工作目录及其若干父目录下的 Config/
    cwd = Path.cwd().resolve()
    cur = cwd
    for _ in range(5):  # 向上找几层，够用了
        yield cur / "Config" / filename
        cur = cur.parent
        if cur == cur.parent:
            break

    # 4) sys.path 中的任意目录下存在 Config/filename 的
    for p in map(Path, sys.path):
        try:
            pp = p.resolve()
        except Exception:
            continue
        yield pp / "Config" / filename


def _resolve_and_load(
    filename: str,
    explicit_path: Optional[str],
    env_var: Optional[str],
    package_name_for_resources: str,
) -> Dict[str, Any]:
    # A. 显式参数优先
    if explicit_path:
        p = Path(explicit_path).expanduser().resolve()
        if p.is_file():
            return _load_yaml_file(p)
        raise FileNotFoundError(f"Explicit path not found: {p}")

    # B. 环境变量
    if env_var:
        env_val = os.getenv(env_var)
        if env_val:
            p = Path(env_val).expanduser().resolve()
            if p.is_file():
                return _load_yaml_file(p)
            raise FileNotFoundError(f"Env {env_var} points to missing file: {p}")

    # C. 打包资源（同包内）
    # 例如 package_name_for_resources = 'BigVisionDatabase.Config'
    data = _search_yaml_in_package(package_name_for_resources, filename)
    if data is not None:
        return data

    # D. 文件系统多策略搜索
    for cand in _iter_candidate_paths(filename):
        if cand.is_file():
            return _load_yaml_file(cand)

    # E. 全部失败：报错并给出提示
    tried = [
        str(c) for c in _iter_candidate_paths(filename)
    ]
    hint = (
        f"Unable to locate '{filename}'. Tried typical locations near the module, "
        f"current working directory and sys.path.\n"
        f"Solutions:\n"
        f"  1) Set environment variable to the absolute file path:\n"
        f"     export {env_var}=/abs/path/to/{filename}\n"
        f"  2) Call Get*Config(path='/abs/path/to/{filename}') explicitly.\n"
        f"  3) Ensure a 'Config/{filename}' exists relative to your CWD or module folder."
    )
    raise FileNotFoundError(f"{hint}\nCandidates (examples):\n  - " + "\n  - ".join(tried[:10]))


# -------------------- 对外 API --------------------

def GetDatabaseConfig(path: Optional[str] = None) -> Dict[str, Any]:
    """
    读取 database_config.yaml
    优先级：显式 path > 环境变量 BIGVISION_DB_CONFIG > 包内资源 > 多策略搜索
    返回 dict
    """
    return _resolve_and_load(
        filename="database_config.yaml",
        explicit_path=path,
        env_var="BIGVISION_DB_CONFIG",
        package_name_for_resources=__package__ or "BigVisionDatabase.Config",
    )


def GetOrigindataConfig(path: Optional[str] = None) -> Dict[str, Any]:
    """
    读取 origindata_config.yaml
    优先级：显式 path > 环境变量 BIGVISION_ORIGINDATA_CONFIG > 包内资源 > 多策略搜索
    返回 dict
    """
    return _resolve_and_load(
        filename="origindata_config.yaml",
        explicit_path=path,
        env_var="BIGVISION_ORIGINDATA_CONFIG",
        package_name_for_resources=__package__ or "BigVisionDatabase.Config",
    )
