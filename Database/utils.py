# Database/utils.py
from __future__ import annotations

import importlib
import inspect
import json
import os
import shutil
import zipfile
from pathlib import Path
from typing import Optional, Tuple, Dict, Union, Generator, List, Any


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

def sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()

def is_zip_path(p: Union[str, Path]) -> bool:
    p = Path(p)
    return p.is_file() and p.suffix.lower() == ".zip"

def bundle_read_json(bundle_path: Union[str, Path], inner_relpath: str) -> Dict:
    """
    读取 bundle 内部 JSON（支持目录或 ZIP），返回 dict。
    inner_relpath 如 'manifest.json' / 'protocol.json'。
    """
    bp = Path(bundle_path)
    if is_zip_path(bp):
        with zipfile.ZipFile(str(bp), "r") as zf:
            with zf.open(inner_relpath, "r") as fh:
                return json.loads(fh.read().decode("utf-8"))
    else:
        return json.loads((bp / inner_relpath).read_text(encoding="utf-8"))

def bundle_iter_jsonl(bundle_path: Union[str, Path], inner_relpath: str) -> Generator[Dict, None, None]:
    """
    逐行读取 JSONL（支持目录或 ZIP）。每行解析失败会跳过。
    """
    bp = Path(bundle_path)
    if is_zip_path(bp):
        with zipfile.ZipFile(str(bp), "r") as zf:
            with zf.open(inner_relpath, "r") as fh:
                for raw in fh:
                    try:
                        yield json.loads(raw.decode("utf-8"))
                    except Exception:
                        continue
    else:
        with (bp / inner_relpath).open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    yield json.loads(line)
                except Exception:
                    continue

def bundle_member_exists(bundle_path: Union[str, Path], inner_relpath: str) -> bool:
    bp = Path(bundle_path)
    if is_zip_path(bp):
        with zipfile.ZipFile(str(bp), "r") as zf:
            try:
                zf.getinfo(inner_relpath)
                return True
            except KeyError:
                return False
    else:
        return (bp / inner_relpath).exists()

def bundle_copy_or_link_member(
    bundle_path: Union[str, Path],
    inner_relpath: str,
    dst_path: Union[str, Path],
    *,
    mode: str = "copy",          # 'copy' | 'hardlink' | 'symlink'
    overwrite: bool = True,
) -> None:
    """
    将 bundle 内部的单个文件复制/硬链/软链到 dst_path。
    - 若 bundle 是 ZIP：不支持硬链/软链，自动退化为 copy（按需流式拷贝，不全量解压）。
    - 若 bundle 是目录：支持 'copy'|'hardlink'|'symlink'。
    """
    dst_path = Path(dst_path)
    ensure_dir(dst_path.parent)

    bp = Path(bundle_path)
    if dst_path.exists():
        if overwrite:
            if dst_path.is_file():
                try: dst_path.unlink()
                except Exception: pass
        else:
            return

    if is_zip_path(bp):
        # zip 内按需流式复制
        with zipfile.ZipFile(str(bp), "r") as zf:
            with zf.open(inner_relpath, "r") as src, open(dst_path, "wb") as out:
                shutil.copyfileobj(src, out, length=1024 * 1024)
        return

    # 目录形式
    src_path = (bp / inner_relpath)
    if not src_path.exists():
        raise FileNotFoundError(f"bundle member not found: {inner_relpath}")
    if mode == "copy":
        shutil.copy2(src_path, dst_path)
    elif mode == "hardlink":
        # 硬链要求同一文件系统
        os.link(src_path, dst_path)
    elif mode == "symlink":
        # 软链跨盘也可，但后续移动 bundle 会失效
        rel = os.path.relpath(src_path, start=dst_path.parent)
        os.symlink(rel, dst_path)
    else:
        raise ValueError("mode must be one of: copy|hardlink|symlink")

def finalize_export_zip(
    export_dir: Path,
    out_path: str,
    zip_output: bool,
    overwrite: bool,
) -> Dict[str, Any]:
    """
    将 export_dir 打包/落地到 out_path。
    - 若 zip_output=True：生成 .zip，且**包含顶层目录**（即解压后会有一个包裹文件夹）。
    - 若 zip_output=False：将 export_dir 作为最终目录输出（移动/覆盖）。

    返回:
        {"ok": True, "path": <zip或目录路径>, "type": "zip"|"dir"}
    """
    export_dir = Path(export_dir)
    out_path = str(out_path)
    if not zip_output:
        # 输出为目录
        out_dir = Path(out_path)
        if out_dir.exists():
            if not overwrite:
                raise FileExistsError(f"Target directory exists: {out_dir}")
            shutil.rmtree(out_dir, ignore_errors=True)
        # 直接移动 export_dir 到 out_dir
        out_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(export_dir), str(out_dir))
        return {"ok": True, "path": str(out_dir), "type": "dir"}

    # 输出为 zip：确保包含顶层目录
    out_zip = Path(out_path)
    if out_zip.suffix.lower() != ".zip":
        # 若没给 .zip 后缀，自动补上
        out_zip = out_zip.with_suffix(".zip")
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    if out_zip.exists():
        if not overwrite:
            raise FileExistsError(f"Target zip exists: {out_zip}")
        out_zip.unlink()

    # 关键点：以 export_dir 的父目录为 root_dir，base_dir 指向 export_dir.name，
    # 这样 zip 内会包含顶层目录 <export_dir.name>/...
    base = out_zip.with_suffix("")  # shutil 会自己加 .zip
    tmp_zip = shutil.make_archive(
        base_name=str(base),
        format="zip",
        root_dir=str(export_dir.parent),
        base_dir=str(export_dir.name),
    )
    # shutil.make_archive 已经生成了 <base>.zip；若 out_zip 与之不同则移动
    tmp_zip_path = Path(tmp_zip)
    if tmp_zip_path != out_zip:
        if out_zip.exists():
            out_zip.unlink()
        tmp_zip_path.replace(out_zip)

    # 打包完成后，源构建目录可以由调用方清理；这里不删除 export_dir 以安全起见
    return {"ok": True, "path": str(out_zip), "type": "zip"}


def export_thumbnails_for_ids(
    *,
    conn, # duckdb 连接
    database_root: Union[str, Path],
    image_ids: List[str],
    out_dir: Union[str, Path],
    color_order: str = "bgr",
    max_side: int = 256,
    root_override: Optional[str] = None,
    ) -> Dict[str, object]:
    """
    为一组 image_id 生成 PNG 缩略图到 out_dir。
    - color_order: 'bgr'|'rgb'，三通道颜色解释（与 export_images_by_ids 一致）
    - max_side:    最长边缩放到该值（保持纵横比）
    - root_override: 若提供，则从该根目录寻找 images/<dataset>/<iid>.npy；
                     默认 None 表示从 DB 标准路径 <database_root> 加载

    返回: { "out_dir": str, "count": int }
    """
    import numpy as np
    from pathlib import Path

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 写 PNG：优先 imageio，回退 cv2
    try:
        import imageio.v3 as iio  # type: ignore
        def _save_png(dst: Path, img: np.ndarray) -> bool:
            iio.imwrite(str(dst), img)
            return True

        writer_backend = "imageio"  # 期望 RGB
    except Exception:
        try:
            import cv2  # type: ignore
            def _save_png(dst: Path, img: np.ndarray) -> bool:
                return bool(cv2.imwrite(str(dst), img))

            writer_backend = "cv2"  # 期望 BGR
        except Exception:
            raise RuntimeError("Neither imageio nor OpenCV is available to write PNG files.")

    # 简单 resize
    def _resize(img: np.ndarray) -> np.ndarray:
        if img.ndim < 2:
            return img
        h, w = img.shape[:2]
        if max(h, w) <= max_side:
            return img
        scale = max_side / float(max(h, w))
        new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        try:
            import cv2  # type: ignore
            return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        except Exception:
            try:
                from PIL import Image  # type: ignore
                return np.array(Image.fromarray(img).resize((new_w, new_h)))
            except Exception:
                # 最粗暴的最近邻
                ys = np.linspace(0, h - 1, new_h).astype(np.int64)
                xs = np.linspace(0, w - 1, new_w).astype(np.int64)
                return img[ys][:, xs]

    # uint8 归一化与轴修正
    def _prep_uint8(arr: np.ndarray) -> np.ndarray:
        a = np.asarray(arr)
        if a.ndim == 3 and a.shape[0] in (1, 3) and a.shape[2] not in (1, 3):
            a = np.moveaxis(a, 0, 2)
        while a.ndim > 3:
            a = a.squeeze(axis=0)
        if a.dtype != np.uint8:
            a = a.astype(np.float32)
            amin, amax = np.nanmin(a), np.nanmax(a)
            if amax > amin:
                a = (a - amin) / (amax - amin)
            else:
                a = np.zeros_like(a, dtype=np.float32)
            a = (a * 255.0).round().clip(0, 255).astype(np.uint8)
        # 统一成 HxW 或 HxWx3
        if a.ndim == 2:
            return a
        if a.ndim == 3:
            if a.shape[2] == 1:
                return a[:, :, 0]
            if a.shape[2] >= 3:
                return a[:, :, :3]
        return a.reshape(a.shape[0], a.shape[1]).astype(np.uint8)

    # 查询 uri/alias 信息
    rows = []
    if image_ids:
        placeholders = ",".join(["?"] * len(image_ids))
        q = f"SELECT image_id, uri, alias, modality FROM images WHERE image_id IN ({placeholders})"
        rows = conn.execute(q, image_ids).fetchall()

    db_root = Path(root_override) if root_override else Path(database_root)

    count = 0
    for idx, (iid, uri, alias, modality) in enumerate(rows):
        try:
            src = (db_root / str(uri)).resolve()
            import numpy as _np2
            arr = _np2.load(src, allow_pickle=False)
            img = _prep_uint8(arr)

            # 颜色顺序修正（与 export_images_by_ids 一致）
            if isinstance(img, _np2.ndarray) and img.ndim == 3 and img.shape[2] == 3:
                if writer_backend == "imageio" and color_order == "bgr":
                    img = img[:, :, ::-1]  # BGR -> RGB
                elif writer_backend == "cv2" and color_order == "rgb":
                    img = img[:, :, ::-1]  # RGB -> BGR

            img = _resize(img)

            base = f"{idx:05d}_{alias or modality or iid}.png"
            safe = "".join(ch for ch in base if ch.isalnum() or ch in ("-", "_", ".", "+"))
            dst = out_dir / (safe if safe else f"{idx:05d}.png")
            _save_png(dst, img)
            count += 1
        except Exception:
            continue

    return {"out_dir": str(out_dir), "count": count}

def _repo_misc_dir() -> Path:
    # <repo_root>/Database/utils.py -> <repo_root>/misc
    return Path(__file__).resolve().parents[1] / "misc"

def _resolve_template_path(filename: str) -> Path:
    """
    在以下位置查找 README 模板（现在是 .md，而不是 .tmpl）：
      1) <repo_root>/misc/<filename>
      2) ./misc/<filename>（当前工作目录）
    """
    p1 = _repo_misc_dir() / filename
    if p1.exists():
        return p1
    p2 = Path.cwd() / "misc" / filename
    if p2.exists():
        return p2
    raise FileNotFoundError(f"Template not found: {filename} (searched in {p1} and {p2})")

def _render_template(filename: str, variables: Dict[str, Any]) -> str:
    """
    读取 .md 模板并用 str.format(**variables) 渲染。
    兼容模板中把下划线写成 \\_ 的占位符（如 {loader\\_entry}）。
    模板里如需字面量大括号，请用 {{ 和 }}。
    """
    text = _resolve_template_path(filename).read_text(encoding="utf-8")

    # 构造一个带“转义下划线”别名的变量字典，兼容 {loader\_entry} 这种写法
    vars2 = dict(variables)
    for k, v in list(variables.items()):
        k_escaped = k.replace("_", "\\_")
        # 只有在确实不同且未被占用时才添加别名
        if k_escaped != k and k_escaped not in vars2:
            vars2[k_escaped] = v

    return text.format(**vars2)

def render_readme_bundle(protocol_name: str) -> str:
    """
    渲染 Bundle 的 README（来自 misc/readme_bundle.md）
    可用占位符：
      - {protocol_name}
      - {created_utc}
      - {schema}（固定 bvbundle.v1）
    """
    from datetime import datetime, timezone
    vars = {
        "protocol_name": protocol_name,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "schema": "bvbundle.v1",
    }
    return _render_template("readme_bundle.md", vars)

def render_readme_compact(protocol_name: str) -> str:
    """
    渲染 Compact 的 README（来自 misc/readme_compact.md）
    可用占位符：
      - {protocol_name}
      - {created_utc}
      - {schema}（固定 compact.v1）
      - {loader_entry}（通常 'loader.py'）
    """
    from datetime import datetime, timezone
    vars = {
        "protocol_name": protocol_name,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "schema": "compact.v1",
        "loader_entry": "loader.py",
    }
    return _render_template("readme_compact.md", vars)

def copy_compact_loader_from_misc(dst_dir: str | Path, module_path: str = "misc.loader") -> str:
    """
    尝试从 `module_path`（默认 misc.loader）复制源码文件到 dst_dir/loader.py。
    返回目标路径字符串。
    如果模块不可导入或没有 __file__，抛出异常让调用方决定兜底策略。
    """
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    mod = importlib.import_module(module_path)
    src_path = Path(inspect.getfile(mod))  # e.g. /path/to/your/project/misc/loader.py
    if not src_path.exists():
        raise FileNotFoundError(f"Resolved loader source not found: {src_path}")

    dst_path = dst_dir / "loader.py"
    shutil.copy2(src_path, dst_path)
    return str(dst_path)