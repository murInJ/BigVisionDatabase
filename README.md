# BigVisionDatabase
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![DuckDB](https://img.shields.io/badge/DuckDB-%F0%9F%A6%85-ffcc00)](https://duckdb.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](#license)
## Version
0.1.2
## Update Plan
None

---

## ✨ 特性

- **数据布局标准化**
  - `<db_root>/images/<dataset_name>/<image_id>.npy` —— 原样 NPY（不做颜色/尺度变换）
  - `<db_root>/db/catalog.duckdb` —— 元信息存储（**单进程写**）
- **导出/回导**
  - **Bundle**：完整可回导备份（`protocol.json`、`relations.jsonl`、`images_index.jsonl`、`manifest.json`、`images/`、…）
  - **Compact**：训练就绪（`images/` + `samples-*.parquet` + `manifest.json` + `loader.py`）
- **组织与采样**：Protocol 的创建/重命名/合并/采样/删除，Relation 的 CRUD
- **清理与统计**：孤儿文件发现/删除、缺失文件报告、DB 概览
- **加载方式**：
  - `misc/loader.py`（Compact 专用）
  - `Dataset/loader.py`（**DB 直连**）

---

## 🧱 目录结构约定

```
<database_root>/
  images/<dataset_name>/<image_id>.npy
  db/catalog.duckdb
  tmp/
src/
  Database/db.py
  Database/utils.py
  OriginDataset/writer.py
  Dataset/loader.py         # ← DB 直连 Dataset
  misc/loader.py            # ← Compact 数据集 Dataset
scripts/
  *.sh
```

---

## 🗃️ 表结构（DuckDB）

- `images(image_id TEXT PRIMARY KEY, uri TEXT, modality TEXT, dataset_name TEXT, alias TEXT, extra TEXT)`  
- `relations(relation_id TEXT PRIMARY KEY, payload TEXT)`  
  - `payload` 至少含 `image_ids: List[str]`；可选 `task_type`、`annotation`、`extra`、`image_aliases`、`image_names`…
- `protocol(protocol_name TEXT, relation_id TEXT, relation_set TEXT)`  
  - 约定：`relation_set == protocol_name`

---

## 🔒 不变式与行为

- **单进程写**：DB 只允许一个进程写；`BigVisionDatabase` 内部持有**唯一连接**并注入到 `DatasetWriter`
- **写入原子性**：失败回滚；可能有**孤儿 .npy** → `garbage_collect()` 可扫描/清理
- **颜色/通道**：库内 NPY 原样；导出 PNG 默认按 **BGR 来源** 写出（可切换）
- **别名优先级（Compact 导出）**：`relation.image_aliases` > `relation.image_names` > `images.alias` > `images.modality` > `img{序号}`

---

## ⚙️ 安装与依赖

```bash
# 推荐创建虚拟环境
python -m venv .venv && source .venv/bin/activate
pip install -U numpy duckdb pyarrow tqdm
# 可选（使用 torch DataLoader）
pip install torch  # 或按照你的 CUDA 版本安装
```

---

## 🧰 配置

默认从 `Config.setting.GetDatabaseConfig()` 读取：
请从`database_config.yaml`进行配置


也支持通过环境变量覆盖：`DB_ROOT=/path/to/bigvision_db`。

---

## 🚀 快速开始

### 1) 初始化与写入（烟测）

```python
from Database.db import BigVisionDatabase
db = BigVisionDatabase()
print(db.get_db_summary())  # 空库概览
```

```bash
# 写入原始数据（仅 writer 流程）
NO_PROGRESS=1 scripts/writeOriginDataset.sh
# 报告孤儿/缺失
python - <<'PY'
from Database.db import BigVisionDatabase
db = BigVisionDatabase()
print(db.garbage_collect(remove_orphan_files=False, check_db_missing_files=True))
PY
```

### 2) Relations / Protocols 示例

```python
from Database.db import BigVisionDatabase
db = BigVisionDatabase()

rel = db.add_relation(
    payload={"image_ids": ["img_0001","img_0002"], "task_type":"pair"},
    protocols=["demo_proto"],
    dry_run=False
)
db.create_protocol("demo_proto", [rel["relation_id"]], replace=True)
print(db.get_db_summary())
```

### 3) Bundle（导出→校验→回导）

```bash
scripts/export_protocol_bundle.sh -p demo_proto
scripts/verify_bundle.sh -i "$DB_ROOT/tmp/bundles/demo_proto.bvbundle.zip"
scripts/load_bundle.sh   -i "$DB_ROOT/tmp/bundles/demo_proto.bvbundle.zip" -m overwrite
```

### 4) Compact（训练集导出）

```bash
scripts/export_protocol_compact.sh -p demo_proto
python - <<'PY'
from misc.loader import CompactDataset
ds = CompactDataset(f"{r'$DB_ROOT'}/tmp/compact/demo_proto.compact")
for i in range(min(5, len(ds))): _ = ds[i]
print("Compact OK. Size =", len(ds))
PY
```

### 5) **DB 直连 Dataset（无需导出）**

```python
from Dataset.loader import DBProtocolDataset
from torch.utils.data import DataLoader

# db_root 默认走 Config，可显式覆盖：DBProtocolDataset(db_root="/abs/path", ...)
ds = DBProtocolDataset(protocol_name="demo_proto",
                       normalize=True, to_tensor=True, mmap=True, color_order="bgr")
dl = DataLoader(ds, batch_size=4, shuffle=True,
                collate_fn=DBProtocolDataset.collate_batch, num_workers=0)

batch = next(iter(dl))
print(batch["relation_id"][:2], len(batch["images"]))
```

**关于 `--mmap`**：以 `np.load(..., mmap_mode='r')` 方式按需映射，降低内存占用；开启颜色翻转或转 Tensor 时会发生一次合理拷贝（已处理负 stride 兼容）。

---

## 🧪 CLI 自检

```bash
python Dataset/loader.py --protocol demo_proto --limit 4 --mmap --normalize
# 或包方式
python -m Dataset.loader --protocol demo_proto
```

CLI 解析 `db_root` 顺序：`--db-root` > `DB_ROOT` > `Config.setting.GetDatabaseConfig()`（内部包含鲁棒的模块定位）。

---

## 📜 脚本速查（scripts/*）

- 写入：`scripts/writeOriginDataset.sh`  
- Bundle：`scripts/export_protocol_bundle.sh`、`scripts/verify_bundle.sh`、`scripts/load_bundle.sh`  
- Protocol 组织：`scripts/merge_protocols.sh`、`scripts/sample_protocol.sh`、`scripts/delete_protocol.sh`  
- Compact：`scripts/export_protocol_compact.sh`

---

## ⚡ 性能与实践建议

- DB 写入**单进程**，但图片写盘可多线程/进程并发（由 `DatasetWriter` 控制）
- 使用 **NVMe 本地盘** 存放 `images/`、`db/` 显著提升吞吐；网络盘（如 SMB/NFS）建议先本地缓存
- 大数据量训练时：`mmap=True` + `num_workers>0` + 合理的 DataLoader prefetch
- 如果 NPY 本身为 RGB，可设置 `color_order='none'` 避免翻转复制

---

## ❓ 常见问题（FAQ）

- **为什么 PNG 导出默认按 BGR 来源写？**  
  历史上原始数据多为 OpenCV BGR，默认翻转可避免“蓝脸”。可通过参数关闭或改为 RGB。

- **Compact 与 DB 直连选哪个？**  
  - **Compact**：可移植、可归档、可复用（跨机器/集群）  
  - **DB 直连**：开发调试快、无需导出、依赖库内布局稳定

---

## 🤝 贡献

欢迎 PR / Issue！请在提交前：

- 遵循表结构与目录约定
- 保持 `BigVisionDatabase` 作为**唯一连接持有者**
- 补充最小可复现示例与单元测试（如有）

---

## License

Apache-2.0


