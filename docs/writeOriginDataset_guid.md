# OriginDataset 写入指南（DuckDB + `.npy` 方案）

> 元数据写入到 `<database_root>/db/catalog.duckdb`；图像按**每张图的来源数据集**写入到 `<database_root>/images/<dataset_name>/`，文件为 `.npy`。
> 进度显示：**每个 adaptor 一个进度条**，仅在进度条后缀汇总统计（不刷屏）。

---

## 1) 安装与准备

```bash
# 建议 Python 3.9+
python3 -m pip install duckdb pandas numpy tqdm

# 如 adaptor 提供 torch.Tensor，需要安装 PyTorch（任选你环境匹配的方式）
# pip install torch

# （可选）如果你的 adaptor 仍用 OpenCV 读图，再安装：
# pip install opencv-python
```

项目结构示例（关键部分）：

```
.
├─ OriginDataset/
│  ├─ writer.py          # 写入器（使用 duckdb + .npy）
│  ├─ base.py            # BaseAdaptor（你已有）
│  ├─ PADISI/            # 示例数据集
│  │  └─ adaptor.py
│  └─ utils.py           # ensure_dir / require_duckdb 等
└─ Config/
   └─ setting.py         # 提供 GetDatabaseConfig()
```

---

## 2) 快速开始

### 2.1 从注册器批量写入（推荐）

```python
from Config.setting import GetDatabaseConfig
from OriginDataset.writer import write

cfg = GetDatabaseConfig()  # 需包含 database_root
write(cfg, dry_run=False, max_workers=16, show_progress=True)
```

* `dry_run=True`：不落盘，仅走一遍流程（便于检查 adaptor 输出结构是否正确）。
* `show_progress=True`：每个 adaptor 一个 `tqdm` 进度条，后缀动态显示累计写入的 images/relations/protocol/err。

### 2.2 细粒度：按条写入（直接调用 `DatasetWriter.write_dataset`）

```python
import numpy as np
from OriginDataset.writer import DatasetWriter

# 构造示例数据（images 的键是人类可读名；alias 默认为键名）
images = {
    "RGB":   {"image": np.zeros((480, 640, 3), dtype=np.uint8), "dataset_name": "MYDS", "modality": "RGB"},
    "DEPTH": {"image": np.zeros((480, 640),     dtype=np.uint16), "dataset_name": "MYDS", "modality": "DEPTH"},
    "IR":    {"image": np.zeros((480, 640),     dtype=np.uint8),  "dataset_name": "MYDS", "modality": "IR"},
}

# relation 可用 image_names 选择任意子集（元素可以是键名或 alias）
relations = {
    "FAS": {
        "image_names": ["RGB", "IR"],      # 若省略则默认使用 images 的全部，按插入顺序
        "task_type": "classification",
        "annotation": {"label": 1},
        "meta": {"source": "demo", "version": 1}
    }
}

# protocol 的键既是 protocol_name 也是 relation_set（你也可以用不同命名拆分集合）
protocols = {
    "MYDS_train": ["FAS"]
}

w = DatasetWriter(database_root="/data/mydb", max_workers=8)
w.write_dataset(images=images, relations=relations, protocols=protocols, dry_run=False)
```

---

## 3) Adaptor 返回结构要求

Adaptor 的 `__getitem__` 必须返回以下结构（顶层三键固定）：

```python
{
  "images":   Dict[str, ImageSpec],
  "relation": Dict[str, RelationSpec],
  "protocol": Dict[str, List[str]],
}
```

### 3.1 `images`（必填）

* 字典键：人类可读名（如 "RGB"/"DEPTH"/"IR"）。如 `ImageSpec.alias` 未显式给出，则默认使用该键作为别名。
* `ImageSpec`：

  ```python
  {
    "image":        np.ndarray | torch.Tensor,  # 必填；writer 会自动转 numpy
    "dataset_name": str,                        # 必填；这张图像的来源数据集（决定落盘目录）
    "modality":     str,                        # 可选；不填则回退为键名
    "alias":        str,                        # 可选；导出别名；不填回退为键名
    "extra":        dict                        # 可选；图像级元信息（JSON）
  }
  ```
* 落盘路径：`images/<dataset_name>/<UUID>.npy`。

### 3.2 `relation`（必填，≥1）

* `RelationSpec`：自由负载；`image_names` 可选。

  ```python
  {
    "image_names": ["RGB", "IR"],   # 可选；元素可写“键名”或“alias”；保持顺序
    "task_type": "classification",  # 以及任意业务字段
    "annotation": {...},
    "meta": {...}
  }
  ```
* 写入时会把 `image_names` → `image_ids` 注入到 `payload.image_ids`；其它键（含 `meta`）原样存入 `relations.payload`。

### 3.3 `protocol`（可多个）

* 类型：`Dict[str, List[str]]`，键是 `protocol_name`/`relation_set`，值为包含的 relation 名称列表（即上面 `relation` 的键）。
* 写入时将 `(protocol_name, relation_name)` 映射为 `relation_id`，生成 `protocol` 表记录。

---

## 4) 数据库存储结构

```
<database_root>/
├─ images/
│  └─ <dataset_name>/<UUID>.npy
└─ db/
   └─ catalog.duckdb
```

**DuckDB：**

```sql
-- 1) 每张图像
CREATE TABLE IF NOT EXISTS images (
  image_id     TEXT PRIMARY KEY,
  uri          TEXT NOT NULL,     -- 'images/<dataset_name>/<uuid>.npy'
  modality     TEXT,
  dataset_name TEXT NOT NULL,
  alias        TEXT,
  extra        JSON
);
CREATE INDEX IF NOT EXISTS idx_images_dataset ON images(dataset_name);

-- 2) 每条关系（任务实例）
CREATE TABLE IF NOT EXISTS relations (
  relation_id  TEXT PRIMARY KEY,
  payload      JSON NOT NULL      -- 包含 image_ids + 任意业务键（task_type/annotation/meta/...）
);

-- 3) 协议/集合分组
CREATE TABLE IF NOT EXISTS protocol (
  protocol_name TEXT NOT NULL,
  relation_id   TEXT NOT NULL,
  relation_set  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_protocol_set  ON protocol(relation_set);
CREATE INDEX IF NOT EXISTS idx_protocol_name ON protocol(protocol_name);
```

---

## 5) 进度显示策略

* 仅在 `write_from_registry` 层对**每个 adaptor**显示一个 `tqdm` 进度条。
* 进度条后缀（postfix）实时汇总：已写入 `images/relations/protocol` 的行数与 `err`（错误数）。
* 子步骤（写 `.npy`、构建 DataFrame、事务插入）**静默**执行，不逐条打印，避免刷屏。
* 若可估算总数（`len(adaptor)` 或 `len(adaptor.dataset)`），显示百分比；否则显示累计条。

---

## 6) 维护/清理（示例）

> 本方案**没有**“导入批次/原子替换/.trash”概念。清理建议基于业务集合（如特定 `protocol_name`/`relation_set`）或你自己的标记。

### 6.1 按集合名删除一批关系（及其协议行）

```sql
-- 找到某集合的 relation_id
WITH rels AS (
  SELECT DISTINCT relation_id
  FROM protocol
  WHERE relation_set = 'MYDS_train'
)
-- 先删 protocol，再删 relations（注意：images 是否删除看你是否被其它 relation 复用）
DELETE FROM protocol WHERE relation_id IN (SELECT relation_id FROM rels);
DELETE FROM relations WHERE relation_id IN (SELECT relation_id FROM rels);
```

> 若要删除对应图片：需要先解析 `relations.payload.image_ids` 聚合成集合，与 `images.image_id` 关联，再删文件与表行（注意有“被其它 relation 复用”的情况）。

---

## 7) 导出样例（按 alias 还原字典）

```python
import duckdb, json, numpy as np
from pathlib import Path

db_path = "/data/mydb/db/catalog.duckdb"
root = Path("/data/mydb")

con = duckdb.connect(db_path)

# 例：导出某个集合 'MYDS_train' 中第一条 relation 的字典 {alias: ndarray}
rid = con.execute("""
SELECT r.relation_id
FROM protocol p
JOIN relations r ON p.relation_id = r.relation_id
WHERE p.relation_set = 'MYDS_train'
LIMIT 1
""").fetchone()[0]

payload = con.execute("SELECT payload FROM relations WHERE relation_id = ?", [rid]).fetchone()[0]
payload = json.loads(payload)
image_ids = payload["image_ids"]

rows = con.execute(
    "SELECT image_id, uri, alias FROM images WHERE image_id IN ({})".format(
        ",".join(["?"] * len(image_ids))
    ),
    image_ids
).fetchall()

# 建映射并按 image_ids 顺序重建
id2row = {r[0]: (r[1], r[2]) for r in rows}  # image_id -> (uri, alias)

result = {}
for iid in image_ids:
    uri, alias = id2row[iid]
    arr = np.load(root / uri, allow_pickle=False)
    result[alias] = arr

# result 即 {alias: ndarray}
```

---

## 8) 常见问题

* **必须是 `.npy` 吗？** 是的，本实现固定写 `.npy`（读写快、无损且支持任意维度）；如需 `.jpg`/`.png`，建议另做导出工具。
* **`image_names` 必填吗？** 否。缺省则“使用该样本全部 images 且保序”。
* **`alias` 必填吗？** 否。缺省用 `images` 的键名。
* **`meta` 是否支持？** 是。任意键（`meta`/`task_type`/`annotation`/…）都会原样写入 `relations.payload`。
* **如何避免终端刷屏？** 已默认仅每 adaptor 一个进度条，子步骤静默，无逐条打印。
* **能否强制覆盖？** 本方案不提供“批次覆盖”。请按集合或条件执行清理 SQL，再重写入。

