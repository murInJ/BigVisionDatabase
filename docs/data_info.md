# 数据集写入规范（DuckDB + `.npy` 方案）

## 文件系统组织方式

```
<database_root>/
├─ images/                               # 所有图像根目录
│  ├─ <dataset_name>/                    # 按“每张图像自身的来源数据集”分目录
│  │  ├─ <UUID>.npy                      # 使用 .npy 无损持久化（ndarray/tensor）
│  │  └─ ...
└─ db/
   └─ catalog.duckdb                     # DuckDB 元数据库（仅此处存 images / relations / protocol）
```

* **没有导入批次命名空间**；一次写入可以是**多来源数据集混合**，图片会落到各自的 `images/<dataset_name>/` 目录下。
* **不使用原子替换目录/隔离区**；如果需要“回滚/清理”，请基于 `protocol`/`relations` 的查询结果自行删除相关行与对应文件。

---

## DuckDB 存储（文件：`<database_root>/db/catalog.duckdb`）

### 1) 表：`images`

> 记录每张图像的来源、路径与导出别名

```sql
CREATE TABLE IF NOT EXISTS images (
  image_id     TEXT PRIMARY KEY,
  uri          TEXT NOT NULL,     -- 'images/<dataset_name>/<uuid>.npy'
  modality     TEXT,              -- 可选：来自 adaptor；缺省回退为 images 字典的键名
  dataset_name TEXT NOT NULL,     -- 必填：这张图像的来源数据集（决定磁盘目录）
  alias        TEXT,              -- 可选：导出时使用的别名；缺省回退为键名
  extra        JSON               -- 可选：与单张图像相关的自定义元信息
);

CREATE INDEX IF NOT EXISTS idx_images_dataset ON images(dataset_name);
```

### 2) 表：`relations`

> 每条“关系/任务实例”，只存一个 JSON 负载（payload）

```sql
CREATE TABLE IF NOT EXISTS relations (
  relation_id  TEXT PRIMARY KEY,
  payload      JSON NOT NULL      -- 写入前已将 image_names → image_ids；其余键原样保留
);
```

### 3) 表：`protocol`（由 `protocol_entries` 重命名）

> 将关系分配到一个或多个“协议/集合”中

```sql
CREATE TABLE IF NOT EXISTS protocol (
  protocol_name TEXT NOT NULL,    -- 协议/集合名（由 adaptor 决定）
  relation_id   TEXT NOT NULL,    -- 指向 relations.relation_id（未强制外键）
  relation_set  TEXT NOT NULL     -- 由 adaptor 决定；可与 protocol_name 相同或不同
);

CREATE INDEX IF NOT EXISTS idx_protocol_set  ON protocol(relation_set);
CREATE INDEX IF NOT EXISTS idx_protocol_name ON protocol(protocol_name);
```

---

## `relations.payload` 示例

> 除 `image_ids` 外，其它键（如 `task_type`/`annotation`/`meta`）**完全由 Adaptor 决定**并原样写入。

```json
{
  "task_type": "classification",
  "annotation": {
    "label": 1
  },
  "meta": {
    "source": "PADISI",
    "version": 2,
    "difficulty": "hard",
    "note": "produced by adaptor v1.1"
  },
  "image_ids": [
    "a1b2c3d4e5f6...",
    "0f9e8d7c6b5a..."
  ]
}
```

* `image_ids` 的**顺序**取自该 relation 的 `image_names` 顺序；若 relation 未提供 `image_names`，则使用该样本 `images` 字典的**插入顺序**（Python 3.7+ 保序）。

---

## 写入流程（高层逻辑）

1. **预创建目录**：对本条样本使用到的所有 `images[*].dataset_name`，确保 `images/<dataset_name>/` 目录存在。
2. **并发写图**：把 `np.ndarray/torch.Tensor` 以 `.npy` 保存为 `images/<dataset_name>/<UUID>.npy`；同时收集 `image_id/uri/modality/dataset_name/alias/extra`。
3. **构建 relations**：

   * 若指定了 `image_names`：把其中的每个名称（可为“键名或 alias”）映射为对应的 `image_id`；保持顺序。
   * 若未指定：默认把当前样本 `images` 的所有条目按插入顺序映射为 `image_id`。
   * 将 `image_ids` 注入到 relation 的 `payload` JSON 中（其它字段原样保留）。
4. **构建 protocol**：

   * 对每个 `protocol_name`/`relation_set` 的键，将其值（relation 名称列表）映射到相应的 `relation_id`，写入 `protocol` 表。
5. **一次事务写入 DuckDB**：批量插入 `images`、`relations`、`protocol`。
6. **进度显示**：仅在 `write_from_registry` 层为**每个 adaptor**显示一个 `tqdm` 进度条；后缀（postfix）汇总 `images/relations/protocol/err` 数量，避免终端刷屏。

> 说明：由于不再有“导入批次”的概念，也就没有 `.trash/` 隔离区与整目录原子替换。如果需要“重新导入/清理”，建议：
>
> * 先按业务键（如某个 `protocol_name` 或 `relation_set`）选出 `relation_id`；
> * 读取对应 `relations.payload.image_ids`；
> * 事务删除 `protocol/relations` 行，并视需要删除 `images` 行与其 `.npy` 文件。

---

# Adaptor 的数据组织要求

Adaptor 的 `__getitem__` 必须返回这个顶层结构（键名固定）：

```python
{
  "images":   Dict[str, ImageSpec],
  "relation": Dict[str, RelationSpec],
  "protocol": Dict[str, List[str]],
}
```

## 1) `images`（必填）

* 类型：`Dict[str, ImageSpec]`
* **字典键**：人类可读的名称（如 "RGB" / "DEPTH" / "IR"）。如果 `ImageSpec.alias` 未显式给出，默认使用该键作为别名。
* **ImageSpec 字段**：

```python
ImageSpec = {
  "image":        np.ndarray | torch.Tensor,  # 必填；支持 (H,W) / (H,W,C) 等
  "dataset_name": str,                        # 必填；这张图像的来源数据集（决定磁盘目录 & images.dataset_name）
  "modality":     str,                        # 可选；如 "RGB"/"DEPTH"/"IR"；不填回退为字典键名
  "alias":        str,                        # 可选；导出用别名；不填回退为字典键名
  "extra":        dict                        # 可选；图像级元信息（JSON 存储到 images.extra）
}
```

* **落盘路径**：`images/<dataset_name>/<UUID>.npy`

## 2) `relation`（必填，至少 1 个）

* 类型：`Dict[str, RelationSpec]`
* **RelationSpec**：自由的业务负载，**可选**的 `image_names` 控制参与图像子集：

```python
RelationSpec = {
  # 可选：限制该 relation 使用的图像子集；元素可写“images 的键名”或“alias”。
  "image_names": ["RGB", "IR"],

  # 下面任意业务字段都会原样写入 relations.payload：
  "task_type": "classification",
  "annotation": {...},   # 例如 {"label": int 或 List[...] }
  "meta": {...},         # 可选：任意扩展元信息，如 {"source": "PADISI", "version": 2}
  # ... 其它你需要的键
}
```

* **规则**：

  * 若提供 `image_names`：严格按其顺序映射为 `image_ids`。
  * 若不提供：默认使用本样本 `images` 的**全部条目**，顺序为字典插入顺序。
  * 写入时 `writer` 会把 `image_ids` 注入到 `payload` 中，其它键（包括 `meta`）**不做改名或过滤**。

## 3) `protocol`（可多个）

* 类型：`Dict[str, List[str]]`
* **键**：`protocol_name` / `relation_set`，例如 `"PADISI_train"`, `"PADISI_eval"`, `"ALL"`, …
* **值**：该集合包含的 relation 名称列表（即 `relation` 字典的键名）。
* 写入时会将 `(protocol_name, relation_name)` 映射为具体 `relation_id` 并生成 `protocol` 表记录。

---

## 参考示例（PADISI）

```python
images = {
  "RGB":   {"image": rgb_nd,   "dataset_name": "PADISI", "modality": "RGB",   "alias": "RGB"},
  "DEPTH": {"image": depth_nd, "dataset_name": "PADISI", "modality": "DEPTH", "alias": "DEPTH"},
  "IR":    {"image": ir_nd,    "dataset_name": "PADISI", "modality": "IR",    "alias": "IR"},
}

relation = {
  "FAS": {
    "image_names": ["RGB", "DEPTH", "IR"],   # 也可只选 ["RGB","IR"]
    "task_type": "classification",
    "annotation": {"label": label},
    "meta": {"source": "PADISI", "version": 2}
  },
  # 也可以定义更多 relation，使用不同子集：
  # "FAS_rgb_ir": { "image_names": ["RGB", "IR"], "task_type": "classification", "annotation": {...} }
}

protocol = {
  "PADISI_train": ["FAS"],      # 一个或多个集合
  # "PADISI_all": ["FAS", "FAS_rgb_ir"]
}
```

---

## 进度显示策略

* **每个 adaptor 一个进度条**（`tqdm`）。
* 进度条 `postfix` 动态汇总：已写入的 `images`/`relations`/`protocol` 行数与 `err`（错误数）。
* 子步骤（写图/构建 DF/DB 插入）**静默**，不逐条打印，避免终端刷屏。
* 若可估算总数（`len(adaptor)` 或 `len(adaptor.dataset)`），进度条显示百分比；否则显示累计条。

---

