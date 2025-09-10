

# OriginDataset 写入指南（DuckDB + `.npy`）

> **连接与建表现在统一由 `BigVisionDatabase` 管理**；`DatasetWriter` 仅专注写入。  
> 元数据：`<database_root>/db/catalog.duckdb`；图像：`<database_root>/images/<dataset_name>/<UUID>.npy`。  
> 进度：每个 adaptor 一个进度条（后缀汇总，不刷屏）。

---

## 1) 安装与准备

```bash
python3 -m pip install duckdb pandas numpy tqdm
# 若 adaptor 返回 torch.Tensor，需要安装 PyTorch（按你的环境）
# python3 -m pip install torch
````

项目结构（关键）：

```
.
├─ Database/
│  ├─ db.py             # BigVisionDatabase（集中管理 duckdb 连接 & schema）
│  └─ utils.py          # require_duckdb / init_duckdb_schema / ensure_dir
├─ OriginDataset/
│  ├─ writer.py         # DatasetWriter（仅写入，接收现有连接）
│  └─ utils.py          # ensure_dir（数据集侧）
├─ scripts/
│  └─ writeOriginDataset.sh
└─ Config/
   └─ setting.py        # 提供 GetDatabaseConfig() -> {"database_root": "..."}
```

---

## 2) 快速开始

### 2.1 从注册器批量写入（默认）

```bash
./scripts/writeOriginDataset.sh
```

等价于：

```bash
python -m Database.db --registry
```

* 会从 `Config.setting.GetDatabaseConfig()` 读取 `database_root`，并显示“每个 adaptor 一个”进度条。
* 若未配置 `Config.setting`，请显式指定 `-d`：

```bash
python -m Database.db --registry -d /data/mydb
```

### 2.2 DEMO 单样本写入（自检）

```bash
python -m Database.db --demo-sample -d /data/mydb
```

会写入一个名为 `DEMO` 的 3 模态样本，验证目录/scheme/事务是否工作正常。

---

## 3) 常用参数

```bash
python -m Database.db \
  --registry \                # 或 --demo-sample
  -d /data/mydb \             # 数据库根目录；不提供则尝试从 Config.setting 读取
  --duckdb-path /data/mydb/db/catalog.duckdb \  # 可选；缺省为 <database_root>/db/catalog.duckdb
  -w 16 \                     # 写图并发（默认 8）
  -t 8 \                      # DuckDB PRAGMA threads（0=让 DuckDB 自己决定）
  -n \                        # dry-run：演练模式，不落盘
  --no-progress               # 禁用进度条（仅对 registry 模式有效）
```

也可用环境变量（见脚本）：

```bash
DB_ROOT=/data/mydb MAX_WORKERS=16 THREADS=8 ./scripts/writeOriginDataset.sh
```

---

## 4) Adaptor 输出结构要求（复述）

`__getitem__` 必须返回：

```python
{
  "images":   Dict[str, ImageSpec],
  "relation": Dict[str, RelationSpec],
  "protocol": Dict[str, List[str]],
}
```

* `images`：每项 `ImageSpec` 至少包含

  ```python
  {
    "image": np.ndarray | torch.Tensor,
    "dataset_name": str,         # 这张图像来自哪个数据集
    "modality": "RGB/DEPTH/IR",  # 可选；缺省回退为键名
    "alias": "RGB",              # 可选；导出别名；缺省回退为键名
    "extra": {...}               # 可选；图像级元信息（JSON）
  }
  ```
* `relation`：

  ```python
  {
    "image_names": ["RGB", "IR"],  # 可选；元素可为键名或 alias；缺省=用全部 images（按插入顺序）
    "task_type": "classification",
    "annotation": {...},
    "meta": {...}                  # 任意扩展字段都会原样写入 payload
  }
  ```
* `protocol`：`{ "PADISI_train": ["FAS"], ... }`，键既是 `protocol_name` 也是 `relation_set`。

---

## 5) 行为说明

* `.npy` 落盘，文件名为随机 `UUID`；路径：`images/<dataset_name>/<UUID>.npy`。
* `relations.payload` 写入前会把 `image_names` 映射成 `image_ids`（不提供时默认取全部 images，保序）。
* `protocol` 表记录集合与 `relation_id` 的多对多关系。
* 连接 & 事务 & schema 由 `BigVisionDatabase` 统一管理，`DatasetWriter` 只负责写入。

---

## 6) 故障排查

* **报错 `relation references unknown image name/alias`**：检查 `relation.image_names` 是否与 `images` 的键或 `alias` 对齐。
* **无法自动找到 `database_root`**：显式传 `-d`，或检查 `Config.setting.GetDatabaseConfig()` 是否返回 `{"database_root": "..."}`。
* **多进程/多线程并发**：DuckDB 写入建议保持**单进程**，这个实现通过在 `BigVisionDatabase` 内持有唯一连接来规避并发写问题。

