# 使用示例（Markdown）

> 下述示例基于 **DuckDB-only** 实现，元数据写入到 `<database_root>/db/catalog.duckdb`，图像写入到 `<database_root>/images/<dataset>/`。当未开启 `force_rewrite` 且数据集已存在时会**自动跳过**。

## 1) 安装与准备

```bash
# Python 3.9+ 建议
python3 -m pip install duckdb pandas opencv-python

# 赋予脚本执行权限
chmod +x writeOriginDataset.sh
```

项目（关键部分）结构示例：

```text
.
├─ writeOriginDataset.sh
└─ OriginDataset/
   ├─ writer.py             # 本次实现的 DuckDB 写入器
   └─ utils_duckdb.py
```

## 2) 快速开始（命令行）

### 2.1 演练（不落盘）

```bash
./writeOriginDataset.sh -n
```

### 2.2 正式写入（从 `Config.setting.GetDatabaseConfig()` 读取 database\_root）

```bash
./writeOriginDataset.sh
```

### 2.3 指定并发与数据库根目录

```bash
./writeOriginDataset.sh -w 16 -d /data/mydb
```

### 2.4 强制重写（危险操作，会删除同名数据集的历史元数据与图像）

```bash
# 交互确认
./writeOriginDataset.sh -r

# 跳过确认
./writeOriginDataset.sh -r -y
```

### 2.5 使用环境变量覆盖参数

```bash
# 与 -d/-w/-r/-n 等价
DB_ROOT=/data/mydb MAX_WORKERS=32 FORCE_REWRITE=1 ./writeOriginDataset.sh
```

## 3) 以 Python 方式调用

### 3.1 从注册器写入（与你原始流程等价）

```python
from Config.setting import GetDatabaseConfig
from OriginDataset.writer import write

cfg = GetDatabaseConfig()
write(cfg, force_rewrite=False, dry_run=False, max_workers=16)
```

### 3.2 细粒度控制：单数据集写入

```python
import cv2
from OriginDataset.writer import DatasetWriter

# 构造示例数据
images = {
    "imgA.jpg": cv2.imread("/path/to/imgA.jpg"),
    "imgB.jpg": cv2.imread("/path/to/imgB.jpg"),
}
relations = {
    "rel1": {"image_names": ["imgA.jpg", "imgB.jpg"], "type": "pairing", "labels": ["cat", "dog"]},
}
protocols = {
    "train_v1": ["rel1"]
}

writer = DatasetWriter(database_root="/data/mydb", max_workers=16)
writer.write_dataset(
    dataset_name="MyDataset",
    images=images,
    relations=relations,
    protocols=protocols,
    force_rewrite=True,   # 不想覆盖就设为 False；已存在会自动跳过
    dry_run=False,
)
```

## 4) 行为说明与小贴士

* **存在即跳过**：`force_rewrite=False` 时，如果 DuckDB `relations` 存在该 `dataset_name` 的记录，或 `images/<dataset_name>/` 非空，写入将被跳过并打印 `[SKIP]`。
* **安全重写**：`force_rewrite=True` 时，会先将旧图像目录移动到 `.trash/<dataset>-<timestamp>.quarantine/`，DuckDB 在事务中删除旧条目；写入成功后再清理隔离目录。
* **并发**：图片写入使用 `ThreadPoolExecutor` 并发落盘到临时目录，完成后原子替换。
* **性能**：DuckDB 采用批量 `INSERT` + 索引（`dataset_name`/`relation_set`/`protocol_name`）加速查询与清理。
