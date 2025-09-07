# 文件系统组织方式
```
<database_root>/
├─ images/                               # 所有图像根目录
│  ├─ <dataset_name>/                    # 每个数据集单独一个目录（原子替换写入）
│  │  ├─ <UUID>.jpg                      # 随机 UUID 命名的 JPEG 文件
│  │  └─ ...
├─ db/
│  └─ catalog.duckdb                     # DuckDB 元数据库（仅此处存 relations / protocols）
└─ .trash/                               # 强制重写时的隔离区（安全回滚）
   └─ <dataset_name>-<timestamp>.quarantine/
      └─ (旧的 images/... 被整体移动到这里)
```
## DuckDB 存储（文件：<database_root>/db/catalog.duckdb）

```
CREATE TABLE IF NOT EXISTS relations (
  relation_id   TEXT PRIMARY KEY,   -- 关系条目的唯一 ID（UUID）
  dataset_name  TEXT NOT NULL,      -- 该关系所属的数据集名（用于按数据集清理/查询）
  payload       JSON NOT NULL       -- 关系对象的 JSON：已将 image_names 映射为 image_ids
);
```
```
CREATE TABLE IF NOT EXISTS protocol_entries (
  protocol_name TEXT NOT NULL,      -- 协议名称（如 train_v1 / eval 等）
  relation_id   TEXT NOT NULL,      -- 指向 relations.relation_id（外键不强制）
  relation_set  TEXT NOT NULL       -- 等同于数据集名（用于按数据集清理/查询）
);
```
-- 索引（加速查询与按数据集删除）
```
CREATE INDEX IF NOT EXISTS idx_relations_dataset ON relations(dataset_name);
CREATE INDEX IF NOT EXISTS idx_protocols_set    ON protocol_entries(relation_set);
CREATE INDEX IF NOT EXISTS idx_protocols_name   ON protocol_entries(protocol_name);
```

## relations.payload 示例（具体键由各 Adaptor 产出，写入前已将 image_names -> image_ids）
```
{
  "type": "pairing",
  "labels": ["cat", "dog"],
  "image_ids": [
    "a1b2c3d4e5f6...",      // 与 images/<dataset_name>/<UUID>.jpg 对应
    "0f9e8d7c6b5a..."
  ],
  "meta": {
    "source": "AdaptorX",
    "version": 1
  }
}
```

# 写入与覆盖策略（逻辑概览）

1) 预检查：_dataset_exists(dataset_name)
   - 若 DuckDB 的 relations 存在该 dataset_name 的任何行 → 视为已写入
   - 或 images/<dataset_name> 目录存在且非空 → 视为已写入
   - 未设置 force_rewrite 时遇到已写入 → 直接跳过

2) 正常写入
   - 并发将图像写入临时目录 images/<dataset_name>__tmp-<rand>/
   - relations: 生成 {relation_id, dataset_name, payload(JSON)} 批量插入 DuckDB
   - protocol_entries: 生成 {protocol_name, relation_id, relation_set} 批量插入 DuckDB
   - 临时目录原子替换为 images/<dataset_name>/

3) 强制重写（force_rewrite=True）
   - 将旧的 images/<dataset_name>/ 移至 .trash/<dataset_name>-<ts>.quarantine/
   - DuckDB 事务内删除：
       DELETE FROM protocol_entries WHERE relation_set = <dataset_name>;
       DELETE FROM relations         WHERE dataset_name = <dataset_name>;
   - 写入新数据成功并提交后，清理 quarantine 目录
