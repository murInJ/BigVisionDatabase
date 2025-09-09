# BigVision **Bundle** — Protocol: **{protocol_name}**

**Schema:** `{schema}`
**Created (UTC):** `{created_utc}`

本包是用于**备份/迁移/复现**的完整快照（可回灌 DB）。它包含：
- `relations.jsonl`：每行 `{{"relation_id", "payload"}}`，其中 `payload.image_ids` 已解析为数据库中的 `image_id`。
- `images_index.jsonl`：每行 `{{"image_id","rel_path","dataset_name","modality","alias","dtype","shape","checksum_sha256","extra"}}`，描述图像在本 Bundle 内的位置与元数据。
- `protocol.json`：`{{"protocol_name","relation_ids","counts"}}`，列出本协议的所有关系条目 ID。
- `manifest.json`：导出元数据与关键文件的 `sha256` 校验和。
- `images/`：按 `images/<dataset_name>/<image_id>.npy` 布局存放的图像（若导出使用了 `manifest-only`，此目录可能为空或不齐全）。
- `thumbnails/`（可选）：导出的缩略图（便捷预览），不参与校验。
- `README.md`：当前说明文件。

---

## 文件结构

```
{protocol_name}.bvbundle/
  images/
    <dataset/_name>/
      <image_id>.npy
  relations.jsonl
  images_index.jsonl
  protocol.json
  manifest.json
  README.md
  thumbnails/             # 可选
````

---

## 快速校验

- 校验清单：
  - `manifest.json` 包含关键文件 (`relations.jsonl`, `images_index.jsonl`, `protocol.json`) 的 sha256。
  - 你也可以使用项目的 `verify_bundle` 工具函数进行更严格校验（包括尺寸/类型的轻量检查）。

---

## 回灌到数据库（DB Import）

使用本项目的 `BigVisionDatabase.load_bundle` 将本 Bundle 导入到目标数据库：

```python
from Database.db import BigVisionDatabase

db = BigVisionDatabase(database_root="/path/to/your/dbroot")

# 严格模式：若已有同名行会报错
db.load_bundle("/path/to/{protocol_name}.bvbundle.zip", mode="strict", copy_mode="copy", verify=True)

# 覆盖模式：允许覆盖已存在的 images/relations/protocol 映射
# db.load_bundle("/path/to/{protocol_name}.bvbundle.zip", mode="overwrite", copy_mode="copy", verify=True)

# 跳过已存在：仅插入缺失的项
# db.load_bundle("/path/to/{protocol_name}.bvbundle.zip", mode="skip-existing", copy_mode="hardlink", verify=True)
````

**参数说明：**

* `mode`：

  * `strict`：任何冲突直接中止；
  * `overwrite`：覆盖已有行与文件；
  * `skip-existing`：跳过已存在的行与文件；
* `copy_mode`：

  * `copy`：普通复制，跨盘通用；
  * `hardlink`：同盘零拷贝，最快但受文件系统限制；
  * `symlink`：软链接，跨盘也快，但移动目标后可能失效。

> **注意**：若输入是 `.zip`，导入流程会自动做流式复制到 DB 布局，不会整包解压占满磁盘。

---

## `relations.jsonl` 与 `images_index.jsonl` 示例

**relations.jsonl（行示例）**

```json
{{"relation_id":"abc123...", "payload":{{"task_type":"classification","annotation":{{"label":1}},"image_ids":["id1","id2"]}}}}
```

**images_index.jsonl（行示例）**

```json
{{"image_id":"id1","rel_path":"images/PADISI/id1.npy","dataset_name":"PADISI","modality":"RGB","alias":"RGB","dtype":"uint8","shape":[224,224,3],"checksum_sha256":"...","extra":"{{}}"}}
```

---

## 提示与约定

* **图像以 NPY 原样保存**（无颜色/量化改动），可用 `numpy.load` 直接读取。
* `modality/alias` 仅用于语义标记与导出时的文件命名，不影响数据内容。
* 如需与其他工程共享图像目录，可在导出时选择 `manifest-only`，此时 Bundle 仅包含索引而不携带图片本体（适合共享盘/镜像场景）。

祝使用愉快！🚀

````
