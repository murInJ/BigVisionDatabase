# Compact Dataset — **{protocol_name}**

**Schema:** `{schema}`
**Created (UTC):** `{created_utc}`

这是一个 **开箱即用（train-ready）** 的打包数据集，包含：
- `images/`                  ：原样 **NPY** 图像（数组形状与 dtype 保持数据库一致）
- `samples/`                 ：训练样本清单（分片 Parquet：`samples-00000.parquet`, ...）
- `manifest.json`            ：分片校验和等元数据
- `{loader_entry}`           ：直接可用的 PyTorch `Dataset` 加载器
- `README.md`                ：当前说明文件

---

## 目录结构

````
<protocol>.compact/
    images/
        <dataset_name>/
            <image_id>.npy
    samples/
        samples-00000.parquet
        samples-00001.parquet
        ...
    {loader_entry}
    manifest.json
    README.md
````

---

## 快速开始（PyTorch）

`{loader_entry}` 已包含 `CompactDataset`，可直接用于训练/验证。

```python
from loader import CompactDataset
from torch.utils.data import DataLoader

# 支持传入目录或 .zip
ds = CompactDataset("/path/to/<protocol>.compact/")  # 或 "/path/to/<protocol>.compact.zip"
dl = DataLoader(ds, batch_size=4, num_workers=0, shuffle=False, collate_fn=lambda x: x)

for batch in dl:
    sample = batch[0]
    # sample["images"] 是 {{alias: numpy.ndarray}}
    # sample["annotation"] 是 dict
    # sample["task_type"] 等同导出时的字段
    ...
````

---

## Parquet Schema（每行）

* `protocol_name` : `string`
* `relation_id`   : `string`
* `image_paths`   : `list<string>`（相对数据集根；指向 `images/*.npy`）
* `image_aliases` : `list<string>`（与 `image_paths` 一一对应）
* `task_type`     : `string | null`
* `annotation`    : `JSON string`（用于自定义标注）
* `extra`         : `JSON string`（用于附加信息）

> **注意**：图像是 **NPY 原样**（无归一化/颜色变换）。你可在 DataLoader 中自行转换为 tensor/做增强。

---

## 校验

我们在 `manifest.json` 中记录了 `samples-*.parquet` 的 `sha256`，可快速检测损坏或不完整：

* 使用我们的 `loader.py` 自带 `verify_compact_dataset(path)`；
* 或者自行对照 `manifest.json` 计算校验和。

---

## 常见问题

* **能直接读取 zip 吗？**
  是的，`CompactDataset` 支持传入 `.zip` 路径，会在本地缓存目录自动解压并复用。

* **为什么使用 NPY？**
  避免重复转码和精度损失；保持与 DB 中的数组一致，训练时更可控。

* **image_aliases 与键名**
  导出时会保留每张图像的 `alias`（若缺省则回退为键名），训练时可据此路由分支或拼接通道。

祝你训练顺利！🔥

