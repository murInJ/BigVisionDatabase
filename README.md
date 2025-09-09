# BigVisionDatabase

## Version
0.0.2

## Update Plan
- `v0.0.3` Database Entry
  - 修改images磁盘存储方式为.npy,OriginDataset/writer.py新建images信息表存储图像信息
  - 指定一组images进行可视化查看和导出(增删查改)
  - relation增删查改
  - protocol增删查改，组织现有protocol或relation形成新protocol,采样protocol
  - 导出protocol为可直接使用的训练测试数据集
  - 导出指定datasets,relation,protocol为指定格式的备份数据集，方便后续以adaptor形式重新迁移导入。


