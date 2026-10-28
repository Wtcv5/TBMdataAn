# 原始数据（Git LFS）

本目录保存全部1,531份原始CSV和1份原始Excel的无损ZIP压缩包，按原数据目录分包。
重复来源、缺失字段以及在当前基线中隔离的数据也完整保留。
压缩包不包含原目录中的Python环境、第三方库、旧处理代码或已生成的周期切分结果。

## 下载与还原

安装Git LFS后，在仓库根目录运行：

```bash
git lfs install
git lfs pull
python analysis/restore_raw_data.py
```

脚本会核对压缩包和每份解压文件的SHA-256，将文件还原至原路径`TBMData/TBMData/`。
已有文件若一致则保留；若内容不同则停止，不覆盖。
仅检查压缩包及内部文件而不解压：

```bash
python analysis/restore_raw_data.py --verify-only
```

`manifest.json`记录原文件路径、大小、SHA-256和对应压缩包；`verification.json`记录最近一次完整校验结果。
压缩包可以用常规ZIP工具解压，但建议通过脚本同时验证完整性。

原始数据的单位、设备编号归属和跨设备重复问题见[审计报告](../../analysis/outputs/审计与预测基线报告.md)。
