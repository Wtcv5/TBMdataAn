# TBMdataAn

TBM运行数据审计与初始预测基线。项目检查多设备导出数据的字段完整性、时间连续性和跨设备重复，并建立严格按时间划分的多步预测比较。

## 本轮结果

- 审计1,531个原始CSV，共7,628,284行，另审计Excel汇总表1,035,283行。
- 发现不同TBM编号下的时间戳和四个核心信号存在完全相同记录；在当前基线中隔离相关源文件双方，共210个文件。
- 清理后，TBM5、TBM6、TBM8、TBM9、TBM10满足本轮训练、验证和测试条件。
- 比较持久性、Ridge增量回归和梯度提升树，预测30、60、120秒后的推力、扭矩和推进速度。
- 当前回归模型尚未稳定超过持久性基线；设备单位及部分数据来源仍待核实。

## 查看结果

- [完整审计与预测基线报告](analysis/outputs/审计与预测基线报告.md)
- [逐设备指标](analysis/outputs/baseline_metrics.csv)
- [设备等权汇总指标](analysis/outputs/baseline_macro.csv)
- [隔离文件清单](analysis/outputs/quarantined_files.csv)
- [核验结果](analysis/outputs/verification.json)
- [复现步骤和评估协议](analysis/README.md)

## 复现

使用Python 3.11，依赖版本见[requirements.txt](analysis/requirements.txt)。将原始数据放置在项目根目录的`TBMData/TBMData/`，按[分析说明](analysis/README.md)依次运行审计、重复核查、基线、报告和验证脚本。

仓库包含代码、审计统计、模型指标、报告和图表。原始数据、压缩备份、本地Python环境及生成的Parquet数据集、模型二进制、逐窗口预测缓存不纳入Git；这些派生文件可通过脚本重建。

原始数据的设备编号与真实来源需进一步核对。当前指标使用未经单位转换的导出数值，不应据此推断地质状态、故障风险或最优控制策略。
