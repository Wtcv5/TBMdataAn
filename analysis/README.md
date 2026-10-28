# TBM数据审计与初始预测基线

原始数据位于 `../TBMData/TBMData`。脚本不修改原始文件，全部输出写入 `outputs`。
本项目使用当前已安装的Python环境，依赖版本见 `requirements.txt`。

从项目根目录运行：

```powershell
$env:PYTHONIOENCODING='utf-8'
python -u analysis/audit_data.py
python -u analysis/check_overlap.py
python -u analysis/investigate_overlap.py
python -u analysis/build_baselines.py
python -u analysis/report_results.py
python -u analysis/verify_outputs.py
```

`audit_data.py`全量读取原始 `tbm_data*.csv` 的表头、时间和核心信号，并扫描未标机器编号的Excel文件。
其他传感器仅清点表头；缺失率、数值分布等质量统计聚焦F、T、n、v、p，不代表已验证全部辅助字段。
UTF-8和GB18030导出文件分别读取。`codes`、`common_using`、已分周期数据和zip备份不计作新增原始样本。
Excel单独审计，不因它有一百多万行就将它当成一台新机器。

`build_baselines.py`建立时间戳、缺口和数据集边界严格约束的多步预测：

- 输入仅为F、T、n、v当前和前3次观测及其差分、时间间隔；p不作为输入。
- 目标为未来F、T、v；30、60、120秒后首个真实观测作为标签，最大延迟5秒。
- 不插值、不提前使用未来转速，不跨越超过60秒的采样缺口，不跨字段口径不同的目录构造窗口。
- 预测起点要求四信号为正；历史和标签必须有限且非负。未来停机零值仍纳入评估。
- 按每台机器全部去重原始记录的60%/20%/20%时间戳划分训练/验证/测试；跨边界标签窗口清除。
- 同时间戳核心信号冲突的所有记录排除；相同记录仅保留一次。
- 对不同机器下时间戳和四个有效信号完全相同的记录，隔离涉及的原始文件双方，清单保存在`quarantined_files.csv`；不移动原文件。
- 每台机器训练最多6000窗口、验证和测试各最多10000窗口，种子42，均匀随机选取后恢复时间顺序。
- 基线为持久性、Ridge增量回归、HistGradientBoosting增量回归；回归模型输出裁剪到非负。
- 超参数预先固定，不使用验证/测试数据选择模型或调参。验证集指标单独保存，供后续研究使用。
- 同机器时间外推与留一机器两种协议使用同一测试窗口。留一机器完全不拟合目标机数据。
- 各时域仅纳入具有至少300训练、50验证、50测试窗口的机器；留一机器源域取这些合格机器的训练部分。
- 各目标原始尺度MAE/RMSE/R²按机器报告。跨机器总体结果取各机相对持久性MAE降低率的等权平均。
- 置信区间按预测起点所在的日分块重采样1000次；少于5日不报告区间。它不证明跨独立机器总体上的显著性。

`outputs/models`保存可重新加载的模型；`outputs/predictions`保存逐窗口真值、预测值、起止时间和实际时域。
`outputs/datasets`保留输入窗口及划分标签；`split_manifest.csv`给出每机分界时间。
单位未经设备字典确认，当前指标均以原始导出数值计量；不将推进压力替代推力，不构造功率、比能或地质标签。

再次运行审计会重建CSV缓存；已存在的Excel缓存会复用。如Excel原文件被替换，需要在确认后先清理对应生成缓存再重新审计。
