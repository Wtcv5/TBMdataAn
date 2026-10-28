# TBM数据审计与初始预测基线

原始数据位于 `../TBMData/TBMData`。脚本不修改原始文件，全部输出写入 `outputs`。
本项目使用当前已安装的Python环境，依赖版本见 `requirements.txt`。

从GitHub克隆后，先按[原始数据说明](../data/raw/README.md)运行`git lfs pull`和`python analysis/restore_raw_data.py`，以还原完整输入。

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

## 最小验证方案

在设计新模型之前，先检验固定基线为何难以超过持久性：时间标签核查、A训练样本量、B简单基线与验证集选择、C按未来状态分层的事后误差归因、D公平的同机与留一机器比较、E毛刺污染（探索性）。
方案和预设判定规则见[最小验证方案与初步诊断](outputs/最小验证方案与初步诊断.md)，真实数据上的结果见[最小验证方案实验结果](outputs/最小验证方案实验结果.md)。

```powershell
python -u analysis/preliminary_from_aggregates.py   # 仅用已提交汇总，无需原始数据
python -u analysis/smoke_test_validation.py         # 合成数据自检，无需原始数据
python -u analysis/validation_experiments.py all    # 或 horizon、A、B、C、D、E 单独运行
python -u analysis/validation_experiments.py A --clean-train   # 训练池剔除毛刺的敏感性版本，D同理
python -u analysis/report_validation.py             # 按预设规则机械判定
```

实验读取基线生成的`outputs/datasets/`和`outputs/canonical/`；验证和测试窗口与基线完全相同，训练只用训练期窗口，模型选择只用验证窗口，未来状态只用于事后统计，不进入特征。
结果写入`outputs/validation/`；逐窗口预测、真值与未来状态写入`outputs/validation_predictions/`，不纳入Git。
不要同时运行两个实验进程：每个进程占4个线程，超额订阅会使拟合慢百倍以上。

## P0 可信机器与数据源清单

路线图第一步：解决跨机器数据重合、来源和单位问题。报告见[P0可信机器与数据源清单](outputs/P0可信机器与数据源清单.md)，输出在`outputs/p0/`。脚本只读原始CSV，依次运行：

```powershell
python -u analysis/p0_file_provenance.py   # 逐文件指纹：里程计数器、比值、采样、行哈希（行哈希缓存在outputs/p0_cache，不纳入Git）
python -u analysis/p0_counter_events.py    # 里程计数器跳变
python -u analysis/p0_trust_list.py        # 副本、外来行、单位检验，生成trusted_sources.csv和trusted_machines.csv
python -u analysis/p0_tbm9_regime.py       # TBM9转速列的证据
python -u analysis/p0_copy_columns.py      # 副本文件的逐列比较
python -u analysis/p0_near_duplicates.py   # 相同时间戳的近重复筛查（需要audit_data.py写出的outputs/canonical）
python -u analysis/smoke_test_p0.py        # 合成数据自检，无需原始数据
```

每个文件得到一个状态（`trusted`、`flagged`、`repair`、`signal_invalid`、`truncate`、`exclude`），判定规则和阈值写在`p0_trust_list.py`开头。
清单是文件级和行级的，不改动原始文件；基线和验证实验目前仍使用旧的“核心列相同的文件双方一起隔离”的处理，要到P5才按新清单重跑。

## 标准时序数据v1（P1）

路线图第二步：统一字段、时间戳和质量标记。数据卡见[标准时序数据v1数据卡](outputs/标准时序数据v1数据卡.md)。输入是P0的清单（`outputs/p0/`），输出在`outputs/standard_v1/`：每台机器一个parquet文件（不纳入Git，约80秒重建）以及字段字典、列映射、行账目、标记统计、采样与缺口、manifest和校验结果。

```powershell
python -u analysis/p1_standardize.py   # 读P0清单，写 outputs/standard_v1/（--machines TBM8,TBM9 可只建部分机器）
python -u analysis/p1_verify.py        # 81项检查：manifest、行账目、时间、标记、转速修复、原始数据抽查、与canonical缓存逐值对比
python -u analysis/smoke_test_p1.py    # 合成数据自检，无需原始数据
```

v1只改动三类东西：排除P0判定的副本和外来行、丢弃整行完全相同的重复行、按默认处理修复TBM9等机器的转速列（原值保留在`n_raw`）。其余读入的值与`canonical`缓存逐值一致。异常值只标记不删除，默认筛选是`valid_core`。

## 连续掘进段与阶段划分（P2）

路线图第三步：建立连续掘进段及阶段划分。报告见[P2分段与阶段划分报告](outputs/P2分段与阶段划分报告.md)。输入是v1（`outputs/standard_v1/`），输出在`outputs/segments_v1/`：段表、段表字典、汇总、参数敏感性、推进量对照、人工抽样验证包，以及每台机器一个逐行标签的parquet文件（不纳入Git，约10秒重建，逐字节一致）。

```powershell
python -u analysis/p2_segment.py       # 读v1，写 outputs/segments_v1/（--machines TBM8,TBM9 可只建部分机器）
python -u analysis/p2_checks.py        # 间隔限、停顿限和阶段参数的敏感性，推进速度积分与里程计数器对照
python -u analysis/p2_review.py        # 分层抽样98个段，写抽样表、figures.pdf和答题说明（--png 目录 同时写每段一张PNG）
python -u analysis/p2_verify.py        # 72项检查：表与标签互相一致、与v1逐行对应、独立重算分段和稳定阶段
python -u analysis/smoke_test_p2.py    # 合成数据自检，无需原始数据，含三种篡改必须被验证脚本发现
python -u analysis/p2_review_score.py --sheet outputs/segments_v1/review/<填好的文件>.csv
```

规则和参数在`p2_segment.py`开头的`PARAMS`并写入`manifest.json`。一个段是机器自己的掘进状态为开的连续行（TBM5在2023年前半段取推进工作状态等于3，两个状态都缺的行取信号），读数间隔不超过`max_gap_s`，且没有推力不大于0达`interrupt_s`的行；长段（≥`long_s`）按推力和推进速度达到本段稳定水平的90%和80%划分上升、稳定、下降。阶段码：0段外、1上升、2稳定、3下降、4段内但未命名阶段；运行状态：0掘进、1刀盘在转而不掘进、2刀盘不转、3转速缺失。标签与v1行序相同，`pd.concat([v1, labels[['seg_no','phase','op_state']]], axis=1)`即可使用。

