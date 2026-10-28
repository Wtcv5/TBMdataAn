from pathlib import Path
import json,re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'analysis/outputs'

def table(df):
    def fmt(v):
        if pd.isna(v): return '—'
        if isinstance(v,(float,np.floating)): return f'{v:.3f}'
        return str(v).replace('|','/')
    lines=['| '+' | '.join(map(str,df.columns))+' |','| '+' | '.join(['---']*len(df.columns))+' |']
    lines += ['| '+' | '.join(fmt(v) for v in row)+' |' for row in df.itertuples(index=False,name=None)]
    return '\n'.join(lines)

def run():
    dirs=pd.read_csv(OUT/'directory_audit.csv'); files=pd.read_csv(OUT/'file_audit.csv')
    machines=pd.read_csv(OUT/'machine_audit.csv'); quarantine=pd.read_csv(OUT/'quarantined_files.csv')
    metrics=pd.read_csv(OUT/'baseline_metrics.csv'); macro=pd.read_csv(OUT/'baseline_macro.csv')
    overlaps=pd.read_csv(OUT/'directory_overlap.csv'); exclusions=pd.read_csv(OUT/'baseline_exclusions.csv')
    wb=json.loads((OUT/'workbook_audit.json').read_text(encoding='utf-8'))
    figures=OUT/'figures'; figures.mkdir(exist_ok=True)
    # Labels in English keep the standalone figures portable.
    fig,axs=plt.subplots(1,3,figsize=(13,4),sharey=True)
    for ax,c in zip(axs,['F','T','v']):
        d=metrics[(metrics.split=='test')&(metrics.protocol=='leave_one_machine_out')&(metrics.horizon_s==60)&(metrics.target==c)]
        pivot=d.pivot(index='machine',columns='model',values='MAE_skill_vs_persistence')[['Ridge_delta','HistGB_delta']]*100
        pivot.plot.barh(ax=ax); ax.axvline(0,color='black',lw=.8); ax.set_title(c+' : 60 s forecast')
        ax.set_xlabel('MAE reduction vs persistence (%)'); ax.legend(fontsize=8)
    fig.suptitle('Held-out machines: fixed initial baselines (raw export units)')
    fig.tight_layout(); fig.savefig(figures/'cross_machine_skill_60s.png',dpi=180); plt.close(fig)
    heat=np.column_stack([machines['missing_F']/machines.unique_rows,machines['missing_T']/machines.unique_rows])*100
    fig,ax=plt.subplots(figsize=(6,5)); im=ax.imshow(heat,vmin=0,vmax=100,cmap='YlOrRd',aspect='auto')
    ax.set_xticks([0,1],['F missing (%)','T missing (%)']); ax.set_yticks(range(len(machines)),machines.machine)
    for i in range(len(machines)):
        for j in range(2): ax.text(j,i,f'{heat[i,j]:.1f}',ha='center',va='center')
    ax.set_title('After overlap quarantine and timestamp deduplication'); fig.colorbar(im,ax=ax)
    fig.tight_layout(); fig.savefig(figures/'missing_core_fields.png',dpi=180); plt.close(fig)
    overview=dirs[['source','rows','time_min','time_max','interval_mode_s','F_missing','T_missing','positive_four_rows']].copy()
    overview.columns=['目录','行数','起始时间','结束时间','众数间隔/秒','F缺失','T缺失','四信号为正行数']
    mm=machines[['machine','raw_rows','quarantined_rows','conflict_timestamps','identical_timestamp_duplicates','unique_rows','missing_F','missing_T']].copy()
    mm.columns=['设备','原始行数','隔离行数','冲突时间戳数','相同重复行数','保留行数','F缺失','T缺失']
    comparison=macro[macro.model!='Persistence'].copy()
    comparison['mean_MAE_skill']=comparison.mean_MAE_skill*100
    comparison['median_MAE_skill']=comparison.median_MAE_skill*100
    comparison=comparison[['protocol','horizon_s','model','target','machines','mean_MAE_skill','positive_skill_machines']]
    comparison.columns=['协议','时域/秒','模型','目标','设备数','平均MAE降低/%','优于持久性设备数']
    sixty=metrics[(metrics.split=='test')&(metrics.horizon_s==60)&(metrics.protocol=='leave_one_machine_out')&(metrics.target=='T')]
    detail=sixty.pivot(index='machine',columns='model',values='MAE').reset_index()
    detail=detail[['machine','Persistence','Ridge_delta','HistGB_delta']]
    gain=sixty[sixty.model=='HistGB_delta'].set_index('machine')
    detail['HistGB降低/%']=detail.machine.map(gain.MAE_skill_vs_persistence)*100
    detail['测试窗口数']=detail.machine.map(gain.n)
    detail['日块数']=detail.machine.map(gain.day_blocks)
    detail['skill区间下限/%']=detail.machine.map(gain.skill_ci_low)*100
    detail['skill区间上限/%']=detail.machine.map(gain.skill_ci_high)*100
    ov=overlaps.rename(columns={'a':'目录A','b':'目录B','matches':'相同有效记录','a_active':'A有效记录','b_active':'B有效记录','fraction_a':'占A比例','fraction_b':'占B比例'})
    eligible=sorted(metrics.machine.unique()); tests=metrics[(metrics.split=='test')&(metrics.protocol=='within_machine')&(metrics.model=='Persistence')&(metrics.target=='T')]
    windows=tests[['machine','horizon_s','n','actual_horizon_median_s']].copy()
    windows.columns=['设备','标称时域/秒','实际评估窗口数','实际时域中位数/秒']
    qfolders=quarantine.assign(directory=quarantine.file.str.split('\\').str[0]).groupby('directory').agg(files=('file','size'),rows=('rows','sum')).reset_index()
    text=f'''# TBM数据审计与初始预测基线报告

审计日期：2026-10-28（Asia/Shanghai）。范围：当前解压数据目录，原始文件未修改。

## 结论

读取全部 **{len(files):,}个原始CSV、{int(files.rows.sum()):,}行、{files.bytes.sum()/1e9:.3f} GB**，覆盖16个目录和9个TBM编号；CSV读取失败0个。
另完整审计Excel **{wb['rows']:,}行**。9个编号不等于9套已验证独立的数据源：跨编号存在大量逐时刻一致的数据。
隔离{len(quarantine)}个存在跨机器重合的源文件（双方均隔离），合计{int(quarantine.rows.sum()):,}行；这些文件仍保留在原目录。
同机器按核心信号去重后，剩余{int(machines.unique_rows.sum()):,}行供资格审查。
当前能够完成共同时间外推与留一机器评估的是 **{', '.join(eligible)}**，各时域5台；其余4台缺少足够完整且覆盖训练、验证、测试期的四信号窗口。

固定初始基线未稳定超过持久性。60秒留一机器的HistGB模型，扭矩平均MAE降低率为{macro.query("protocol=='leave_one_machine_out' and horizon_s==60 and model=='HistGB_delta' and target=='T'").mean_MAE_skill.iloc[0]*100:.2f}%，仅2/5台改善。
负降低率表示模型比持久性更差。这些结果不支持当前简单模型已有迁移优势；也不代表更复杂方法一定无效。

## 原始目录审计

下表在隔离和去重之前统计，缺失包括“文件没有该字段”及“值无法解析”。仅核心字段做全量数值质量审计，辅助传感器仅清点表头。

{table(overview)}

采样间隔来自真实时间戳，不是文件夹名称。数据含5、10、11、20、30、31秒等采样尺度及长缺口；TBM2的“9s_interval”名称不能替代时间审计。
`file_audit.csv`保存每文件起止时间、倒序、重复、编码、缺失字段、字节数与schema编号；`schemas.json`保存全部表头和映射。

## 数据来源与重复风险

在四个核心信号均为正的观测中，对“时间戳、F、T、n、v”做完全相同值检查，跨机器匹配如下（比例分母为目录有效运行记录）。

{table(ov)}

TBM2-2024的41,751条有效运行观测全部出现在TBM5/6-2024；TBM6-2024的105,649条有效运行观测全部出现在TBM5-2024。
这说明目录标签与数据独立性尚需核实；无法仅凭这些文件判断哪台设备才是真实来源。
隔离整个相关源文件双方，避免邻近时间窗口仍携带同一序列。对应源文件名与source_id见`file_overlap.csv`与`quarantined_files.csv`。

{table(qfolders)}

Excel没有机器编号。对其中每10条有效观测取一条，共25,093条作重合检查，24,948条与TBM9目录精确匹配；也有127条与TBM10-2023匹配，其中存在TBM9/10本身重合的记录。
Excel未用作训练或新增机器；少量不匹配记录及其真实来源未完全追溯，不能宣称它是独立外部验证集。
`segmented_cycles_all`、`segmented_cycles_150`、`codes`、`common_using`和zip备份没有计入新增样本。

## 核心字段、单位与清理

F按“总推进力/总推力/推力”映射；T按“刀盘转矩/刀盘扭矩”映射；n优先“刀盘速度”，其次“刀盘实际转速”；v按“推进速度/推进速度(mm/M)”映射。
“刀盘转速”在部分机型上远大于“刀盘速度”，只作辅助核对，未直接替代刀盘速度。p保留用于审计，不作为基线输入或预测标签。
TBM1及TBM2-2023部分导出只有“推进压力”，不以压力替代推力。TBM3和TBM4大量导出缺少总刀盘转矩，不用各电机扭矩随意拼造总扭矩。
`directory_audit.csv`含p*n/v一致性检查与分位数，代数比值接近1只用于口径核对，不能单独确认单位或传感器正确。

用户确认TBM编号对应独立设备；未获得明确设备类型、推力/扭矩/转速单位、刀盘直径、齿轮比及额定载荷表。因此指标采用原始导出数值，无SI转换，无功率/比能特征。
Excel含F最小值{wb['F_q0']:,.0f}、v最大值{wb['v_q1']:,.3f}等异常候选；原始CSV数值范围详见审计表。
基线剔除非有限和负的历史/标签观测，不按测试集分位数截尾，也未默默删除所有高值。正值尖峰是否故障需后续设备字典核实。

{table(mm)}

“相同重复”按同机器同时间戳四个核心信号相同认定；若任一核心信号冲突，剔除该时间戳所有行。p和其他辅助字段的重复冲突不用于当前四信号预测资格。
TBM1、2、3、4未进入当前比较的具体样本不足原因见`baseline_exclusions.csv`；这不代表这些设备所有数据都不可用，可在来源修复后重新评估。

![核心字段缺失](figures/missing_core_fields.png)

## 预测任务与防泄漏协议

- 当前及前3次真实观测的F、T、n、v，加历史差分和时间间隔，预测未来F、T、v；不使用未来控制设定、未来转速或同刻贯入代数重构。
- 起点必须处于F、T、n、v均为正的观测状态。未来停机零值纳入标签。因此结论适用于运行起点预测，不能推广到任意停机起点。
- 标签取t+h之后首个真实观测，h为30、60、120秒，最大迟到5秒；实际时域随样本略变，不是严格等间隔标签。
- 不插值，不跨超过60秒缺口、无效观测或不同目录构造窗口。历史是4次观测，物理时间长度随采样间隔而变。
- 每机清理后全部唯一原始时间戳按60%/20%/20%划分；训练目标必须早于训练边界；验证目标必须早于测试边界；跨界窗口剔除。
- 每机训练最多6000窗口、验证和测试各最多10000窗口，种子42；等概率抽取后恢复时间顺序，保留全量窗口缓存。
- 同机器模型仅用该机训练期；留一机器模型仅用其余4台合格机训练期，完全排除目标机拟合数据。两个协议用同一目标机测试窗口。
- 三个模型：Persistence；StandardScaler+Ridge(alpha=100)预测增量；HistGB(80轮、15叶子)预测增量。回归预测限制非负。超参数固定，未用测试结果调参；验证结果另存，未进行模型选择。
- MAE、RMSE、R²按机报告；跨机仅等权汇总相对持久性MAE降低率，避免混合未确认单位的绝对误差。
- 置信区间以预测起点所在的“日”为块重采样1000次；少于5日不报告。区间是每机时间块不确定性，不是对独立机器总体泛化的显著性证明。

分界时间见`split_manifest.csv`。本轮未做预训练、少样本适配、深度模型比较或概率校准，属于数据就绪审计与固定初始基线。

{table(windows)}

## 实测结果

下表为测试期各机等权平均，正值优于持久性，负值差于持久性。协议`within_machine`为同机时间外推，`leave_one_machine_out`为留一机器。

{table(comparison)}

60秒留一机器扭矩MAE明细（原始导出单位）：

{table(detail)}

![跨机60秒基线](figures/cross_machine_skill_60s.png)

全部540条指标（含验证/测试、全部时域和三个目标）见`baseline_metrics.csv`；逐样本真值、预测和实际时域见`predictions/`，模型见`models/`。
不能因秒级相邻样本很多就宣称强统计证据，也不能用部分设备的好结果覆盖其余设备退化。

## 后续研究应先解决的事项

1. 按`quarantined_files.csv`向数据提供方核实TBM2/5/6-2024和TBM9/10-2023的源设备、导出过程及是否误拷贝。在核实前保留当前保守隔离。
2. 补齐设备类型、总推力与油压区别、扭矩与速度单位、齿轮比和列名字典；恢复TBM3/4缺少的刀盘总转矩及TBM1/2缺少的力值。
3. 先解释持续性强、自相关和正值尖峰的影响；在冻结的新验证协议中比较仅历史可用的稳健损失、设备尺度校准与简单适配，保留当前基线作对照。
4. 数据来源可信后再增加自监督快慢状态建模与少样本新机适配。当前结果不足以支撑围岩识别、卡机预警或最优控制结论。

复现入口见上级`README.md`。源文件保持原样，所有隔离均为派生数据筛选，不执行文件搬移或删除。
'''
    (OUT/'审计与预测基线报告.md').write_text(text,encoding='utf-8')
    print('Report written:',OUT/'审计与预测基线报告.md')

if __name__=='__main__': run()
