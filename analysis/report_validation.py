"""Apply the pre-set decision rules (outputs/最小验证方案与初步诊断.md, section 3) to the experiment CSVs.
Thresholds are the fixed conventions written there before any real run; nothing here is tuned on results."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
STATES=['stopped_at_target','interrupted_in_path','running']

def md(df,digits=3):
    def fmt(v):
        if isinstance(v,(float,np.floating)): return '' if np.isnan(v) else f'{v:.{digits}f}'
        return '' if pd.isna(v) else str(v)
    return '\n'.join(['| '+' | '.join(map(str,df.columns))+' |','| '+' | '.join('---' for _ in df.columns)+' |']+['| '+' | '.join(fmt(v) for v in r)+' |' for r in df.itertuples(index=False)])

def horizon(V):
    a=pd.read_csv(V/'H_actual_horizon.csv'); a=a[a.split=='test']
    s=pd.read_csv(V/'H_strict_sensitivity.csv'); s=s[s.model!='Persistence']
    w=s.pivot_table(index=['machine','horizon_s','model','target'],columns='subset',values='skill').reset_index()
    flips=[]
    for sub in [c for c in ['excess<=1s','excess==0s'] if c in w]:
        f=w[(np.sign(w[sub])!=np.sign(w['all']))&w[sub].notna()].copy(); f['subset']=sub; f['skill_all']=f['all']; f['skill_subset']=f[sub]; flips.append(f[['machine','horizon_s','model','target','subset','skill_all','skill_subset']])
    flips=pd.concat(flips) if flips else pd.DataFrame()
    text=['## 时间标签核查','',md(a[['machine','horizon_s','n','actual_median_s','actual_mean_s','excess_p95','excess_max','sampling_median_s','history_span_median_s','share_excess_0s']]),'',
          f'严格子集（超出名义值不超过1秒，或恰为0）与全体方向不同的（机器×时域×模型×目标）共{len(flips)}个，共{len(w)}个可比。']
    if len(flips): text+=['',md(flips)]
    hs=s[s.subset.isin(['history<=train_p90','history>train_p90'])]
    if len(hs):
        g=hs.groupby(['machine','horizon_s','model','subset']).agg(skill=('skill','mean'),n=('n','first')).reset_index()
        t=g.pivot_table(index=['machine','horizon_s','model'],columns='subset',values=['skill','n']); t.columns=[f'{a}_{b}' for a,b in t.columns]; t=t.reset_index()
        shares=a[a.split=='test'][['machine','horizon_s','train_span_p90_s','share_history_longer_than_train_p90']]
        text+=['','### 历史跨度对照（探索性）','','测试窗口的4次观测历史跨度是否落在训练期分布内（不超过训练期90%分位）。skill为三个目标的平均，未做跨目标加权。','',md(shares),'',md(t)]
    return '\n'.join(text)

def exp_a(V,sfx='',note=''):
    s=pd.read_csv(V/f'A{sfx}_summary.csv'); p=pd.read_csv(V/f'A{sfx}_paired.csv'); p=p[p.machine=='macro'].set_index(['horizon_s','model','target'])
    wide=s.pivot_table(index=['horizon_s','model','target'],columns='train_config',values='macro_skill').join(p[['skill_difference','ci_low','ci_high']]).reset_index()
    def verdict(r):
        helpful=r.ci_low>0 and r.skill_difference>=0.02
        if helpful and r['N=all']>0 and r.skill_difference>=0.05: return 'A1 训练量是主要原因之一'
        if helpful and r['N=all']<=0: return 'A2 有帮助但不足以超过持久性'
        if helpful: return 'A3 有帮助（2到5个百分点），未达主要原因阈值'   # fills a gap in the written rule
        return 'A4 不是主要瓶颈'
    wide['verdict']=wide.apply(verdict,axis=1)
    cols=['horizon_s','model','target']+[c for c in ['baseline_replica','N=1500','N=6000','N=24000','N=all'] if c in wide]+['skill_difference','ci_low','ci_high','verdict']
    return '\n'.join([f'## A 训练样本量{note}','','宏平均skill；skill_difference为N=all减去基线复现，区间为日块bootstrap 95%。','',md(wide[cols])])

def exp_b(V):
    m=pd.read_csv(V/'B_macro.csv'); simple=m[m.rule=='SIMPLE']; diff=m[m.rule=='ALL minus SIMPLE'].copy()
    diff['verdict']=np.where(diff.ci_low>0,'复杂模型有额外信息',np.where(diff.ci_high<0,'复杂模型更差','复杂模型无额外信息（区间含0）'))
    per_h=simple.assign(positive=simple.ci_low>0).groupby('horizon_s').positive.sum().rename('SIMPLE区间全大于0的目标数').reset_index()
    per_h['verdict']=np.where(per_h.iloc[:,1]>=2,'B1 简单方法已能改进持久性，应先修协议','B2 简单方法未稳定改进持久性')
    return '\n'.join(['## B 简单基线与验证集选择','',md(m[m.rule!='ALL minus SIMPLE'][['horizon_s','target','rule','machines','macro_test_skill','positive_machines','ci_low','ci_high']]),'',
                      '### 复杂模型是否带来额外信息（ALL减SIMPLE）','',md(diff[['horizon_s','target','macro_test_skill','positive_machines','ci_low','ci_high','verdict']]),'','### 简单方法判定','',md(per_h)])

def exp_c(V):
    c=pd.read_csv(V/'C_strata.csv'); rows=[]
    for (h,model,t),g in c.groupby(['horizon_s','model','target']):
        if model=='Persistence': continue
        mach=g.machine.unique(); overall=g.groupby('machine').overall_skill.first().mean(); r=dict(horizon_s=h,model=model,target=t,overall_skill=overall)
        concentrated=[]
        for st in STATES:
            x=g[g.state==st].set_index('machine'); share=x.share_windows.reindex(mach).fillna(0).mean(); contrib=x.contribution_to_skill.reindex(mach).fillna(0).mean()
            r[f'share_{st}']=share; r[f'contribution_{st}']=contrib
            if overall<0 and share<0.2 and contrib/overall>0.5: concentrated.append(st)
        r['verdict']='退化集中于'+'、'.join(concentrated) if concentrated else ('未见集中' if overall<0 else '总体不差于持久性')
        rows.append(r)
    return '\n'.join(['## C 未来状态分层（事后）','','contribution为该状态对总skill的贡献，三个状态之和等于总skill；负值表示该状态拖累。','',md(pd.DataFrame(rows))])

def exp_d(V,sfx='',note=''):
    p=pd.read_csv(V/f'D{sfx}_paired.csv'); p=p[p.machine=='macro']
    g=p.groupby(['horizon_s','model','target','variant','N']).agg(skill_difference=('skill_difference','mean'),ci_low=('ci_low','first'),ci_high=('ci_high','first'),
        variant_train_n=('variant_train_n','mean'),within_train_n=('within_train_n','mean')).reset_index()
    g['size_matched']=(g.variant_train_n/g.within_train_n-1).abs()<=0.02
    g['verdict']=np.where(~g.size_matched,'训练量未匹配，不下结论',np.where(g.ci_low>0,'帮助',np.where(g.ci_high<0,'妨碍','无法判断')))
    wide=g.pivot_table(index=['horizon_s','model','target','N'],columns='variant',values='verdict',aggfunc='first').reset_index()
    strict=[c for c in ['lomo_same_sampling','lomo_time_causal','lomo_no_overlap_partner'] if c in wide]
    def conflict(r):
        base=r.get('lomo_all'); return '；'.join(f'{c}与lomo_all相反' for c in strict if {base,r[c]}=={'帮助','妨碍'})
    wide['冲突']=wide.apply(conflict,axis=1)
    return '\n'.join([f'## D 公平跨机器比较{note}','','skill_difference为变体减同机（同样训练量）；点估计取各种子平均，区间取第一个种子。','',md(g[['horizon_s','model','target','variant','N','variant_train_n','within_train_n','skill_difference','ci_low','ci_high','verdict']]),'','### 各变体结论并排','',md(wide)])

def exp_e(V):
    p=pd.read_csv(V/'E_prevalence.csv'); s=pd.read_csv(V/'E_summary.csv'); key=['horizon_s','model','target']
    prev=p[p.split=='test'].pivot_table(index=['machine','horizon_s'],columns='factor',values='glitch_any').reset_index()
    prev.columns=['machine','horizon_s']+[f'测试窗口含毛刺占比（阈值{int(c)}倍中位数）' for c in prev.columns[2:]]
    a=s[(s.train=='train_all')&(s.subset=='all_windows')].set_index(key).macro_skill.rename('全部窗口_训练含毛刺')
    b=s[(s.train=='train_all')&(s.subset=='clean_windows')].set_index(key)[['macro_skill','ci_low','ci_high']].add_prefix('清洁窗口_训练含毛刺_')
    c=s[(s.train=='train_clean')&(s.subset=='clean_windows')].set_index(key)[['macro_skill','ci_low','ci_high','diff_vs_train_all_ci_low','diff_vs_train_all_ci_high']].add_prefix('清洁窗口_训练清洁_')
    t=pd.concat([a,b,c],axis=1).reset_index()
    t['E1']=np.where((t['全部窗口_训练含毛刺']<=0)&(t['清洁窗口_训练含毛刺_ci_low']>0),'评价被毛刺窗口拖累','否')
    t['E2']=np.where(t['清洁窗口_训练清洁_diff_vs_train_all_ci_low']>0,'训练被毛刺污染','否')
    return '\n'.join(['## E 毛刺污染（探索性）','','毛刺：任一历史观测或标签超过训练期中位数的10倍（F、T、v分别计）；阈值不使用测试期统计。','',md(prev),'',
                      'E1：用含毛刺窗口训练的模型，全部窗口skill不大于0，而清洁窗口skill区间下限大于0。E2：训练集剔除毛刺窗口相对不剔除的配对差，区间下限大于0。','',md(t)])

def per_machine(V):
    a=pd.read_csv(V/'A_metrics.csv'); a=a[(a.split=='test')&(a.model=='HistGB_L1_delta')&(a.train_config=='N=all')].copy()
    a['skill [95%区间]']=a.apply(lambda r:f'{r.skill:+.3f} [{r.skill_ci_low:+.3f},{r.skill_ci_high:+.3f}]',axis=1)
    t=a.pivot_table(index=['horizon_s','target'],columns='machine',values='skill [95%区间]',aggfunc='first').reset_index()
    parts=['## 逐机器视图','','宏平均会掩盖机器间的差异，这里给出MAE目标的梯度提升树（用全部训练窗口）在每台机器测试期的skill。','',md(t)]
    if (V/'B_selected.csv').exists():
        b=pd.read_csv(V/'B_selected.csv').pivot_table(index=['horizon_s','target','rule'],columns='machine',values='test_skill').reset_index()
        parts+=['','验证集选出的最优方法（SIMPLE只含平滑与收缩，ALL还含学习模型）在每台机器测试期的skill：','',md(b)]
    return '\n'.join(parts)

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--dir',type=Path,default=ROOT/'analysis/outputs/validation'); a=ap.parse_args()
    parts=['# 判定规则的机械执行结果','','由`analysis/report_validation.py`生成。阈值是写在方案里的约定；区间为每台机器内部时间块的bootstrap，不代表对独立机器总体的显著性。']
    sections=[('horizon',['H_actual_horizon.csv','H_strict_sensitivity.csv'],horizon),('A',['A_summary.csv','A_paired.csv'],exp_a),('A剔除毛刺训练',['A_clean_summary.csv','A_clean_paired.csv'],lambda V:exp_a(V,'_clean','（训练池剔除毛刺窗口，敏感性分析）')),
              ('B',['B_macro.csv'],exp_b),('C',['C_strata.csv'],exp_c),('D',['D_paired.csv'],exp_d),('D剔除毛刺训练',['D_clean_paired.csv'],lambda V:exp_d(V,'_clean','（训练池剔除毛刺窗口，敏感性分析）')),
              ('E',['E_prevalence.csv','E_summary.csv'],exp_e),('逐机器',['A_metrics.csv'],per_machine)]
    for name,need,fn in sections:
        if all((a.dir/f).exists() for f in need): parts+=['',fn(a.dir)]
        else: parts+=['',f'## {name}','','未运行（缺少输出文件）。']
    (a.dir/'decision_summary.md').write_text('\n'.join(parts)+'\n',encoding='utf-8'); print('wrote',a.dir/'decision_summary.md')

if __name__=='__main__': main()
