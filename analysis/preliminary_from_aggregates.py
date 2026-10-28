"""Preliminary diagnostics that need only the committed aggregate outputs, not the raw data."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'analysis/outputs'
DEST=OUT/'validation'
MODELS=['Ridge_delta','HistGB_delta']

def save(df,name):
    DEST.mkdir(exist_ok=True)
    df.to_csv(DEST/name,index=False,encoding='utf-8-sig')

def validation_vs_test(metrics):
    reg=metrics[metrics.model.isin(MODELS)].copy(); reg['skill_pct']=reg.MAE_skill_vs_persistence*100
    wide=reg.pivot_table(index=['protocol','horizon_s','model','target','machine'],columns='split',values='skill_pct').reset_index()
    macro=wide.groupby(['protocol','horizon_s','model','target'])[['validation','test']].mean().reset_index()
    macro['test_minus_validation']=macro.test-macro.validation
    flips=dict(settings=len(wide),validation_positive=int((wide.validation>0).sum()),
               validation_positive_test_negative=int(((wide.validation>0)&(wide.test<=0)).sum()),
               validation_negative_test_positive=int(((wide.validation<=0)&(wide.test>0)).sum()),
               both_positive=int(((wide.validation>0)&(wide.test>0)).sum()))
    return macro,flips

def mae_vs_rmse(metrics):
    """Squared-loss models are judged by MAE here. MAE worse but RMSE better means the objective, not missing information, may be the problem."""
    p=metrics[metrics.model=='Persistence'].set_index(['protocol','machine','horizon_s','split','target'])[['MAE','RMSE']].rename(columns={'MAE':'MAE0','RMSE':'RMSE0'})
    r=metrics[metrics.model.isin(MODELS)].join(p,on=['protocol','machine','horizon_s','split','target'])
    r['mae_skill_pct']=100*(1-r.MAE/r.MAE0); r['rmse_skill_pct']=100*(1-r.RMSE/r.RMSE0)
    macro=r.groupby(['split','protocol','model','target'])[['mae_skill_pct','rmse_skill_pct']].mean().reset_index()
    counts={}
    for split,g in r.groupby('split'):
        counts[split]=dict(settings=len(g),mae_worse_rmse_better=int(((g.mae_skill_pct<0)&(g.rmse_skill_pct>0)).sum()),
                           mae_worse_rmse_worse=int(((g.mae_skill_pct<0)&(g.rmse_skill_pct<=0)).sum()),
                           mae_better_rmse_better=int(((g.mae_skill_pct>=0)&(g.rmse_skill_pct>0)).sum()),
                           mae_better_rmse_worse=int(((g.mae_skill_pct>=0)&(g.rmse_skill_pct<=0)).sum()))
    return macro,counts

def protocol_comparison(metrics):
    t=metrics[(metrics.split=='test')&metrics.model.isin(MODELS)]
    w=t.pivot_table(index=['machine','horizon_s','model','target'],columns='protocol',values=['MAE','train_n']).reset_index()
    w.columns=['_'.join(c).strip('_') for c in w.columns]
    w['lomo_over_within_MAE']=w.MAE_leave_one_machine_out/w.MAE_within_machine
    w['lomo_better']=w.lomo_over_within_MAE<1
    summary=w.groupby(['horizon_s','model','target']).agg(machines=('machine','nunique'),lomo_better_machines=('lomo_better','sum'),
              median_MAE_ratio=('lomo_over_within_MAE','median'),within_train_n=('train_n_within_machine','first'),
              lomo_train_n=('train_n_leave_one_machine_out','first')).reset_index()
    return w,summary

def machine_span(directory_audit):
    d=directory_audit.copy(); d['machine']=d.source.str.extract(r'^(TBM\d+)')[0]
    d['t0']=pd.to_datetime(d.time_min); d['t1']=pd.to_datetime(d.time_max)
    return d.groupby('machine').agg(first_observation=('t0','min'),last_observation=('t1','max'))

def lomo_calendar_overlap(split,span,eligible):
    s=split.set_index('machine')[['train_end_exclusive','validation_end_exclusive','test_end']].apply(pd.to_datetime)
    rows=[]
    for target in eligible:
        test_start=s.loc[target,'validation_end_exclusive']; test_end=s.loc[target,'test_end']
        for source in eligible:
            if source==target: continue
            tr0=span.loc[source,'first_observation']; tr1=s.loc[source,'train_end_exclusive']
            overlap=max(pd.Timedelta(0),min(tr1,test_end)-max(tr0,test_start))
            rows.append(dict(target=target,source=source,target_test_start=test_start,target_test_end=test_end,
                             source_train_start=tr0,source_train_end=tr1,
                             source_train_extends_past_target_test_start=bool(tr1>test_start),
                             source_train_inside_target_test_days=overlap.total_seconds()/86400))
    return pd.DataFrame(rows)

def window_availability(machine_audit,eligible):
    rows=[]
    for _,r in machine_audit.iterrows():
        if r.machine not in eligible: continue
        for h,c in json.loads(r.windows).items():
            if not c: continue
            train=int(c.get('train',0))
            rows.append(dict(machine=r.machine,horizon_s=int(h),unique_rows=int(r.unique_rows),train_windows=train,
                             validation_windows=int(c.get('validation',0)),test_windows=int(c.get('test',0)),
                             baseline_train_cap=6000,baseline_train_fraction_used=min(6000,train)/train if train else np.nan))
    return pd.DataFrame(rows).sort_values(['horizon_s','machine'])

def horizon_offsets(metrics):
    t=metrics[(metrics.model=='Persistence')&(metrics.target=='F')&(metrics.protocol=='within_machine')]
    o=t[['machine','horizon_s','split','actual_horizon_median_s','n']].copy()
    o['median_excess_s']=o.actual_horizon_median_s-o.horizon_s; o['median_excess_pct']=100*o.median_excess_s/o.horizon_s
    return o.sort_values(['horizon_s','machine','split'])

def main():
    metrics=pd.read_csv(OUT/'baseline_metrics.csv'); split=pd.read_csv(OUT/'split_manifest.csv')
    directory=pd.read_csv(OUT/'directory_audit.csv'); audit=pd.read_csv(OUT/'machine_audit.csv')
    eligible=sorted(metrics.machine.unique())
    macro,flips=validation_vs_test(metrics); per_machine,summary=protocol_comparison(metrics)
    save(macro,'prelim_validation_vs_test.csv'); (DEST/'prelim_validation_flips.json').write_text(json.dumps(flips,indent=2),encoding='utf-8')
    save(per_machine,'prelim_protocol_per_machine.csv'); save(summary,'prelim_protocol_summary.csv')
    mismatch,counts=mae_vs_rmse(metrics); save(mismatch,'prelim_mae_vs_rmse.csv'); (DEST/'prelim_mae_vs_rmse_counts.json').write_text(json.dumps(counts,indent=2),encoding='utf-8')
    save(lomo_calendar_overlap(split,machine_span(directory),eligible),'prelim_lomo_calendar_overlap.csv')
    save(window_availability(audit,eligible),'prelim_window_availability.csv')
    save(horizon_offsets(metrics),'prelim_actual_horizon_offsets.csv')
    print(json.dumps(flips,indent=2))

if __name__=='__main__': main()
