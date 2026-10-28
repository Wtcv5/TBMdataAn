"""Causal native-timestamp forecasts; no resampling, interpolation or future inputs."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('MKL_NUM_THREADS','4')
from pathlib import Path
import json, re, time
import numpy as np
import pandas as pd
import sklearn, joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'analysis/outputs'
CORE=['F','T','n','v']; TARGETS=['F','T','v']; HORIZONS=[30,60,120]
CONFIG=dict(seed=42,horizons_s=HORIZONS,target_tolerance_s=5,history_rows=4,
            maximum_gap_s=60,train_fraction=.6,validation_fraction=.2,
            max_train_rows_per_machine=6000,max_eval_rows_per_machine=10000,
            ridge_alpha=100,hgb_max_iter=80,hgb_max_leaf_nodes=15,
            feature_policy='four raw signals at current and previous 3 observations, their differences and time gaps',
            origin_policy='F,T,n,v strictly positive at origin; history and target finite/nonnegative; future stops included',
            split_policy='per-machine raw unique timestamp 60/20/20; training labels must precede boundary; test origins in final 20%',
            cross_machine_policy='held-out machine entirely excluded from fitting; evaluate same final-20% windows as local models',
            unit_policy='raw export values, no conversion or energy feature; units not yet confirmed')
CONFIG['quarantine_policy']='exclude whole source files on BOTH sides of exact active (time,F,T,n,v) overlap across different machine IDs'

def save_json(obj,p): p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str),encoding='utf-8')

def cap(df,n):
    if len(df)<=n: return df
    return df.iloc[np.sort(np.random.default_rng(42).choice(len(df),n,replace=False))]

def prepare():
    ds=OUT/'datasets'; ds.mkdir(exist_ok=True)
    overlap=pd.read_csv(OUT/'file_overlap.csv')
    quarantined=set(overlap.source_id_a)|set(overlap.source_id_b)
    file_audit=pd.read_csv(OUT/'file_audit.csv')
    quarantine=file_audit.loc[sorted(quarantined)].copy()
    quarantine['reason']='cross-machine exact active timestamp and four-signal overlap; both files excluded'
    quarantine.to_csv(OUT/'quarantined_files.csv',index=False,encoding='utf-8-sig')
    machines={}; stats=[]; splits=[]; feature_names=None
    for p in (OUT/'canonical').glob('TBM*.parquet'):
        mid=re.match(r'TBM\d+',p.name)[0]
        machines.setdefault(mid,[]).append(p)
    for mid,ps in sorted(machines.items()):
        raw=pd.concat([pd.read_parquet(p) for p in sorted(ps)],ignore_index=True)
        raw_rows=len(raw)
        quarantined_rows=int(raw.source_id.isin(quarantined).sum())
        raw=raw[~raw.source_id.isin(quarantined)].copy()
        invalid=int(raw.time.isna().sum())
        raw=raw.dropna(subset=['time']).sort_values(['time','source_id'],kind='stable')
        dup=raw[raw.time.duplicated(keep=False)]
        conflicts=dup.groupby('time')[CORE].nunique(dropna=False).max(axis=1).gt(1)
        conflict_times=conflicts.index[conflicts]
        raw=raw[~raw.time.isin(conflict_times)]
        same_duplicates=int(raw.time.duplicated().sum())
        raw=raw.drop_duplicates('time',keep='first').reset_index(drop=True)
        assert raw.time.is_unique and raw.time.is_monotonic_increasing
        b1=raw.time.iloc[int(.6*len(raw))]; b2=raw.time.iloc[int(.8*len(raw))]
        splits.append(dict(machine=mid,train_end_exclusive=str(b1),validation_end_exclusive=str(b2),
                           test_end=str(raw.time.max()),raw_unique_rows=len(raw)))
        counts={}
        for h in HORIZONS:
            parts=[]
            for directory,g in raw.groupby('directory',sort=False):
                g=g.sort_values('time').reset_index(drop=True)
                ts=g.time.astype('int64').to_numpy()//10**9
                a=g[CORE].to_numpy(dtype=float); n=len(g)
                # Invalid observations and long outages break both history and forecast paths.
                bad=(~np.isfinite(a).all(axis=1))|(a<0).any(axis=1)
                breaks=np.r_[True,np.diff(ts)>60]|bad|np.r_[False,bad[:-1]]
                seg=np.cumsum(breaks)
                origin=np.arange(3,n)
                j=np.searchsorted(ts,ts[origin]+h,side='left'); inside=j<n
                origin=origin[inside]; j=j[inside]
                ok=(ts[j]-ts[origin]-h<=5)&(seg[origin-3]==seg[j])&(a[origin]>0).all(axis=1)
                origin=origin[ok]; j=j[ok]
                if not len(origin): continue
                features={}
                for lag in range(4):
                    for k,c in enumerate(CORE): features[f'{c}_lag{lag}']=a[origin-lag,k]
                for lag in range(1,4):
                    features[f'age_lag{lag}_s']=ts[origin]-ts[origin-lag]
                    for k,c in enumerate(CORE): features[f'{c}_change{lag}']=a[origin,k]-a[origin-lag,k]
                feature_names=list(features)
                f=pd.DataFrame(features)
                f['origin']=g.time.iloc[origin].to_numpy(); f['target_time']=g.time.iloc[j].to_numpy()
                f['actual_horizon_s']=ts[j]-ts[origin]; f['directory']=directory
                for c in TARGETS: f['y_'+c]=g[c].to_numpy()[j]
                f['split']=np.where(f.target_time<b1,'train',np.where((f.origin>=b1)&(f.target_time<b2),'validation',np.where(f.origin>=b2,'test','purged')))
                parts.append(f)
            if not parts:
                counts[str(h)]={}; continue
            df=pd.concat(parts,ignore_index=True).sort_values('origin').reset_index(drop=True)
            assert (df.target_time>df.origin).all()
            assert df.actual_horizon_s.between(h,h+5).all()
            assert df[feature_names].notna().all().all()
            assert (df[df.split=='train'].target_time<b1).all()
            assert (df[df.split=='test'].origin>=b2).all()
            df.to_parquet(ds/f'{mid}_{h}s.parquet',index=False)
            counts[str(h)]=df.split.value_counts().to_dict()
        stats.append(dict(machine=mid,raw_rows=raw_rows,quarantined_rows=quarantined_rows,invalid_time=invalid,conflict_timestamps=len(conflict_times),
                          identical_timestamp_duplicates=same_duplicates,unique_rows=len(raw),
                          missing_F=int(raw['F'].isna().sum()),missing_T=int(raw['T'].isna().sum()),
                          windows=json.dumps(counts)))
        print('PREPARED',mid,counts,flush=True)
    pd.DataFrame(stats).to_csv(OUT/'machine_audit.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(splits).to_csv(OUT/'split_manifest.csv',index=False,encoding='utf-8-sig')
    save_json(feature_names,OUT/'feature_names.json')

def metric(y,p,persistence):
    e=y-p; base=np.mean(np.abs(y-persistence)); den=np.sum((y-y.mean())**2)
    return dict(MAE=float(np.mean(np.abs(e))),RMSE=float(np.sqrt(np.mean(e**2))),
                R2=float(1-np.sum(e**2)/den) if den else None,
                MAE_skill_vs_persistence=float(1-np.mean(np.abs(e))/base) if base else None)

def block_interval(frame,c,pred):
    # Resample entire origin-days; no claim that seconds are independent replicates.
    d=pd.DataFrame({'day':frame.origin.dt.floor('D'),'e':np.abs(frame['y_'+c].to_numpy()-pred),
                    'b':np.abs(frame['y_'+c]-frame[c+'_lag0'])})
    b=d.groupby('day')[['e','b']].agg(['sum','count']); k=len(b)
    if k<5: return dict(day_blocks=k,skill_ci_low=None,skill_ci_high=None)
    err=b[('e','sum')].to_numpy(); base=b[('b','sum')].to_numpy()
    ix=np.random.default_rng(42).integers(0,k,(1000,k)); den=base[ix].sum(axis=1)
    skills=1-err[ix].sum(axis=1)/np.maximum(den,1e-12)
    return dict(day_blocks=k,skill_ci_low=float(np.quantile(skills,.025)),skill_ci_high=float(np.quantile(skills,.975)))

def train():
    names=json.loads((OUT/'feature_names.json').read_text(encoding='utf-8'))
    results=[]; exclusions=[]; model_dir=OUT/'models'; pred_dir=OUT/'predictions'
    model_dir.mkdir(exist_ok=True); pred_dir.mkdir(exist_ok=True)
    for h in HORIZONS:
        sets={}; allsets={}
        for p in sorted((OUT/'datasets').glob(f'*_{h}s.parquet')):
            mid=p.name.split('_')[0]; f=pd.read_parquet(p)
            splits={s:cap(f[f.split==s].copy(),6000 if s=='train' else 10000) for s in ['train','validation','test']}
            allsets[mid]=splits
            if len(splits['train'])<300 or len(splits['validation'])<50 or len(splits['test'])<50:
                exclusions.append(dict(machine=mid,horizon=h,reason='requires >=300 train, >=50 validation and >=50 test windows',**{s:len(d) for s,d in splits.items()})); continue
            sets[mid]=splits
        for mid,ss in sets.items():
            for protocol in ['within_machine','leave_one_machine_out']:
                source_machines=[mid] if protocol=='within_machine' else [m for m in sets if m!=mid]
                assert protocol!='leave_one_machine_out' or mid not in source_machines
                fit=pd.concat([sets[m]['train'] for m in source_machines],ignore_index=True)
                X=fit[names].to_numpy(); train_base=fit[[c+'_lag0' for c in TARGETS]].to_numpy()
                delta=fit[['y_'+c for c in TARGETS]].to_numpy()-train_base
                start=time.perf_counter()
                ridge=make_pipeline(StandardScaler(),Ridge(alpha=100)).fit(X,delta)
                hgb=[]
                for k,c in enumerate(TARGETS):
                    m=HistGradientBoostingRegressor(max_iter=80,max_leaf_nodes=15,learning_rate=.08,
                                                   min_samples_leaf=30,l2_regularization=1,early_stopping=False,random_state=42)
                    hgb.append(m.fit(X,delta[:,k]))
                elapsed=time.perf_counter()-start
                joblib.dump(dict(ridge=ridge,hgb=hgb,features=names,targets=TARGETS,source_machines=source_machines),
                            model_dir/f'{protocol}_{mid}_{h}s.joblib',compress=3)
                for split in ['validation','test']:
                    f=ss[split]; x=f[names].to_numpy(); base=f[[c+'_lag0' for c in TARGETS]].to_numpy()
                    predictions={'Persistence':base,'Ridge_delta':np.maximum(0,base+ridge.predict(x)),
                                 'HistGB_delta':np.maximum(0,base+np.column_stack([m.predict(x) for m in hgb]))}
                    artifact=f[['origin','target_time','actual_horizon_s','directory']].copy()
                    for k,c in enumerate(TARGETS): artifact['y_'+c]=f['y_'+c].to_numpy()
                    for model,pr in predictions.items():
                        for k,c in enumerate(TARGETS):
                            artifact[model+'_'+c]=pr[:,k]
                            row=dict(protocol=protocol,machine=mid,horizon_s=h,split=split,model=model,target=c,
                                     n=len(f),train_n=len(fit),source_machines=','.join(source_machines),fit_seconds=elapsed,
                                     actual_horizon_median_s=float(f.actual_horizon_s.median()),
                                     **metric(f['y_'+c].to_numpy(),pr[:,k],base[:,k]))
                            if split=='test': row.update(block_interval(f,c,pr[:,k]))
                            results.append(row)
                    artifact.to_parquet(pred_dir/f'{protocol}_{mid}_{h}s_{split}.parquet',index=False)
                pd.DataFrame(results).to_csv(OUT/'baseline_metrics.csv',index=False,encoding='utf-8-sig')
                print('FIT',h,mid,protocol,'train',len(fit),'seconds',round(elapsed,2),flush=True)
    pd.DataFrame(exclusions).to_csv(OUT/'baseline_exclusions.csv',index=False,encoding='utf-8-sig')
    metrics=pd.DataFrame(results); test=metrics[metrics.split=='test']
    macro=test.groupby(['protocol','horizon_s','model','target']).agg(machines=('machine','nunique'),
                    mean_MAE_skill=('MAE_skill_vs_persistence','mean'),median_MAE_skill=('MAE_skill_vs_persistence','median'),
                    positive_skill_machines=('MAE_skill_vs_persistence',lambda x:int((x>0).sum()))).reset_index()
    macro.to_csv(OUT/'baseline_macro.csv',index=False,encoding='utf-8-sig')
    save_json(dict(**CONFIG,sklearn=sklearn.__version__,features=names,targets=TARGETS),OUT/'baseline_config.json')
    print('BASELINES COMPLETE',len(results),'metric rows',flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=4):
        prepare(); train()
