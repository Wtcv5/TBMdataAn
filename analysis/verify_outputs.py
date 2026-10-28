"""Independent artifact checks: splits, provenance, overlap, metrics and saved models."""
from pathlib import Path
import json,re
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'analysis/outputs'
CORE=['F','T','n','v']; TARGETS=['F','T','v']

def run():
    q=pd.read_csv(OUT/'quarantined_files.csv')
    files=pd.read_csv(OUT/'file_audit.csv')
    quarantined=set(files.index[files.file.isin(q.file)])
    assert not json.loads((OUT/'read_failures.json').read_text(encoding='utf-8'))
    split=pd.read_csv(OUT/'split_manifest.csv').set_index('machine')
    names=json.loads((OUT/'feature_names.json').read_text(encoding='utf-8'))
    metrics=pd.read_csv(OUT/'baseline_metrics.csv'); checks=0; active=[]; allowed={}
    for p in (OUT/'canonical').glob('TBM*.parquet'):
        df=pd.read_parquet(p)
        df=df[~df.source_id.isin(quarantined)]
        mid=re.match(r'TBM\d+',p.name)[0]
        # Verify hashes denote identical tuples before using them as overlap evidence.
        a=df[(df[CORE]>0).all(axis=1)].copy()
        a['hash']=pd.util.hash_pandas_object(a[['time']+CORE].astype({c:'float64' for c in CORE}),index=False).to_numpy()
        a['machine']=mid; active.append(a[['hash','machine','time']+CORE])
        allowed[p.stem]=pd.Index(df.time.dropna().unique())
    a=pd.concat(active,ignore_index=True)
    n=a.groupby('hash').machine.nunique()
    assert not (n>1).any(), 'Cross-machine overlap survived quarantine'
    checks+=1
    for p in (OUT/'datasets').glob('*.parquet'):
        mid,hstr=p.stem.split('_'); h=int(hstr[:-1]); f=pd.read_parquet(p)
        assert f.origin.is_monotonic_increasing
        assert f.origin.is_unique
        assert (f.actual_horizon_s==(f.target_time-f.origin).dt.total_seconds()).all()
        assert f.actual_horizon_s.between(h,h+5).all()
        assert f[names].notna().all().all()
        assert (f[[c+'_lag0' for c in CORE]]>0).all().all()
        assert (f[['y_'+c for c in TARGETS]]>=0).all().all()
        b1=pd.Timestamp(split.loc[mid,'train_end_exclusive']); b2=pd.Timestamp(split.loc[mid,'validation_end_exclusive'])
        assert (f[f.split=='train'].target_time<b1).all()
        v=f[f.split=='validation']; assert ((v.origin>=b1)&(v.target_time<b2)).all()
        assert (f[f.split=='test'].origin>=b2).all()
        for directory,g in f.groupby('directory'):
            assert g.origin.isin(allowed[directory]).all()
            assert g.target_time.isin(allowed[directory]).all()
        for lag in range(1,4):
            assert f[f'age_lag{lag}_s'].between(0,60*lag,inclusive='right').all()
            for c in CORE:
                np.testing.assert_allclose(f[f'{c}_change{lag}'],f[f'{c}_lag0']-f[f'{c}_lag{lag}'])
        checks+=1
    checked_metrics=0; checked_models=0
    for p in (OUT/'predictions').glob('*.parquet'):
        match=re.fullmatch(r'(within_machine|leave_one_machine_out)_(TBM\d+)_(\d+)s_(validation|test)',p.stem)
        protocol,mid,h,s=match.groups(); h=int(h)
        pred=pd.read_parquet(p)
        original=pd.read_parquet(OUT/'datasets'/f'{mid}_{h}s.parquet')
        original=original[original.split==s].set_index('origin').loc[pred.origin].reset_index()
        assert (pred.target_time==original.target_time).all()
        for model in ['Persistence','Ridge_delta','HistGB_delta']:
            for c in TARGETS:
                y=pred['y_'+c].to_numpy(); yhat=pred[model+'_'+c].to_numpy()
                np.testing.assert_allclose(y,original['y_'+c])
                row=metrics[(metrics.protocol==protocol)&(metrics.machine==mid)&(metrics.horizon_s==h)&(metrics.split==s)&(metrics.model==model)&(metrics.target==c)]
                assert len(row)==1; row=row.iloc[0]
                assert len(y)==row.n
                mae=np.mean(np.abs(y-yhat)); rmse=np.sqrt(np.mean((y-yhat)**2))
                np.testing.assert_allclose([mae,rmse],[row.MAE,row.RMSE],rtol=1e-10)
                base=np.mean(np.abs(y-original[c+'_lag0'].to_numpy()))
                if base: np.testing.assert_allclose(1-mae/base,row.MAE_skill_vs_persistence,atol=1e-12)
                checked_metrics+=1
        if s=='test':
            bundle=joblib.load(OUT/'models'/f'{protocol}_{mid}_{h}s.joblib')
            assert protocol!='leave_one_machine_out' or mid not in bundle['source_machines']
            assert bundle['features']==names
            x=original[names].iloc[:20].to_numpy(); base=original[[c+'_lag0' for c in TARGETS]].iloc[:20].to_numpy()
            ridge=np.maximum(0,base+bundle['ridge'].predict(x))
            tree=np.maximum(0,base+np.column_stack([m.predict(x) for m in bundle['hgb']]))
            for k,c in enumerate(TARGETS):
                np.testing.assert_allclose(ridge[:,k],pred['Ridge_delta_'+c].iloc[:20],atol=1e-8)
                np.testing.assert_allclose(tree[:,k],pred['HistGB_delta_'+c].iloc[:20],atol=1e-8)
            checked_models+=1
        checks+=1
    assert checked_metrics==len(metrics)==540
    assert checked_models==30
    result=dict(status='passed',dataset_and_prediction_checks=checks,metric_rows_recomputed=checked_metrics,
                saved_model_bundles_reloaded=checked_models,cross_machine_active_overlap_after_quarantine=0,
                source_csv_read_failures=0)
    (OUT/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(result)

if __name__=='__main__':
    with threadpool_limits(limits=4): run()
