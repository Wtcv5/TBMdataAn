"""Shared pieces of the minimal validation experiments; evaluation windows are exactly the baseline's."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('MKL_NUM_THREADS','4')
import json, re
from functools import lru_cache
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from build_baselines import cap, CORE, TARGETS, OUT, CONFIG

DEST=OUT/'validation'; ART=OUT/'validation_predictions'
BASE=[c+'_lag0' for c in TARGETS]; LABEL=['y_'+c for c in TARGETS]
MAX_TRAIN=CONFIG['max_train_rows_per_machine']; MAX_EVAL=CONFIG['max_eval_rows_per_machine']
LEARNED=['Ridge_delta','HistGB_delta','HistGB_L1_delta']
# Same fixed hyperparameters as build_baselines.train(); early stopping stays off so the
# training-set size is the only thing that changes (sklearn would otherwise switch it on above 10,000 rows).
HGB=dict(max_iter=CONFIG['hgb_max_iter'],max_leaf_nodes=CONFIG['hgb_max_leaf_nodes'],learning_rate=.08,
         min_samples_leaf=30,l2_regularization=1,early_stopping=False,random_state=42)

def feature_names():
    names=json.loads((OUT/'feature_names.json').read_text(encoding='utf-8'))
    assert not any(n.startswith(('y_','oracle_')) or n in ('origin','target_time','actual_horizon_s','split','directory') for n in names)
    return names

def save_rows(rows,name):
    DEST.mkdir(exist_ok=True)
    df=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows)
    df.to_csv(DEST/name,index=False,encoding='utf-8-sig')
    return df

def overlap_partners():
    """Machine -> machines it shares exact active (time, F, T, n, v) records with, from the committed directory overlap table."""
    path=OUT/'directory_overlap.csv'; out={}
    if not path.exists(): return out
    for a,b in pd.read_csv(path)[['a','b']].itertuples(index=False):
        ma,mb=re.match(r'TBM\d+',a)[0],re.match(r'TBM\d+',b)[0]
        out.setdefault(ma,set()).add(mb); out.setdefault(mb,set()).add(ma)
    return out

def machine_sets(h,machines=None):
    """Eligible machines as in the baseline. Validation/test are the baseline's capped windows; train is the full pool."""
    sets={}
    for p in sorted((OUT/'datasets').glob(f'*_{h}s.parquet')):
        mid=p.name.split('_')[0]
        if machines and mid not in machines: continue
        f=pd.read_parquet(p)
        pool=f[f.split=='train'].reset_index(drop=True)
        val=cap(f[f.split=='validation'].copy(),MAX_EVAL).reset_index(drop=True)
        test=cap(f[f.split=='test'].copy(),MAX_EVAL).reset_index(drop=True)
        if min(len(pool),MAX_TRAIN)<300 or len(val)<50 or len(test)<50: continue
        sets[mid]=dict(pool=pool,validation=val,test=test)
    return sets

def draw(pool,n,seed):
    """Nested uniform subset: the first n of a seeded permutation, restored to time order. None or n>=len returns the pool."""
    if n is None or n>=len(pool): return pool
    return pool.iloc[np.sort(np.random.default_rng(seed).permutation(len(pool))[:n])]

def fit_models(fit,names,kinds=LEARNED):
    x=fit[names].to_numpy(); delta=fit[LABEL].to_numpy()-fit[BASE].to_numpy(); out={}
    if 'Ridge_delta' in kinds: out['Ridge_delta']=make_pipeline(StandardScaler(),Ridge(alpha=CONFIG['ridge_alpha'])).fit(x,delta)
    if 'HistGB_delta' in kinds: out['HistGB_delta']=[HistGradientBoostingRegressor(**HGB).fit(x,delta[:,k]) for k in range(3)]
    if 'HistGB_L1_delta' in kinds:
        out['HistGB_L1_delta']=[HistGradientBoostingRegressor(loss='absolute_error',**HGB).fit(x,delta[:,k]) for k in range(3)]
    return out

def predict(models,frame,names):
    x=frame[names].to_numpy(); base=frame[BASE].to_numpy(); out={}
    for name,m in models.items():
        d=m.predict(x) if name=='Ridge_delta' else np.column_stack([e.predict(x) for e in m])
        out[name]=np.maximum(0,base+d)
    return out

def eval_rows(frame,preds,meta):
    """One row per model and target. skill = 1 - MAE/MAE(persistence), as in the baseline. Actual horizons are always recorded."""
    y=frame[LABEL].to_numpy(); base=frame[BASE].to_numpy(); excess=frame.actual_horizon_s.to_numpy()-meta['horizon_s']
    rows=[]
    for model,p in {'Persistence':base,**preds}.items():
        for k,c in enumerate(TARGETS):
            e=np.abs(y[:,k]-p[:,k]); b=np.abs(y[:,k]-base[:,k])
            rows.append(dict(**meta,model=model,target=c,n=len(frame),MAE=float(e.mean()),RMSE=float(np.sqrt((e**2).mean())),
                             skill=float(1-e.sum()/b.sum()) if b.sum()>0 else np.nan,
                             actual_horizon_median_s=float(np.median(frame.actual_horizon_s)),
                             actual_horizon_mean_s=float(frame.actual_horizon_s.mean()),excess_gt0_share=float((excess>0).mean())))
    return rows

def glitch_thresholds(pool,factor):
    """Spike threshold per signal: factor x the median of the training pool's origin values. Test statistics are never used."""
    return {c:factor*float(pool[c+'_lag0'].median()) for c in TARGETS}

def glitch_flags(frame,thresholds):
    """Spike flags from observed values only: any of the four history observations, or the label, above the threshold."""
    out={}; hist=np.zeros(len(frame),bool); target=np.zeros(len(frame),bool)
    for c in TARGETS:
        h=(frame[[f'{c}_lag{j}' for j in range(4)]]>thresholds[c]).any(axis=1).to_numpy(); t=(frame['y_'+c]>thresholds[c]).to_numpy()
        out[f'glitch_hist_{c}']=h; out[f'glitch_target_{c}']=t; hist|=h; target|=t
    out['glitch_hist']=hist; out['glitch_target']=target; out['glitch_any']=hist|target
    return pd.DataFrame(out,index=frame.index)

def day_codes(frame):
    # Frames are time-ordered, so first-appearance order is calendar order.
    return pd.factorize(frame.origin.dt.floor('D'))[0]

class Boot:
    """Day-block bootstrap of skill vs persistence. Variants added under one (key, machine) share their resamples, so
    differences between them are paired. Macro values average machines with equal weight, as in the baseline."""
    def __init__(self,B=1000): self.B=B; self.skill={}
    def add(self,key,mid,codes,reference,errors):
        k=int(codes.max())+1
        if k<5: return                       # same rule as the baseline: fewer than 5 day blocks, no interval
        ix=np.random.default_rng(42+int(re.sub(r'\D','',mid))).integers(0,k,(self.B,k))
        ref=np.maximum(np.bincount(codes,weights=reference,minlength=k)[ix].sum(axis=1),1e-12)
        for variant,e in errors.items():
            self.skill[(key,variant,mid)]=1-np.bincount(codes,weights=e,minlength=k)[ix].sum(axis=1)/ref
    def interval(self,vec):
        return (float(np.quantile(vec,.025)),float(np.quantile(vec,.975))) if vec is not None else (np.nan,np.nan)
    def machine_ci(self,key,variant,mid):
        return self.interval(self.skill.get((key,variant,mid)))
    def diff_ci(self,key,a,b,mid):
        va,vb=self.skill.get((key,a,mid)),self.skill.get((key,b,mid))
        return self.interval(None if va is None or vb is None else va-vb)
    def macro_ci(self,key,a,mids,b=None):
        vs=[self.skill[(key,a,m)]-(self.skill[(key,b,m)] if b else 0) for m in mids if (key,a,m) in self.skill and (b is None or (key,b,m) in self.skill)]
        return (self.interval(np.mean(vs,axis=0)),len(vs)) if vs else ((np.nan,np.nan),0)

# --- post-hoc future state: only for error attribution, never a model input ---------------------------------------

@lru_cache(maxsize=8)
def clean_series(mid):
    """Per-directory cleaned observations of one machine; replicates build_baselines.prepare() step by step. Callers must not modify it."""
    ov=pd.read_csv(OUT/'file_overlap.csv'); quarantined=set(ov.source_id_a)|set(ov.source_id_b)
    paths=sorted(p for p in (OUT/'canonical').glob('TBM*.parquet') if re.match(r'TBM\d+',p.name)[0]==mid)
    raw=pd.concat([pd.read_parquet(p) for p in paths],ignore_index=True)
    raw=raw[~raw.source_id.isin(quarantined)].copy()
    raw=raw.dropna(subset=['time']).sort_values(['time','source_id'],kind='stable')
    dup=raw[raw.time.duplicated(keep=False)]
    conflicts=dup.groupby('time')[CORE].nunique(dropna=False).max(axis=1).gt(1)
    raw=raw[~raw.time.isin(conflicts.index[conflicts])]
    raw=raw.drop_duplicates('time',keep='first').reset_index(drop=True)
    return {d:g.sort_values('time').reset_index(drop=True) for d,g in raw.groupby('directory',sort=False)}

STATES=['running','interrupted_in_path','stopped_at_target']

def future_state(frame,series):
    """Stratum from the true observations in (origin, target]; stop means any of F,T,n,v <= 0.
    The label check below fails loudly if the window-to-observation mapping is wrong."""
    state=np.empty(len(frame),dtype=object); n_target=np.full(len(frame),np.nan); steps=np.zeros(len(frame),int)
    pos={i:k for k,i in enumerate(frame.index)}
    for d,g in frame.groupby('directory'):
        s=series[d]; times=s.time.to_numpy(); a=s[CORE].to_numpy(float)
        o=np.searchsorted(times,g.origin.to_numpy()); j=np.searchsorted(times,g.target_time.to_numpy())
        assert (times[o]==g.origin.to_numpy()).all() and (times[j]==g.target_time.to_numpy()).all(),'window not found in cleaned series'
        assert np.allclose(a[j][:,[CORE.index(c) for c in TARGETS]],g[LABEL].to_numpy()),'label mismatch with cleaned series'
        stopped=(a<=0).any(axis=1); span=j-o; between=np.zeros(len(g),bool)
        for step in range(1,int(span.max())):
            between|=(step<span)&stopped[np.minimum(o+step,len(a)-1)]
        st=np.where(stopped[j],'stopped_at_target',np.where(between,'interrupted_in_path','running'))
        ix=[pos[i] for i in g.index]
        state[ix]=st; n_target[ix]=a[j,CORE.index('n')]; steps[ix]=span
    return pd.DataFrame(dict(oracle_state=state,oracle_n_target=n_target,oracle_steps=steps),index=frame.index)
