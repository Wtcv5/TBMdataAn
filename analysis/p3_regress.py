"""P3 regression against the committed baseline (writes outputs/analysis_v1/window_regression.csv and window_difference.csv).
1. The window rules of p3_windows.py rebuild the committed baseline windows (outputs/datasets/<machine>_<h>s.parquet) from the cached canonical series, preparing it as build_baselines.py does:
   all 9 machines and the 3 horizons, every feature, target, origin and target time equal.
2. The windows of the analysis dataset against the committed ones, for the machines of the dataset: how many are the same, and for the others what separates them (rows the committed
   series lacks, rows that valid_core drops, the repaired rotation speed)."""
import argparse, re
from pathlib import Path
import numpy as np
import pandas as pd
from p1_standardize import OUT
from p2_segment import finish
from p3_build import DEST
from p3_windows import HORIZONS, CORE, run_ids, target_rows, samples, seconds, feature_names

DATASETS=OUT/'datasets'

def canonical_machines():
    """The canonical series of every machine as build_baselines.prepare() hands it to the window code: whole files quarantined for cross-machine overlap left out, rows without a time dropped,
    conflicting timestamps removed, repeated timestamps reduced to the first."""
    overlap=pd.read_csv(OUT/'file_overlap.csv'); quarantined=set(overlap.source_id_a)|set(overlap.source_id_b); machines={}
    for p in (OUT/'canonical').glob('TBM*.parquet'): machines.setdefault(re.match(r'TBM\d+',p.name)[0],[]).append(p)
    for mid,ps in sorted(machines.items(),key=lambda kv:int(kv[0][3:])):
        raw=pd.concat([pd.read_parquet(p) for p in sorted(ps)],ignore_index=True); raw=raw[~raw.source_id.isin(quarantined)].copy()
        raw=raw.dropna(subset=['time']).sort_values(['time','source_id'],kind='stable'); dup=raw[raw.time.duplicated(keep=False)]
        conflicts=dup.groupby('time')[list(CORE)].nunique(dropna=False).max(axis=1).gt(1); raw=raw[~raw.time.isin(conflicts.index[conflicts])]
        yield mid,raw.drop_duplicates('time',keep='first').reset_index(drop=True)

def rebuild(raw,h):
    """The windows of one machine and horizon by the rules of p3_windows.py, stream (directory) by stream, in the layout and order of the committed dataset."""
    parts=[]
    for directory,g in raw.groupby('directory',sort=False):
        g=g.sort_values('time').reset_index(drop=True); ts=seconds(g.time); a=g[list(CORE)].to_numpy(float)
        good=np.isfinite(a).all(axis=1)&~(a<0).any(axis=1); tgt=target_rows(ts,run_ids(ts,good),(a>0).all(axis=1),h)
        f=samples(g,tgt,(tgt>=0).astype('int8'),1,context=False); f['directory']=directory; parts.append(f)
    return pd.concat(parts,ignore_index=True).sort_values('origin').reset_index(drop=True) if parts else None

def same(a,b):
    return a.dtype.kind in 'fiM' and np.array_equal(a.to_numpy(),b.to_numpy(),equal_nan=a.dtype.kind=='f') or a.dtype.kind not in 'fiM' and (a.to_numpy()==b.to_numpy()).all()

def difference(old,new,frame,old_t,mid,h):
    """The windows of the dataset (new) against the committed ones (old), matched by origin time: counts, and for the windows only one side has the first reason found in the dataset's series."""
    m=old[['origin','target_time']+feature_names()].merge(new[['origin','target_time']+feature_names()],on='origin',how='outer',suffixes=('_old','_new'),indicator=True)
    both=m[m._merge=='both']; same_target=both.target_time_old==both.target_time_new; names=feature_names(); others=[c for c in names if not c.startswith('n_')]
    equal=lambda cols: np.all([(both[c+'_old'].to_numpy()==both[c+'_new'].to_numpy()) for c in cols],axis=0); same_feat=equal(names); others_same=equal(others)
    r=dict(machine=mid,horizon_s=h,committed=len(old),dataset=len(new),identical=int((same_target&same_feat).sum()),other_target=int((~same_target).sum()),features_differ_only_in_n=int((same_target&~same_feat&others_same).sum()),
           features_differ_elsewhere=int((same_target&~others_same).sum()),only_committed=int((m._merge=='left_only').sum()),only_dataset=int((m._merge=='right_only').sum()))
    t=frame.time.to_numpy(); valid=frame.valid_core.to_numpy(bool); act=frame.active.to_numpy(bool); bad=np.cumsum(np.r_[0,~valid]); rep=np.cumsum(np.r_[0,frame.n_source.astype(str).str.startswith('repaired').to_numpy(bool)])
    lo=m[m._merge=='left_only']; po=np.searchsorted(t,lo.origin.to_numpy()); pt=np.searchsorted(t,lo.target_time_old.to_numpy()); inside=(po<len(t))&(pt<len(t))
    po,pt=np.minimum(po,len(t)-1),np.minimum(pt,len(t)-1); present=inside&(t[po]==lo.origin.to_numpy())&(t[pt]==lo.target_time_old.to_numpy())
    invalid=present&((bad[pt+1]-bad[np.maximum(po-3,0)])>0); inactive=present&~invalid&~act[po]
    r.update(only_committed_row_not_in_dataset=int((~present).sum()),only_committed_invalid_row_in_window=int(invalid.sum()),only_committed_origin_not_active=int(inactive.sum()),only_committed_other=int((present&~invalid&~inactive).sum()))
    ro=new[~new.origin.isin(old.origin)]; first=np.searchsorted(t,ro.origin.to_numpy())-3; last=np.searchsorted(t,ro.target_time.to_numpy()); lacks=np.searchsorted(old_t,t[last],'right')-np.searchsorted(old_t,t[first],'left')!=last-first+1
    touched=~lacks&((rep[last+1]-rep[first])>0)
    r.update(only_dataset_rows_missing_in_committed_series=int(lacks.sum()),only_dataset_with_repaired_n=int(touched.sum()),only_dataset_other=int((~lacks&~touched).sum()))
    return r

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--dest',default=str(DEST)); a=ap.parse_args(); dest=Path(a.dest); rows=[]; diffs=[]; ok=True
    machines_in_dataset=[p.stem for p in sorted(dest.glob('TBM*.parquet'),key=lambda p:int(p.stem[3:]))]
    for mid,raw in canonical_machines():
        frame=pd.read_parquet(dest/f'{mid}.parquet',columns=['time','F','T','n','v','active','valid_core','n_source']+[c for h in HORIZONS for c in (f'win{h}_pos',f'win{h}_level')]) if mid in machines_in_dataset else None
        for h in HORIZONS:
            old=pd.read_parquet(DATASETS/f'{mid}_{h}s.parquet'); new=rebuild(raw,h); n_new=0 if new is None else len(new); differing=[]
            if new is not None and len(new)==len(old): differing=[c for c in old.columns if c!='split' and not same(old[c],new[c])]
            ident=n_new==len(old) and not differing; ok&=ident
            rows.append(dict(machine=mid,horizon_s=h,committed_windows=len(old),rebuilt_windows=n_new,identical=ident,differing_columns=';'.join(differing)))
            print(f'{mid:>6} {h:>4} s  committed {len(old):>8,}  rebuilt {n_new:>8,}  {"identical" if ident else "DIFFERENT "+";".join(differing)}',flush=True)
            if frame is not None:
                new_ds=samples(frame,frame[f'win{h}_pos'].to_numpy(),frame[f'win{h}_level'].to_numpy(),1,context=False); diffs.append(difference(old,new_ds,frame,raw.time.to_numpy(),mid,h))
    pd.DataFrame(rows).to_csv(dest/'window_regression.csv',index=False,encoding='utf-8-sig'); d=pd.DataFrame(diffs); d.to_csv(dest/'window_difference.csv',index=False,encoding='utf-8-sig')
    pd.set_option('display.width',250); pd.set_option('display.max_columns',30); print(d.to_string(index=False)); print('the rules reproduce the committed windows' if ok else 'THE RULES DO NOT REPRODUCE THE COMMITTED WINDOWS'); finish(0 if ok else 1)

if __name__=='__main__': main()
