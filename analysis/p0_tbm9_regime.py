"""P0 step 4: evidence for what is wrong with TBM9's cutterhead-speed column (read-only).
A) In every TBM9 file, the cutterhead speed, the commanded speed and the commanded frequency are compared with v/p, the only
   rotation speed that does not depend on those three columns. B) Within one window where affected and healthy files alternate, the
   median of every column is compared between the two groups to see how many other columns changed meaning."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'TBMData/TBMData'; OUT=ROOT/'analysis/outputs'; DEST=OUT/'p0'
WINDOW=('2023-11-20','2024-05-10')   # affected and healthy files alternate here, so operating conditions are comparable
MIN_BORING=20; MIN_ACTIVE_WINDOW=200
CORE=['推力','刀盘扭矩','刀盘实际转速','推进速度']

def load(args):
    sid,rel,enc=args
    return sid,pd.read_csv(DATA/rel,encoding=enc,low_memory=False)

def n_check(args):
    sid,raw=load(args)
    num=raw.apply(pd.to_numeric,errors='coerce'); a=num[(num[CORE]>0).all(axis=1)&(num['贯入度']>0)]
    if len(a)<MIN_BORING: return None
    rpm=a['推进速度']/a['贯入度']
    return dict(source_id=sid,boring_rows=len(a),n_col_over_v_over_p=float((a['刀盘实际转速']/rpm).median()),f_set_over_v_over_p=float((a['刀盘给定频率']/rpm).median()),
                n_set_over_v_over_p=float((a['刀盘给定转速']/rpm).median()))

def column_medians(args):
    sid,raw=load(args)
    num=raw.drop(columns=[raw.columns[0]]).apply(pd.to_numeric,errors='coerce'); a=num[(num[['推力','刀盘扭矩','推进速度']]>0).all(axis=1)&(num['刀盘实际转速']>0)]
    return a.median().rename(sid)

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--workers',type=int,default=4); a=ap.parse_args()
    t=pd.read_csv(DEST/'trusted_sources.csv',parse_dates=['time_min']); fa=pd.read_csv(OUT/'file_audit.csv'); fa['file']=fa.file.str.replace('\\','/')
    tb=t[(t.machine=='TBM9')&(t.active_rows>=MIN_BORING)]; jobs=[(i,fa.file[i],fa.encoding[i]) for i in tb.source_id]
    with ProcessPoolExecutor(a.workers) as ex: rows=[r for r in ex.map(n_check,jobs,chunksize=4) if r]
    chk=pd.DataFrame(rows).merge(t[['source_id','file','time_min','n_state']],on='source_id'); chk.to_csv(DEST/'tbm9_n_regime_check.csv',index=False,encoding='utf-8-sig')
    print(chk.groupby('n_state')[['n_col_over_v_over_p','f_set_over_v_over_p','n_set_over_v_over_p']].describe().T.round(3).to_string())
    w=t[(t.machine=='TBM9')&(t.time_min>=WINDOW[0])&(t.time_min<=WINDOW[1])&(t.active_rows>MIN_ACTIVE_WINDOW)&t.n_state.isin(['shifted','ok'])]
    with ProcessPoolExecutor(a.workers) as ex: prof=pd.concat(list(ex.map(column_medians,[(i,fa.file[i],fa.encoding[i]) for i in w.source_id],chunksize=4)),axis=1).T
    sh,ok=w[w.n_state=='shifted'].source_id,w[w.n_state=='ok'].source_id
    cmp=pd.DataFrame({'shifted_median':prof.loc[sh].median(),'ok_median':prof.loc[ok].median()}); cmp['ratio']=cmp.shifted_median/cmp.ok_median.replace(0,np.nan)
    cmp['abs_log10_ratio']=np.log10(cmp.ratio.abs().replace(0,np.nan)).abs(); cmp.index.name='column'
    cmp.sort_values('abs_log10_ratio',ascending=False).to_csv(DEST/'tbm9_column_comparison.csv',encoding='utf-8-sig')
    print(f'window {WINDOW[0]}..{WINDOW[1]}: {len(sh)} affected and {len(ok)} healthy files; columns {len(cmp)}; median differs by more than 10x in {int((cmp.abs_log10_ratio>1).sum())}')

if __name__=='__main__': main()
