"""P0 step 1: per-file provenance fingerprints from the raw exports (read-only).
Counters (chainage, ring, running time), the machine's own boring state, core-signal medians on active rows, physical
consistency ratios that expose unit or scale changes, and the sampling pattern of the time column. Row hashes go to the
git-ignored p0_cache for duplicate analysis."""
import argparse, json, re, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from audit_data import ALIASES

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'TBMData/TBMData'; OUT=ROOT/'analysis/outputs'; DEST=OUT/'p0'; CACHE=OUT/'p0_cache'
CORE=['F','T','n','v']
COUNTERS=['刀盘里程','主梁里程','导向盾首里程','导向盾中里程','导向盾尾里程','导向掘进环数','刀盘运行时间','隧道总长度']
STATE='掘进状态'
SETPOINTS={'n_set':'刀盘给定转速','f_set':'刀盘给定频率'}   # commanded cutterhead speed and commanded drive frequency
ADVANCE_COUNTERS={'chainage':'导向盾首里程','ring':'导向掘进环数','cutter':'刀盘里程'}   # position counters in metres; v is mm/min
V_IMPLAUSIBLE=250.0     # mm/min (15 m/h): no file median of any machine here exceeds a third of this
N_RPM_LIMIT=12.0        # no cutterhead here turns faster than this
DROPOUT_FRACTION=0.01  # counter readings below this fraction of the typical magnitude are dropouts
TORQUE_PATTERNS=[r'^主驱动电机\d+扭矩$',r'^M\d+电机扭矩$',r'^\d+#电机扭矩$']
SPEED_PATTERNS=[r'^主驱动电机\d+速度$',r'^M\d+电机频率$',r'^\d+#电机输出频率$']

def motor_columns(header,patterns):
    for p in patterns:
        found=[c for c in header if re.match(p,c)]
        if found: return found
    return []

def median_ratio(num,den):
    ok=(den>0)&(num>0)
    return float((num[ok]/den[ok]).median()) if ok.sum() else np.nan

def advance_closure(df,raw):
    """Distance advanced according to the speed channel vs according to each position counter (metres, small steps only)."""
    order=df.time.sort_values().index; dt=df.time.loc[order].diff().dt.total_seconds(); ok=(dt>0)&(dt<=120)
    have=[c for c in ('F','T','n') if df[c].notna().any()]; boring=(df[have]>0).all(axis=1)&(df.v>0)     # cylinder movement while re-gripping is not advance
    out={}
    for key,v in (('adv_v_m',df.v.where(boring,0)),('adv_v_all_m',df.v)):
        v=v.loc[order].fillna(0); out[key]=float((((v+v.shift())/2*(dt/60)/1000).where(ok,0)).sum())
    for key,c in ADVANCE_COUNTERS.items():
        if c not in raw: continue
        x=pd.to_numeric(raw[c],errors='coerce').set_axis(df.index).loc[order]; x=x.where(x.abs()>DROPOUT_FRACTION*x.abs().median()); d=(x-x.shift()).where(ok)   # zero and near-zero readings are dropouts
        d=d.where(d.abs()<5); out['adv_'+key+'_m']=float(max(d.clip(lower=0).sum(),(-d).clip(lower=0).sum()))   # small steps in the direction of travel; counter jitter inflates it
        v=x.dropna(); out['adv_'+key+'_net_m']=float(abs(v.iloc[-1]-v.iloc[0])) if len(v) else np.nan            # first to last valid reading
    return out

def summarize(args):
    sid,rel,enc,header=args
    path=DATA/rel.replace('\\','/')
    mapping={k:next((c for c in names if c in header),None) for k,names in ALIASES.items()}
    torque,speed=motor_columns(header,TORQUE_PATTERNS),motor_columns(header,SPEED_PATTERNS)
    wanted=[c for c in set(v for v in mapping.values() if v)|set(COUNTERS)|{STATE}|set(torque)|set(speed)|set(SETPOINTS.values()) if c in header]
    raw=pd.read_csv(path,usecols=wanted,encoding=enc,low_memory=False)
    df=pd.DataFrame({'time':pd.to_datetime(raw[mapping['time']],errors='coerce')})
    for k,c in mapping.items():
        if k!='time': df[k]=pd.to_numeric(raw[c],errors='coerce') if c else np.nan
    df=df.replace([np.inf,-np.inf],np.nan)
    row=dict(source_id=sid,file=rel.replace('\\','/'),rows=len(df),time_min=df.time.min(),time_max=df.time.max())
    dt=df.time.sort_values().diff().dt.total_seconds().dropna(); sec=df.time.dt.second.dropna()
    row.update(dt_median_s=float(dt.median()) if len(dt) else np.nan,dt_mode_s=float(dt.mode().iloc[0]) if len(dt) else np.nan,
               dup_time_share=float(df.time.duplicated().mean()) if len(df) else np.nan,second_zero_share=float((sec==0).mean()) if len(sec) else np.nan)
    active=(df[CORE]>0).all(axis=1); a=df[active]; row['active_rows']=int(active.sum())
    for c in CORE+['p','n_aux']: row['med_'+c]=float(a[c].median()) if len(a) and a[c].notna().any() else np.nan
    row['ratio_p_n_over_v']=median_ratio(a.p*a.n,a.v); row['n_gt12_share']=float((a.n>N_RPM_LIMIT).mean()) if len(a) else np.nan
    row['v_implausible_share']=float((a.v>V_IMPLAUSIBLE).mean()) if len(a) else np.nan
    if torque:
        tq=pd.DataFrame({c:pd.to_numeric(raw[c],errors='coerce') for c in torque}).loc[a.index]; ts=tq.where(tq>0).sum(axis=1,min_count=1)
        row['motor_torque_sum_median']=float(ts[ts>0].median()) if (ts>0).any() else np.nan; row['ratio_T_over_torque_sum']=median_ratio(a['T'],ts)
    if speed:
        sp=pd.DataFrame({c:pd.to_numeric(raw[c],errors='coerce') for c in speed}).loc[a.index]; on=sp>0; fm=sp.where(on).mean(axis=1)
        row['motors_total']=len(speed); row['motors_on_median']=float(on.sum(axis=1).median()) if len(sp) else np.nan
        row['motor_freq_median']=float(fm[fm>0].median()) if (fm>0).any() else np.nan; row['ratio_n_over_motor_freq']=median_ratio(a.n,fm)
    for k,c in SETPOINTS.items():
        if c in raw:
            x=pd.to_numeric(raw[c],errors='coerce').loc[a.index]; row['med_'+k]=float(x.median()) if x.notna().any() else np.nan
            if k=='n_set': row['ratio_n_over_n_set']=median_ratio(a.n,x)
    row.update(advance_closure(df,raw))
    for c in COUNTERS:
        if c not in raw: continue
        x=pd.to_numeric(raw[c],errors='coerce').set_axis(df.index); order=df.time.sort_values().index; x=x.loc[order]
        row[f'{c}_valid_share']=float(((x!=0)&x.notna()).mean()); x=x.dropna()
        if len(x)==0: continue
        d=x.diff().dropna()
        row.update({f'{c}_first':float(x.iloc[0]),f'{c}_last':float(x.iloc[-1]),f'{c}_min':float(x.min()),f'{c}_max':float(x.max()),
                    f'{c}_decreases':int((d<0).sum()),f'{c}_max_up_step':float(d.max()) if len(d) else np.nan})
    if STATE in raw:
        s=pd.to_numeric(raw[STATE],errors='coerce'); vc=s.value_counts(normalize=True).head(6)
        row['state_shares']=json.dumps({str(int(k)) if float(k).is_integer() else str(k):round(float(v),4) for k,v in vc.items()})
        row['state_nan_share']=float(s.isna().mean())
    # hashes: active core tuples (as in the overlap audit) and every extracted column
    t=df.time.to_numpy().astype('datetime64[ns]').view('int64')      # NaT stays a fixed sentinel instead of raising
    core=pd.DataFrame({'t':t,**{c:df[c].astype('float64') for c in CORE}})[active.to_numpy()]
    full=pd.DataFrame({'t':t,**{c:df[c].astype('float64') for c in CORE+['p']},**{c:pd.to_numeric(raw[c],errors='coerce') for c in COUNTERS+[STATE] if c in raw}})
    return row,pd.util.hash_pandas_object(core,index=False).to_numpy(),pd.util.hash_pandas_object(full,index=False).to_numpy()

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--limit',type=int,default=None); ap.add_argument('--workers',type=int,default=4); a=ap.parse_args()
    fa=pd.read_csv(OUT/'file_audit.csv'); schemas={s['id']:s['columns'] for s in json.loads((OUT/'schemas.json').read_text(encoding='utf-8'))}
    jobs=[(i,r.file,r.encoding,schemas[r.schema]) for i,r in enumerate(fa.itertuples())][:a.limit]
    DEST.mkdir(exist_ok=True); CACHE.mkdir(exist_ok=True); rows=[]; core={}; full={}; start=time.perf_counter()
    with ProcessPoolExecutor(a.workers) as ex:
        for n,(row,hc,hf) in enumerate(ex.map(summarize,jobs,chunksize=4),1):
            rows.append(row); core[f'c{row["source_id"]}']=hc; full[f'f{row["source_id"]}']=hf
            if n%100==0: print(n,'/',len(jobs),round(time.perf_counter()-start),'s',flush=True)
    df=pd.DataFrame(rows).sort_values('source_id'); df.to_csv(DEST/'file_provenance.csv',index=False,encoding='utf-8-sig')
    np.savez_compressed(CACHE/'row_hashes.npz',**core,**full)
    print('wrote',DEST/'file_provenance.csv',len(df),'files',round(time.perf_counter()-start),'s',flush=True)

if __name__=='__main__': main()
