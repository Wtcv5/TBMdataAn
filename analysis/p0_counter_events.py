"""P0 step 2: jumps of the position counter, in time order per machine and across file boundaries (read-only).
Exact zeros are treated as dropouts. A jump inside a file is either a counter glitch that returns to its old level or a
change of source; a jump between files is normal progress during a gap in the record unless the implied speed is impossible."""
import argparse, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'TBMData/TBMData'; OUT=ROOT/'analysis/outputs'; DEST=OUT/'p0'
POSITION={**{f'TBM{i}':['刀盘里程'] for i in range(1,7)},**{f'TBM{i}':['导向盾首里程','导向掘进环数'] for i in (8,9,10)}}   # position counters per machine, metres or rings
JUMP=20.0               # larger steps than this between consecutive valid values are events
RETURN_TOL=5.0          # the series has come back when it is within this of the pre-jump level
LOOKAHEAD=5000          # valid rows searched for the return
MAX_RATE_PER_DAY=60.0   # no machine here advances faster than this (metres or rings per day)
DROPOUT_FRACTION=0.01  # readings below this fraction of the typical magnitude are dropouts

def read_counter(args):
    sid,rel,enc,counter=args
    hdr=pd.read_csv(DATA/rel.replace('\\','/'),encoding=enc,nrows=0).columns
    tcol='运行时间-time' if '运行时间-time' in hdr else '时间'
    raw=pd.read_csv(DATA/rel.replace('\\','/'),encoding=enc,usecols=[tcol]+([counter] if counter in hdr else []),low_memory=False)
    x=pd.to_numeric(raw[counter],errors='coerce') if counter in raw else pd.Series(np.nan,index=raw.index)
    return pd.DataFrame({'source_id':sid,'row':np.arange(len(raw)),'time':pd.to_datetime(raw[tcol],errors='coerce'),'value':x})

def machine_events(machine,counter,d):
    d=d.reset_index(drop=True); d['value']=d.value.where(d.value.abs()>DROPOUT_FRACTION*d.value.abs().median())   # zero or near-zero readings are dropouts
    v=d[d.value.notna()]; idx=v.index.to_numpy(); val=v.value.to_numpy(); sid=v.source_id.to_numpy(); t=v.time.to_numpy()
    step=np.diff(val); rows=[]
    for k in np.flatnonzero(np.abs(step)>JUMP):
        a,b=k,k+1; same=sid[a]==sid[b]
        ev=dict(machine=machine,counter=counter,delta=step[k],source_id_prev=int(sid[a]),row_prev=int(d.row[idx[a]]),time_prev=t[a],value_prev=val[a],
                source_id=int(sid[b]),row=int(d.row[idx[b]]),time=t[b],value=val[b],same_file=bool(same))
        gap_days=(t[b]-t[a])/np.timedelta64(1,'D') if pd.notna(t[a]) and pd.notna(t[b]) else np.nan
        ev['gap_days']=gap_days; ev['rate_per_day']=step[k]/gap_days if gap_days and gap_days>0 else np.nan
        if same:
            tail=val[b:b+LOOKAHEAD]; tail_sid=sid[b:b+LOOKAHEAD]; back=np.flatnonzero((np.abs(tail-val[a])<=RETURN_TOL)&(tail_sid==sid[b]))
            ev['returns_after_valid_rows']=int(back[0]) if len(back) else np.nan
            ev['kind']='within_file_excursion' if len(back) else 'within_file_shift'
        else:
            ev['returns_after_valid_rows']=np.nan
            ev['kind']='between_files_'+('too_fast' if not abs(ev['rate_per_day'])<=MAX_RATE_PER_DAY else 'step')
        rows.append(ev)
    gaps=[e for e in rows if e['kind']=='between_files_step']; direction=np.sign(np.median([e['delta'] for e in gaps])) if len(gaps)>=3 else 1.0
    for e in gaps: e['kind']='between_files_progress' if np.sign(e['delta'])==direction else 'between_files_regress'
    for e in rows: e['machine_direction']=direction
    for e in rows:                      # the event that closes an excursion is its return, not a new shift
        if e['kind']=='within_file_excursion':
            back=next((r for r in rows if r is not e and r['source_id']==e['source_id'] and r['row']>e['row'] and abs(r['value']-e['value_prev'])<=RETURN_TOL),None)
            if back is not None and back['kind']=='within_file_shift': back['kind']='within_file_return'
    return rows

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--workers',type=int,default=4); a=ap.parse_args()
    fa=pd.read_csv(OUT/'file_audit.csv'); fa['file']=fa.file.str.replace('\\','/'); fa['machine']=fa.file.str.extract(r'^(TBM\d+)')[0]
    fa['tmin']=pd.to_datetime(fa.time_min,errors='coerce'); start=time.perf_counter(); events=[]
    for machine,counters in POSITION.items():
        sub=fa[fa.machine==machine].sort_values(['tmin']).copy()
        if sub.empty: continue
        for counter in counters:
            with ProcessPoolExecutor(a.workers) as ex: parts=list(ex.map(read_counter,[(i,r.file,r.encoding,counter) for i,r in sub.iterrows()],chunksize=4))
            events+=machine_events(machine,counter,pd.concat(parts,ignore_index=True))
    df=pd.DataFrame(events); df['file']=df.source_id.map(fa.file); df['file_prev']=df.source_id_prev.map(fa.file)
    DEST.mkdir(exist_ok=True); df.to_csv(DEST/'counter_events.csv',index=False,encoding='utf-8-sig')
    print('wrote',DEST/'counter_events.csv',len(df),'events',round(time.perf_counter()-start),'s')
    print(df.groupby(['machine','counter','kind']).size().to_string())

if __name__=='__main__': main()
