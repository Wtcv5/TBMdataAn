"""P1: standard time series v1. Reads every raw export that the P0 list keeps, maps the machine-specific column names to one set of
standard fields, normalises the time axis and attaches row-level quality flags. Nothing is interpolated or smoothed; values that are
repaired keep their exported value next to the repaired one. One parquet file per machine in outputs/standard_v1 (git-ignored,
rebuilt by this script) plus the field dictionary, the column map, the row accounting and the flag summary."""
import argparse, hashlib, json, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from audit_data import ALIASES
from p0_file_provenance import N_RPM_LIMIT, V_IMPLAUSIBLE, DROPOUT_FRACTION, TORQUE_PATTERNS, SPEED_PATTERNS, motor_columns

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'TBMData/TBMData'; OUT=ROOT/'analysis/outputs'; P0=OUT/'p0'; DEST=OUT/'standard_v1'
T_IMPLAUSIBLE=12000.     # kN*m: the bulk of every machine ends near 7,000; readings above this are sparse and mostly taken while stopped
F_IMPLAUSIBLE=60000.     # kN: the largest reading in the data is 44,414
DRIVE_SHIFT_RATIO=0.5    # n divided by the drive-motor speed: about 0.06 (or 0.003) when n is a rotation speed, about 1 when it holds the motor frequency
N_MIN_FOR_RATIO=0.5      # below this the cutterhead speed is at the resolution floor and its ratio to the drive speed says nothing
PNV_BAND=(0.5,2.0)       # penetration x speed / advance speed outside this band, where the three are large enough to be read
PNV_MIN=dict(v=10.,p=1.,n=.5)
GLITCH_TOLERANCE=5.      # a counter that sits further than this from its level at the start of the file (metres or rings) is off
STATE_DTYPE='Int16'

SIGNAL_ALIASES={k:ALIASES[k] for k in ('F','T','n','v','p')}
STATE_ALIASES={'state_boring':['掘进状态','推进状态'],'state_cutter':['刀盘工作状态'],'state_thrust':['推进工作状态'],'state_regrip':['换步状态']}
COUNTER_COLUMNS={'chainage':'导向盾首里程','ring':'导向掘进环数','cutter_mileage':'刀盘里程'}
FLAGS=['q_time_backward','q_time_coarse','q_time_conflict','q_neg','q_core_missing','q_n_invalid','q_v_high','q_T_high','q_F_high','q_pnv_off','q_v_invalid_file','q_counter_glitch']
VALUE_COLUMNS=['F','T','n_raw','v','p','state_boring','state_cutter','state_thrust','state_regrip','chainage','ring','cutter_mileage','drive_speed_med','drive_torque_med','drive_motors_on']
ORDER=['time','machine','stream','project','file_status','source_id','row','F','T','n','v','p','n_raw','n_source','state_boring','state_cutter','state_thrust','state_regrip',
       'chainage','ring','cutter_mileage','drive_speed_med','drive_torque_med','drive_motors_on','dt_prev_s','active','valid_core']+FLAGS

def resolve(header):
    """Standard field -> source column for one export header."""
    cols={'time':next((c for c in ALIASES['time'] if c in header),None)}
    for k,names in {**SIGNAL_ALIASES,**STATE_ALIASES}.items(): cols[k]=next((c for c in names if c in header),None)
    for k,c in COUNTER_COLUMNS.items(): cols[k]=c if c in header else None
    cols['_torque']=motor_columns(header,TORQUE_PATTERNS); cols['_speed']=motor_columns(header,SPEED_PATTERNS)
    return cols

def row_median_positive(frame):
    """Median over the motors that read above zero, and how many do; NaN where none does."""
    a=np.where(frame.to_numpy(float)>0,frame.to_numpy(float),np.nan) if frame.shape[1] else np.full((len(frame),1),np.nan)
    a.sort(axis=1); k=np.isfinite(a).sum(axis=1); med=np.full(len(a),np.nan); ok=np.flatnonzero(k>0)
    med[ok]=(a[ok,(k[ok]-1)//2]+a[ok,k[ok]//2])/2
    return med,k

def glitch_mask(events,n):
    """Rows where a position counter sits off its level at the start of the file, from the jumps found in P0."""
    mask=np.zeros(n,bool)
    for _,g in events.groupby('counter'):
        g=g.sort_values('row'); rows=g.row.to_numpy(); offset=np.cumsum(g.delta.to_numpy()); ends=np.r_[rows[1:],n]
        for r0,r1,o in zip(rows,ends,offset):
            if abs(o)>GLITCH_TOLERANCE: mask[r0:r1]=True
    return mask

def standardize(job):
    sid,rel,enc,header,meta=job
    cols=resolve(header); use=sorted({c for k,c in cols.items() if k[0]!='_' and c}|set(cols['_torque'])|set(cols['_speed']))
    raw=pd.read_csv(DATA/rel,usecols=use,encoding=enc,low_memory=False); n0=len(raw)
    t=pd.to_datetime(raw[cols['time']],errors='coerce'); d=pd.DataFrame({'time':t,'source_id':np.int16(sid),'row':np.arange(n0,dtype=np.int32)})
    backward=(t<t.shift()).to_numpy()
    num=lambda c: pd.to_numeric(raw[c],errors='coerce').replace([np.inf,-np.inf],np.nan).astype('float64') if c else pd.Series(np.nan,index=raw.index)
    for k in SIGNAL_ALIASES: d[k]=num(cols[k])
    for k,c in COUNTER_COLUMNS.items():
        x=num(cols[k]); typical=meta['counter_typical'].get(k,np.nan); d[k]=x.where(x.abs()>DROPOUT_FRACTION*typical) if np.isfinite(typical) else np.nan
    for k in STATE_ALIASES: d[k]=num(cols[k]).round().astype(STATE_DTYPE)
    speed,on=row_median_positive(raw[cols['_speed']].apply(pd.to_numeric,errors='coerce')) if cols['_speed'] else (np.full(n0,np.nan),np.zeros(n0,int))
    torque,_=row_median_positive(raw[cols['_torque']].apply(pd.to_numeric,errors='coerce')) if cols['_torque'] else (np.full(n0,np.nan),0)
    d['drive_speed_med']=speed; d['drive_torque_med']=torque.astype('float32'); d['drive_motors_on']=on.astype('int8') if cols['_speed'] else np.int8(0)
    # cutterhead speed: a value above the rpm limit, or one that is as large as the drive-motor speed, is not a rotation speed
    n_raw=d.n.copy(); fm=d.drive_speed_med; shifted=(n_raw>N_RPM_LIMIT)|((n_raw>=N_MIN_FOR_RATIO)&(fm>0)&(n_raw/fm.where(fm>0)>DRIVE_SHIFT_RATIO))
    by_vp=(d.v/d.p.where(d.p>0)).where(d.v>0); k_n=meta['drive_to_n']; by_drive=k_n*fm if np.isfinite(k_n) else pd.Series(np.nan,index=d.index)
    use_vp=shifted&(by_vp<=N_RPM_LIMIT)&by_vp.notna(); use_drive=shifted&~use_vp&(by_drive<=N_RPM_LIMIT)&by_drive.notna()
    d['n']=n_raw.where(~shifted); d.loc[use_vp,'n']=by_vp[use_vp]; d.loc[use_drive,'n']=by_drive[use_drive]; d['n_raw']=n_raw
    src=np.where(n_raw.isna(),'missing',np.where(use_vp,'repaired_v_over_p',np.where(use_drive,'repaired_drive_speed',np.where(shifted,'unrecoverable','exported'))))
    d['n_source']=pd.Categorical(src,categories=['exported','repaired_v_over_p','repaired_drive_speed','unrecoverable','missing'])
    d['drive_speed_med']=d.drive_speed_med.astype('float32'); d['active']=(d[['F','T','n','v']]>0).all(axis=1)
    ratio=(d.p*d.n/d.v.where(d.v>0)).where((d.v>=PNV_MIN['v'])&(d.p>=PNV_MIN['p'])&(d.n>=PNV_MIN['n']))
    d['q_time_backward']=backward; d['q_time_coarse']=meta['coarse']; d['q_neg']=(d[['F','T','n_raw','v','p']]<0).any(axis=1)
    d['q_core_missing']=d[['F','T','n','v']].isna().any(axis=1); d['q_n_invalid']=shifted
    d['q_v_high']=d.v>V_IMPLAUSIBLE; d['q_T_high']=d['T']>T_IMPLAUSIBLE; d['q_F_high']=d.F>F_IMPLAUSIBLE; d['q_pnv_off']=(ratio<PNV_BAND[0])|(ratio>PNV_BAND[1])
    d['q_v_invalid_file']=meta['status']=='signal_invalid'; d['q_counter_glitch']=glitch_mask(meta['events'],n0) if len(meta['events']) else False
    for k in ('machine','stream','project','file_status'): d[k]=meta[k if k!='file_status' else 'status']
    excluded=np.zeros(n0,bool)
    for a,b in meta['exclusions']: excluded[a:b+1]=True
    keep=t.notna().to_numpy()&~excluded
    acct=dict(source_id=sid,rows_raw=n0,rows_invalid_time=int((~t.notna().to_numpy()&~excluded).sum()),rows_p0_excluded=int(excluded.sum()))
    return d[keep],acct

def counter_typical(prov,machine):
    """Typical magnitude of each position counter on a machine: median over files of the largest absolute reading."""
    out={}
    for k,c in COUNTER_COLUMNS.items():
        if f'{c}_max' not in prov: continue
        g=prov[prov.machine==machine]; m=np.maximum(g[f'{c}_max'].abs(),g[f'{c}_min'].abs()); m=m[m>0]
        if len(m): out[k]=float(m.median())
    return out

def drive_to_n(t,machine):
    """Rotation speed per unit of drive-motor speed, from the files of the machine whose cutterhead speed passed the P0 checks."""
    g=t[(t.machine==machine)&(t.n_state=='ok')&t.status.isin(['trusted','flagged'])&t.ratio_n_over_motor_freq.notna()]
    return float(g.ratio_n_over_motor_freq.median()) if len(g) else np.nan

def build_jobs(machines,limit):
    fa=pd.read_csv(OUT/'file_audit.csv'); schemas={s['id']:s['columns'] for s in json.loads((OUT/'schemas.json').read_text(encoding='utf-8'))}
    t=pd.read_csv(P0/'trusted_sources.csv'); prov=pd.read_csv(P0/'file_provenance.csv'); prov['machine']=prov.file.str.extract(r'^(TBM\d+)')[0]
    excl=pd.read_csv(P0/'row_exclusions.csv'); ev=pd.read_csv(P0/'counter_events.csv'); ev=ev[ev.kind.isin(['within_file_excursion','within_file_return','within_file_shift'])]
    typ={m:counter_typical(prov,m) for m in t.machine.unique()}; kn={m:drive_to_n(t,m) for m in t.machine.unique()}
    jobs=[]
    for r in t.itertuples():
        if r.status=='exclude' or (machines and r.machine not in machines): continue
        meta=dict(machine=r.machine,stream=r.stream,project=r.project,status=r.status,coarse=r.timestamps=='minute_resolution',counter_typical=typ[r.machine],drive_to_n=kn[r.machine],
                  exclusions=[(int(a),int(b)) for a,b in excl[excl.source_id==r.source_id][['row_start','row_end']].itertuples(index=False)],events=ev[ev.source_id==r.source_id][['counter','row','delta']])
        jobs.append((int(r.source_id),fa.file[r.source_id].replace('\\','/'),fa.encoding[r.source_id],schemas[fa.schema[r.source_id]],meta))
    return jobs[:limit],t,kn

def assemble(frames):
    d=pd.concat(frames,ignore_index=True).sort_values(['time','source_id','row'],kind='stable',ignore_index=True)
    key=pd.DataFrame({'t':d.time.astype('int64'),'h':pd.util.hash_pandas_object(d[VALUE_COLUMNS].astype('float64'),index=False).to_numpy()})
    dup=key.duplicated(keep='first').to_numpy(); dropped=d[dup].groupby('source_id').size(); d=d[~dup].reset_index(drop=True)
    d['q_time_conflict']=d.time.duplicated(keep=False)
    d['dt_prev_s']=d.time.diff().dt.total_seconds().astype('float32')
    d['valid_core']=~d[['q_neg','q_core_missing','q_v_high','q_T_high','q_F_high','q_v_invalid_file','q_time_conflict']].any(axis=1)
    for k in ('machine','stream','project','file_status'): d[k]=d[k].astype('category')
    return d[ORDER],dropped

def time_axis(d):
    """Sampling and gaps per stream, from consecutive rows of the same stream."""
    rows=[]
    for stream,g in d.groupby('stream',observed=True):
        dt=g.time.diff().dt.total_seconds().dropna(); mode=dt.mode().iloc[0] if len(dt) else np.nan
        rows.append(dict(stream=stream,machine=g.machine.iloc[0],rows=len(g),time_min=g.time.min(),time_max=g.time.max(),days_with_data=int(g.time.dt.normalize().nunique()),
                         dt_median_s=float(dt.median()) if len(dt) else np.nan,dt_mode_s=float(mode),share_dt_at_mode=float((dt==mode).mean()) if len(dt) else np.nan,dt_p95_s=float(dt.quantile(.95)) if len(dt) else np.nan,
                         gaps_over_60s=int((dt>60).sum()),gaps_over_300s=int((dt>300).sum()),longest_gap_days=float(dt.max()/86400) if len(dt) else np.nan))
    return pd.DataFrame(rows)

def availability(d):
    """Share of rows with a value, per stream and field."""
    cols=['F','T','n','v','p','state_boring','state_cutter','state_thrust','state_regrip','chainage','ring','cutter_mileage','drive_speed_med','drive_torque_med']
    a=d.groupby('stream',observed=True)[cols].agg(lambda x: x.notna().mean()); a.insert(0,'machine',d.groupby('stream',observed=True).machine.first().astype(str)); a.insert(1,'rows',d.groupby('stream',observed=True).size())
    return a.reset_index()

def field_dictionary():
    """Standard fields with unit status. Units are the readings of the P0 report; only some are tied to a second quantity."""
    rows=[('time','datetime64[ns]','','','Time of the reading as exported; local clock, time zone unconfirmed','运行时间-time / 时间'),
          ('machine','category','','','TBM number from the directory name',''),('stream','category','','','Directory (and part) the file came from',''),
          ('project','category','','','Project named in the directory; 未标明 when it names none',''),('file_status','category','','','P0 status of the source file',''),
          ('source_id','int16','','','Index of the source file in file_audit.csv / source_files.csv',''),('row','int32','','','0-based row number in the source file',''),
          ('F','float64','kN','unverified','Total thrust',' / '.join(SIGNAL_ALIASES['F'])),('T','float64','kN*m','unverified (constant multiple of the drive motors, P0)','Cutterhead torque',' / '.join(SIGNAL_ALIASES['T'])),
          ('n','float64','rpm','consistent with v/p, absolute unit unconfirmed','Cutterhead speed after repair (see n_source)',' / '.join(SIGNAL_ALIASES['n'])),
          ('v','float64','mm/min','verified against the position counters (P0)','Advance speed',' / '.join(SIGNAL_ALIASES['v'])),('p','float64','mm/rev','follows from v = p*n','Penetration per revolution',' / '.join(SIGNAL_ALIASES['p'])),
          ('n_raw','float64','as exported','','Cutterhead speed exactly as exported; differs from n only where n_source starts with repaired or is unrecoverable',' / '.join(SIGNAL_ALIASES['n'])),
          ('n_source','category','','','exported, repaired_v_over_p, repaired_drive_speed, unrecoverable or missing',''),
          ('state_boring','Int16','0/1','','The machine\'s own boring flag (TBM8: advance state); empty where the export has none','掘进状态 / 推进状态'),
          ('state_cutter','Int16','code','','Cutterhead working state code','刀盘工作状态'),('state_thrust','Int16','code','','Thrust working state code','推进工作状态'),('state_regrip','Int16','0/1','','Re-gripping state','换步状态'),
          ('chainage','float64','m','verified against v (P0)','Guide shield head chainage; zero and near-zero readings are NaN','导向盾首里程'),('ring','float64','ring (about 1 m)','','Guide boring ring count; zero readings are NaN','导向掘进环数'),
          ('cutter_mileage','float64','m','verified against v (P0)','Cutterhead mileage of TBM1 to TBM6 (offset by about 735,000 on TBM5 and TBM6); zero readings are NaN','刀盘里程'),
          ('drive_speed_med','float32','family specific','not comparable across drive families','Median speed (TBM1-TBM6) or frequency (others) over the drive motors that read above zero','主驱动电机i速度 / Mi电机频率 / i#电机输出频率'),
          ('drive_torque_med','float32','family specific','not comparable across drive families','Median torque over the drive motors that read above zero','主驱动电机i扭矩 / Mi电机扭矩 / i#电机扭矩'),
          ('drive_motors_on','int8','count','','Drive motors reading above zero',''),('dt_prev_s','float32','s','','Seconds since the previous reading of the machine',''),
          ('active','bool','','','F, T, n and v all above zero',''),('valid_core','bool','','','No hard flag: not q_neg, q_core_missing, q_v_high, q_T_high, q_F_high, q_v_invalid_file or q_time_conflict','')]
    flags={'q_time_backward':'Time is earlier than the previous row of the file','q_time_coarse':'File has timestamps only to the minute (P0)','q_time_conflict':'Another reading of the machine has the same timestamp and different values',
           'q_neg':'A core signal is negative','q_core_missing':'F, T, n or v is empty (also where the export has no such column)','q_n_invalid':'Exported n is not a rotation speed (above the rpm limit, or as large as the drive-motor speed)',
           'q_v_high':f'v above {V_IMPLAUSIBLE:g} mm/min','q_T_high':f'T above {T_IMPLAUSIBLE:g}','q_F_high':f'F above {F_IMPLAUSIBLE:g}','q_pnv_off':'p*n/v outside 0.5 to 2 where v, p and n are large enough to be read',
           'q_v_invalid_file':'P0 found v and p of the whole file unusable','q_counter_glitch':'A position counter is off its level at the start of the file (jumps found in P0)'}
    rows+=[(k,'bool','','',desc,'') for k,desc in flags.items()]
    return pd.DataFrame(rows,columns=['name','dtype','unit','unit_status','description','source_columns'])

def column_map(jobs):
    """One row per distinct way the standard fields resolve to source columns, with the number of files that resolve that way."""
    rows=[]
    for sid,rel,enc,header,meta in jobs:
        c=resolve(header)
        rows.append({**{k:c[k] for k in c if k[0]!='_'},'drive_torque_columns':len(c['_torque']),'drive_torque_first_last':'|'.join(c['_torque'][:1]+c['_torque'][-1:]),
                     'drive_speed_columns':len(c['_speed']),'drive_speed_first_last':'|'.join(c['_speed'][:1]+c['_speed'][-1:])})
    m=pd.DataFrame(rows); g=m.groupby(list(m.columns),dropna=False).size().reset_index(name='files')
    return g.sort_values('files',ascending=False)

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda: f.read(1<<20),b''): h.update(chunk)
    return h.hexdigest()

def checksums(d):
    return {c:(dict(nan=int(d[c].isna().sum()),sum=float(np.nansum(d[c].astype('float64')))) if d[c].dtype.kind in 'fiub' or str(d[c].dtype).startswith(('Int','float')) else dict(nunique=int(d[c].nunique()))) for c in d.columns if c!='time'}

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--workers',type=int,default=4); ap.add_argument('--machines',default=''); ap.add_argument('--limit',type=int,default=None); a=ap.parse_args()
    DEST.mkdir(exist_ok=True); start=time.perf_counter(); jobs,t,kn=build_jobs(set(filter(None,a.machines.split(','))),a.limit)
    with ProcessPoolExecutor(a.workers) as ex: results=list(ex.map(standardize,jobs,chunksize=4))
    print(f'{len(jobs)} files standardised in {time.perf_counter()-start:.0f} s',flush=True)
    by_machine={}; acct=[]; dropped_parts=[]; axes=[]; avail=[]
    for (sid,rel,enc,header,meta),(d,ac) in zip(jobs,results): by_machine.setdefault(meta['machine'],[]).append(d); acct.append(ac)
    acct=pd.DataFrame(acct).set_index('source_id'); manifest={}; flags=[]
    for machine in sorted(by_machine,key=lambda m:int(m[3:])):
        d,dropped=assemble(by_machine[machine]); dropped_parts.append(dropped); path=DEST/f'{machine}.parquet'
        d.to_parquet(path,engine='pyarrow',compression='zstd',index=False); axes.append(time_axis(d)); avail.append(availability(d))
        manifest[machine]=dict(file=path.name,rows=len(d),bytes=path.stat().st_size,sha256=sha256(path),time_min=str(d.time.min()),time_max=str(d.time.max()),columns=list(d.columns),checksums=checksums(d))
        f=d.groupby('stream',observed=True)[FLAGS+['active','valid_core']].sum(); f.insert(0,'rows',d.groupby('stream',observed=True).size()); f.insert(0,'machine',machine); flags.append(f.reset_index())
        print(machine,len(d),'rows',f'{time.perf_counter()-start:.0f} s',flush=True)
    full=t.set_index('source_id'); a_all=pd.DataFrame(index=full.index); a_all['file']=full.file; a_all['stream']=full.stream; a_all['status']=full.status; a_all['rows_raw']=full.rows
    a_all['rows_p0_excluded']=np.where(full.status=='exclude',full.rows,0); a_all=a_all.join(acct.drop(columns='rows_raw'),rsuffix='_w'); a_all['rows_p0_excluded']=a_all.rows_p0_excluded_w.fillna(a_all.rows_p0_excluded).astype(int)
    a_all['rows_invalid_time']=a_all.rows_invalid_time.fillna(0).astype(int); a_all['rows_exact_duplicate_dropped']=(pd.concat(dropped_parts) if dropped_parts else pd.Series(dtype='int64')).reindex(a_all.index).fillna(0).astype(int)
    kept=pd.concat([pd.read_parquet(DEST/m['file'],columns=['source_id']) for m in manifest.values()]).groupby('source_id').size() if manifest else pd.Series(dtype=int)
    a_all['rows_kept']=kept.reindex(a_all.index).fillna(0).astype(int); a_all=a_all.drop(columns=['rows_p0_excluded_w']).reset_index().rename(columns={'index':'source_id'})
    if not a.machines and a.limit is None: assert (a_all.rows_raw==a_all.rows_p0_excluded+a_all.rows_invalid_time+a_all.rows_exact_duplicate_dropped+a_all.rows_kept).all(),'row accounting does not reconcile'
    a_all.to_csv(DEST/'row_accounting.csv',index=False,encoding='utf-8-sig'); pd.concat(flags).to_csv(DEST/'flag_summary.csv',index=False,encoding='utf-8-sig')
    pd.concat(axes).to_csv(DEST/'time_axis.csv',index=False,encoding='utf-8-sig'); pd.concat(avail).to_csv(DEST/'field_availability.csv',index=False,encoding='utf-8-sig'); field_dictionary().to_csv(DEST/'field_dictionary.csv',index=False,encoding='utf-8-sig'); column_map(jobs).to_csv(DEST/'column_map.csv',index=False,encoding='utf-8-sig')
    t[['source_id','file','stream','machine','project','status','action']].to_csv(DEST/'source_files.csv',index=False,encoding='utf-8-sig')
    inputs={p.name:sha256(p) for p in (P0/'trusted_sources.csv',P0/'row_exclusions.csv',P0/'counter_events.csv',P0/'file_provenance.csv')}
    (DEST/'manifest.json').write_text(json.dumps(dict(machines=manifest,p0_inputs_sha256=inputs,drive_to_n={m:v for m,v in kn.items()},limits=dict(rpm=N_RPM_LIMIT,v=V_IMPLAUSIBLE,T=T_IMPLAUSIBLE,F=F_IMPLAUSIBLE)),ensure_ascii=False,indent=1),encoding='utf-8')
    print('done',f'{time.perf_counter()-start:.0f} s')

if __name__=='__main__': main()
