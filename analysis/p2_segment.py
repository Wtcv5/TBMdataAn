"""P2: continuous boring segments and boring phases on the standard time series v1.
A segment is a maximal run of rows in which the machine says it is boring (its own boring flag; where a file has none the thrust state, and
where that is missing too the signals) with no gap between readings above max_gap_s and no stretch of at least interrupt_s with the flag on but the
thrust off. Segment edges next to a data hole are marked censored.
Segments of at least long_s seconds get phases: ramp (thrust and advance speed rising to their steady level), steady, end (falling away from the level they had). Every row of
the machine series gets a segment number, a phase code and an operating state. Nothing is smoothed or interpolated in the data itself.
Outputs in outputs/segments_v1: segments.csv, segment_summary.csv, phase_summary.csv, state_consistency.csv, label_dictionary.csv, manifest.json
and labels/<machine>.parquet (git-ignored, rebuilt by this script)."""
import argparse, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from p1_standardize import DEST as SRC, OUT, P0, sha256, checksums

DEST=OUT/'segments_v1'
PARAMS=dict(max_gap_s=60.,        # a reading gap above this ends a segment (the baseline's window rule); the sensitivity is in p2_checks.py
            interrupt_s=120.,     # a stretch in which the flag is on but the thrust is off (F not above zero) is no boring when it lasts this long
            blip_s=60.,           # a segment shorter than this, or without one active row, is too short to be a boring stretch
            long_s=300.,          # phases are assigned from this length on
            phase_min_rows=12, valid_min=.9,
            rho_F=.9, rho_T=0., rho_v=.8,   # share of the steady level that ends the ramp and starts the end phase: thrust and advance speed decide, the torque only has to be above zero (0 drops the condition; p2_checks.py shows the alternatives)
            hold_rows=3, smooth_rows=5,     # the shares must hold for hold_rows readings of the rolling median over smooth_rows readings
            trail_s=300.,         # the end phase starts where the signals fall below their level over the preceding trail_s seconds (0: below the steady level of the whole segment)
            end_max_s=300.,       # a stretch below that level that lasts longer than this at the end is a drift or a change of level, not the end of the stroke
            steady_from=.25, steady_to=.75, # the steady level is the median over this part of the segment (by rows)
            stall_v=2.,           # mm/min; a segment whose median advance speed over all its rows is below this is stalled (flag on and thrust on, but no advance)
            n_rotating=.5)                  # rpm; above this the cutterhead counts as turning
PHASE={'none':0,'ramp':1,'steady':2,'end':3,'unassigned':4}
OP_STATE={'boring':0,'rotating':1,'stopped':2,'unknown':3}
REF_SOURCE=['state_boring','state_thrust','signal']
THRUST_ADVANCING=3       # value of the thrust working state while advancing (TBM5 before its boring flag exists)
LABELLED_EDGES=('state_change','pause','hole_other')   # a ramp or end phase is named only where the state change itself was read (or a hole hides only a state that was off), not next to a boring row that a hole cuts off
SEG_COLUMNS=['segment_id','machine','seg_no','stream','start','end','rows','duration_s','dt_median_s','ref_source','left_edge','right_edge','left_censored','right_censored','gap_before_s','gap_after_s',
             'seg_class','active_rows','active_share','valid_share','lead_inactive_s','trail_inactive_s','max_inactive_s','v_all_med','stalled','F_med','T_med','n_med','v_med','adv_v_m','counter','adv_counter_m',
             'phase_status','steady_start','steady_end','ramp_s','steady_s','end_s','tail_s','steady_ok_share','F_level','T_level','v_level','F_steady','T_steady','n_steady','v_steady']
COLUMNS=['time','stream','source_id','row','F','T','n','v','state_boring','state_thrust','dt_prev_s','active','valid_core','chainage','ring','cutter_mileage','q_counter_glitch']
COUNTERS=('chainage','cutter_mileage','ring')

def finish(code=0):
    """Leave without interpreter finalisation: under load pyarrow's thread pool can abort while the process exits ('terminate called without an active exception'), which turns a finished run into exit status -6."""
    sys.stdout.flush(); sys.stderr.flush(); os._exit(code)

def reference(d):
    """Boring flag of every row and where it comes from: the machine's own boring flag, else the thrust state, else the signals (active)."""
    sb,st=d.state_boring,d.state_thrust; hb,ht=sb.notna().to_numpy(),st.notna().to_numpy()
    boring=np.where(hb,sb.eq(1).fillna(False).to_numpy(bool),np.where(ht,st.eq(THRUST_ADVANCING).fillna(False).to_numpy(bool),d.active.to_numpy(bool)))
    return boring.astype(bool),np.where(hb,0,np.where(ht,1,2)).astype('int8')

def pauses(boring,F,ts,dt,p):
    """Rows of the stretches in which the boring flag is on but the thrust is off (F not above zero) for at least interrupt_s: the machine is not boring there,
    whatever the flag says. Shorter drops of the thrust stay in the segment; so do spin-up and spin-down rows that last less than interrupt_s."""
    n=len(boring); paused=np.zeros(n,bool); idx=np.flatnonzero(boring&(F<=0))        # a missing F compares False
    if not len(idx): return paused
    step=np.nanmedian(dt); cut=np.r_[0,np.flatnonzero((np.diff(idx)>1)|~(dt[idx[1:]]<=p['max_gap_s']))+1,len(idx)]
    for c0,c1 in zip(cut[:-1],cut[1:]):
        a,b=idx[c0],idx[c1-1]
        if ts[b]-ts[a]+step>=p['interrupt_s']: paused[a:b+1]=True
    return paused

def find_segments(boring,dt,max_gap):
    """First and last row of every maximal run of boring rows without a reading gap above max_gap, and the segment number of each row (-1 outside)."""
    linked=boring[1:]&boring[:-1]&(dt[1:]<=max_gap)       # a missing dt compares False and so breaks the run
    starts=boring&~np.r_[False,linked]; ends=boring&~np.r_[linked,False]
    no=np.where(boring,np.cumsum(starts)-1,-1).astype('int32')
    return np.flatnonzero(starts),np.flatnonzero(ends),no

def detect(F,T,v,p,dtm):
    """Steady rows of one segment. The steady level of F, T and v is their median over the middle part of the segment and must be above zero in all three, else the
    segment has no steady state. The steady rows start at the first run of hold_rows readings in which the smoothed F, T and v reach rho_F, rho_T and rho_v of that
    level (a share of 0 drops the condition) and end at the last run in which they reach those shares of their own level over the preceding trail_s seconds
    (the level of the whole segment where trail_s is 0). Returns (first,last,levels,share of the steady rows that reach the segment level) or None."""
    m=len(F); lo,hi=int(m*p['steady_from']),int(m*p['steady_to'])+1; k=p['hold_rows']
    lev={c:np.nanmedian(x[lo:hi]) if np.isfinite(x[lo:hi]).any() else np.nan for c,x in (('F',F),('T',T),('v',v))}
    if not all(np.isfinite(list(lev.values()))) or min(lev.values())<=0: return None
    sm={c:pd.Series(x).rolling(p['smooth_rows'],center=True,min_periods=1).median().to_numpy() for c,x in (('F',F),('T',T),('v',v))}
    share=dict(F=p['rho_F'],T=p['rho_T'],v=p['rho_v'])
    def reach(ref):
        ok=np.ones(m,bool)
        for c,r in share.items():
            if r>0: ok&=sm[c]>=r*ref[c]
        return ok
    def runs(ok): return np.convolve(ok.astype(int),np.ones(k,int),'valid')==k          # run[i]: readings i to i+k-1 all reach their level
    ok=reach(lev); run=runs(ok)
    if not run.any(): return None
    first=int(np.argmax(run)); end=ok
    if p['trail_s']>0:
        w=max(2*k,int(round(p['trail_s']/dtm))) if np.isfinite(dtm) and dtm>0 else 2*k
        end=reach({c:pd.Series(sm[c]).rolling(w,min_periods=3).median().shift(1).to_numpy() for c in sm}); end[:first+k]=True
    run=runs(end); last=len(run)-1-int(np.argmax(run[::-1]))+k-1
    return first,last,lev,float(ok[first:last+1].mean())

def longest_inactive(act,ts,dtm):
    """Longest spell of inactive readings between the first and the last active one, in seconds (readings times the typical interval)."""
    idx=np.flatnonzero(act)
    if len(idx)<2: return 0.
    inner=~act[idx[0]:idx[-1]+1]
    if not inner.any(): return 0.
    pos=np.flatnonzero(inner); cut=np.r_[0,np.flatnonzero(np.diff(pos)>1)+1,len(pos)]
    return float(max(ts[idx[0]+pos[c1-1]]-ts[idx[0]+pos[c0]] for c0,c1 in zip(cut[:-1],cut[1:]))+dtm)

def build(d,p,machine):
    """Segments (one dict per segment) and the row labels of one machine series."""
    n=len(d); boring,ref=reference(d); dt=d.dt_prev_s.to_numpy(float); t=d.time.to_numpy(); ts=(t-t[0])/np.timedelta64(1,'s'); act=d.active.to_numpy(bool); mg=p['max_gap_s']
    F,T,nn,v=(d[c].to_numpy(float) for c in ('F','T','n','v')); paused=pauses(boring,F,ts,dt,p); inseg=boring&~paused; first,last,seg_no=find_segments(inseg,dt,mg)
    valid=d.valid_core.to_numpy(bool); v_ok=np.where(valid,v,np.nan)
    counter=next((c for c in COUNTERS if d[c].notna().mean()>=.5),None); cnt=d[counter].to_numpy(float) if counter else np.full(n,np.nan); glitch=d.q_counter_glitch.to_numpy(bool)
    phase=np.zeros(n,'int8'); stream=d.stream.astype(str).to_numpy(); recs=[]
    for i,(a,b) in enumerate(zip(first,last)):
        sl=slice(a,b+1); m=b-a+1; dur=float(ts[b]-ts[a]); dtm=float(np.median(dt[a+1:b+1])) if m>1 else np.nan; ac=act[sl]; na=int(ac.sum())
        le='series_edge' if a==0 else ('pause' if paused[a-1] else 'state_change') if dt[a]<=mg else 'hole_boring' if boring[a-1] else 'hole_other'
        re='series_edge' if b==n-1 else ('pause' if paused[b+1] else 'state_change') if dt[b+1]<=mg else 'hole_boring' if boring[b+1] else 'hole_other'
        cls='blip' if na==0 or dur<p['blip_s'] else 'short' if dur<p['long_s'] else 'long'; first_active=np.flatnonzero(ac)
        ok=~np.isnan(cnt[sl])&~glitch[sl]; c=cnt[sl][ok]
        rec=dict(segment_id=f'{machine}-{i:05d}',seg_no=i,machine=machine,stream=stream[a],start=t[a],end=t[b],rows=m,duration_s=dur,dt_median_s=dtm,ref_source=REF_SOURCE[int(np.bincount(ref[sl],minlength=3).argmax())],
                 left_edge=le,right_edge=re,left_censored=le not in ('state_change','pause'),right_censored=re not in ('state_change','pause'),
                 gap_before_s=dt[a] if a>0 and dt[a]>mg else np.nan,gap_after_s=dt[b+1] if b<n-1 and dt[b+1]>mg else np.nan,
                 seg_class=cls,active_rows=na,active_share=na/m,valid_share=float(valid[sl].mean()),lead_inactive_s=float(ts[a+first_active[0]]-ts[a]) if na else dur,trail_inactive_s=float(ts[b]-ts[a+first_active[-1]]) if na else dur,v_all_med=float(np.nanmedian(v_ok[sl])) if np.isfinite(v_ok[sl]).any() else np.nan,max_inactive_s=longest_inactive(ac,ts[sl]-ts[a],dtm if m>1 else 0.),
                 adv_v_m=float(np.nansum(np.clip(v_ok[a+1:b+1],0,None)*np.minimum(dt[a+1:b+1],mg))/60/1000),counter=counter,adv_counter_m=float(abs(c[-1]-c[0])) if len(c)>=2 else np.nan)
        rec['stalled']=bool(rec['v_all_med']<p['stall_v']); rec['v_all_med']=round(rec['v_all_med'],2)
        for col,x in (('F',F),('T',T),('n',nn),('v',v)): rec[col+'_med']=float(np.median(x[sl][ac])) if na else np.nan
        status='ok' if cls=='long' else 'not_long'
        if cls=='long' and m<p['phase_min_rows']: status='few_rows'
        elif cls=='long' and rec['valid_share']<p['valid_min']: status='few_valid'
        phase[sl]=PHASE['unassigned']
        if status=='ok':
            r=detect(F[sl],T[sl],v[sl],p,dtm)
            if r is None: status='no_steady'
            else:
                f,l,lev,okshare=r; s=slice(a+f,a+l+1); rec.update(steady_start=t[a+f],steady_end=t[a+l],steady_ok_share=okshare,F_level=lev['F'],T_level=lev['T'],v_level=lev['v'])
                for col,x in (('F',F),('T',T),('n',nn),('v',v)): rec[col+'_steady']=float(np.nanmedian(x[s])) if np.isfinite(x[s]).any() else np.nan
                phase[s]=PHASE['steady']
                if le in LABELLED_EDGES: phase[a:a+f]=PHASE['ramp']; rec['ramp_s']=float(ts[a+f]-ts[a])
                tail=float(ts[b]-ts[a+l]); rec['tail_s']=tail
                if re in LABELLED_EDGES and tail<=p['end_max_s']: phase[a+l+1:b+1]=PHASE['end']; rec['end_s']=tail
                rec['steady_s']=float(ts[a+l]-ts[a+f])
        rec['phase_status']=status; recs.append(rec)
    seg=pd.DataFrame(recs).reindex(columns=SEG_COLUMNS)
    nn_=np.where(np.isnan(nn),3,np.where(nn>=p['n_rotating'],1,2)); op=np.where(inseg,OP_STATE['boring'],nn_).astype('int8')
    labels=pd.DataFrame({'time':d.time.to_numpy(),'source_id':d.source_id.to_numpy(),'row':d.row.to_numpy(),'seg_no':seg_no,'phase':phase,'op_state':op})
    return seg,labels,boring,ref,paused

def consistency(d,boring,ref,paused,machine,max_gap):
    """How the machine's boring flag agrees with the signals, by where the flag comes from; advance (integral of the valid advance speed) inside and outside the segments."""
    act=d.active.to_numpy(bool); F=d.F.to_numpy(float); thrust=F>0; rows=[]
    step=np.clip(np.where(d.valid_core.to_numpy(bool),d.v.to_numpy(float),np.nan),0,None)*np.minimum(d.dt_prev_s.to_numpy(float),max_gap)/60/1000
    for k,name in enumerate(REF_SOURCE):
        s=ref==k
        if not s.any(): continue
        b,a,f=boring[s],act[s],thrust[s]; adv=step[s]; pz=paused[s]
        rows.append(dict(machine=machine,ref_source=name,rows=int(s.sum()),boring_rows=int(b.sum()),paused_rows=int(pz.sum()),active_rows=int(a.sum()),boring_and_active=int((b&a).sum()),boring_not_active=int((b&~a).sum()),
                         active_not_boring=int((~b&a).sum()),thrust_on_not_boring=int((~b&f).sum()),active_share_of_boring=float((b&a).sum()/max(b.sum(),1)),boring_share_of_active=float((b&a).sum()/max(a.sum(),1)),
                         advance_m_inside=float(np.nansum(adv[b])),advance_m_outside=float(np.nansum(adv[~b])),advance_m_outside_with_thrust=float(np.nansum(adv[~b&f]))))
    return rows

def process(job):
    machine,src,dest,p=job; d=pd.read_parquet(Path(src)/f'{machine}.parquet',columns=COLUMNS); seg,labels,boring,ref,paused=build(d,p,machine)
    path=Path(dest)/'labels'/f'{machine}.parquet'; labels.to_parquet(path,engine='pyarrow',compression='zstd',index=False)
    return machine,seg,consistency(d,boring,ref,paused,machine,p['max_gap_s']),dict(file=f'labels/{machine}.parquet',rows=len(labels),bytes=path.stat().st_size,sha256=sha256(path),segments=len(seg),checksums=checksums(labels))

def summaries(seg):
    """Segment counts and hours by machine and class; durations and shares of the phases."""
    seg=seg.assign(hours=seg.duration_s/3600); rows=[]
    for (m,c),g in seg.groupby(['machine','seg_class'],sort=False):
        rows.append(dict(machine=m,seg_class=c,segments=len(g),hours=g.hours.sum(),median_duration_s=g.duration_s.median(),left_censored=int(g.left_censored.sum()),right_censored=int(g.right_censored.sum()),
                         left_hole_boring=int((g.left_edge=='hole_boring').sum()),right_hole_boring=int((g.right_edge=='hole_boring').sum()),left_hole_other=int((g.left_edge=='hole_other').sum()),right_hole_other=int((g.right_edge=='hole_other').sum()),
                         left_series_edge=int((g.left_edge=='series_edge').sum()),right_series_edge=int((g.right_edge=='series_edge').sum()),left_pause=int((g.left_edge=='pause').sum()),right_pause=int((g.right_edge=='pause').sum()),max_inactive_ge_60s=int((g.max_inactive_s>=60).sum())))
    ph=[]
    for m,g in seg[seg.seg_class=='long'].groupby('machine',sort=False):
        o=g[g.phase_status=='ok']; tot=o.duration_s.sum(); q=lambda x: x.quantile([.05,.5,.95]).tolist() if len(x.dropna()) else [np.nan]*3
        ph.append(dict(machine=m,long_segments=len(g),**{f'status_{s}':int((g.phase_status==s).sum()) for s in ('ok','no_steady','few_valid','few_rows')},
                       ramp_s_p05_p50_p95='/'.join(f'{x:.0f}' for x in q(o.ramp_s)),steady_s_p05_p50_p95='/'.join(f'{x:.0f}' for x in q(o.steady_s)),end_s_p05_p50_p95='/'.join(f'{x:.0f}' for x in q(o.end_s)),
                       ramp_share_of_time=float(o.ramp_s.sum()/tot) if tot else np.nan,steady_share_of_time=float(o.steady_s.sum()/tot) if tot else np.nan,end_share_of_time=float(o.end_s.sum()/tot) if tot else np.nan,
                       ramp_not_named=int(o.ramp_s.isna().sum()),end_not_named=int(o.end_s.isna().sum())))
    return pd.DataFrame(rows),pd.DataFrame(ph)

def segment_dictionary():
    d={'segment_id':('str','','Machine and segment number, e.g. TBM9-00123'),'machine':('str','',''),'seg_no':('int','','Number of the segment within the machine series; the value of seg_no in labels/<machine>.parquet'),
       'stream':('str','','Data stream of the first row'),'start':('time','','Time of the first row'),'end':('time','','Time of the last row'),'rows':('int','',''),'duration_s':('float','s','end minus start'),
       'dt_median_s':('float','s','Median interval between the readings inside the segment'),'ref_source':('str','','Where the boring flag of the rows comes from, the most common source in the segment: state_boring, state_thrust or signal'),
       'left_edge':('str','','What bounds the segment before it: state_change (the flag turned on, previous reading within max_gap_s), pause (a stretch of at least interrupt_s with the flag on and the thrust off was cut away), hole_boring (a reading gap above max_gap_s next to a boring row), hole_other (a gap next to a row that is not boring: the true start may lie in the gap), series_edge (first row of the machine series)'),
       'right_edge':('str','','The same after the segment: state_change, pause, hole_boring, hole_other or series_edge (last row of the series)'),'left_censored':('bool','','The left edge is neither a state_change nor a pause: the true start is not observed'),
       'right_censored':('bool','','The right edge is neither a state_change nor a pause: the true end is not observed'),'gap_before_s':('float','s','Length of the gap before the first row, for hole edges'),'gap_after_s':('float','s','Length of the gap after the last row, for hole edges'),
       'seg_class':('str','','blip (shorter than blip_s or without an active row), short (shorter than long_s), long'),'active_rows':('int','','Rows with F, T, n and v all above zero'),'active_share':('float','','active_rows / rows'),
       'valid_share':('float','','Share of the rows with valid_core'),'lead_inactive_s':('float','s','Time from the first row to the first active row'),'trail_inactive_s':('float','s','Time from the last active row to the last row'),
       'max_inactive_s':('float','s','Longest spell of rows that are not active between the first and the last active row'),'v_all_med':('float','mm/min','Median advance speed over all rows, zeros included, invalid rows left out'),
       'stalled':('bool','','v_all_med below stall_v: flag on and thrust on but almost no advance'),'F_med':('float','as exported','Median over the active rows'),'T_med':('float','as exported','Median over the active rows'),
       'n_med':('float','rpm','Median over the active rows'),'v_med':('float','mm/min','Median over the active rows'),'adv_v_m':('float','m','Integral of the valid advance speed over the segment (reading intervals capped at max_gap_s)'),
       'counter':('str','','Position counter used for adv_counter_m: chainage, cutter_mileage or ring'),'adv_counter_m':('float','m','Absolute difference between the last and the first valid counter reading in the segment (counter steps are 0.1 to 1 m)'),
       'phase_status':('str','','ok (steady phase found), not_long, few_rows (long but fewer than phase_min_rows readings), few_valid (valid_share below valid_min), no_steady (no positive steady level or no run of readings at the level)'),
       'steady_start':('time','','Time of the first steady reading'),'steady_end':('time','','Time of the last steady reading'),'ramp_s':('float','s','steady_start minus start; empty where the left edge hides the start (hole_boring, series_edge) or there is no steady phase'),
       'steady_s':('float','s','steady_end minus steady_start'),'end_s':('float','s','end minus steady_end; empty where the right edge is hidden or the stretch is longer than end_max_s'),'tail_s':('float','s','end minus steady_end whether or not it is named an end phase'),
       'steady_ok_share':('float','','Share of the steady readings in which the smoothed thrust and advance speed reach their share of the segment level'),'F_level':('float','as exported','Steady level of the thrust: median over the middle half of the segment'),
       'T_level':('float','as exported','Steady level of the torque'),'v_level':('float','mm/min','Steady level of the advance speed'),'F_steady':('float','as exported','Median thrust over the steady rows'),'T_steady':('float','as exported','Median torque over the steady rows'),
       'n_steady':('float','rpm','Median cutterhead speed over the steady rows'),'v_steady':('float','mm/min','Median advance speed over the steady rows')}
    return pd.DataFrame([(c,)+d[c] for c in SEG_COLUMNS],columns=['name','dtype','unit','description'])

def label_dictionary():
    rows=[('seg_no','int32','Segment number within the machine series (index into segments.csv, seg_no); -1 outside every segment'),('phase','int8','0 outside a segment, 1 ramp, 2 steady, 3 end, 4 inside a segment without a named phase (blip, short, no steady level found, too few valid rows, or an edge hidden by a data hole)'),
          ('op_state','int8','0 boring (inside a segment), 1 cutterhead turning without boring, 2 cutterhead not turning, 3 cutterhead speed missing')]
    return pd.DataFrame(rows,columns=['name','dtype','description'])

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--src',default=str(SRC)); ap.add_argument('--dest',default=str(DEST)); ap.add_argument('--machines',default=''); ap.add_argument('--workers',type=int,default=4); a=ap.parse_args()
    src,dest=Path(a.src),Path(a.dest); (dest/'labels').mkdir(parents=True,exist_ok=True); start=time.perf_counter()
    if a.machines: machines=a.machines.split(',')
    else:
        tm=pd.read_csv(P0/'trusted_machines.csv'); machines=sorted(tm[tm.verdict.str.startswith('usable')].machine.unique(),key=lambda m:int(m[3:]))
    with ProcessPoolExecutor(min(a.workers,len(machines))) as ex: results=list(ex.map(process,[(m,str(src),str(dest),PARAMS) for m in machines]))
    seg=pd.concat([r[1] for r in results],ignore_index=True); cons=pd.DataFrame([x for r in results for x in r[2]])
    out=seg.copy()
    for c in ('start','end','steady_start','steady_end'):
        if c in out: out[c]=out[c].dt.strftime('%Y-%m-%d %H:%M:%S')
    for c,k in (('duration_s',1),('dt_median_s',1),('gap_before_s',1),('gap_after_s',1),('active_share',4),('valid_share',4),('lead_inactive_s',1),('trail_inactive_s',1),('max_inactive_s',1),('adv_v_m',3),('adv_counter_m',3),('steady_ok_share',4),('ramp_s',1),('steady_s',1),('end_s',1),('tail_s',1)):
        if c in out: out[c]=out[c].round(k)
    for c in [c for c in out if c.endswith(('_med','_steady','_level'))]: out[c]=out[c].round(2)
    out.to_csv(dest/'segments.csv',index=False,encoding='utf-8-sig',lineterminator='\n')
    ss,ps=summaries(seg); ss.round(3).to_csv(dest/'segment_summary.csv',index=False,encoding='utf-8-sig'); ps.round(4).to_csv(dest/'phase_summary.csv',index=False,encoding='utf-8-sig')
    cons.round(5).to_csv(dest/'state_consistency.csv',index=False,encoding='utf-8-sig'); label_dictionary().to_csv(dest/'label_dictionary.csv',index=False,encoding='utf-8-sig'); segment_dictionary().to_csv(dest/'segment_dictionary.csv',index=False,encoding='utf-8-sig')
    (dest/'manifest.json').write_text(json.dumps(dict(parameters=PARAMS,phase_codes=PHASE,op_state_codes=OP_STATE,source=dict(standard_v1_manifest_sha256=sha256(src/'manifest.json')),
        segments=dict(file='segments.csv',rows=len(seg),sha256=sha256(dest/'segments.csv')),machines={r[0]:r[3] for r in results}),ensure_ascii=False,indent=1),encoding='utf-8')
    print(f'{len(seg)} segments from {len(machines)} machines in {time.perf_counter()-start:.0f} s'); print(ss.groupby('seg_class')[['segments','hours']].sum().round(1))

if __name__=='__main__': main(); finish()
