"""P2 verification (read-only): the segment table and the row labels against each other, against the standard time series v1 and against an independent
re-derivation (a separate implementation of the segmentation and of the steady-phase rule). Every check prints PASS or FAIL; the exit status is 1 if any fails.
Results go to outputs/segments_v1/verification.json."""
import argparse, json
from itertools import groupby
from pathlib import Path
import numpy as np
import pandas as pd
from p1_standardize import checksums, sha256
from p2_segment import DEST, SRC, PARAMS, PHASE, OP_STATE, LABELLED_EDGES, COLUMNS, THRUST_ADVANCING, finish

SPOT_SEGMENTS=300; SEED=20240101

def reference_flag(d):
    """Boring flag by a plain loop over the three sources, in the order boring flag, thrust state, signals."""
    out=np.zeros(len(d),bool); sb,st,ac=d.state_boring.tolist(),d.state_thrust.tolist(),d.active.tolist()
    for i in range(len(d)):
        out[i]=(sb[i]==1) if not pd.isna(sb[i]) else ((st[i]==THRUST_ADVANCING) if not pd.isna(st[i]) else bool(ac[i]))
    return out

def resegment(d,p):
    """Segments by run-length grouping: pauses are the runs of boring rows with F not above zero that last interrupt_s; segments are the runs of the remaining boring rows with no gap above max_gap_s."""
    n=len(d); flag=reference_flag(d); t=(d.time-d.time.iloc[0]).dt.total_seconds().to_numpy(); dt=d.dt_prev_s.to_numpy(float); F=d.F.to_numpy(float); step=float(np.nanmedian(dt)); keep=flag.copy()
    i=0
    for off,grp in groupby(zip(flag,F<=0,dt),key=lambda r:r[0] and r[1]):
        m=sum(1 for _ in grp); j=i+m
        if off:                                                    # a run of boring rows with the thrust off, split where a reading gap breaks it
            cuts=[i]+[k for k in range(i+1,j) if not dt[k]<=p['max_gap_s']]+[j]
            for a,b in zip(cuts[:-1],cuts[1:]):
                if t[b-1]-t[a]+step>=p['interrupt_s']: keep[a:b]=False
        i=j
    segs=[]; start=None
    for k in range(n):
        if keep[k]:
            if start is None: start=k
            elif not dt[k]<=p['max_gap_s']: segs.append((start,k-1)); start=k
        elif start is not None: segs.append((start,k-1)); start=None
    if start is not None: segs.append((start,n-1))
    return segs,flag

def steady_rows(F,T,v,dtm,p):
    """The steady rows of one segment by explicit windows instead of pandas rolling: (first,last) or None."""
    m=len(F); lo,hi=int(m*p['steady_from']),int(m*p['steady_to'])+1
    def level(x):
        w=x[lo:hi]; w=w[np.isfinite(w)]; return float(np.median(w)) if len(w) else np.nan
    lev=dict(F=level(F),T=level(T),v=level(v))
    if not all(np.isfinite(list(lev.values()))) or min(lev.values())<=0: return None
    def smooth(x,w):
        h=w//2; return np.array([np.nanmedian(x[max(0,i-h):i+h+1]) if np.isfinite(x[max(0,i-h):i+h+1]).any() else np.nan for i in range(m)])
    sm={'F':smooth(F,p['smooth_rows']),'T':smooth(T,p['smooth_rows']),'v':smooth(v,p['smooth_rows'])}; rho=dict(F=p['rho_F'],T=p['rho_T'],v=p['rho_v']); k=p['hold_rows']
    def reach(ref,i):
        return all(rho[c]<=0 or sm[c][i]>=rho[c]*(ref[c][i] if hasattr(ref[c],'__len__') else ref[c]) for c in 'FTv')
    def runs(flags): return [i for i in range(m-k+1) if all(flags[i:i+k])]
    r=runs([reach({c:lev[c] for c in 'FTv'},i) for i in range(m)])
    if not r: return None
    first=r[0]; end=[True]*m
    if p['trail_s']>0:
        w=max(2*k,int(round(p['trail_s']/dtm))) if np.isfinite(dtm) and dtm>0 else 2*k
        loc={c:np.array([np.nanmedian(sm[c][max(0,i-w):i]) if i>=3 and np.isfinite(sm[c][max(0,i-w):i]).any() else np.nan for i in range(m)]) for c in 'FTv'}   # the median of the w smoothed readings before row i
        end=[reach(loc,i) if i>=first+k else True for i in range(m)]
    r=runs(end); return first,r[-1]+k-1

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--src',default=str(SRC)); ap.add_argument('--dest',default=str(DEST)); a=ap.parse_args(); dest=Path(a.dest); src=Path(a.src); results=[]
    def check(name,cond,detail=''):
        print(('PASS ' if cond else 'FAIL ')+name+(f'  [{detail}]' if detail else ''),flush=True); results.append(dict(check=name,passed=bool(cond),detail=str(detail)))
    man=json.loads((dest/'manifest.json').read_text(encoding='utf-8')); seg=pd.read_csv(dest/'segments.csv'); p=man['parameters']; machines=list(man['machines'])
    check('the parameters of the manifest are those of the code',p==PARAMS)
    check('standard time series v1 is the one the labels were built from',sha256(src/'manifest.json')==man['source']['standard_v1_manifest_sha256'])
    check('segments.csv is the file of the manifest',sha256(dest/'segments.csv')==man['segments']['sha256'] and len(seg)==man['segments']['rows'])
    check('the segment dictionary names every column of segments.csv',list(pd.read_csv(dest/'segment_dictionary.csv').name)==list(seg.columns))
    check('segment ids are unique and numbered 0.. within each machine',seg.segment_id.is_unique and all((g.seg_no.to_numpy()==np.arange(len(g))).all() for _,g in seg.groupby('machine')))
    check('every segment has a class, an edge on each side and a status',seg[['seg_class','left_edge','right_edge','phase_status']].notna().all().all() and set(seg.seg_class)<={'blip','short','long'}
          and set(seg.left_edge)|set(seg.right_edge)<={'state_change','pause','hole_boring','hole_other','series_edge'})
    check('class follows from duration and activity',(seg.seg_class==np.where((seg.active_rows==0)|(seg.duration_s<p['blip_s']),'blip',np.where(seg.duration_s<p['long_s'],'short','long'))).all())
    check('censored means the edge is neither a state change nor a pause',((seg.left_censored==~seg.left_edge.isin(['state_change','pause']))&(seg.right_censored==~seg.right_edge.isin(['state_change','pause']))).all())
    check('phase status: ok only for long segments, and then with a steady start and end',((seg.phase_status=='ok')<=(seg.seg_class=='long')).all() and (seg.steady_start.notna()==(seg.phase_status=='ok')).all()
          and (seg[seg.phase_status!='ok'][['ramp_s','steady_s','end_s']].isna().all().all()))
    check('the inactive stretches at the start and end fit into the segment, and a segment without an active row is inactive throughout',((seg.lead_inactive_s+seg.trail_inactive_s)<=seg.duration_s+1e-6)[seg.active_rows>0].all() and (seg.lead_inactive_s[seg.active_rows==0]==seg.duration_s[seg.active_rows==0]).all())
    check('stalled follows from the median advance speed',(seg.stalled==(seg.v_all_med<p['stall_v'])).all())
    rng=np.random.default_rng(SEED)
    for m in machines:
        mm=man['machines'][m]; lab=pd.read_parquet(dest/mm['file']); std=pd.read_parquet(src/f'{m}.parquet',columns=COLUMNS); sg=seg[seg.machine==m].reset_index(drop=True)
        check(f'{m}: label file matches the manifest (rows and column checksums)',len(lab)==mm['rows'] and all(abs(c.get('sum',0)-mm['checksums'][k].get('sum',0))<=1e-9*max(1,abs(c.get('sum',0))) and c.get('nan',0)==mm['checksums'][k].get('nan',0)
              and c.get('nunique',0)==mm['checksums'][k].get('nunique',0) for k,c in checksums(lab).items()))
        check(f'{m}: labels are row for row the standard series (time, file, row)',len(lab)==len(std) and (lab.time.to_numpy()==std.time.to_numpy()).all() and (lab.source_id.to_numpy()==std.source_id.to_numpy()).all() and (lab.row.to_numpy()==std.row.to_numpy()).all())
        sn=lab.seg_no.to_numpy(); idx=np.flatnonzero(sn>=0); blocks=np.flatnonzero(np.diff(sn[idx])!=0)+1 if len(idx) else np.array([],int); first=idx[np.r_[0,blocks]] if len(idx) else idx; last=idx[np.r_[blocks-1,len(idx)-1]] if len(idx) else idx
        check(f'{m}: every segment is one block of consecutive rows, in order, as many as the table says',len(first)==len(sg) and (sn[first]==np.arange(len(sg))).all() and ((last-first+1)==sg.rows.to_numpy()).all() and (sn[idx]>=0).all())
        t=std.time.to_numpy(); tt=lambda s: np.array(pd.to_datetime(s).to_numpy(),dtype='datetime64[ns]')
        check(f'{m}: start and end times of the table are the times of the first and last row',(t[first]==tt(sg.start)).all() and (t[last]==tt(sg.end)).all())
        check(f'{m}: operating state 0 exactly inside segments',((lab.op_state==OP_STATE['boring']).to_numpy()==(sn>=0)).all() and lab.op_state.isin(list(OP_STATE.values())).all())
        ph=lab.phase.to_numpy(); check(f'{m}: phase 0 exactly outside segments, 1 to 4 inside',((ph==0)==(sn<0)).all() and ((ph>=0)&(ph<=4)).all())
        ok=sg[sg.phase_status=='ok']; bad=0; order_bad=0
        for r in ok.itertuples():
            rows=slice(first[r.seg_no],last[r.seg_no]+1); x=ph[rows]; tm=t[rows]; s0=np.datetime64(r.steady_start); s1=np.datetime64(r.steady_end)
            steady=(tm>=s0)&(tm<=s1); bad+=int(not ((x==PHASE['steady'])==steady).all())
            order_bad+=int(not (np.diff(x[x!=PHASE['unassigned']])>=0).all())
            ramp=x==PHASE['ramp']; end=x==PHASE['end']
            if r.left_edge in LABELLED_EDGES: bad+=int(not (ramp==(tm<s0)).all())
            else: bad+=int(ramp.any())
            if r.right_edge in LABELLED_EDGES and r.tail_s<=p['end_max_s']: bad+=int(not (end==(tm>s1)).all())
            else: bad+=int(end.any())
        check(f'{m}: steady, ramp and end rows follow the steady start and end of the table; phases never run backwards',bad==0 and order_bad==0,f'{len(ok)} phased segments')
        other=sg[sg.phase_status!='ok']; check(f'{m}: segments without phases carry only the code {PHASE["unassigned"]}',all((ph[first[r.seg_no]:last[r.seg_no]+1]==PHASE['unassigned']).all() for r in other.itertuples()))
        # independent segmentation
        segs,flag=resegment(std,p); mine=list(zip(first.tolist(),last.tolist()))
        check(f'{m}: an independent run-length re-derivation finds the same segments',segs==mine,f'{len(segs)} vs {len(mine)}')
        dt=std.dt_prev_s.to_numpy(float); gaps=sum(int(not dt[k]<=p['max_gap_s']) for a,b in mine for k in range(a+1,b+1)); check(f'{m}: no reading gap above {p["max_gap_s"]:g} s inside a segment',gaps==0)
        F=std.F.to_numpy(float); ts=(std.time-std.time.iloc[0]).dt.total_seconds().to_numpy(); step=float(np.nanmedian(dt)); off=sum(1 for a,b in mine if _longest_off(F[a:b+1],ts[a:b+1],step)>=p['interrupt_s'])
        check(f'{m}: no segment holds a stretch of {p["interrupt_s"]:g} s or more with the thrust off',off==0)
        if (sg.phase_status=='ok').any():
            pick=rng.choice(len(ok),size=min(SPOT_SEGMENTS,len(ok)),replace=False); diff=0; Fa,Ta,va=(std[c].to_numpy(float) for c in ('F','T','v'))
            for r in ok.iloc[np.sort(pick)].itertuples():
                a,b=first[r.seg_no],last[r.seg_no]; got=steady_rows(Fa[a:b+1],Ta[a:b+1],va[a:b+1],r.dt_median_s,p)
                diff+=int(got is None or t[a+got[0]]!=np.datetime64(r.steady_start) or t[a+got[1]]!=np.datetime64(r.steady_end))
            check(f'{m}: an independent windowed re-derivation of the steady rows agrees on {len(pick)} sampled segments',diff==0,f'{diff} differ')
    cons=pd.read_csv(dest/'state_consistency.csv'); tot=int(sum((pd.read_parquet(dest/man['machines'][m]['file'],columns=['op_state']).op_state==0).sum() for m in machines))
    check('state consistency: boring rows minus paused rows equal the rows inside segments',int((cons.boring_rows-cons.paused_rows).sum())==tot,f'{tot} rows')
    n_ok=sum(r['passed'] for r in results); (dest/'verification.json').write_text(json.dumps(dict(passed=n_ok,total=len(results),all_passed=n_ok==len(results),checks=results),ensure_ascii=False,indent=1),encoding='utf-8')
    print(f'{n_ok}/{len(results)} checks passed'); finish(0 if n_ok==len(results) else 1)

def _longest_off(F,ts,step):
    """Longest stretch of consecutive rows with the thrust off in one segment, in seconds, by the definition of the segmentation: first to last row plus the typical interval."""
    best=0.; i=0; n=len(F)
    while i<n:
        if F[i]<=0:
            j=i
            while j+1<n and F[j+1]<=0: j+=1
            best=max(best,ts[j]-ts[i]+step); i=j+1
        else: i+=1
    return best

if __name__=='__main__': main()
