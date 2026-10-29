"""P3 window rules, shared by p3_build.py, p3_verify.py and the baseline re-runs of P5. A window is an origin row, the three rows before it and a target row
at the horizon, all readings of one machine on its native timestamps (nothing resampled or interpolated). The rules are those of the committed baseline
(build_baselines.py), written over the rows of the standard series: the rows of a window must be good (the caller says which), no two neighbouring rows may be
more than MAX_GAP_S apart, the origin must be active (F, T, n and v above zero), and the target is the first row at or after origin + horizon, at most
TOLERANCE_S late. The two views then ask more of the same window: the rows of a boring-view window all lie in one boring-view segment, those of a steady-view window
in one steady stretch of it. Levels are nested: 0 no window, 1 full-run, 2 boring view, 3 steady view."""
import numpy as np
import pandas as pd

HORIZONS=(30,60,120); TOLERANCE_S=5; HISTORY_ROWS=4; MAX_GAP_S=60
CORE=('F','T','n','v'); TARGETS=('F','T','v')
LEVEL=dict(none=0,full=1,boring=2,steady=3)
PHASE_STEADY=2

def seconds(time):
    """Timestamps (datetime64[ns]) as whole seconds."""
    return np.asarray(time).astype('datetime64[ns]').astype('int64')//10**9

def run_ids(ts,good,max_gap=MAX_GAP_S):
    """Number of the run of consecutive good rows each row belongs to (-1 for rows that are not good). A run ends at a row that is not good or where the time to the next row exceeds max_gap."""
    good=np.asarray(good,bool); joined=good[1:]&good[:-1]&(np.diff(ts)<=max_gap); start=good&~np.r_[False,joined]
    return np.where(good,np.cumsum(start)-1,-1).astype('int32')

def target_rows(ts,rid,active,horizon,history=HISTORY_ROWS,tolerance=TOLERANCE_S):
    """For every row the row number of its target at `horizon` seconds, or -1 where the row is not the origin of a window: the origin must be active and good,
    the history-1 rows before it must be in the same run, and the target (first row at or after origin + horizon, at most `tolerance` seconds late) must be in that run too."""
    n=len(ts); tgt=np.full(n,-1,np.int64); cand=np.flatnonzero(np.asarray(active,bool)&(rid>=0)&(np.arange(n)>=history-1))
    cand=cand[rid[cand-(history-1)]==rid[cand]]; j=np.searchsorted(ts,ts[cand]+horizon,side='left'); inside=j<n; cand,j=cand[inside],j[inside]
    ok=(ts[j]-ts[cand]-horizon<=tolerance)&(rid[j]==rid[cand]); tgt[cand[ok]]=j[ok]; return tgt

def window_levels(tgt,seg_no,boring,phase,history=HISTORY_ROWS):
    """Nested level of every window (0 where there is none). Rows of a segment are consecutive, so the first row of the window and the target decide: the same boring-view
    segment is level 2, and both in the steady stretch of it level 3."""
    lev=np.zeros(len(tgt),np.int8); w=np.flatnonzero(tgt>=0); s=w-(history-1); j=tgt[w]; lev[w]=LEVEL['full']
    same=boring[s]&boring[j]&(seg_no[s]==seg_no[j]); lev[w[same]]=LEVEL['boring']
    steady=same&(phase[s]==PHASE_STEADY)&(phase[j]==PHASE_STEADY); lev[w[steady]]=LEVEL['steady']; return lev

def feature_names(history=HISTORY_ROWS):
    names=[f'{c}_lag{k}' for k in range(history) for c in CORE]
    for k in range(1,history): names+=[f'age_lag{k}_s']+[f'{c}_change{k}' for c in CORE]
    return names

def samples(frame,tgt,level,minimum,history=HISTORY_ROWS,context=True):
    """The windows of level >= `minimum` in the layout of the committed baseline's datasets (features of the origin and the history rows, origin, target time, actual horizon, targets);
    with `context` also where each window sits (row positions, segment numbers and phases of the origin and target rows, operating state of the target, level). frame holds the
    time and the core signals, and for the context the P2 labels, in row order."""
    o=np.flatnonzero((level>=minimum)&(tgt>=0)); j=tgt[o]; ts=seconds(frame['time']); a={c:frame[c].to_numpy(float) for c in CORE}; out={}
    for k in range(history):
        for c in CORE: out[f'{c}_lag{k}']=a[c][o-k]
    for k in range(1,history):
        out[f'age_lag{k}_s']=ts[o]-ts[o-k]
        for c in CORE: out[f'{c}_change{k}']=a[c][o]-a[c][o-k]
    t=frame['time'].to_numpy(); out.update(origin=t[o],target_time=t[j],actual_horizon_s=ts[j]-ts[o])
    for c in TARGETS: out['y_'+c]=a[c][j]
    if context:
        lab={c:frame[c].to_numpy() for c in ('seg_no','phase','op_state')}
        out.update(row_origin=o,row_target=j,seg_no_origin=lab['seg_no'][o],seg_no_target=lab['seg_no'][j],phase_origin=lab['phase'][o],phase_target=lab['phase'][j],op_state_target=lab['op_state'][j],level=level[o])
    return pd.DataFrame(out)
