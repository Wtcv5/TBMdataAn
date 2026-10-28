"""P2 checks: how much the segments and phases depend on their parameters, and whether the segments agree with the position counters.
Re-runs the segmentation of p2_segment.py with one parameter changed at a time (nothing is tuned: the table shows what the choice costs) and
compares the advance each long segment covers (integral of the advance speed) with the position counter. Writes sensitivity_segmentation.csv,
sensitivity_phase.csv and advance_check.csv into outputs/segments_v1."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from p2_segment import PARAMS, COLUMNS, DEST, SRC, build, finish, P0

GAPS=[30.,45.,60.,90.,120.,300.]
INTERRUPTS=[60.,120.,300.,1e9]   # 1e9: no split at pauses
VARIANTS={'strict_levels':dict(rho_F=.95,rho_v=.9),'loose_levels':dict(rho_F=.85,rho_v=.7),'thrust_only':dict(rho_v=0.),'with_torque':dict(rho_T=.8),'with_torque_strict':dict(rho_F=.95,rho_T=.9,rho_v=.9),'end_vs_segment_level':dict(trail_s=0.),'no_end_cap':dict(end_max_s=1e9),'trail_120s':dict(trail_s=120.),'smooth_3':dict(smooth_rows=3),'smooth_7':dict(smooth_rows=7),
          'hold_2':dict(hold_rows=2),'hold_5':dict(hold_rows=5),'level_window_narrow':dict(steady_from=.4,steady_to=.6),'level_window_wide':dict(steady_from=.1,steady_to=.9)}
COUNTER_STEPS=3.           # segments are compared with the counter only where it moved at least this many of its own steps
MIN_COMPARED=30            # and a machine is summarised only where at least this many segments qualify

def one_machine(job):
    machine,src=job; d=pd.read_parquet(Path(src)/f'{machine}.parquet',columns=COLUMNS); base,_,boring,_,_=build(d,PARAMS,machine); step=counter_step(d,boring); gap_rows=[]; phase_rows=[]
    for par,vals in (('max_gap_s',GAPS),('interrupt_s',INTERRUPTS)):
        for g in vals:
            s=base if g==PARAMS[par] else build(d,{**PARAMS,par:g},machine)[0]; lg=s[s.seg_class=='long']
            gap_rows.append(dict(machine=machine,parameter=par,value=g,segments=len(s),long_segments=len(lg),long_hours=lg.duration_s.sum()/3600,short_and_blip=int((s.seg_class!='long').sum()),
                                 hole_boring_edges=int((s.left_edge=='hole_boring').sum()+(s.right_edge=='hole_boring').sum()),pause_edges=int((s.left_edge=='pause').sum()+(s.right_edge=='pause').sum()),long_phase_ok=int((lg.phase_status=='ok').sum())))
    b=base[base.phase_status=='ok'].set_index('seg_no')
    for name,change in {'default':{},**VARIANTS}.items():
        s=build(d,{**PARAMS,**change},machine)[0] if change else base; lg=s[s.seg_class=='long']; o=s[s.phase_status=='ok'].set_index('seg_no'); both=o.index.intersection(b.index)
        sh0=(o.loc[both,'steady_start']-b.loc[both,'steady_start']).dt.total_seconds().abs(); sh1=(o.loc[both,'steady_end']-b.loc[both,'steady_end']).dt.total_seconds().abs()
        tot=o.duration_s.sum()
        phase_rows.append(dict(machine=machine,variant=name,long_segments=len(lg),steady_found=len(o),steady_found_share=len(o)/max(len(lg),1),compared=len(both),
                               start_shift_median_s=sh0.median(),start_shift_p90_s=sh0.quantile(.9),start_shift_over_60s=float((sh0>60).mean()) if len(both) else np.nan,
                               end_shift_median_s=sh1.median(),end_shift_p90_s=sh1.quantile(.9),end_shift_over_60s=float((sh1>60).mean()) if len(both) else np.nan,
                               ramp_share_of_time=float(o.ramp_s.sum()/tot) if tot else np.nan,steady_share_of_time=float(o.steady_s.sum()/tot) if tot else np.nan,end_share_of_time=float(o.end_s.sum()/tot) if tot else np.nan))
    return pd.DataFrame(gap_rows),pd.DataFrame(phase_rows),(machine,step)

def counter_step(d,boring):
    """Typical size of one step of the position counter while boring (median of the non-zero small changes); NaN if the machine has none."""
    c=next((x for x in ('chainage','cutter_mileage','ring') if d[x].notna().mean()>=.5),None)
    if c is None: return np.nan
    x=d[c].to_numpy(float); dx=np.abs(np.diff(x)); ok=boring[1:]&boring[:-1]&~d.q_counter_glitch.to_numpy(bool)[1:]&(dx>0)&(dx<=2)
    return float(np.median(dx[ok])) if ok.any() else np.nan

def advance_check(seg,steps):
    """Per machine: for long segments whose rows are all valid and whose counter moved at least COUNTER_STEPS of its own steps, the advance from the speed integral against the counter."""
    rows=[]
    for m,g in seg.groupby('machine',sort=False):
        l=g[(g.seg_class=='long')&(g.valid_share>=.99)]; st=steps.get(m,np.nan); c=l[l.adv_counter_m>=COUNTER_STEPS*st] if np.isfinite(st) else l.iloc[:0]
        r=c.adv_v_m/c.adv_counter_m; q=r.quantile([.1,.25,.5,.75,.9]) if len(c)>=MIN_COMPARED else pd.Series([np.nan]*5)
        rows.append(dict(machine=m,counter=l.counter.iloc[0] if len(l) else '',counter_step_m=st,long_segments_valid=len(l),counter_moved_3_steps=len(c),ratio_p10=q.iloc[0],ratio_p25=q.iloc[1],ratio_p50=q.iloc[2],ratio_p75=q.iloc[3],ratio_p90=q.iloc[4],
                         within_20_percent=float(((r>=.8)&(r<=1.25)).mean()) if len(c)>=MIN_COMPARED else np.nan,sum_speed_integral_m=c.adv_v_m.sum(),sum_counter_m=c.adv_counter_m.sum(),
                         sum_speed_integral_all_long_m=l.adv_v_m.sum(),sum_counter_all_long_m=l.adv_counter_m.sum()))
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--src',default=str(SRC)); ap.add_argument('--dest',default=str(DEST)); ap.add_argument('--machines',default=''); ap.add_argument('--workers',type=int,default=4); a=ap.parse_args()
    dest=Path(a.dest)
    if a.machines: machines=a.machines.split(',')
    else: tm=pd.read_csv(P0/'trusted_machines.csv'); machines=sorted(tm[tm.verdict.str.startswith('usable')].machine.unique(),key=lambda m:int(m[3:]))
    with ProcessPoolExecutor(min(a.workers,len(machines))) as ex: res=list(ex.map(one_machine,[(m,a.src) for m in machines]))
    gap=pd.concat([r[0] for r in res],ignore_index=True); ph=pd.concat([r[1] for r in res],ignore_index=True); adv=advance_check(pd.read_csv(dest/'segments.csv'),dict(r[2] for r in res))
    gap.round(3).to_csv(dest/'sensitivity_segmentation.csv',index=False,encoding='utf-8-sig'); ph.round(4).to_csv(dest/'sensitivity_phase.csv',index=False,encoding='utf-8-sig'); adv.round(4).to_csv(dest/'advance_check.csv',index=False,encoding='utf-8-sig')
    pd.set_option('display.width',250); pd.set_option('display.max_columns',30); print(gap.round(2).to_string(index=False)); print(ph.round(3).to_string(index=False)); print(adv.round(3).to_string(index=False))

if __name__=='__main__': main(); finish()
