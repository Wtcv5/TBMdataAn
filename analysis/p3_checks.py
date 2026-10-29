"""P3 sensitivity of the window and view choices (read-only except outputs/analysis_v1/sensitivity_windows.csv): how many windows each machine and horizon would have if one choice were made
otherwise. Variants: the reading-gap limit (60 s as in the baseline, 120 s), the target rule (the first reading at or after origin + horizon, at most 5 s late, as in the baseline; or the reading
nearest to origin + horizon within 5 s either side), the origin (active as in the baseline, or any valid row), and the definition of the boring view (long segments with a steady phase that advance;
long segments that advance, with or without a steady phase; every long segment; every segment). The full-run count depends on the first three, the boring count on all four."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from p3_build import DEST, row_lookup
from p3_windows import HORIZONS, TOLERANCE_S, HISTORY_ROWS, MAX_GAP_S, LEVEL, run_ids, target_rows, window_levels, seconds
from p2_segment import DEST as SEG, finish

SEGMENTS=SEG/'segments.csv'

def nearest_rows(ts,rid,origin,horizon,history=HISTORY_ROWS,tolerance=TOLERANCE_S):
    """Like target_rows, but the target is the reading nearest to origin + horizon (a tie goes to the later one), at most `tolerance` seconds away on either side."""
    n=len(ts); tgt=np.full(n,-1,np.int64); cand=np.flatnonzero(np.asarray(origin,bool)&(rid>=0)&(np.arange(n)>=history-1)); cand=cand[rid[cand-(history-1)]==rid[cand]]
    want=ts[cand]+horizon; later=np.minimum(np.searchsorted(ts,want,side='left'),n-1); earlier=np.maximum(later-1,0)
    use_later=np.abs(ts[later]-want)<=np.abs(want-ts[earlier]); j=np.where(use_later,later,earlier); ok=(np.abs(ts[j]-want)<=tolerance)&(j>cand)&(rid[j]==rid[cand]); tgt[cand[ok]]=j[ok]; return tgt

def variants(seg):
    """Definitions of the boring view as per-segment flags (seg_no is the index)."""
    s=seg.set_index('seg_no').sort_index(); long_=s.seg_class.eq('long').to_numpy(); ok=s.phase_status.eq('ok').to_numpy(); advancing=~s.stalled.to_numpy(bool)
    return {'boring_default':long_&ok&advancing,'boring_with_stalled':long_&ok,'boring_long_advancing':long_&advancing,'boring_long':long_,'boring_any_segment':np.ones(len(s),bool)}

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--dest',default=str(DEST)); ap.add_argument('--seg',default=str(SEGMENTS)); a=ap.parse_args(); dest=Path(a.dest); seg=pd.read_csv(a.seg); rows=[]
    for p in sorted(dest.glob('TBM*.parquet'),key=lambda p:int(p.stem[3:])):
        m=p.stem; d=pd.read_parquet(p,columns=['time','active','valid_core','seg_no','phase']); ts=seconds(d.time); valid=d.valid_core.to_numpy(bool); active=d.active.to_numpy(bool)
        seg_no=d.seg_no.to_numpy(); phase=d.phase.to_numpy(); flags=variants(seg[seg.machine==m]); inseg={k:row_lookup(v,seg_no) for k,v in flags.items()}
        settings=[('default',dict(gap=MAX_GAP_S,nearest=False,origin=active)),('gap_120_s',dict(gap=120,nearest=False,origin=active)),('target_nearest_reading',dict(gap=MAX_GAP_S,nearest=True,origin=active)),
                  ('origin_any_valid_row',dict(gap=MAX_GAP_S,nearest=False,origin=valid))]
        for h in HORIZONS:
            for name,s in settings:
                rid=run_ids(ts,valid,s['gap']); tgt=(nearest_rows if s['nearest'] else target_rows)(ts,rid,s['origin'],h); lev=window_levels(tgt,seg_no,inseg['boring_default'],phase)
                rows.append(dict(machine=m,horizon_s=h,variant=name,full_run_windows=int((tgt>=0).sum()),boring_windows=int((lev>=LEVEL['boring']).sum()),steady_windows=int((lev>=LEVEL['steady']).sum())))
            rid=run_ids(ts,valid,MAX_GAP_S); tgt=target_rows(ts,rid,active,h)
            for name in list(flags)[1:]:
                lev=window_levels(tgt,seg_no,inseg[name],phase); rows.append(dict(machine=m,horizon_s=h,variant=name,full_run_windows=int((tgt>=0).sum()),boring_windows=int((lev>=LEVEL['boring']).sum()),steady_windows=int((lev>=LEVEL['steady']).sum())))
    out=pd.DataFrame(rows); out.to_csv(dest/'sensitivity_windows.csv',index=False,encoding='utf-8-sig'); pd.set_option('display.width',200)
    t=out[out.horizon_s==60].pivot(index='variant',columns='machine',values='boring_windows'); print('boring windows at 60 s'); print(t.to_string()); t=out[out.horizon_s==60].pivot(index='variant',columns='machine',values='full_run_windows'); print('full-run windows at 60 s'); print(t.to_string())
    finish()

if __name__=='__main__': main()
