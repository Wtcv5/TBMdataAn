"""P3: the canonical analysis dataset v1 with its two views.
The dataset is the standard time series v1 of every machine that P2 labelled (P1 rows and order, nothing dropped, smoothed, resampled or interpolated) with the segment
number, phase and operating state of P2 and the views written next to every row. Full-run view: every row with valid_core; a window needs good rows, no reading gap above
60 s, an active origin and a target at the horizon (p3_windows.py has the rules and the baseline's layout). Continuous-boring view: the rows of long segments that have a
steady phase and advance (phase_status ok, not stalled); a window of this view lies inside one such segment. Strict variant: both edges of the segment are observed. Steady
variant: the window lies inside the steady phase. Levels of a window are nested: win<h>_level is 0 (no window), 1 full-run, 2 boring view, 3 steady view.
Outputs in outputs/analysis_v1: <machine>.parquet (git-ignored, rebuilt by this script), manifest.json, field_dictionary.csv, view_summary.csv, window_summary.csv, window_accounting.csv."""
import argparse, json, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from p1_standardize import OUT, sha256, checksums
from p2_segment import SRC, DEST as SEG, PHASE, OP_STATE, finish
from p3_windows import HORIZONS, TOLERANCE_S, HISTORY_ROWS, MAX_GAP_S, LEVEL, run_ids, target_rows, window_levels, seconds

DEST=OUT/'analysis_v1'
KEEP=['time','machine','stream','project','file_status','source_id','row','F','T','n','v','p','n_source','state_boring','state_cutter','state_thrust','state_regrip','dt_prev_s','active','valid_core',
      'q_time_backward','q_time_coarse','q_time_conflict','q_neg','q_core_missing','q_n_invalid','q_v_high','q_T_high','q_F_high','q_pnv_off','q_v_invalid_file']
LABELS=['seg_no','phase','op_state']
WINDOW_COLUMNS=[c for h in HORIZONS for c in (f'win{h}_pos',f'win{h}_level')]
COLUMNS=KEEP+LABELS+['view_boring','view_strict','view_steady','run_id']+WINDOW_COLUMNS
DROPPED=['n_raw','chainage','ring','cutter_mileage','drive_speed_med','drive_torque_med','drive_motors_on','q_counter_glitch']
VIEWS=('full','boring','strict','steady','steady_strict')
REASONS=('wide','short','phase_not_ok','stalled')      # why a segment is outside or inside the boring view, in the order of precedence

def segment_views(seg):
    """Per segment (index seg_no) whether it is in the boring view, in its strict variant, and why not (REASONS)."""
    n=int(seg.seg_no.max())+1; assert len(seg)==n and (np.sort(seg.seg_no.to_numpy())==np.arange(n)).all()
    s=seg.set_index('seg_no').sort_index(); short=s.seg_class.ne('long').to_numpy(); notok=s.phase_status.ne('ok').to_numpy(); stalled=s.stalled.to_numpy(bool)
    reason=np.where(short,1,np.where(notok,2,np.where(stalled,3,0))).astype('int8'); boring=reason==0; strict=boring&~s.left_censored.to_numpy(bool)&~s.right_censored.to_numpy(bool)
    return boring,strict,reason

def row_lookup(table,seg_no):
    """table[seg_no] with False (0 for integers) outside every segment (seg_no -1)."""
    out=np.zeros(len(seg_no),table.dtype); inside=seg_no>=0; out[inside]=table[seg_no[inside]]; return out

def interior_hours(ts,member):
    """Hours between neighbouring readings that both belong to `member` and are at most MAX_GAP_S apart: the time the member rows cover."""
    d=np.diff(ts); both=member[1:]&member[:-1]&(d<=MAX_GAP_S); return float(d[both].sum()/3600)

def window_accounting(ts,rid,active,valid,tgt,lev,h,seg_no,reason,machine):
    """Where the active rows go: a partition of the active rows by the first rule that ends their window, then of the full-run windows by what keeps them out of the boring view and the steady view."""
    n=len(ts); idx=np.arange(n); act=np.asarray(active,bool); rows=[]
    def add(stage,k): rows.append(dict(machine=machine,horizon_s=h,stage=stage,windows=int(k)))
    history=(idx>=HISTORY_ROWS-1)&(rid>=0); history[HISTORY_ROWS-1:]&=rid[:n-HISTORY_ROWS+1]==rid[HISTORY_ROWS-1:]
    cand=np.flatnonzero(act&valid&history); j=np.searchsorted(ts,ts[cand]+h,side='left'); inside=j<n; found=np.zeros(len(cand),bool); same=np.zeros(len(cand),bool)
    found[inside]=ts[j[inside]]-ts[cand[inside]]-h<=TOLERANCE_S; same[found]=rid[j[found]]==rid[cand[found]]
    add('active_rows',act.sum()); add('origin_not_valid',(act&~valid).sum()); add('history_incomplete',(act&valid&~history).sum()); add('target_not_found',(~found).sum())
    add('target_in_other_run',(found&~same).sum()); w=np.flatnonzero(tgt>=0); add('full_run_windows',len(w)); assert len(w)==int(same.sum())
    s=w-(HISTORY_ROWS-1); j=tgt[w]; a,b=seg_no[s],seg_no[j]
    outside=(a<0)|(b<0); crossing=~outside&(a!=b); one=~outside&(a==b); why=np.where(one,reason[np.maximum(a,0)],-1)
    add('not_boring:window_touches_rows_outside_segments',outside.sum()); add('not_boring:window_crosses_segments',crossing.sum())
    for r,name in enumerate(REASONS[1:],1): add(f'not_boring:segment_{name}',(why==r).sum())
    add('boring_windows',(lev>=LEVEL['boring']).sum()); add('boring_not_steady',(lev==LEVEL['boring']).sum()); add('steady_windows',(lev>=LEVEL['steady']).sum())
    assert sum(r['windows'] for r in rows if r['stage'] in ('origin_not_valid','history_incomplete','target_not_found','target_in_other_run','full_run_windows'))==int(act.sum())
    assert sum(r['windows'] for r in rows if r['stage'].startswith('not_boring'))+int((lev>=LEVEL['boring']).sum())==len(w)
    return rows

def window_summary(d,tgt,lev,h,strict,machine,repaired,pnv):
    """Descriptions of the windows of every view: how many, what the target row is, where the origin lies in the stroke, how much the target moves away from the origin."""
    w=np.flatnonzero(lev>=LEVEL['full']); s=w-(HISTORY_ROWS-1); j=tgt[w]; L=lev[w]; days=d.time.dt.floor('D').to_numpy()[w]; act=d.active.to_numpy(bool); op=d.op_state.to_numpy(); ph=d.phase.to_numpy()
    a={c:d[c].to_numpy(float) for c in ('F','T','v')}; o=w; srep=np.cumsum(np.r_[0,repaired]); spnv=np.cumsum(np.r_[0,pnv])
    both_strict=strict[s]&strict[j]
    masks={'full':L>=LEVEL['full'],'boring':L>=LEVEL['boring'],'strict':(L>=LEVEL['boring'])&both_strict,'steady':L>=LEVEL['steady'],'steady_strict':(L>=LEVEL['steady'])&both_strict}
    rows=[]
    for view,m in masks.items():
        k=int(m.sum()); r=dict(machine=machine,horizon_s=h,view=view,windows=k,share_of_full=k/max(len(w),1))
        if k:
            r.update(origin_days=int(len(np.unique(days[m]))),target_active_share=float(act[j[m]].mean()),
                     **{f'target_{name}_share':float((op[j[m]]==code).mean()) for name,code in OP_STATE.items()},
                     **{f'origin_{name}_share':float((ph[o[m]]==code).mean()) for name,code in PHASE.items()},
                     with_repaired_n_share=float((srep[j[m]+1]-srep[s[m]]>0).mean()),with_pnv_off_share=float((spnv[j[m]+1]-spnv[s[m]]>0).mean()))
            for c in ('F','T','v'):
                y=a[c][j[m]]; x=a[c][o[m]]; r[f'persistence_rel_mae_{c}']=float(np.mean(np.abs(y-x))/max(np.mean(np.abs(y)),1e-12))
        rows.append(r)
    return rows

def process(job):
    machine,src,seg_dir,dest=job; src,seg_dir,dest=Path(src),Path(seg_dir),Path(dest)
    d=pd.read_parquet(src/f'{machine}.parquet',columns=KEEP); lab=pd.read_parquet(seg_dir/'labels'/f'{machine}.parquet')
    for c in ('time','source_id','row'):
        if not (d[c].to_numpy()==lab[c].to_numpy()).all(): raise SystemExit(f'{machine}: the P2 labels do not line up with the standard series in {c}')
    seg=pd.read_csv(seg_dir/'segments.csv'); seg=seg[seg.machine==machine]; boring_seg,strict_seg,reason_seg=segment_views(seg)
    seg_no=lab.seg_no.to_numpy(); phase=lab.phase.to_numpy(); d['seg_no']=seg_no; d['phase']=phase; d['op_state']=lab.op_state.to_numpy()
    boring=row_lookup(boring_seg,seg_no); strict=row_lookup(strict_seg,seg_no); d['view_boring']=boring; d['view_strict']=strict; d['view_steady']=boring&(phase==PHASE['steady'])
    ts=seconds(d.time); valid=d.valid_core.to_numpy(bool); active=d.active.to_numpy(bool); rid=run_ids(ts,valid); d['run_id']=rid
    acct=[]; wins=[]; counts={}
    repaired=d.n_source.astype(str).str.startswith('repaired').to_numpy(bool); pnv=d.q_pnv_off.to_numpy(bool)
    tg={}
    for h in HORIZONS:
        tgt=target_rows(ts,rid,active,h); lev=window_levels(tgt,seg_no,boring,phase); tg[h]=(tgt,lev); d[f'win{h}_pos']=tgt.astype('int32'); d[f'win{h}_level']=lev
    d=d[COLUMNS]
    for h in HORIZONS:
        tgt,lev=tg[h]; acct+=window_accounting(ts,rid,active,valid,tgt,lev,h,seg_no,reason_seg,machine); wins+=window_summary(d,tgt,lev,h,strict,machine,repaired,pnv)
        counts[str(h)]=dict(full_run_view=int((lev>=LEVEL['full']).sum()),boring_view=int((lev>=LEVEL['boring']).sum()),steady_view=int((lev>=LEVEL['steady']).sum()))
    views=[]
    for view,member in (('full',valid),('boring',boring),('strict',strict),('steady',d.view_steady.to_numpy(bool)),('steady_strict',d.view_steady.to_numpy(bool)&strict)):
        r=dict(machine=machine,view=view,rows=int(member.sum()),share_of_rows=float(member.mean()),hours=interior_hours(ts,member),active_rows=int((member&active).sum()))
        if view in ('boring','strict'): r['segments']=int((boring_seg if view=='boring' else strict_seg).sum())
        views.append(r)
    path=dest/f'{machine}.parquet'; d.to_parquet(path,engine='pyarrow',compression='zstd',index=False)
    return machine,dict(file=f'{machine}.parquet',rows=len(d),bytes=path.stat().st_size,sha256=sha256(path),windows=counts,checksums=checksums(d)),views,wins,acct

def field_dictionary(std_dict):
    """Dictionary of the dataset: the columns taken from the standard series keep their description, the new ones are described here."""
    new={'seg_no':('int32','','Number of the P2 segment the row lies in (index into segments_v1/segments.csv, same machine); -1 outside every segment','P2 labels'),
         'phase':('int8','','P2 phase: 0 outside a segment, 1 ramp, 2 steady, 3 end, 4 inside a segment without a named phase','P2 labels'),
         'op_state':('int8','','P2 operating state: 0 boring (inside a segment), 1 cutterhead turning without boring, 2 cutterhead not turning, 3 cutterhead speed missing','P2 labels'),
         'view_boring':('bool','','Row lies in a segment of the continuous-boring view: long, phase_status ok (a steady phase was found) and not stalled','P3'),
         'view_strict':('bool','','Row lies in a segment of the continuous-boring view whose two edges are both observed (neither is censored by a data hole or the end of the series)','P3'),
         'view_steady':('bool','','Row is in the steady phase (phase 2) of a segment of the continuous-boring view','P3'),
         'run_id':('int32','','Number of the run of consecutive valid_core rows, no two neighbours more than 60 s apart, within the machine series; -1 where valid_core is false','P3')}
    for h in HORIZONS:
        new[f'win{h}_pos']=('int32','',f'Row position (0-based, in this file) of the target of the {h} s window that starts at this row (origin), -1 where no window starts here: the origin is active, the origin and the 3 rows before it and the target row are valid_core rows of one run, and the target is the first reading at or after origin+{h} s, at most 5 s late','P3')
        new[f'win{h}_level']=('int8','','Level of that window: 0 none, 1 full-run view, 2 also continuous-boring view (all rows of the window inside one boring-view segment), 3 also steady view (first history row and target in the steady phase of it)','P3')
    rows=[]
    for c in COLUMNS:
        if c in new: rows.append((c,)+new[c])
        else:
            r=std_dict[c]; rows.append((c,r['dtype'],r['unit'],r['description'],'standard series v1'))
    return pd.DataFrame(rows,columns=['name','dtype','unit','description','origin'])

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--src',default=str(SRC)); ap.add_argument('--seg',default=str(SEG)); ap.add_argument('--dest',default=str(DEST)); ap.add_argument('--workers',type=int,default=4); a=ap.parse_args()
    src,seg_dir,dest=Path(a.src),Path(a.seg),Path(a.dest); dest.mkdir(parents=True,exist_ok=True); start=time.perf_counter()
    seg_manifest=json.loads((seg_dir/'manifest.json').read_text(encoding='utf-8')); machines=list(seg_manifest['machines'])
    with ProcessPoolExecutor(min(a.workers,len(machines))) as ex: results=list(ex.map(process,[(m,str(src),str(seg_dir),str(dest)) for m in machines]))
    views=pd.DataFrame([x for r in results for x in r[2]]); wins=pd.DataFrame([x for r in results for x in r[3]]); acct=pd.DataFrame([x for r in results for x in r[4]])
    views.round(5).to_csv(dest/'view_summary.csv',index=False,encoding='utf-8-sig'); wins.round(5).to_csv(dest/'window_summary.csv',index=False,encoding='utf-8-sig'); acct.to_csv(dest/'window_accounting.csv',index=False,encoding='utf-8-sig')
    std=pd.read_csv(src/'field_dictionary.csv',encoding='utf-8-sig',keep_default_na=False).set_index('name').to_dict('index'); field_dictionary(std).to_csv(dest/'field_dictionary.csv',index=False,encoding='utf-8-sig')
    (dest/'manifest.json').write_text(json.dumps(dict(
        windows=dict(horizons_s=list(HORIZONS),history_rows=HISTORY_ROWS,max_gap_s=MAX_GAP_S,tolerance_s=TOLERANCE_S,origin='F, T, n and v above zero',rows_of_a_window='valid_core, one run',levels=LEVEL),
        views=dict(full='valid_core rows',boring='segments with seg_class long, phase_status ok and not stalled',strict='boring view with both segment edges uncensored',steady='phase 2 rows of the boring view'),
        phase_codes=PHASE,op_state_codes=OP_STATE,dropped_from_standard_series=DROPPED,
        source=dict(standard_v1_manifest_sha256=sha256(src/'manifest.json'),segments_v1_manifest_sha256=sha256(seg_dir/'manifest.json'),segments_sha256=sha256(seg_dir/'segments.csv')),
        machines={r[0]:r[1] for r in results}),ensure_ascii=False,indent=1),encoding='utf-8')
    print(f'{len(machines)} machines, {sum(r[1]["rows"] for r in results):,} rows in {time.perf_counter()-start:.0f} s')
    pd.set_option('display.width',200); print(views.groupby('view')[['rows','hours']].sum().round(0)); print(wins[wins.horizon_s==60].groupby('view').windows.sum())

if __name__=='__main__': main(); finish()
