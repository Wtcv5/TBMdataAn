"""P3 verification (read-only except outputs/analysis_v1/verification.json): the analysis dataset against the standard time series v1 and the P2 labels, and against independent
re-derivations: the views from the segment table by a table merge, the runs by a plain loop, the windows once in the style of the committed baseline (breaks at bad rows and gaps, segment
numbers compared) and once by reading the definition row by row on sampled rows, the levels by running sums, and the summary tables from the stored columns. Every check prints PASS or FAIL;
the exit status is 1 if any fails. --synthetic leaves out what only makes sense on the real data (the numbers of the P2 report, the regression file)."""
import argparse, json, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from p1_standardize import checksums, sha256
from p2_segment import SRC, DEST as SEG, PHASE, OP_STATE, finish
from p3_build import DEST, KEEP, COLUMNS, DROPPED, VIEWS
from p3_windows import HORIZONS, TOLERANCE_S, HISTORY_ROWS, MAX_GAP_S, LEVEL, samples, seconds, feature_names

REPORTED=dict(boring=(8949,3739),strict=(8200,3523),steady_rows=847291,steady_strict_rows=801193)      # P2 report, section 8
SPOT=200; SEED=20240102; HERE=Path(__file__).resolve().parent

def same(a,b):
    """Equal values, missing values equal to each other."""
    if a.dtype.kind=='M': return bool((a.to_numpy()==b.to_numpy()).all())
    if str(a.dtype)=='category' or str(b.dtype)=='category' or a.dtype.kind=='O': return bool((a.astype(str).to_numpy()==b.astype(str).to_numpy()).all())
    return bool(np.array_equal(a.to_numpy(dtype='float64',na_value=np.nan),b.to_numpy(dtype='float64',na_value=np.nan),equal_nan=True))

def runs_by_loop(valid,ts):
    """Run numbers by a plain loop: a run is a stretch of valid rows with no two neighbours more than MAX_GAP_S apart."""
    out=[-1]*len(ts); run=-1; before=False; valid=valid.tolist(); t=ts.tolist()
    for i in range(len(t)):
        if valid[i]:
            if not (before and t[i]-t[i-1]<=MAX_GAP_S): run+=1
            out[i]=run
        before=valid[i]
    return np.array(out,dtype='int32')

def baseline_style(ts,valid,active,stream,h):
    """Window targets as the committed baseline finds them: per stream, a row breaks the path if it is bad or follows a bad row or a gap above 60 s; the origin needs 3 rows before it on the same path as the target."""
    pos=np.full(len(ts),-1,np.int64)
    for s in pd.unique(stream):
        idx=np.flatnonzero(stream==s); t=ts[idx]; bad=~valid[idx]; n=len(idx); breaks=np.r_[True,np.diff(t)>MAX_GAP_S]|bad|np.r_[False,bad[:-1]]; seg=np.cumsum(breaks)
        origin=np.arange(HISTORY_ROWS-1,n); j=np.searchsorted(t,t[origin]+h,side='left'); inside=j<n; origin,j=origin[inside],j[inside]
        ok=(t[j]-t[origin]-h<=TOLERANCE_S)&(seg[origin-(HISTORY_ROWS-1)]==seg[j])&active[idx][origin]; pos[idx[origin[ok]]]=idx[j[ok]]
    return pos

def levels_by_sums(pos,boring,steady,seg_no):
    """Level of every window by running sums over the rows of the window: all rows in the boring view and no change of segment number is level 2, all rows in the steady view level 3."""
    lev=np.zeros(len(pos),np.int8); o=np.flatnonzero(pos>=0); s=o-(HISTORY_ROWS-1); j=pos[o]; lev[o]=LEVEL['full']
    nb=np.cumsum(np.r_[0,~boring]); ns=np.cumsum(np.r_[0,~steady]); chg=np.r_[0,np.cumsum(seg_no[1:]!=seg_no[:-1])]
    b=((nb[j+1]-nb[s])==0)&(chg[j]==chg[s]); st=b&((ns[j+1]-ns[s])==0); lev[o[b]]=LEVEL['boring']; lev[o[st]]=LEVEL['steady']; return lev

def explicit(o,h,t,valid,active,boring,seg_no,phase):
    """Target and level of the window that starts at row o, by reading the definition row by row."""
    n=len(t)
    if not active[o] or o<HISTORY_ROWS-1: return -1,0
    j=next((k for k in range(o+1,n) if t[k]>=t[o]+h),None)
    if j is None or t[j]-t[o]-h>TOLERANCE_S: return -1,0
    rows=list(range(o-(HISTORY_ROWS-1),j+1))
    if not all(valid[k] for k in rows) or any(t[k+1]-t[k]>MAX_GAP_S for k in rows[:-1]): return -1,0
    level=LEVEL['full']
    if all(boring[k] for k in rows) and len({seg_no[k] for k in rows})==1:
        level=LEVEL['boring']
        if all(phase[k]==PHASE['steady'] for k in rows): level=LEVEL['steady']
    return j,level

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--src',default=str(SRC)); ap.add_argument('--seg',default=str(SEG)); ap.add_argument('--dest',default=str(DEST))
    ap.add_argument('--synthetic',action='store_true'); ap.add_argument('--no-rebuild',action='store_true'); a=ap.parse_args(); src,seg_dir,dest=Path(a.src),Path(a.seg),Path(a.dest); results=[]
    def check(name,cond,detail=''):
        print(('PASS ' if cond else 'FAIL ')+name+(f'  [{detail}]' if detail else ''),flush=True); results.append(dict(check=name,passed=bool(cond),detail=str(detail)))
    man=json.loads((dest/'manifest.json').read_text(encoding='utf-8')); machines=list(man['machines']); seg=pd.read_csv(seg_dir/'segments.csv'); rng=np.random.default_rng(SEED)
    check('the inputs are those the dataset was built from (standard series, P2 labels, segment table)',
          man['source']==dict(standard_v1_manifest_sha256=sha256(src/'manifest.json'),segments_v1_manifest_sha256=sha256(seg_dir/'manifest.json'),segments_sha256=sha256(seg_dir/'segments.csv')))
    check('the window rules of the manifest are those of the code',man['windows']==dict(horizons_s=list(HORIZONS),history_rows=HISTORY_ROWS,max_gap_s=MAX_GAP_S,tolerance_s=TOLERANCE_S,origin='F, T, n and v above zero',rows_of_a_window='valid_core, one run',levels=LEVEL)
          and man['phase_codes']==PHASE and man['op_state_codes']==OP_STATE)
    check('the machines are those of the P2 labels',machines==list(json.loads((seg_dir/'manifest.json').read_text(encoding='utf-8'))['machines']))
    fd=pd.read_csv(dest/'field_dictionary.csv',encoding='utf-8-sig',keep_default_na=False); check('the field dictionary names every column, in order, with a description and its origin',list(fd.name)==COLUMNS and (fd.description!='').all()
          and set(fd.origin)<={'standard series v1','P2 labels','P3'})
    vs=pd.read_csv(dest/'view_summary.csv'); ws=pd.read_csv(dest/'window_summary.csv'); wa=pd.read_csv(dest/'window_accounting.csv'); tot=dict(boring_rows=0,strict_rows=0,steady_rows=0,steady_strict_rows=0)
    for m in machines:
        mm=man['machines'][m]; d=pd.read_parquet(dest/mm['file']); std=pd.read_parquet(src/f'{m}.parquet',columns=KEEP); lab=pd.read_parquet(seg_dir/'labels'/f'{m}.parquet'); sg=seg[seg.machine==m].set_index('seg_no').sort_index(); n=len(d)
        check(f'{m}: file, rows and column checksums are those of the manifest',sha256(dest/mm['file'])==mm['sha256'] and len(d)==mm['rows'] and checksums(d)==mm['checksums'])
        check(f'{m}: columns are the documented ones, with the documented types',list(d.columns)==COLUMNS and all(str(d[c].dtype)==t for c,t in zip(fd.name,fd.dtype)),'; '.join(f'{c}: {d[c].dtype} vs {t}' for c,t in zip(fd.name,fd.dtype) if str(d[c].dtype)!=t)[:200])
        check(f'{m}: every row of the standard series is here, in the same order, with the same values in {len(KEEP)} columns',len(d)==len(std) and all(same(d[c],std[c]) for c in KEEP))
        check(f'{m}: the columns left out are exactly those listed in the manifest',sorted(set(pq.read_schema(src/f"{m}.parquet").names)-set(d.columns))==sorted(DROPPED))
        check(f'{m}: the P2 labels are copied unchanged',all(same(d[c],lab[c]) for c in ('seg_no','phase','op_state')))
        check(f'{m}: time increases strictly',(np.diff(seconds(d.time))>0).all())
        wide=sg.seg_class.eq('long')&sg.phase_status.eq('ok')&~sg.stalled; strict=wide&~sg.left_censored&~sg.right_censored
        by_merge=d[['seg_no','phase']].merge(pd.DataFrame(dict(seg_no=sg.index,boring=wide.to_numpy(),strict=strict.to_numpy())),on='seg_no',how='left').fillna(False)
        check(f'{m}: the views follow from the segment table (merge on the segment number)',(d.view_boring.to_numpy()==by_merge.boring.to_numpy(bool)).all() and (d.view_strict.to_numpy()==by_merge.strict.to_numpy(bool)).all()
              and (d.view_steady.to_numpy()==(by_merge.boring.to_numpy(bool)&(d.phase.to_numpy()==PHASE['steady']))).all() and (d.view_strict<=d.view_boring).all() and (d.view_steady<=d.view_boring).all())
        check(f'{m}: the rows of the boring view are the rows of its segments, the segments those of the table',int(d.view_boring.sum())==int(sg.rows[wide].sum()) and int(d.view_strict.sum())==int(sg.rows[strict].sum())
              and int(wide.sum())==int(vs[(vs.machine==m)&(vs.view=='boring')].segments.iloc[0]) and int(strict.sum())==int(vs[(vs.machine==m)&(vs.view=='strict')].segments.iloc[0]),f'{int(wide.sum())} and {int(strict.sum())} segments')
        ts=seconds(d.time); valid=d.valid_core.to_numpy(bool); active=d.active.to_numpy(bool); seg_no=d.seg_no.to_numpy(); phase=d.phase.to_numpy(); boring=d.view_boring.to_numpy(bool); steady=d.view_steady.to_numpy(bool)
        check(f'{m}: run numbers agree with a plain loop',(runs_by_loop(valid,ts)==d.run_id.to_numpy()).all())
        t_l,v_l,a_l,b_l,s_l,p_l=ts.tolist(),valid.tolist(),active.tolist(),boring.tolist(),seg_no.tolist(),phase.tolist(); stream=d.stream.astype(str).to_numpy()
        rep=np.cumsum(np.r_[0,d.n_source.astype(str).str.startswith('repaired').to_numpy(bool)]); pnv=np.cumsum(np.r_[0,d.q_pnv_off.to_numpy(bool)])
        for h in HORIZONS:
            pos=d[f'win{h}_pos'].to_numpy(np.int64); lev=d[f'win{h}_level'].to_numpy()
            check(f'{m}, {h} s: the targets are those found the way the committed baseline finds them',(baseline_style(ts,valid,active,stream,h)==pos).all())
            check(f'{m}, {h} s: levels agree with the running-sum derivation; a window exists exactly where the level is above 0',(levels_by_sums(pos,boring,steady,seg_no)==lev).all() and ((lev>0)==(pos>=0)).all())
            o=np.flatnonzero(pos>=0); s=o-(HISTORY_ROWS-1); j=pos[o]
            ok=(o>=HISTORY_ROWS-1).all() and (j>o).all() and (j<n).all() and active[o].all() and (d.run_id.to_numpy()[s]==d.run_id.to_numpy()[j]).all() and (d.run_id.to_numpy()[o]>=0).all() and (np.diff(j)>=0).all()
            gap=np.diff(ts); inside=np.cumsum(np.r_[0,(gap>MAX_GAP_S)|~valid[1:]|~valid[:-1]]); hz=ts[j]-ts[o]
            check(f'{m}, {h} s: every window has an active origin, valid rows of one run without a gap above {MAX_GAP_S} s, and a target {h} to {h+TOLERANCE_S} s after the origin',bool(ok) and (inside[j]==inside[s]).all() and ((hz>=h)&(hz<=h+TOLERANCE_S)).all() and valid[s].all() and valid[j].all(),f'{len(o)} windows')
            pick_w=rng.choice(o,size=min(SPOT,len(o)),replace=False) if len(o) else np.array([],int); pick_n=rng.choice(np.flatnonzero(active&(pos<0)),size=min(SPOT,int((active&(pos<0)).sum())),replace=False)
            bad=sum(explicit(int(r),h,t_l,v_l,a_l,b_l,s_l,p_l)!=(int(pos[r]),int(lev[r])) for r in np.r_[pick_w,pick_n])
            check(f'{m}, {h} s: reading the definition row by row gives the stored target and level for {len(pick_w)} windows and {len(pick_n)} active origins without one',bad==0,f'{bad} differ')
            fr=samples(d,pos,lev,LEVEL['full']); sel=rng.choice(len(fr),size=min(SPOT,len(fr)),replace=False) if len(fr) else []; names=feature_names(); wrong=0
            for q in sel:
                r=fr.iloc[q]; oo=int(r.row_origin); jj=int(r.row_target)
                for k in range(HISTORY_ROWS):
                    for c in ('F','T','n','v'): wrong+=int(r[f'{c}_lag{k}']!=d[c].iloc[oo-k])
                for k in range(1,HISTORY_ROWS):
                    wrong+=int(r[f'age_lag{k}_s']!=ts[oo]-ts[oo-k])+sum(int(r[f'{c}_change{k}']!=d[c].iloc[oo]-d[c].iloc[oo-k]) for c in ('F','T','n','v'))
                wrong+=sum(int(r['y_'+c]!=d[c].iloc[jj]) for c in ('F','T','v'))+int(r.actual_horizon_s!=ts[jj]-ts[oo])+int(r.origin!=d.time.iloc[oo])+int(r.target_time!=d.time.iloc[jj])
            check(f'{m}, {h} s: the samples (features, targets, times) equal the values read directly from the rows for {len(sel)} windows',wrong==0 and list(fr.columns[:len(names)])==names,f'{wrong} differ')
            a_m=wa[(wa.machine==m)&(wa.horizon_s==h)].set_index('stage').windows; w1=int((pos>=0).sum()); w2=int((lev>=LEVEL['boring']).sum()); w3=int((lev>=LEVEL['steady']).sum())
            part=a_m.origin_not_valid+a_m.history_incomplete+a_m.target_not_found+a_m.target_in_other_run+a_m.full_run_windows
            check(f'{m}, {h} s: the accounting adds up (active rows = rows lost for each reason + windows) and counts the windows of the stored levels',a_m.active_rows==int(active.sum()) and part==a_m.active_rows and a_m.full_run_windows==w1 and a_m.boring_windows==w2
                  and a_m.boring_not_steady==w2-w3 and a_m.steady_windows==w3 and a_m.origin_not_valid==int((active&~valid).sum()) and w1-w2==a_m[[k for k in a_m.index if k.startswith('not_boring')]].sum())
            check(f'{m}, {h} s: the manifest counts the windows of the three views',man['machines'][m]['windows'][str(h)]==dict(full_run_view=w1,boring_view=w2,steady_view=w3))
            both=d.view_strict.to_numpy(bool)[fr.row_origin.to_numpy()-(HISTORY_ROWS-1)]&d.view_strict.to_numpy(bool)[fr.row_target.to_numpy()]
            masks={'full':fr.level>=1,'boring':fr.level>=2,'strict':(fr.level>=2)&both,'steady':fr.level>=3,'steady_strict':(fr.level>=3)&both}; wrong=[]
            for view in VIEWS:
                r=ws[(ws.machine==m)&(ws.horizon_s==h)&(ws.view==view)].iloc[0]; k=masks[view].to_numpy(bool); x=fr[k]; tg=x.row_target.to_numpy()
                if r.windows!=k.sum(): wrong.append((view,'windows')); continue
                if not k.any(): continue
                first=x.row_origin.to_numpy()-(HISTORY_ROWS-1)
                exp=dict(target_active_share=active[tg].mean(),with_repaired_n_share=((rep[tg+1]-rep[first])>0).mean(),with_pnv_off_share=((pnv[tg+1]-pnv[first])>0).mean(),**{f'target_{nm}_share':(d.op_state.to_numpy()[tg]==c).mean() for nm,c in OP_STATE.items()},**{f'origin_{nm}_share':(phase[x.row_origin.to_numpy()]==c).mean() for nm,c in PHASE.items()},
                         persistence_rel_mae_v=np.abs(x.y_v-x.v_lag0).mean()/np.abs(x.y_v).mean(),persistence_rel_mae_F=np.abs(x.y_F-x.F_lag0).mean()/np.abs(x.y_F).mean(),origin_days=x.origin.dt.floor('D').nunique(),share_of_full=k.sum()/len(fr))
                wrong+=[(view,c) for c,e in exp.items() if abs(r[c]-e)>6e-6*max(1,abs(e))]
            check(f'{m}, {h} s: the window summary agrees with the windows of the stored columns, view by view',not wrong,str(wrong[:4]) if wrong else '')
        wrong=[]; dt=d.dt_prev_s.to_numpy(float)
        for view in VIEWS:
            member={'full':valid,'boring':boring,'strict':d.view_strict.to_numpy(bool),'steady':steady,'steady_strict':steady&d.view_strict.to_numpy(bool)}[view]; r=vs[(vs.machine==m)&(vs.view==view)].iloc[0]
            both=member[1:]&member[:-1]&(dt[1:]<=MAX_GAP_S); hours=float(dt[1:][both].sum()/3600)
            if not (r.rows==member.sum() and abs(r.hours-hours)<=2e-5*max(1,hours) and r.active_rows==(member&active).sum() and abs(r.share_of_rows-member.mean())<=6e-6): wrong.append(view)
        check(f'{m}: view summary: rows, share, hours (from the reading intervals) and active rows of the five views agree with the stored columns',not wrong,str(wrong) if wrong else '')
        check(f'{m}: the hours of the boring and strict view equal the durations in the segment table',abs(vs[(vs.machine==m)&(vs.view=="boring")].hours.iloc[0]-sg.duration_s[wide].sum()/3600)<2e-5*sg.duration_s[wide].sum()/3600
              and abs(vs[(vs.machine==m)&(vs.view=="strict")].hours.iloc[0]-sg.duration_s[strict].sum()/3600)<2e-5*sg.duration_s[strict].sum()/3600)
        tot['boring_rows']+=int(boring.sum()); tot['strict_rows']+=int(d.view_strict.sum()); tot['steady_rows']+=int(steady.sum()); tot['steady_strict_rows']+=int((steady&d.view_strict.to_numpy(bool)).sum())
    wide=seg.seg_class.eq('long')&seg.phase_status.eq('ok')&~seg.stalled; strict=wide&~seg.left_censored&~seg.right_censored
    if not a.synthetic:
        check('the views reproduce the numbers of the P2 report: segments, hours and steady rows',(int(wide.sum()),round(seg.duration_s[wide].sum()/3600))==REPORTED['boring'] and (int(strict.sum()),round(seg.duration_s[strict].sum()/3600))==REPORTED['strict']
              and tot['steady_rows']==REPORTED['steady_rows'] and tot['steady_strict_rows']==REPORTED['steady_strict_rows'],f"{int(wide.sum())} segments, {seg.duration_s[wide].sum()/3600:.0f} h; strict {int(strict.sum())}, {seg.duration_s[strict].sum()/3600:.0f} h; steady rows {tot['steady_rows']} and {tot['steady_strict_rows']}")
        reg=dest/'window_regression.csv'; r=pd.read_csv(reg) if reg.exists() else None
        check('the regression against the committed baseline ran: all 9 machines and 3 horizons rebuilt identically',r is not None and len(r)==27 and r.identical.all())
    if not a.no_rebuild:
        with tempfile.TemporaryDirectory() as tmp:
            rc=subprocess.run([sys.executable,'p3_build.py','--src',str(src),'--seg',str(seg_dir),'--dest',tmp,'--workers','2'],cwd=HERE,stdout=subprocess.DEVNULL).returncode
            names=sorted([f'{m}.parquet' for m in machines]+['manifest.json','field_dictionary.csv','view_summary.csv','window_summary.csv','window_accounting.csv'])
            check('a second build reproduces every file byte for byte',rc==0 and names==sorted(p.name for p in Path(tmp).iterdir()) and all(sha256(dest/nm)==sha256(Path(tmp)/nm) for nm in names),f'{len(names)} files')
    n_ok=sum(r['passed'] for r in results); (dest/'verification.json').write_text(json.dumps(dict(passed=n_ok,total=len(results),all_passed=n_ok==len(results),checks=results),ensure_ascii=False,indent=1),encoding='utf-8')
    print(f'{n_ok}/{len(results)} checks passed'); finish(0 if n_ok==len(results) else 1)

if __name__=='__main__': main()
