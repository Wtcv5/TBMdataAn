"""Self-test for P3. Takes the SYNTHETIC standard time series of smoke_test_p2 (two machines with planted strokes, holes, pauses, a stalled stretch, a flicker, a stroke with invalid rows ...),
completes it to the columns of the standard series v1, runs the real p2_segment.py, p3_build.py and p3_verify.py on it, and checks that the windows and views of every planted case come out
as worked out by hand: the clean stroke, the strokes cut by a hole (no window across the hole), the short, stalled and flicker segments (full-run view only), the strict and the steady
variant, rows that are not valid. Checks that the verification fails when a level, a target, a view flag, a signal, a label, a run number, a summary table or a source hash is tampered with
(the manifest is brought up to date after each tampering, so that only the content checks can catch it), that a second build reproduces every byte, and that the window rules of the
regression (p3_regress.rebuild) equal a verbatim copy of the committed baseline's window loop on a messy synthetic series. Nothing here is a research result and nothing is written to the
real analysis/outputs."""
import argparse, json, shutil, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from p1_standardize import checksums, sha256
from p2_segment import finish
from p3_regress import rebuild
from smoke_test_p2 import cases, assemble

def standard(d,repaired=(),pnv=()):
    """The frames of smoke_test_p2 completed to the columns of the standard series v1 (repaired: rows whose n is marked as repaired; pnv: rows flagged q_pnv_off)."""
    d=d.copy(); n=len(d); valid=d.valid_core.to_numpy(bool)
    d['project']='synthetic'; d['file_status']='trusted'; d['p']=d.v/d.n.where(d.n>0); d['n_raw']=d.n; d['n_source']=pd.Categorical(['exported']*n,categories=['exported','repaired_v_over_p'])
    d.loc[list(repaired),'n_source']='repaired_v_over_p'
    for c in ('state_cutter','state_regrip'): d[c]=pd.array([pd.NA]*n,dtype='Int16')
    d['drive_speed_med']=np.float32(np.nan); d['drive_torque_med']=np.float32(np.nan); d['drive_motors_on']=np.zeros(n,'int8')
    for q in ('q_time_backward','q_time_coarse','q_time_conflict','q_neg','q_core_missing','q_n_invalid','q_v_high','q_T_high','q_F_high','q_pnv_off','q_v_invalid_file'): d[q]=False
    d['q_v_high']=~valid; d.loc[list(pnv),'q_pnv_off']=True
    return d

def run(cmd,**kw): return subprocess.run([sys.executable]+cmd,cwd=HERE,**kw)

def retamper(dest,machine,modify):
    """Change the machine file and bring the manifest up to date (size, hash, column checksums), so that only checks of the content can notice."""
    path=dest/f'{machine}.parquet'; d=pd.read_parquet(path); modify(d); d.to_parquet(path,engine='pyarrow',compression='zstd',index=False)
    man=json.loads((dest/'manifest.json').read_text(encoding='utf-8')); man['machines'][machine].update(bytes=path.stat().st_size,sha256=sha256(path),checksums=checksums(d))
    (dest/'manifest.json').write_text(json.dumps(man,ensure_ascii=False,indent=1),encoding='utf-8')

def committed_windows(raw,h):
    """The window loop of build_baselines.prepare(), copied verbatim (apart from the names), for the comparison with p3_regress.rebuild."""
    CORE=['F','T','n','v']; TARGETS=['F','T','v']; parts=[]
    for directory,g in raw.groupby('directory',sort=False):
        g=g.sort_values('time').reset_index(drop=True); ts=g.time.astype('int64').to_numpy()//10**9; a=g[CORE].to_numpy(dtype=float); n=len(g)
        bad=(~np.isfinite(a).all(axis=1))|(a<0).any(axis=1); breaks=np.r_[True,np.diff(ts)>60]|bad|np.r_[False,bad[:-1]]; seg=np.cumsum(breaks); origin=np.arange(3,n)
        j=np.searchsorted(ts,ts[origin]+h,side='left'); inside=j<n; origin=origin[inside]; j=j[inside]
        ok=(ts[j]-ts[origin]-h<=5)&(seg[origin-3]==seg[j])&(a[origin]>0).all(axis=1); origin=origin[ok]; j=j[ok]
        if not len(origin): continue
        features={}
        for lag in range(4):
            for k,c in enumerate(CORE): features[f'{c}_lag{lag}']=a[origin-lag,k]
        for lag in range(1,4):
            features[f'age_lag{lag}_s']=ts[origin]-ts[origin-lag]
            for k,c in enumerate(CORE): features[f'{c}_change{lag}']=a[origin,k]-a[origin-lag,k]
        f=pd.DataFrame(features); f['origin']=g.time.iloc[origin].to_numpy(); f['target_time']=g.time.iloc[j].to_numpy(); f['actual_horizon_s']=ts[j]-ts[origin]; f['directory']=directory
        for c in TARGETS: f['y_'+c]=g[c].to_numpy()[j]
        parts.append(f)
    return pd.concat(parts,ignore_index=True).sort_values('origin').reset_index(drop=True)

def messy(seed):
    """A messy synthetic series in the layout of the canonical caches: two directories, irregular reading intervals with gaps, zeros, negative and missing values."""
    r=np.random.default_rng(seed); frames=[]
    for k,directory in enumerate(('dirA','dirB')):
        n=3000; dt=r.choice([10]*12+[1,2,5,9,11,20,30,61,130],size=n); t=pd.Timestamp('2023-05-01')+pd.to_timedelta(np.cumsum(dt)+k*10**7,unit='s')
        a=r.uniform(.5,5,size=(n,4)); a[r.random((n,4))<.006]=0.; a[r.random((n,4))<.003]=-1.; a[r.random((n,4))<.003]=np.nan
        frames.append(pd.DataFrame(dict(time=t,F=a[:,0],T=a[:,1],n=a[:,2],v=a[:,3],directory=directory)))
    return pd.concat(frames,ignore_index=True)

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--keep',action='store_true'); a=ap.parse_args(); ok=[]
    def check(name,cond,detail=''):
        print(('PASS ' if cond else 'FAIL ')+name+(f'  [{detail}]' if detail and not cond else ''),flush=True); ok.append(bool(cond))
    with tempfile.TemporaryDirectory() as tmp:
        tmp=Path(tmp); src=tmp/'src'; seg=tmp/'seg'; out=tmp/'out'; src.mkdir(); (seg/'labels').mkdir(parents=True); p8,p5=cases(); d8,w8=assemble(p8,'TBM8'); d5,w5=assemble(p5,'TBM5')
        d8=standard(d8,repaired=range(w8['decay'][0],w8['decay'][1]+1),pnv=range(w8['terminal'][0]+20,w8['terminal'][0]+26)); d5=standard(d5)
        d8.to_parquet(src/'TBM8.parquet'); d5.to_parquet(src/'TBM5.parquet'); (src/'manifest.json').write_text(json.dumps(dict(synthetic=True)),encoding='utf-8')
        ref=pd.read_parquet(src/'TBM8.parquet'); pd.DataFrame([(c,str(ref[c].dtype),'','','synthetic','') for c in ref.columns],columns=['name','dtype','unit','unit_status','description','source_columns']).to_csv(src/'field_dictionary.csv',index=False,encoding='utf-8-sig')
        r=run(['p2_segment.py','--src',str(src),'--dest',str(seg),'--machines','TBM5,TBM8','--workers','2'],stdout=subprocess.DEVNULL); check('p2_segment.py runs on the synthetic series',r.returncode==0)
        r=run(['p3_build.py','--src',str(src),'--seg',str(seg),'--dest',str(out),'--workers','2'],stdout=subprocess.DEVNULL); check('p3_build.py runs',r.returncode==0)
        d=pd.read_parquet(out/'TBM8.parquet'); e=pd.read_parquet(out/'TBM5.parquet'); st=pd.read_csv(seg/'segments.csv'); s8=st[st.machine=='TBM8']; vs=pd.read_csv(out/'view_summary.csv'); ws=pd.read_csv(out/'window_summary.csv')
        lev=d.win60_level.to_numpy(); pos=d.win60_pos.to_numpy(); span=lambda name: slice(w8[name][0],w8[name][1]+1)
        count=lambda name,level,h=60: int((d[f'win{h}_level'].to_numpy()[span(name)]>=level).sum())
        check('one row per row of the standard series, in order, and every column documented',len(d)==len(d8) and (d.time.to_numpy()==d8.time.to_numpy()).all() and len(e)==len(d5))
        c=s8[s8.start==d.time.iloc[w8['clean'][0]].strftime('%Y-%m-%d %H:%M:%S')].iloc[0]; a0=w8['clean'][0]
        f0=int(np.flatnonzero(d.time.to_numpy()==np.datetime64(c.steady_start))[0]); l0=int(np.flatnonzero(d.time.to_numpy()==np.datetime64(c.steady_end))[0])
        check('clean stroke (162 rows): every row starts a full-run window; the boring view needs the 3 history rows and the 6 rows ahead inside the stroke; the steady view inside the steady phase',
              (count('clean',1),count('clean',2),count('clean',3))==(162,153,l0-f0-8),f'{count("clean",1)} {count("clean",2)} {count("clean",3)} vs 162 153 {l0-f0-8}')
        check('the 30 s and 120 s windows of the clean stroke: 3 and 12 rows ahead',(count('clean',1,30),count('clean',2,30),count('clean',1,120),count('clean',2,120))==(162,156,162,147))
        check('short stroke (20 rows), stalled stretch (15 active rows), three-row flicker: windows in the full-run view only',(count('short',1),count('short',2))==(20,0) and (count('stalled',1),count('stalled',2))==(15,0) and (count('blip',1),count('blip',2))==(3,0))
        ha,hb=w8['hole_a'],w8['hole_b']
        check('stroke cut by a hole: no window starts in the last 5 rows before the 300 s gap or in the first 3 rows after it; none reaches across',
              (count('hole_a',1),count('hole_a',2))==(106,103) and (count('hole_b',1),count('hole_b',2),count('hole_b',3))==(97,91,91) and (pos[ha[0]:ha[1]+1]<hb[0]).all() and (pos[hb[0]:hb[0]+3]==-1).all())
        check('holes make the segment edges censored: the cut stroke is in the boring view but not in the strict one, the clean one in both',bool(d.view_boring.iloc[ha[0]+10]) and not bool(d.view_strict.iloc[ha[0]+10]) and bool(d.view_strict.iloc[a0+10]))
        wide=['clean','hole_a','hole_b','pause_a','pause_b','regime_a','regime_b','decay','terminal','after_hole']; not_wide=['series_start','short','blip','stalled','invalid','series_end']
        check('boring view: exactly the long segments with a steady phase that advance',all(d.view_boring.iloc[span(k)].all() for k in wide) and not any(d.view_boring.iloc[span(k)].any() for k in not_wide)
              and not d.view_boring.iloc[span('pause')].any() and not d.view_boring.iloc[[i for k in ('i0','i1','i1b','i2','i3','i4','i5','i6','i7','i8','i9','i10','i11') for i in range(w8[k][0],w8[k][1]+1)]].any())
        vb=vs[vs.machine=='TBM8'].set_index('view'); check('view summary: 9 boring-view segments, 6 of them strict (the cut ones and the one after a hole are not)',(int(vb.segments.boring),int(vb.segments.strict))==(9,6),f'{vb.segments.boring} {vb.segments.strict}')
        inv=d.valid_core.to_numpy(bool); o=np.flatnonzero(pos>=0); bad_in=sum(not inv[x-3:pos[x]+1].all() for x in o)
        check('rows that are not valid never lie in a window; the stroke with every fifth row invalid has no window at all (each window covers 10 rows)',bad_in==0 and count('invalid',1)==0)
        pa,pb=w8['pause_a'],w8['pause_b']; into=(pos[pa[0]:pa[1]+1]>=w8['pause'][0])&(pos[pa[0]:pa[1]+1]<=w8['pause'][1]); after=(o>=pb[0])&(o<pb[0]+3)
        check('paused rows are outside every segment: windows reaching into the pause or starting after it are full-run windows only, none in the boring view',(d.seg_no.iloc[span('pause')]==-1).all()
              and after.sum()==3 and (lev[o][after]==1).all() and into.sum()>0 and (lev[pa[0]:pa[1]+1][into]==1).all())
        rp=ws[(ws.machine=='TBM8')&(ws.horizon_s==60)].set_index('view'); check('window summary: windows touching rows of repaired n or flagged q_pnv_off are counted',rp.with_repaired_n_share.full>0 and rp.with_repaired_n_share.steady>0
              and rp.with_pnv_off_share.full>0 and abs(rp.windows.full-(pos>=0).sum())==0)
        check('window summary: the boring view has no stopped or rotating target, the full-run view has some',rp.target_boring_share.boring==1 and rp.target_boring_share.full<1 and rp.persistence_rel_mae_F.steady<=rp.persistence_rel_mae_F.full)
        r=run(['p3_verify.py','--src',str(src),'--seg',str(seg),'--dest',str(out),'--synthetic'],stdout=subprocess.PIPE,text=True); fails=[l for l in r.stdout.splitlines() if l.startswith('FAIL')]
        check('p3_verify.py passes on the synthetic data (including its own rebuild)',r.returncode==0 and not fails and 'checks passed' in r.stdout,'; '.join(fails)[:300])
        # tampering must be caught
        tampers={'one window level lowered':lambda x:x.__setitem__('win60_level',np.where(np.arange(len(x))==a0+50,1,x.win60_level).astype('int8')),
                 'one window level raised':lambda x:x.__setitem__('win60_level',np.where(np.arange(len(x))==w8['short'][0]+5,3,x.win60_level).astype('int8')),
                 'one target moved by a row':lambda x:x.__setitem__('win60_pos',np.where(np.arange(len(x))==a0+50,x.win60_pos+1,x.win60_pos).astype('int32')),
                 'one window deleted':lambda x:(x.__setitem__('win60_pos',np.where(np.arange(len(x))==a0+50,-1,x.win60_pos).astype('int32')),x.__setitem__('win60_level',np.where(np.arange(len(x))==a0+50,0,x.win60_level).astype('int8'))),
                 'one row of the boring view switched off':lambda x:x.__setitem__('view_boring',np.where(np.arange(len(x))==a0+50,False,x.view_boring)),
                 'one row of the strict view switched on':lambda x:x.__setitem__('view_strict',np.where(np.arange(len(x))==hb[0]+50,True,x.view_strict)),
                 'one signal value changed':lambda x:x.__setitem__('F',np.where(np.arange(len(x))==a0+50,x.F+1,x.F)),
                 'one phase code changed':lambda x:x.__setitem__('phase',np.where(np.arange(len(x))==a0+50,3,x.phase).astype('int8')),
                 'one run number changed':lambda x:x.__setitem__('run_id',np.where(np.arange(len(x))==a0+50,x.run_id+1,x.run_id).astype('int32')),
                 'one quality flag changed':lambda x:x.__setitem__('valid_core',np.where(np.arange(len(x))==a0+50,False,x.valid_core))}
        for name,mod in tampers.items():
            bad=tmp/('bad_'+str(abs(hash(name)))); shutil.copytree(out,bad); retamper(bad,'TBM8',mod); r=run(['p3_verify.py','--src',str(src),'--seg',str(seg),'--dest',str(bad),'--synthetic','--no-rebuild'],stdout=subprocess.PIPE,text=True)
            check(f'verification fails when {name}',r.returncode==1)
        for name,fn in {'a window count in the window summary':('window_summary.csv',lambda t:t.assign(windows=np.where(t.index==3,t.windows+1,t.windows))),
                        'the hours in the view summary':('view_summary.csv',lambda t:t.assign(hours=np.where(t.index==2,t.hours+.5,t.hours))),
                        'a stage of the window accounting':('window_accounting.csv',lambda t:t.assign(windows=np.where(t.index==5,t.windows+1,t.windows)))}.items():
            bad=tmp/('bad_'+str(abs(hash(name)))); shutil.copytree(out,bad); t=pd.read_csv(bad/fn[0]); fn[1](t).to_csv(bad/fn[0],index=False,encoding='utf-8-sig'); r=run(['p3_verify.py','--src',str(src),'--seg',str(seg),'--dest',str(bad),'--synthetic','--no-rebuild'],stdout=subprocess.PIPE,text=True)
            check(f'verification fails when {name} is changed',r.returncode==1)
        bad=tmp/'bad_source'; shutil.copytree(out,bad); man=json.loads((bad/'manifest.json').read_text(encoding='utf-8')); man['source']['segments_sha256']='0'*64; (bad/'manifest.json').write_text(json.dumps(man),encoding='utf-8')
        r=run(['p3_verify.py','--src',str(src),'--seg',str(seg),'--dest',str(bad),'--synthetic','--no-rebuild'],stdout=subprocess.PIPE,text=True); check('verification fails when the manifest names other inputs',r.returncode==1)
        bad=tmp/'bad_file'; shutil.copytree(out,bad); p=bad/'TBM8.parquet'; x=pd.read_parquet(p); x.loc[5,'F']+=1; x.to_parquet(p,compression='zstd',index=False)
        r=run(['p3_verify.py','--src',str(src),'--seg',str(seg),'--dest',str(bad),'--synthetic','--no-rebuild'],stdout=subprocess.PIPE,text=True); check('verification fails when a file is changed behind the manifest',r.returncode==1)
        # a second build with another number of workers reproduces every byte
        out2=tmp/'out2'; run(['p3_build.py','--src',str(src),'--seg',str(seg),'--dest',str(out2),'--workers','1'],stdout=subprocess.DEVNULL)
        names=sorted(p.name for p in out.iterdir() if p.name!='verification.json'); check('a second build (one worker) reproduces every file byte for byte',names==sorted(p.name for p in out2.iterdir()) and all(sha256(out/nm)==sha256(out2/nm) for nm in names),f'{len(names)} files')
        # the regression rules against the verbatim loop
        same=0
        for seed in (1,2,3):
            raw=messy(seed)
            for h in (30,60,120):
                old=committed_windows(raw,h); new=rebuild(raw,h); same+=int(len(old)==len(new) and len(old)>100 and all(np.array_equal(old[c].to_numpy(),new[c].to_numpy()) for c in old.columns))
        check('p3_regress.rebuild equals a verbatim copy of the committed window loop on messy series (3 seeds, 3 horizons, 2 streams each)',same==9,f'{same}/9')
        if a.keep: shutil.copytree(tmp,HERE.parent/'p3_smoke_kept')
    print(f'{sum(ok)}/{len(ok)} checks passed'); finish(0 if all(ok) else 1)

if __name__=='__main__': main()
