"""Mechanics self-test for the validation experiments. Builds SYNTHETIC canonical data in a temporary repository copy,
runs the real build_baselines.prepare() on it, then the experiments, and checks leakage and bookkeeping invariants.
Nothing here is a research result and nothing is written to the real analysis/outputs."""
import argparse, shutil, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import lfilter

HERE=Path(__file__).resolve().parent
# sampling steps (s) with probabilities, rows per directory, {directory suffix: start date}, F/T scale
MACHINES={'TBM5':dict(steps=[10,11,9],p=[.9,.06,.04],rows=35000,start={'2023':'2023-04-15','2024':'2024-01-01'},scale=1.0),
          'TBM6':dict(steps=[10,11,9],p=[.9,.06,.04],rows=35000,start={'2023':'2023-04-14','2024':'2024-01-01'},scale=2.5),
          'TBM8':dict(steps=[10,11,12],p=[.65,.3,.05],rows=50000,start={'2024':'2024-03-05'},scale=.6),
          'TBM9':dict(steps=[20,10,30],p=[.5,.25,.25],rows=30000,start={'2023':'2023-09-15','2024':'2024-06-01'},scale=1.7),
          'TBM10':dict(steps=[20,10,30],p=[.5,.25,.25],rows=30000,start={'2023':'2023-09-13','2024':'2024-07-01'},scale=.9)}

def synth_directory(rng,cfg,suffix,source0):
    n=cfg['rows']; steps=rng.choice(cfg['steps'],p=cfg['p'],size=n); gap=rng.random(n)<.002
    steps[gap]=rng.integers(120,900,gap.sum()); shifts=steps[1500::1500]; steps[1500::1500]=rng.integers(60000,90000,len(shifts))  # shift-length outages
    time=pd.Timestamp(cfg['start'][suffix])+pd.to_timedelta(np.cumsum(steps),unit='s')
    run=np.zeros(n,bool); ramp=np.ones(n); level=np.ones((n,4)); idle=np.zeros(n,bool); i=0; state=True
    while i<n:
        j=min(n,i+int(rng.geometric(1/(450 if state else 150))))
        if state:
            run[i:j]=True; level[i:j]=np.exp(rng.normal(0,.35,4)); ramp[i:j]=.2+.8*np.minimum(1,np.arange(j-i)/40)
        else: idle[i:j]=rng.random()<.25       # some stops keep the cutterhead turning
        state=not state; i=j
    noise=lfilter([1],[1,-.93],.25*rng.normal(size=(n,4)),axis=0)
    x=np.array([8000*cfg['scale'],900*cfg['scale'],6.,45*np.sqrt(cfg['scale'])])*level*ramp[:,None]*np.exp(noise)
    x[rng.random(n)<.001,0]*=3                                   # thrust spikes
    x[run&(rng.random(n)<.002)]=0                                # brief dropouts to zero inside runs
    x[~run]=0; x[idle,2]=3.
    x[rng.random(n)<.0003,0]=np.nan; x[rng.random(n)<.0002,1]=-5  # invalid observations
    df=pd.DataFrame(x,columns=['F','T','n','v']); df.insert(0,'time',time)
    df['p']=df.v/df.n.replace(0,np.nan); df['n_aux']=df.n
    df['source_id']=source0+np.arange(n)*4//n
    return df

def build_synthetic(root,seed=7):
    rng=np.random.default_rng(seed); out=root/'analysis/outputs'; (out/'canonical').mkdir(parents=True); sid=0; files=[]; first={}
    for mid,cfg in MACHINES.items():
        for suffix in cfg['start']:
            name=f'{mid}-{suffix}-synthetic'; df=synth_directory(rng,cfg,suffix,sid); df['directory']=name
            df.to_parquet(out/'canonical'/f'{name}.parquet',index=False); first[name]=sid
            files+=[dict(file=f'{name}/file{k}.csv',rows=int((df.source_id==sid+k).sum())) for k in range(4)]; sid+=4
    pd.DataFrame(files).to_csv(out/'file_audit.csv',index=False)
    # pretend two files overlap across machines, so the quarantine path is exercised
    pd.DataFrame([dict(source_id_a=first['TBM5-2024-synthetic']+1,source_id_b=first['TBM6-2024-synthetic']+2)]).to_csv(out/'file_overlap.csv',index=False)
    pd.DataFrame([dict(a='TBM5-2024-synthetic',b='TBM6-2024-synthetic',matches=1),dict(a='TBM10-2023-synthetic',b='TBM9-2023-synthetic',matches=1)]).to_csv(out/'directory_overlap.csv',index=False)

def run(cmd,**kw):
    print('$',' '.join(str(c) for c in cmd)[:200],flush=True)
    subprocess.run([str(c) for c in cmd],check=True,**kw)

def checks(root,with_baseline):
    sys.path.insert(0,str(root/'analysis'))
    import validation_experiments as ve, validation_common as vc
    V=vc.DEST; ok=[]
    def check(name,cond):
        print(('PASS ' if cond else 'FAIL ')+name,flush=True); ok.append(bool(cond))
    names=vc.feature_names(); check('feature list excludes labels and oracle columns',not any(n.startswith(('y_','oracle_')) for n in names))
    a=pd.read_csv(V/'A_metrics.csv'); check('A: persistence skill is exactly 0',a[a.model=='Persistence'].skill.abs().max()==0)
    check('A: every row records the actual horizon',a.actual_horizon_median_s.notna().all())
    rep=a[(a.train_config=='baseline_replica')]; check('A: replica uses min(6000, pool) training windows',(rep.train_n==np.minimum(6000,rep.pool_n)).all())
    full=a[a.train_config=='N=all']; check('A: N=all uses the whole pool',(full.train_n==full.pool_n).all())
    check('A: test intervals present for learned models',a[(a.split=='test')&(a.model!='Persistence')&(a.train_config=='N=all')].skill_ci_low.notna().all())
    ap_=pd.read_csv(V/'A_paired.csv'); asum=pd.read_csv(V/'A_summary.csv')
    gap=ap_[ap_.machine=='macro'].set_index(['horizon_s','model','target']).skill_difference
    wide=asum.pivot_table(index=['horizon_s','model','target'],columns='train_config',values='macro_skill')
    check('A: paired difference equals the gap between the two summary rows',np.allclose(gap.reindex(wide.index).to_numpy(),(wide['N=all']-wide['baseline_replica']).to_numpy()))
    sets=ve.machine_sets(60,['TBM8'])['TBM8']; pool=sets['pool']
    small,large=vc.draw(pool,500,0).index,vc.draw(pool,2000,0).index
    check('A: subsets are nested for one seed',set(small)<=set(large) and len(small)==500 and len(large)==2000)
    check('A: different seeds give different subsets',set(vc.draw(pool,500,0).index)!=set(vc.draw(pool,500,1).index))
    b=pd.read_csv(V/'B_selected.csv'); cand=pd.read_csv(V/'B_candidates.csv')
    s=b[b.rule=='SIMPLE'].set_index(['machine','horizon_s','target']); al=b[b.rule=='ALL'].set_index(['machine','horizon_s','target'])
    check('B: ALL selection never has worse validation MAE than SIMPLE',(al.val_MAE<=s.val_MAE+1e-12).all())
    check('B: SIMPLE never selects a learned model',not s.selected.str.contains('@').any())
    persist=cand[cand.candidate=='Persistence'].set_index(['machine','horizon_s','target']).val_MAE
    check('B: selected validation MAE <= persistence validation MAE',(s.val_MAE<=persist.reindex(s.index)+1e-12).all())
    pred1,picks1=ve.selected_simple(sets,60)
    corrupted=dict(sets); corrupted['test']=sets['test'].assign(y_F=1e9,y_T=1e9,y_v=1e9)
    pred2,picks2=ve.selected_simple(corrupted,60)
    check('B: selection is unchanged when test labels are destroyed',picks1==picks2 and np.allclose(pred1,pred2))
    c=pd.read_csv(V/'C_strata.csv'); check('C: three-way states observed',set(c.state)<= set(vc.STATES) and len(set(c.state))>=2)
    chk=c.groupby(['machine','horizon_s','model','target']).apply(lambda g:abs(g.contribution_to_skill.sum()-g.overall_skill.iloc[0])).max()
    check('C: state contributions add up to the overall skill',chk<1e-9)
    base=c[c.model=='Persistence']; check('C: persistence contributes nothing in every state',base.contribution_to_skill.abs().max()<1e-12)
    d=pd.read_csv(V/'D_training_sets.csv')
    check('D: leave-one-machine-out never trains on the target machine',not any(r.target_machine in [x.split(':')[0] for x in r.sources.split(',')] for r in d.itertuples() if r.variant.startswith('lomo')))
    check('D: time-causal variant trains only before the target cutoff',d[d.variant=='lomo_time_causal'].all_train_before_target_cutoff.all())
    partners={'TBM5':{'TBM6'},'TBM6':{'TBM5'},'TBM9':{'TBM10'},'TBM10':{'TBM9'}}
    np_=d[d.variant=='lomo_no_overlap_partner']
    check('D: no-overlap-partner variant leaves out the overlapping machine',len(np_)>0 and all(not ({x.split(':')[0] for x in r.sources.split(',')}&partners.get(r.target_machine,set())) for r in np_.itertuples()))
    art=pd.read_parquet(vc.ART/'C_TBM8_60s_test.parquet'); pred_cols=[c for c in art if c.startswith('pred_')]
    check('C: per-window artifact keeps oracle columns apart from the model features','oracle_state' in art and not set(names)&set(c for c in art if c.startswith('oracle_')) and len(pred_cols)>=12)
    dp=pd.read_csv(V/'D_paired.csv'); mm=dp[(dp.machine=='macro')&(dp.variant=='lomo_all')]
    check('D: paired table records training sizes on both sides and a macro interval',{'variant_train_n','within_train_n'}<=set(dp) and len(mm)>0 and mm.ci_low.notna().all())
    w=d[(d.variant=='within')]; check('D: within-machine training uses the target machine only',all(set(x.split(':')[0] for x in s_.split(','))=={t} for s_,t in zip(w.sources,w.target_machine)))
    m=d[d.variant.isin(['within','lomo_all'])&d.N.notna()]
    check('D: matched sizes (within vs lomo_all) agree when sources are large enough',(m.groupby(['horizon_s','target_machine','N','seed']).train_n.nunique()==1).all())
    check('D: history span and sampling interval are recorded',d.train_history_span_median_s.notna().all() and d.test_sampling_median_s.notna().all())
    h=pd.read_csv(V/'H_actual_horizon.csv'); check('horizon: excess within the 5 s tolerance',(h.excess_max<=5).all())
    if with_baseline:
        bm=pd.read_csv(vc.OUT/'baseline_metrics.csv'); t=bm[(bm.split=='test')&(bm.model!='Persistence')]
        for proto,variant in [('within_machine','within_baseline_replica'),('leave_one_machine_out','lomo_baseline_replica')]:
            x=t[t.protocol==proto].set_index(['machine','horizon_s','model','target']).MAE
            dd=pd.read_csv(V/'D_metrics.csv'); y=dd[(dd.variant==variant)&(dd.split=='test')&(dd.model.isin(['Ridge_delta','HistGB_delta']))].set_index(['machine','horizon_s','model','target']).MAE
            check(f'D: {variant} reproduces the baseline {proto} test MAE',np.allclose(x.reindex(y.index).to_numpy(),y.to_numpy(),rtol=1e-6))
    print('SMOKE TEST','PASSED' if all(ok) else 'FAILED',sum(ok),'/',len(ok),flush=True)
    return all(ok)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--with-baseline',action='store_true',help='also run the full build_baselines.train() and compare the replicas (minutes)')
    ap.add_argument('--workdir',type=Path,default=None,help='keep the temporary repository here instead of a throw-away directory')
    a=ap.parse_args()
    ctx=tempfile.TemporaryDirectory() if a.workdir is None else None
    root=Path(ctx.name) if ctx else a.workdir
    try:
        (root/'analysis').mkdir(parents=True,exist_ok=True)
        for p in HERE.glob('*.py'): shutil.copy(p,root/'analysis'/p.name)
        build_synthetic(root)
        hook='import sys; sys.path.insert(0,sys.argv[1]); import build_baselines as b; b.prepare(); '+('b.train()' if a.with_baseline else '')
        run([sys.executable,'-c',hook,root/'analysis'])
        run([sys.executable,root/'analysis/validation_experiments.py','all','--ladder','500,2000','--seeds','0','--sizes','1000,3000','--horizons','60'])
        success=checks(root,a.with_baseline)
    finally:
        if ctx: ctx.cleanup()
    sys.exit(0 if success else 1)

if __name__=='__main__': main()
