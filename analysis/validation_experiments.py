"""Minimal validation experiments on top of the frozen baseline: horizon check, A sample size, B simple baselines,
C future-state strata (post-hoc), D fair cross-machine comparison. Outputs go to analysis/outputs/validation."""
import argparse, re, time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from validation_common import (OUT, ART, TARGETS, BASE, LABEL, LEARNED, MAX_TRAIN, STATES, cap, feature_names, save_rows, machine_sets,
                               draw, fit_models, predict, eval_rows, day_codes, Boot, clean_series, future_state, overlap_partners)

def log(*a): print(*a,flush=True)

def show(df,cols=None,digits=3):
    with pd.option_context('display.width',250,'display.max_columns',40,'display.float_format',lambda x:f'{x:.{digits}f}'):
        print(df[cols] if cols else df,flush=True)

# ---- time labels --------------------------------------------------------------------------------------------------

def run_horizon(a):
    """Actual forecast interval per machine/horizon/split, and a strict-horizon sensitivity of the baseline models."""
    names=feature_names(); rows=[]; strict=[]
    for h in a.horizons:
        for p in sorted((OUT/'datasets').glob(f'*_{h}s.parquet')):
            mid=p.name.split('_')[0]
            if a.machines and mid not in a.machines: continue
            f=pd.read_parquet(p); f['excess']=f.actual_horizon_s-h
            for split,g in f.groupby('split'):
                r=dict(machine=mid,horizon_s=h,split=split,n=len(g),actual_median_s=g.actual_horizon_s.median(),
                       actual_mean_s=g.actual_horizon_s.mean(),excess_p05=g.excess.quantile(.05),excess_p95=g.excess.quantile(.95),
                       excess_max=g.excess.max(),sampling_median_s=g.age_lag1_s.median(),sampling_mean_s=g.age_lag1_s.mean(),
                       history_span_median_s=g.age_lag3_s.median())
                for lo,hi in [(0,0),(1,2),(3,5)]:
                    m=g[(g.excess>=lo)&(g.excess<=hi)]; tag=f'{lo}' if lo==hi else f'{lo}-{hi}'
                    r[f'share_excess_{tag}s']=len(m)/len(g)
                    for c in TARGETS: r[f'persistence_MAE_{c}_excess_{tag}s']=float(np.abs(m['y_'+c]-m[c+'_lag0']).mean()) if len(m) else np.nan
                rows.append(r)
        for mid,s in machine_sets(h,a.machines).items():
            models=fit_models(cap(s['pool'],MAX_TRAIN),names,('Ridge_delta','HistGB_delta'))
            f=s['test']; preds=predict(models,f,names); excess=(f.actual_horizon_s-h).to_numpy()
            for tag,mask in [('all',excess>=0),('excess<=1s',excess<=1),('excess==0s',excess==0)]:
                if mask.sum()<50: continue
                strict+=eval_rows(f[mask],{k:v[mask] for k,v in preds.items()},dict(machine=mid,horizon_s=h,split='test',subset=tag))
        log('horizon check done',h)
    out=save_rows(rows,'H_actual_horizon.csv'); sens=save_rows(strict,'H_strict_sensitivity.csv')
    show(out[out.split=='test'],['machine','horizon_s','n','actual_median_s','actual_mean_s','excess_p95','excess_max','sampling_median_s','sampling_mean_s','history_span_median_s','share_excess_0s'])
    if len(sens):
        show(sens[(sens.model!='Persistence')&(sens.target=='T')].pivot_table(index=['machine','horizon_s','model'],columns='subset',values=['skill','n']),digits=3)

# ---- experiment A: training-set size ------------------------------------------------------------------------------

def run_a(a):
    names=feature_names(); rows=[]; boot=Boot(); setsby={}
    for h in a.horizons:
        sets=setsby[h]=machine_sets(h,a.machines)
        for mid,s in sets.items():
            pool,test=s['pool'],s['test']; codes=day_codes(test); ref=np.abs(test[LABEL].to_numpy()-test[BASE].to_numpy())
            configs=[('baseline_replica',cap(pool,MAX_TRAIN),None)]
            for n in a.ladder:
                if n<len(pool): configs+=[(f'N={n}',draw(pool,n,sd),sd) for sd in a.seeds]
            configs.append(('N=all',pool,None))
            for tag,fit,seed in configs:
                start=time.perf_counter(); models=fit_models(fit,names)
                for split in ['validation','test']:
                    f=s[split]; preds=predict(models,f,names)
                    rows+=eval_rows(f,preds,dict(experiment='A',machine=mid,horizon_s=h,split=split,train_config=tag,train_seed=seed,config_key=f'{tag}#{seed}',
                                                 train_n=len(fit),pool_n=len(pool),train_days=int(fit.origin.dt.floor('D').nunique())))
                    if split=='test':
                        for k,c in enumerate(TARGETS):
                            boot.add((h,c),mid,codes,ref[:,k],{f'{tag}#{seed}#{m}':np.abs(f['y_'+c].to_numpy()-p[:,k]) for m,p in preds.items()})
                log('A',h,mid,tag,seed,'train',len(fit),'seconds',round(time.perf_counter()-start,1))
    df=pd.DataFrame(rows)
    # paired comparison with the baseline's own 6,000-window fit, same test windows and same day-block resamples
    point=df[(df.split=='test')&(df.model!='Persistence')].pivot_table(index=['horizon_s','model','target','machine'],columns='config_key',values='skill')
    pairs=[]
    for h in a.horizons:
        mids=list(setsby[h])
        for model in LEARNED:
            for c in TARGETS:
                diffs=[]
                for mid in mids:
                    d=float(point.loc[(h,model,c,mid),'N=all#None']-point.loc[(h,model,c,mid),'baseline_replica#None']); diffs.append(d)
                    lo,hi=boot.diff_ci((h,c),f'N=all#None#{model}',f'baseline_replica#None#{model}',mid)
                    pairs.append(dict(horizon_s=h,model=model,target=c,machine=mid,comparison='N=all minus baseline_replica',skill_difference=d,ci_low=lo,ci_high=hi))
                (lo,hi),k=boot.macro_ci((h,c),f'N=all#None#{model}',mids,f'baseline_replica#None#{model}')
                pairs.append(dict(horizon_s=h,model=model,target=c,machine='macro',comparison='N=all minus baseline_replica',skill_difference=float(np.mean(diffs)),ci_low=lo,ci_high=hi,machines_in_ci=k))
    # attach per-row day-block intervals for the test split
    lo=[];hi=[]
    for r in df.itertuples():
        if r.split=='test' and r.model!='Persistence': l,u=boot.machine_ci((r.horizon_s,r.target),f'{r.config_key}#{r.model}',r.machine)
        else: l=u=np.nan
        lo.append(l);hi.append(u)
    df['skill_ci_low']=lo; df['skill_ci_high']=hi
    save_rows(df,'A_metrics.csv')
    t=df[(df.split=='test')&(df.model!='Persistence')].copy()
    per_machine=t.groupby(['horizon_s','model','target','train_config','machine'],dropna=False).agg(skill=('skill','mean'),train_n=('train_n','mean')).reset_index()
    summary=per_machine.groupby(['horizon_s','model','target','train_config']).agg(machines=('machine','nunique'),macro_skill=('skill','mean'),
            positive_machines=('skill',lambda x:int((x>0).sum())),mean_train_n=('train_n','mean')).reset_index()
    save_rows(summary,'A_summary.csv'); paired=save_rows(pairs,'A_paired.csv')
    order={'baseline_replica':0,'N=1500':1,'N=6000':2,'N=24000':3,'N=all':9}
    summary['o']=summary.train_config.map(lambda x:order.get(x,5))
    show(summary.sort_values(['horizon_s','model','target','o']).drop(columns='o'),['horizon_s','model','target','train_config','mean_train_n','macro_skill','positive_machines'])
    show(paired[paired.machine=='macro'],['horizon_s','model','target','skill_difference','ci_low','ci_high'])

# ---- experiment B: simple baselines with validation-only selection ------------------------------------------------

def simple_candidates(f,h,centers):
    """Simple forecasts from the four-observation history; grids are fixed here and chosen only on validation windows."""
    n=len(f); t=np.column_stack([np.zeros(n),-f.age_lag1_s,-f.age_lag2_s,-f.age_lag3_s]); out={}
    series=[np.column_stack([f[f'{c}_lag{j}'] for j in range(4)]) for c in TARGETS]
    def put(name,fn): out[name]=np.maximum(0,np.column_stack([fn(x,k) for k,x in enumerate(series)]))
    put('Persistence',lambda x,k:x[:,0])
    for m in (2,3,4): put(f'MA{m}',lambda x,k,m=m:x[:,:m].mean(axis=1))
    for m in (3,4): put(f'MED{m}',lambda x,k,m=m:np.median(x[:,:m],axis=1))
    for al in (.2,.4,.6,.8):
        w=np.array([al,al*(1-al),al*(1-al)**2,(1-al)**3]); put(f'ES{al}',lambda x,k,w=w:x@w)
    for m in (2,3,4):
        for lam in (.1,.25,.5,1.):
            def lle(x,k,m=m,lam=lam):
                tk=t[:,:m]; xk=x[:,:m]; tm=tk.mean(axis=1); xm=xk.mean(axis=1); tc=tk-tm[:,None]; sxx=(tc**2).sum(axis=1)
                b=np.divide((tc*(xk-xm[:,None])).sum(axis=1),sxx,out=np.zeros(n),where=sxx>0)
                return xm-b*tm+lam*b*h        # line fitted to the last m observations, evaluated at origin+h; slope shrunk by lam
            put(f'LLE{m}_{lam}',lle)
    for w in (.05,.1,.2,.35,.5): put(f'SHRINK{w}',lambda x,k,w=w:(1-w)*x[:,0]+w*centers[k])
    return out

def choose(val_mae,allowed):
    """Selection sees validation scores only; ties go to the earlier candidate."""
    return min(allowed,key=lambda nm:(val_mae[nm],allowed.index(nm)))

def family(name):
    return name.split('@')[0] if '@' in name else re.match(r'[A-Za-z]+',name)[0]

def selected_simple(s,h):
    centers=s['pool'][LABEL].median().to_numpy()
    val,test=simple_candidates(s['validation'],h,centers),simple_candidates(s['test'],h,centers); picks=[]; pred=np.zeros((len(s['test']),3))
    for k,c in enumerate(TARGETS):
        vm={nm:float(np.abs(s['validation']['y_'+c].to_numpy()-p[:,k]).mean()) for nm,p in val.items()}
        pick=choose(vm,list(val)); picks.append(pick); pred[:,k]=test[pick][:,k]
    return pred,picks

def run_b(a):
    names=feature_names(); cand=[]; sel=[]; boot=Boot()
    for h in a.horizons:
        sets=machine_sets(h,a.machines)
        for mid,s in sets.items():
            pool=s['pool']; centers=pool[LABEL].median().to_numpy(); P={}
            for split in ['validation','test']: P[split]=simple_candidates(s[split],h,centers)
            simple_names=list(P['test'])
            for tag,fit in [('@6000',cap(pool,MAX_TRAIN)),('@all',pool)]:
                models=fit_models(fit,names)
                for split in ['validation','test']:
                    for m,p in predict(models,s[split],names).items(): P[split][m+tag]=p
            all_names=list(P['test']); codes=day_codes(s['test']); ref=np.abs(s['test'][LABEL].to_numpy()-s['test'][BASE].to_numpy())
            for k,c in enumerate(TARGETS):
                yv,yt=s['validation']['y_'+c].to_numpy(),s['test']['y_'+c].to_numpy()
                vm={nm:float(np.abs(yv-P['validation'][nm][:,k]).mean()) for nm in all_names}
                tm={nm:float(np.abs(yt-P['test'][nm][:,k]).mean()) for nm in all_names}
                for nm in all_names:
                    cand.append(dict(machine=mid,horizon_s=h,target=c,candidate=nm,family=family(nm),val_MAE=vm[nm],test_MAE=tm[nm],
                                     val_skill=1-vm[nm]/vm['Persistence'],test_skill=1-tm[nm]/tm['Persistence']))
                picks={'SIMPLE':choose(vm,simple_names),'ALL':choose(vm,all_names)}
                errs={'Persistence':ref[:,k]}
                for rule,pick in picks.items():
                    errs[rule]=np.abs(yt-P['test'][pick][:,k])
                    sel.append(dict(machine=mid,horizon_s=h,target=c,rule=rule,selected=pick,val_MAE=vm[pick],test_MAE=tm[pick],
                                    test_skill=1-tm[pick]/tm['Persistence'],oracle_best_test_MAE=min(tm[nm] for nm in (simple_names if rule=='SIMPLE' else all_names)),
                                    n_test=len(yt),actual_horizon_median_s=float(s['test'].actual_horizon_s.median())))
                for nm in all_names:
                    if '@' in nm: errs[nm]=np.abs(yt-P['test'][nm][:,k])
                boot.add((h,c),mid,codes,ref[:,k],errs)
            log('B',h,mid,'candidates',len(all_names))
    sel=pd.DataFrame(sel)
    sel['skill_ci_low']=[boot.machine_ci((r.horizon_s,r.target),r.rule,r.machine)[0] for r in sel.itertuples()]
    sel['skill_ci_high']=[boot.machine_ci((r.horizon_s,r.target),r.rule,r.machine)[1] for r in sel.itertuples()]
    save_rows(cand,'B_candidates.csv'); save_rows(sel,'B_selected.csv')
    macro=[]
    for h in a.horizons:
        for c in TARGETS:
            mids=sorted(sel[(sel.horizon_s==h)&(sel.target==c)].machine.unique())
            if not mids: continue
            for rule in ['SIMPLE','ALL']:
                x=sel[(sel.horizon_s==h)&(sel.target==c)&(sel.rule==rule)]
                (lo,hi),k=boot.macro_ci((h,c),rule,mids)
                macro.append(dict(horizon_s=h,target=c,rule=rule,machines=len(x),macro_test_skill=x.test_skill.mean(),positive_machines=int((x.test_skill>0).sum()),ci_low=lo,ci_high=hi))
            (lo,hi),k=boot.macro_ci((h,c),'ALL',mids,'SIMPLE')
            x=sel[(sel.horizon_s==h)&(sel.target==c)].pivot(index='machine',columns='rule',values='test_skill')
            macro.append(dict(horizon_s=h,target=c,rule='ALL minus SIMPLE',machines=len(x),macro_test_skill=(x.ALL-x.SIMPLE).mean(),positive_machines=int((x.ALL>x.SIMPLE).sum()),ci_low=lo,ci_high=hi))
    macro=save_rows(macro,'B_macro.csv')
    show(macro,['horizon_s','target','rule','machines','macro_test_skill','positive_machines','ci_low','ci_high'])
    pick_counts=sel.groupby(['rule','selected']).size().rename('times_selected').reset_index().sort_values(['rule','times_selected'],ascending=[True,False])
    show(pick_counts.groupby('rule').head(8))

# ---- experiment C: error attribution by future state --------------------------------------------------------------

def run_c(a):
    names=feature_names(); rows=[]; bins=[]
    for h in a.horizons:
        sets=machine_sets(h,a.machines)
        for mid,s in sets.items():
            f=s['test']; pool=s['pool']; series=clean_series(mid); st=future_state(f,series)
            y=f[LABEL].to_numpy(); base=f[BASE].to_numpy()
            preds={'Persistence':base}
            for tag,fit in [('@6000',cap(pool,MAX_TRAIN)),('@all',pool)]:
                for m,p in predict(fit_models(fit,names),f,names).items(): preds[m+tag]=p
            preds['SIMPLE_selected'],picks=selected_simple(s,h)
            ART.mkdir(exist_ok=True)       # per-window records for tracing individual error events; derived, not committed
            art=pd.concat([f[['origin','target_time','actual_horizon_s','directory']+BASE+LABEL],st],axis=1)
            for model,p in preds.items():
                for k,c in enumerate(TARGETS): art[f'pred_{model}_{c}']=p[:,k]
            art.to_parquet(ART/f'C_{mid}_{h}s_test.parquet',index=False)
            e_pers=np.abs(y-base)
            for model,p in preds.items():
                e=np.abs(y-p)
                for k,c in enumerate(TARGETS):
                    total_ref=e_pers[:,k].sum()
                    for state in STATES:
                        m=(st.oracle_state==state).to_numpy()
                        if not m.any(): continue
                        rows.append(dict(machine=mid,horizon_s=h,model=model,target=c,state=state,n=int(m.sum()),share_windows=m.mean(),
                                         MAE=e[m,k].mean(),persistence_MAE=e_pers[m,k].mean(),share_of_model_error=e[m,k].sum()/e[:,k].sum(),
                                         skill_in_state=1-e[m,k].sum()/e_pers[m,k].sum() if e_pers[m,k].sum()>0 else np.nan,
                                         contribution_to_skill=(e_pers[m,k].sum()-e[m,k].sum())/total_ref,
                                         overall_skill=1-e[:,k].sum()/total_ref,actual_horizon_median_s=float(f.actual_horizon_s.median())))
                    rel=np.abs(y[:,k]-base[:,k])/base[:,k]
                    for lo,hi in [(0,.1),(.1,.25),(.25,.5),(.5,1),(1,np.inf)]:
                        m=(rel>=lo)&(rel<hi)
                        if m.any(): bins.append(dict(machine=mid,horizon_s=h,model=model,target=c,realized_relative_change=f'[{lo},{hi})',n=int(m.sum()),share_windows=m.mean(),
                                                     contribution_to_skill=(e_pers[m,k].sum()-e[m,k].sum())/total_ref,overall_skill=1-e[:,k].sum()/total_ref))
            log('C',h,mid,'states',st.oracle_state.value_counts().to_dict(),'simple picks',picks)
    df=pd.DataFrame(rows)
    chk=df.groupby(['machine','horizon_s','model','target']).apply(lambda g:abs(g.contribution_to_skill.sum()-g.overall_skill.iloc[0])).max()
    assert chk<1e-9,f'state contributions do not add up to the overall skill ({chk})'
    save_rows(df,'C_strata.csv'); save_rows(bins,'C_realized_change_bins.csv')
    macro=df[df.model!='Persistence'].groupby(['horizon_s','model','target','state']).agg(machines=('machine','nunique'),share_windows=('share_windows','mean'),
            skill_in_state=('skill_in_state','mean'),contribution_to_skill=('contribution_to_skill','mean'),overall_skill=('overall_skill','mean')).reset_index()
    save_rows(macro,'C_macro.csv')
    show(macro[macro.model.isin(['HistGB_delta@6000','Ridge_delta@6000','SIMPLE_selected'])],['horizon_s','model','target','state','share_windows','skill_in_state','contribution_to_skill','overall_skill'])

# ---- experiment D: fair same-machine vs leave-one-machine-out ------------------------------------------------------

def allocate(avail,total):
    """Equal split of `total` windows over sources, capped by what each source has; the remainder goes to sources with spare windows."""
    left=total; out={}; order=sorted(avail,key=lambda m:avail[m])
    for i,m in enumerate(order):
        out[m]=min(avail[m],left//(len(order)-i)); left-=out[m]
    return out

def compose(pools,total,seed,cutoff=None):
    pools={m:(p if cutoff is None else p[p.target_time<cutoff]) for m,p in pools.items()}
    pools={m:p for m,p in pools.items() if len(p)}
    if not pools: return None,{}
    quota=allocate({m:len(p) for m,p in pools.items()},total)
    parts=[draw(pools[m],quota[m],seed+i) for i,m in enumerate(sorted(pools)) if quota[m]>0]
    return pd.concat(parts,ignore_index=True),quota

def run_d(a):
    names=feature_names(); manifest=pd.read_csv(OUT/'split_manifest.csv').set_index('machine'); rows=[]; comp=[]; boot=Boot(); partners=overlap_partners()
    for h in a.horizons:
        sets=machine_sets(h,a.machines); spacing={m:float(s['pool'].age_lag1_s.median()) for m,s in sets.items()}
        for tgt,ts in sets.items():
            others={m:s['pool'] for m,s in sets.items() if m!=tgt}; same={m:p for m,p in others.items() if abs(spacing[m]-spacing[tgt])<=2}
            unrelated={m:p for m,p in others.items() if m not in partners.get(tgt,set())}
            cutoff=pd.Timestamp(manifest.loc[tgt,'train_end_exclusive']); test=ts['test']; codes=day_codes(test)
            ref=np.abs(test[LABEL].to_numpy()-test[BASE].to_numpy())
            jobs=[('within_baseline_replica',None,None,cap(ts['pool'],MAX_TRAIN),{tgt:min(len(ts['pool']),MAX_TRAIN)}),
                  ('lomo_baseline_replica',None,None,pd.concat([cap(p,MAX_TRAIN) for p in others.values()],ignore_index=True),{m:min(len(p),MAX_TRAIN) for m,p in others.items()})]
            for n in a.sizes:
                for sd in a.seeds:
                    fit=draw(ts['pool'],n,sd); jobs.append(('within',n,sd,fit,{tgt:len(fit)}))
                    for name,pools,cut in [('lomo_all',others,None),('lomo_same_sampling',same,None),('lomo_time_causal',others,cutoff),
                                           ('lomo_no_overlap_partner',unrelated,None)]:
                        fit,quota=compose(pools,n,sd*101,cut)
                        if fit is not None: jobs.append((name,n,sd,fit,quota))
            for variant,n,sd,fit,quota in jobs:
                start=time.perf_counter(); models=fit_models(fit,names)
                comp.append(dict(horizon_s=h,target_machine=tgt,variant=variant,N=n,seed=sd,train_n=len(fit),sources=','.join(f'{m}:{q}' for m,q in quota.items()),
                                 train_target_time_min=fit.target_time.min(),train_target_time_max=fit.target_time.max(),
                                 train_history_span_median_s=float(fit.age_lag3_s.median()),test_history_span_median_s=float(test.age_lag3_s.median()),
                                 train_sampling_median_s=float(fit.age_lag1_s.median()),test_sampling_median_s=spacing[tgt],
                                 target_train_end=cutoff,all_train_before_target_cutoff=bool((fit.target_time<cutoff).all())))
                for split in ['validation','test']:
                    f=ts[split]; preds=predict(models,f,names)
                    rows+=eval_rows(f,preds,dict(experiment='D',machine=tgt,horizon_s=h,split=split,variant=variant,N=n,seed=sd,train_n=len(fit)))
                    if split=='test':
                        for k,c in enumerate(TARGETS):
                            boot.add((h,c),tgt,codes,ref[:,k],{f'{variant}#{n}#{sd}#{m}':np.abs(f['y_'+c].to_numpy()-p[:,k]) for m,p in preds.items()})
                log('D',h,tgt,variant,n,sd,'train',len(fit),'seconds',round(time.perf_counter()-start,1))
    df=pd.DataFrame(rows); save_rows(comp,'D_training_sets.csv')
    paired=[]
    for h in a.horizons:
        mids=sorted(df[df.horizon_s==h].machine.unique())
        for model in LEARNED:
            for c in TARGETS:
                for variant in ['lomo_all','lomo_same_sampling','lomo_time_causal','lomo_no_overlap_partner']:
                    for n in a.sizes:
                        for sd in a.seeds:
                            per=[]
                            for mid in mids:
                                lo,hi=boot.diff_ci((h,c),f'{variant}#{n}#{sd}#{model}',f'within#{n}#{sd}#{model}',mid)
                                x=df[(df.machine==mid)&(df.horizon_s==h)&(df.model==model)&(df.target==c)&(df.split=='test')&(df.N==n)&(df.seed==sd)]
                                sv=x[x.variant==variant]; sw=x[x.variant=='within']
                                if len(sv)==1 and len(sw)==1: per.append((mid,float(sv.skill.iloc[0]-sw.skill.iloc[0]),lo,hi,int(sv.train_n.iloc[0]),int(sw.train_n.iloc[0])))
                            for mid,d,lo,hi,tv,tw in per:
                                paired.append(dict(horizon_s=h,model=model,target=c,variant=variant,N=n,seed=sd,machine=mid,skill_difference=d,ci_low=lo,ci_high=hi,variant_train_n=tv,within_train_n=tw))
                            if per:
                                (lo,hi),k=boot.macro_ci((h,c),f'{variant}#{n}#{sd}#{model}',[p[0] for p in per],f'within#{n}#{sd}#{model}')
                                paired.append(dict(horizon_s=h,model=model,target=c,variant=variant,N=n,seed=sd,machine='macro',skill_difference=float(np.mean([p[1] for p in per])),
                                                   ci_low=lo,ci_high=hi,machines_in_ci=k,variant_train_n=float(np.mean([p[4] for p in per])),within_train_n=float(np.mean([p[5] for p in per]))))
    paired=save_rows(paired,'D_paired.csv'); save_rows(df,'D_metrics.csv')
    # model-selection rule, identical for every protocol: pick by the target machine's validation MAE, report test
    sel=[]
    for (mach,h,var,n,sd,c),g in df.groupby(['machine','horizon_s','variant','N','seed','target'],dropna=False):
        v=g[g.split=='validation'].set_index('model').MAE; t=g[g.split=='test'].set_index('model')
        pick=choose(v.to_dict(),['Persistence']+LEARNED)
        sel.append(dict(machine=mach,horizon_s=h,variant=var,N=n,seed=sd,target=c,selected=pick,test_skill=t.loc[pick,'skill']))
    save_rows(sel,'D_selected.csv')
    t=df[(df.split=='test')&(df.model!='Persistence')&df.N.notna()]
    macro=t.groupby(['horizon_s','model','target','variant','N']).agg(machines=('machine','nunique'),macro_skill=('skill','mean'),positive_machines=('skill',lambda x:int((x>0).sum())),mean_train_n=('train_n','mean')).reset_index()
    save_rows(macro,'D_macro.csv')
    show(paired[paired.machine=='macro'].groupby(['horizon_s','model','target','variant','N']).agg(skill_difference=('skill_difference','mean'),ci_low=('ci_low','first'),ci_high=('ci_high','first')).reset_index())

# ---- command line -------------------------------------------------------------------------------------------------

def ints(text): return [int(x) for x in text.split(',') if x]

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('experiment',choices=['horizon','A','B','C','D','all'])
    ap.add_argument('--horizons',type=ints,default=[30,60,120]); ap.add_argument('--machines',type=lambda s:[x for x in s.split(',') if x],default=None)
    ap.add_argument('--ladder',type=ints,default=[1500,6000,24000],help='experiment A training sizes below the full pool')
    ap.add_argument('--seeds',type=ints,default=None,help='training-subset seeds (A default 0,1,2; D default 0,1)')
    ap.add_argument('--sizes',type=ints,default=[6000,24000],help='experiment D matched training sizes')
    a=ap.parse_args(); seeds=a.seeds
    with threadpool_limits(limits=4):
        for name,fn,default in [('horizon',run_horizon,None),('A',run_a,[0,1,2]),('B',run_b,None),('C',run_c,None),('D',run_d,[0,1])]:
            if a.experiment in (name,'all'):
                a.seeds=seeds if seeds is not None else default; start=time.perf_counter(); fn(a); log('FINISHED',name,round(time.perf_counter()-start,1),'s')

if __name__=='__main__': main()
