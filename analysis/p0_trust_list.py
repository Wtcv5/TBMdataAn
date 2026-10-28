"""P0 step 3: duplicates, source verdicts and scale checks per file, compiled into the trusted source lists.
Needs file_provenance.csv, counter_events.csv and row_hashes.npz from the two earlier P0 scripts. Every verdict is a rule
over those tables; the thresholds are the constants below."""
import collections, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'TBMData/TBMData'; OUT=ROOT/'analysis/outputs'; DEST=OUT/'p0'; CACHE=OUT/'p0_cache'
FULL_COPY=0.99          # share of the smaller file's rows found unchanged in another file
PARTIAL_COPY_ROWS=100   # fewer shared rows than this is a coincidence of idle rows
FOREIGN_TOL=3.0         # a row is foreign when its chainage lies inside the other file's range widened by this (metres)
N_SHIFTED,N_MIXED=0.5,0.01   # share of active rows with n above the rpm limit (file_provenance): whole file / part of the file
MIN_ACTIVE_ROWS=20000   # fewer usable rows than this in a stream says nothing about forecasting
PNV_OFF=(0.8,1.25)      # penetration x speed / advance speed must be near 1 in every machine
CLOSURE_MIN_M=3.0       # position counters that advanced less than this say nothing about the speed channel
CLOSURE_OFF=2.0         # advance from the speed channel vs from the position counter: more than this factor apart
V_SPIKE_SHARE,V_INVALID_SHARE=0.002,0.5   # share of boring rows with an impossible advance speed: spikes / the channel is unusable
MINUTE_RESOLUTION=0.5   # share of timestamps on a whole minute
MIN_ROWS_FOR_TIMESTAMP_CHECK=100
POSITION={**{f'TBM{i}':'cutter' for i in range(1,7)},**{f'TBM{i}':'chainage' for i in (8,9,10)}}
PROJECTS={'色季拉山':'色季拉山','孜拉山':'孜拉山','伯舒拉岭':'伯舒拉岭'}
SIGNALS=['F','T','n','v','p']

def hash_pairs(z,prefix):
    keys=[k for k in z.files if k.startswith(prefix)]
    ids=np.concatenate([np.full(len(z[k]),int(k[1:]),dtype=np.int32) for k in keys]); h=np.concatenate([z[k] for k in keys])
    o=np.argsort(h,kind='stable'); h,ids=h[o],ids[o]; start=np.flatnonzero(np.r_[True,h[1:]!=h[:-1]]); end=np.r_[start[1:],len(h)]
    pairs=collections.Counter()
    for r in np.flatnonzero(end-start>1):
        u=np.unique(ids[start[r]:end[r]])
        for i in range(len(u)):
            for j in range(i+1,len(u)): pairs[(int(u[i]),int(u[j]))]+=1
    return pairs

def load_files():
    fa=pd.read_csv(OUT/'file_audit.csv'); fa['file']=fa.file.str.replace('\\','/')
    prov=pd.read_csv(DEST/'file_provenance.csv',parse_dates=['time_min','time_max']).set_index('source_id')
    schemas={s['id']:s for s in json.loads((OUT/'schemas.json').read_text(encoding='utf-8'))}
    f=prov.join(fa[['schema','encoding','F_missing','T_missing','n_missing','v_missing','p_missing']])
    f['directory']=f.file.str.split('/').str[0]; f['name']=f.file.str.split('/').str[1]
    f['stream']=f.directory.str.extract(r'^(TBM\d+-\d{4}(?:-part\d)?)')[0]; f['machine']=f.directory.str.extract(r'^(TBM\d+)')[0]
    f['year']=f.directory.str.extract(r'^TBM\d+-(\d{4})')[0].astype(int)
    f['project']=f.directory.map(lambda d: next((v for k,v in PROJECTS.items() if k in d),'未标明'))
    for k in ['F','T','n','v','p']: f['col_'+k]=f.schema.map(lambda s: schemas[s]['mapping'].get(k))
    f['core_columns_present']=f[['col_F','col_T','col_n','col_v']].notna().all(axis=1)
    return f

def duplicate_table(f,z):
    full=hash_pairs(z,'f'); core=hash_pairs(z,'c'); rows=[]
    for (a,b),m in sorted(full.items()):
        ra,rb=f.rows[a],f.rows[b]; small=min(ra,rb)
        rel='full_copy' if m>=FULL_COPY*small else ('partial_copy' if m>=PARTIAL_COPY_ROWS else 'coincidental')
        rows.append(dict(source_id_a=a,source_id_b=b,file_a=f.file[a],file_b=f.file[b],stream_a=f.stream[a],stream_b=f.stream[b],rows_a=ra,rows_b=rb,
                         full_row_matches=m,core_matches=core.get((a,b),0),share_of_a=m/ra,share_of_b=m/rb,relation=rel))
    return pd.DataFrame(rows)

def copy_owner(f,dups):
    """Among streams that copy each other, the owner is the one whose counters continue the same machine's earlier year."""
    full=dups[dups.relation=='full_copy']; links=collections.defaultdict(set)
    for r in full.itertuples(): links[r.stream_a].add(r.stream_b); links[r.stream_b].add(r.stream_a)
    seen=set(); families=[]
    for s in links:
        if s in seen: continue
        comp={s}; todo=[s]
        while todo:
            for t in links[todo.pop()]:
                if t not in comp: comp.add(t); todo.append(t)
        seen|=comp; families.append(sorted(comp))
    bridges=[]
    for fam in families:
        for s in fam:
            m,y=s.split('-')[0],int(s.split('-')[1]); cur=f[(f.stream==s)].sort_values('time_min'); prev=f[(f.machine==m)&(f.year==y-1)].sort_values('time_min')
            cur=cur.dropna(subset=['刀盘里程_first']); prev=prev.dropna(subset=['刀盘里程_last'])
            jump=days=np.nan
            if len(cur) and len(prev):
                jump=cur['刀盘里程_first'].iloc[0]-prev['刀盘里程_last'].iloc[-1]; days=(cur.time_min.iloc[0]-prev.time_max.iloc[-1]).total_seconds()/86400
            bridges.append(dict(family='+'.join(fam),stream=s,bridge_jump_m=jump,bridge_days=days))
    b=pd.DataFrame(bridges); owners={}
    for fam,g in b.groupby('family'):
        g=g.assign(a=g.bridge_jump_m.abs()).sort_values('a'); ok=g.dropna(subset=['a'])
        owners[fam]=ok.stream.iloc[0] if len(ok) and (len(ok)==1 or ok.a.iloc[0]*100<ok.a.iloc[1]) else None
    b['owner']=b.family.map(owners); return b

def counter_series(path,enc,columns):
    hdr=pd.read_csv(path,encoding=enc,nrows=0).columns; tcol='运行时间-time' if '运行时间-time' in hdr else '时间'
    raw=pd.read_csv(path,encoding=enc,usecols=[tcol]+[c for c in columns if c in hdr],low_memory=False)
    out=pd.DataFrame({'time':pd.to_datetime(raw[tcol],errors='coerce')})
    for c in columns: out[c]=pd.to_numeric(raw[c],errors='coerce') if c in raw else np.nan
    return out

def foreign_rows(f,dups,events):
    """Rows of a file whose chainage lies in the range of a file of another machine that shares exact rows with it."""
    out=[]; unresolved=set(); part=dups[(dups.relation=='partial_copy')&(dups.stream_a.str.split('-').str[0]!=dups.stream_b.str.split('-').str[0])]
    for r in part.itertuples():
        score={}
        for sid in (r.source_id_a,r.source_id_b):
            e=events[((events.source_id==sid)&events.kind.isin(['within_file_shift']))|((events.source_id_prev==sid)&events.kind.isin(['between_files_too_fast','between_files_regress']))]
            score[sid]=len(e)
        bad,good=sorted(score,key=score.get,reverse=True)[:2]
        if score[bad]==score[good]: unresolved.update((r.source_id_a,r.source_id_b)); continue
        c='导向盾首里程'; core=[f['col_'+k][bad] for k in 'FTnv']; db=counter_series(DATA/f.file[bad],f.encoding[bad],[c]+core); dg=counter_series(DATA/f.file[good],f.encoding[good],[c])
        lo,hi=dg[c].where(dg[c]>1000).min()-FOREIGN_TOL,dg[c].where(dg[c]>1000).max()+FOREIGN_TOL
        mask=db[c].between(lo,hi).to_numpy(); idx=np.flatnonzero(mask); active=(db[core]>0).all(axis=1).to_numpy()
        if not len(idx): continue
        out.append(dict(source_id=bad,file=f.file[bad],row_start=int(idx[0]),row_end=int(idx[-1]),rows_excluded=int(mask.sum()),file_rows=len(db),active_rows_excluded=int((active&mask).sum()),active_rows_kept=int((active&~mask).sum()),
                        time_start=db.time.iloc[idx[0]],time_end=db.time.iloc[idx[-1]],reason='foreign_data',belongs_to=f.stream[good],partner_source_id=good,partner_file=f.file[good],
                        shared_exact_rows=int(r.full_row_matches),evidence=f'chainage {db[c].iloc[idx[0]]:.0f}..{db[c].iloc[idx[-1]]:.0f} lies in the range of {f.name[good]} of {f.stream[good]} ({lo+FOREIGN_TOL:.0f}..{hi-FOREIGN_TOL:.0f}); {r.full_row_matches} rows are exact copies'))
    return pd.DataFrame(out),unresolved

def closure_ratio(f):
    """Advance by the position counter: small steps summed (robust to resets, inflated by jitter) and first-to-last reading."""
    key=f.machine.map(POSITION)
    adv=pd.Series([f.loc[i,f'adv_{k}_m'] if f'adv_{k}_m' in f else np.nan for i,k in key.items()],index=f.index)
    net=pd.Series([f.loc[i,f'adv_{k}_net_m'] if f'adv_{k}_net_m' in f else np.nan for i,k in key.items()],index=f.index)
    return adv,net,f.adv_v_m/adv.where(adv>=CLOSURE_MIN_M)

def build():
    f=load_files(); z=np.load(CACHE/'row_hashes.npz'); events=pd.read_csv(DEST/'counter_events.csv')
    dups=duplicate_table(f,z); dups.to_csv(DEST/'duplicate_pairs.csv',index=False,encoding='utf-8-sig')
    bridges=copy_owner(f,dups); bridges.to_csv(DEST/'copy_owner_evidence.csv',index=False,encoding='utf-8-sig')
    excl,unresolved=foreign_rows(f,dups,events); excl.to_csv(DEST/'row_exclusions.csv',index=False,encoding='utf-8-sig')
    # copies: every file of a non-owner stream that has a full copy in the owner stream
    f['copy_of']=None; owner_of=dict(zip(bridges.stream,bridges.owner)); full=dups[dups.relation=='full_copy']
    no_owner={s for s,o in owner_of.items() if o is None}
    for r in full.itertuples():
        for s,t,own in ((r.source_id_a,r.source_id_b,owner_of.get(r.stream_a)),(r.source_id_b,r.source_id_a,owner_of.get(r.stream_b))):
            if own and f.stream[t]==own and f.stream[s]!=own: f.loc[s,'copy_of']=f.file[t]
    f['adv_counter_m'],f['adv_counter_net_m'],f['closure_ratio']=closure_ratio(f)
    f['n_state']=np.where(f.n_gt12_share>N_SHIFTED,'shifted',np.where(f.n_gt12_share>N_MIXED,'mixed',np.where(f.active_rows>0,'ok','not_assessable')))
    f['chk_pnv']=np.where(f.ratio_p_n_over_v.isna(),'na',np.where(f.ratio_p_n_over_v.between(*PNV_OFF),'pass','fail'))
    f['chk_closure']=np.where(f.closure_ratio.isna(),'na',np.where((f.closure_ratio>=CLOSURE_OFF)|(f.closure_ratio<1/CLOSURE_OFF),'off','pass'))
    ev_file=events[events.kind.isin(['within_file_excursion','within_file_return','within_file_shift'])].groupby('source_id').size()
    f['counter_events_in_file']=ev_file.reindex(f.index).fillna(0).astype(int)
    f['timestamps']=np.where((f.second_zero_share>=MINUTE_RESOLUTION)&(f.rows>=MIN_ROWS_FOR_TIMESTAMP_CHECK),'minute_resolution',np.where(f.dup_time_share>=0.05,'duplicated','ok'))
    ex=excl.set_index('source_id') if len(excl) else pd.DataFrame()
    full_ids=set(full.source_id_a)|set(full.source_id_b); flags=[]; status=[]; action=[]
    for sid,r in f.iterrows():
        fl=[]
        if r.copy_of: st,ac='exclude','duplicate_of_owner_stream'; fl.append('duplicate_copy')
        elif r.stream in no_owner and sid in full_ids: st,ac='exclude','duplicate_owner_unresolved'; fl.append('duplicate_unresolved')
        elif len(ex) and sid in ex.index and ex.rows_excluded[sid]>=0.99*ex.file_rows[sid]: st,ac='exclude','foreign_whole_file'; fl.append('foreign_data')
        elif len(ex) and sid in ex.index: st,ac='truncate','drop_rows_row_exclusions'; fl.append('foreign_data')
        elif r.n_state in ('shifted','mixed'): st,ac='repair','n_column_holds_motor_frequency'; fl.append('n_'+r.n_state)
        elif r.v_implausible_share>V_INVALID_SHARE: st,ac='signal_invalid','v_and_p_unusable'; fl.append('v_impossible')
        else: st,ac='trusted','use'
        if V_SPIKE_SHARE<r.v_implausible_share<=V_INVALID_SHARE: fl.append('v_spikes')
        if r.chk_closure=='off' and not r.v_implausible_share>V_SPIKE_SHARE: fl.append('advance_mismatch')
        if sid in unresolved: fl.append('shares_rows_with_other_machine')
        if r.counter_events_in_file: fl.append('counter_glitch')
        if r.timestamps!='ok': fl.append('timestamps_'+r.timestamps)
        if st=='trusted' and fl: st='flagged'
        flags.append(';'.join(fl)); status.append(st); action.append(ac)
    f['status'],f['action'],f['flags']=status,action,flags
    f['rows_usable_as_is']=np.where(f.status.isin(['trusted','flagged']),f.rows,0)
    f['active_rows_usable_as_is']=np.where(f.status.isin(['trusted','flagged']),f.active_rows,0)
    for sid,r in (ex.iterrows() if len(ex) else []):
        whole=r.rows_excluded>=0.99*r.file_rows; f.loc[sid,'rows_usable_as_is']=0 if whole else r.row_start; f.loc[sid,'active_rows_usable_as_is']=r.active_rows_kept
    print('n shift vs penetration x speed check:'); print(pd.crosstab(f.n_state,f.chk_pnv).to_string())
    return f,dups,bridges,excl,events

def machine_table(f,bridges):
    owner_of=dict(zip(bridges.stream,bridges.owner)); rows=[]
    for s,g in f.groupby('stream',sort=False):
        mode=lambda c: g[c].dropna().mode().iloc[0] if g[c].notna().any() else None
        n=len(g); k=lambda st: int((g.status==st).sum()); usable=int(g.active_rows_usable_as_is.sum()); repairable=int(g.loc[g.status=='repair','active_rows'].sum())
        notes=[]
        if k('exclude')==n and owner_of.get(s): verdict='copy_of_other_stream'; notes.append(f'全部{n}个文件是{owner_of[s]}的逐行副本,此数据流没有独立数据')
        else:
            verdict='too_few_active_rows' if usable+repairable<MIN_ACTIVE_ROWS else ('usable_after_repair' if k('repair') else 'usable')
            if k('repair'): notes.append(f'{k("repair")}个文件的转速列n装的是电机频率,修复前不可直接使用')
            if k('signal_invalid'): notes.append(f'{k("signal_invalid")}个文件的推进速度v与贯入度p不可用')
            if k('truncate') or k('exclude'): notes.append(f'{k("truncate")+k("exclude")}个文件含其他机器的数据,已按行排除')
            if int(g.core_columns_present.sum())<n/2: notes.append((f'仅{int(g.core_columns_present.sum())}个文件' if g.core_columns_present.any() else '没有文件')+f'同时有推力、扭矩、转速、推进速度四列,活动行共{int(g.active_rows.sum())}')
            if (g.timestamps!='ok').any(): notes.append(f'{int((g.timestamps!="ok").sum())}个文件时间戳异常({"、".join(sorted(set(g.timestamps)-{"ok"}))})')
        rows.append(dict(stream=s,machine=g.machine.iloc[0],project=g.project.iloc[0],files=n,rows=int(g.rows.sum()),active_rows=int(g.active_rows.sum()),
            time_min=g.time_min.min(),time_max=g.time_max.max(),dt_median_s=g.dt_median_s.median(),
            **{'col_'+k:mode('col_'+k) for k in SIGNALS},core_columns_files=int(g.core_columns_present.sum()),
            T_over_torque_sum=g.ratio_T_over_torque_sum.median(),n_over_motor_freq=g.ratio_n_over_motor_freq.median(),
            **{f'files_{st}':k(st) for st in ['trusted','flagged','repair','signal_invalid','truncate','exclude']},
            rows_usable_as_is=int(g.rows_usable_as_is.sum()),active_rows_usable_as_is=usable,active_rows_after_repair=usable+repairable,verdict=verdict,note=';'.join(notes)))
    return pd.DataFrame(rows)

def unit_evidence(f):
    """Internal consistency per stream: the speed channel against the position counter, penetration x rotation against speed, and the
    fixed multipliers that tie torque and rotation to the individual drive motors. None of these fixes an absolute unit."""
    rows=[]
    for s,g in f.groupby('stream',sort=False):
        clean=g[(g.v_implausible_share.fillna(0)<=V_SPIKE_SHARE)&(g.adv_counter_m>=CLOSURE_MIN_M)]
        q=lambda c,p: g[c].quantile(p) if g[c].notna().any() else np.nan
        rows.append(dict(stream=s,files=len(g),closure_files=len(clean),closure_total_ratio=clean.adv_v_m.sum()/clean.adv_counter_m.sum() if len(clean) else np.nan,
            closure_median=clean.closure_ratio.median(),p_n_over_v_median=g.ratio_p_n_over_v.median(),p_n_over_v_p05=q('ratio_p_n_over_v',.05),p_n_over_v_p95=q('ratio_p_n_over_v',.95),
            n_over_n_set_median=g.ratio_n_over_n_set.median(),T_over_torque_sum_median=g.ratio_T_over_torque_sum.median(),T_over_torque_sum_p05=q('ratio_T_over_torque_sum',.05),T_over_torque_sum_p95=q('ratio_T_over_torque_sum',.95),
            n_over_motor_freq_median=g.ratio_n_over_motor_freq.median(),n_over_motor_freq_p05=q('ratio_n_over_motor_freq',.05),n_over_motor_freq_p95=q('ratio_n_over_motor_freq',.95)))
    return pd.DataFrame(rows)

def main():
    f,dups,bridges,excl,events=build()
    cols=['file','directory','stream','machine','year','project','time_min','time_max','rows','active_rows','status','action','flags','copy_of','n_state','n_gt12_share','ratio_p_n_over_v','ratio_n_over_n_set',
          'ratio_n_over_motor_freq','ratio_T_over_torque_sum','v_implausible_share','closure_ratio','adv_v_m','adv_counter_m','adv_counter_net_m','chk_pnv','chk_closure','counter_events_in_file','timestamps','dt_median_s','dup_time_share',
          'core_columns_present','col_F','col_T','col_n','col_v','col_p','rows_usable_as_is','active_rows_usable_as_is']
    f.reset_index().rename(columns={'index':'source_id'})[['source_id']+cols].to_csv(DEST/'trusted_sources.csv',index=False,encoding='utf-8-sig')
    m=machine_table(f,bridges); m.to_csv(DEST/'trusted_machines.csv',index=False,encoding='utf-8-sig'); unit_evidence(f).to_csv(DEST/'unit_evidence.csv',index=False,encoding='utf-8-sig')
    print(f.status.value_counts().to_string()); print(m[['stream','files','rows','files_trusted','files_flagged','files_repair','files_signal_invalid','files_truncate','files_exclude','active_rows_usable_as_is','active_rows_after_repair']].to_string())

if __name__=='__main__': main()
