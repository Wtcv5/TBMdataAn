"""Self-test for the P0 source checks. Writes SYNTHETIC raw exports into a temporary repository copy, runs the real audit and the
three P0 scripts on them and checks that planted problems are found and clean files are left alone. The planted problems mirror
what was found in the real data: a stream copied into other directories, another machine's rows inside a file, a cutterhead-speed
column that holds the motor frequency, an advance-speed channel that is garbage, spikes, counter glitches and minute timestamps.
Nothing here is a research result and nothing is written to the real analysis/outputs."""
import argparse, shutil, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
SCRIPTS=['audit_data.py','p0_file_provenance.py','p0_counter_events.py','p0_trust_list.py']
KIND={'K1':dict(k_T=.3686,k_n=.0027,motor_T='主驱动电机{}扭矩',motor_f='主驱动电机{}速度'),'K2':dict(k_T=.3249,k_n=.061,motor_T='M{}电机扭矩',motor_f='M{}电机频率')}
ROWS,STEP,MOTORS=3000,10,6

def simulate(rng,start,kind,pos0,sign=1.0,ring0=0.0):
    """One day-file of boring/idle blocks; every counter advances with the integral of the advance speed, as a healthy machine's does."""
    cfg=KIND[kind]; t=pd.Timestamp(start)+pd.to_timedelta(np.arange(ROWS)*STEP,unit='s'); run=np.zeros(ROWS,bool); i=0; state=bool(rng.random()<.6)
    while i<ROWS: j=min(ROWS,i+int(rng.integers(60,240))); run[i:j]=state; state=not state; i=j
    n=np.where(run,5.4+.05*rng.standard_normal(ROWS),0.); v=np.where(run,np.clip(35+4*rng.standard_normal(ROWS),5,None),0.)
    p=np.where(run,v/np.where(n>0,n,1)*(1 if kind=='K2' else 1+.01*rng.standard_normal(ROWS)),0.)
    F=np.where(run,25000+2000*rng.standard_normal(ROWS),0.); T=np.where(run,4000+300*rng.standard_normal(ROWS),0.)
    advance=np.cumsum(v*STEP/60/1000); df=pd.DataFrame({'运行时间-time':t.strftime('%Y-%m-%d %H:%M:%S'),'掘进状态':run.astype(int),'推进速度':v,'贯入度':p})
    dead=int(rng.integers(0,2))                                         # one drive motor may read zero for the whole file
    tq=np.where(run,T/(cfg['k_T']*(MOTORS-dead)),0.); fq=np.where(run,n/cfg['k_n'],0.)
    for m in range(1,MOTORS+1):
        off=dead and m==MOTORS; df[cfg['motor_T'].format(m)]=0. if off else tq; df[cfg['motor_f'].format(m)]=0. if off else fq
    T=T if not dead else np.where(run,tq*cfg['k_T']*(MOTORS-dead),0.)
    if kind=='K1':
        df['总推进力']=F; df['刀盘转矩']=T; df['刀盘速度']=n; df['刀盘转速']=n/cfg['k_n']; df['刀盘里程']=np.round(pos0+sign*advance,1)
    else:
        df['推力']=F; df['刀盘扭矩']=T; df['刀盘实际转速']=n; df['刀盘给定转速']=n*1.012; df['刀盘给定频率']=n/cfg['k_n']
        df['导向盾首里程']=np.round(pos0+sign*advance); df['导向掘进环数']=ring0+sign*advance
    return df,float(pos0+sign*advance[-1]),float(ring0+sign*advance[-1])

def write(root,directory,name,df):
    d=root/'TBMData/TBMData'/directory; d.mkdir(parents=True,exist_ok=True); df.to_csv(d/name,index=False,encoding='utf-8-sig')

def stream(rng,root,directory,kind,start,days,pos0,sign=1.0,ring0=0.0,gap_days=0,names=None):
    """Consecutive day-files; the counters carry on from file to file. Returns the files so that copies can be made."""
    files=[]; pos,ring=pos0,ring0
    for k in range(days):
        day=pd.Timestamp(start)+pd.Timedelta(days=k*(1+gap_days)); df,pos,ring=simulate(rng,day,kind,pos,sign,ring)
        name=(names[k] if names else f'tbm_data ({k}).csv'); write(root,directory,name,df); files.append((name,df))
        pos+=sign*gap_days*8.; ring+=sign*gap_days*8.
    return files,pos,ring

def build(root,seed=11):
    rng=np.random.default_rng(seed); D=dict(t5a='TBM5-2023-孜拉山进口右线',t5b='TBM5-2024-孜拉山进口右线',t6a='TBM6-2023-孜拉山进口左线',t6b='TBM6-2024-孜拉山进口左线',
                                         t2b='TBM2-2024-色季拉山进口左线',t8='TBM8-2024-2f',t9='TBM9-2023-6f-伯舒拉岭右线',t10='TBM10-2023-6f-伯舒拉岭左线')
    _,pos,_=stream(rng,root,D['t5a'],'K1','2023-04-15',6,-735607.,gap_days=1)
    b,_,_=stream(rng,root,D['t5b'],'K1','2024-01-01',6,pos+2.1,gap_days=1)                  # the same machine, counters carry on across the year
    stream(rng,root,D['t6a'],'K1','2023-04-14',4,735988.,gap_days=1)
    for k,(name,df) in enumerate(b[:5]): write(root,D['t6b'],name,df.assign(备注=''))        # TBM6-2024 is a copy of TBM5-2024 ...
    for k,(name,df) in enumerate(b[1:4]): write(root,D['t2b'],name,df)                         # ... and so is part of TBM2-2024
    stream(rng,root,D['t8'],'K2','2024-03-05',6,950000.,sign=-1.0,gap_days=2)                 # chainage falls while boring: not a fault
    # TBM10 first: its files are the originals that TBM9 files will contain
    t10,_,_=stream(rng,root,D['t10'],'K2','2023-10-10',12,1014100.,ring0=1500.,gap_days=0)
    t9,_,_=stream(rng,root,D['t9'],'K2','2023-10-10',12,1013300.,ring0=1200.,gap_days=0)
    cut=2000                                                                               # TBM9 file 4: rows from `cut` on are TBM10's, with TBM10's own timestamps
    def retime(df,rows): df=df.copy(); df.loc[rows,'运行时间-time']=(pd.to_datetime(df.loc[rows,'运行时间-time'])+pd.Timedelta(seconds=3)).dt.strftime('%Y-%m-%d %H:%M:%S'); return df
    foreign=t10[4][1].iloc[cut:]; mixed=pd.concat([t9[4][1].iloc[:cut],foreign],ignore_index=True); mixed=retime(mixed,mixed.index[cut:][::4])
    write(root,D['t9'],t9[4][0],mixed)                                                     # most foreign rows are exact copies, a quarter are not
    whole=t10[7][1]; write(root,D['t9'],t9[7][0],retime(whole,whole.index[::4]))           # a whole TBM9 file holds TBM10 data
    # TBM9 cutterhead-speed column holds the motor frequency: whole files 9 and 10, alternating rows in file 11
    for k in (9,10):
        df=t9[k][1].copy(); df['刀盘给定频率']=df['刀盘实际转速']*1.012; df['刀盘实际转速']=df['刀盘实际转速']/.061*1.04; df['刀盘给定转速']=30.; write(root,D['t9'],t9[k][0],df)
    df=t9[11][1].copy(); odd=df.index[1::2]; df.loc[odd,'刀盘给定频率']=df.loc[odd,'刀盘实际转速']*1.012; df.loc[odd,'刀盘实际转速']=df.loc[odd,'刀盘实际转速']/.061*1.04; write(root,D['t9'],t9[11][0],df)
    # TBM10: garbage advance speed (file 1), spikes (file 2), counter excursion (file 3), minute timestamps (file 5)
    df=t10[1][1].copy(); boring=df.index[df['推进速度']>0]; bad=rng.choice(boring,int(.7*len(boring)),replace=False); df.loc[bad,'推进速度']*=rng.uniform(40,100,len(bad)); df['贯入度']=df['推进速度']/df['刀盘实际转速'].replace(0,np.nan); write(root,D['t10'],t10[1][0],df.fillna(0))
    df=t10[2][1].copy(); boring=df.index[df['推进速度']>0]; sp=rng.choice(boring,max(3,int(.02*len(boring))),replace=False); df.loc[sp,'推进速度']=1500.; df['贯入度']=df['推进速度']/df['刀盘实际转速'].replace(0,np.nan); write(root,D['t10'],t10[2][0],df.fillna(0))
    df=t10[3][1].copy(); df.loc[df.index[1000:1300],['导向盾首里程','导向掘进环数']]+=90.; write(root,D['t10'],t10[3][0],df)
    df=t10[5][1].copy(); df['运行时间-time']=pd.to_datetime(df['运行时间-time']).dt.floor('min').dt.strftime('%Y-%m-%d %H:%M:%S'); write(root,D['t10'],t10[5][0],df)
    return D,dict(cut=cut,t9=[n for n,_ in t9],t10=[n for n,_ in t10])

def run(cmd,**kw):
    print('$',' '.join(str(c) for c in cmd)[:160],flush=True); subprocess.run([str(c) for c in cmd],check=True,**kw)

def checks(root,D,info):
    P=root/'analysis/outputs/p0'; t=pd.read_csv(P/'trusted_sources.csv'); m=pd.read_csv(P/'trusted_machines.csv'); ex=pd.read_csv(P/'row_exclusions.csv'); own=pd.read_csv(P/'copy_owner_evidence.csv')
    u=pd.read_csv(P/'unit_evidence.csv'); ok=[]
    def check(name,cond):
        print(('PASS ' if cond else 'FAIL ')+name,flush=True); ok.append(bool(cond))
    def rec(d,name): r=t[(t.directory==D[d])&(t.file.str.endswith('/'+name))]; assert len(r)==1,(d,name); return r.iloc[0]
    def stat(d): return t[t.directory==D[d]].status
    def flag(r): return '' if pd.isna(r['flags']) else r['flags']
    check('copy owner of the TBM2/TBM5/TBM6 2024 family is TBM5-2024',set(own.owner)=={'TBM5-2024'} and len(own)==3)
    check('TBM5-2024 (the owner) is fully trusted',(stat('t5b')=='trusted').all())
    check('TBM6-2024 files that copy TBM5-2024 are excluded as duplicates',(t[t.directory==D['t6b']].action=='duplicate_of_owner_stream').all())
    check('TBM2-2024 files that copy TBM5-2024 are excluded as duplicates',(t[t.directory==D['t2b']].action=='duplicate_of_owner_stream').all())
    check('TBM5-2023 and TBM6-2023 are fully trusted',(stat('t5a')=='trusted').all() and (stat('t6a')=='trusted').all())
    check('TBM8 with falling chainage is fully trusted and unflagged',(stat('t8')=='trusted').all() and t[t.directory==D['t8']]['flags'].isna().all())
    f4=rec('t9',info['t9'][4]); check('TBM9 file with TBM10 rows from row 2000 is truncated there',f4.status=='truncate' and int(ex[ex.source_id==f4.source_id].row_start.iloc[0])==info['cut'])
    check('truncated file keeps exactly the rows before the cut',int(f4.rows_usable_as_is)==info['cut'])
    f7=rec('t9',info['t9'][7]); check('TBM9 file that is entirely TBM10 data is excluded',f7.status=='exclude' and f7.action=='foreign_whole_file')
    check('both foreign cases are attributed to TBM10-2023',set(ex.belongs_to)=={'TBM10-2023'} and len(ex)==2)
    check('TBM10 originals of the foreign rows stay trusted',rec('t10',info['t10'][4]).status=='trusted' and rec('t10',info['t10'][7]).status=='trusted')
    check('whole files with the motor frequency in n need repair',all(rec('t9',info['t9'][k]).status=='repair' and rec('t9',info['t9'][k]).n_state=='shifted' for k in (9,10)))
    check('alternating rows are found as a mixed file needing repair',rec('t9',info['t9'][11]).status=='repair' and rec('t9',info['t9'][11]).n_state=='mixed')
    check('healthy TBM9 files are trusted',all(rec('t9',info['t9'][k]).status in ('trusted','flagged') for k in (0,1,2,3,5,6,8)))
    check('garbage advance speed makes the signal invalid',rec('t10',info['t10'][1]).status=='signal_invalid')
    r=rec('t10',info['t10'][2]); check('a few advance-speed spikes are only flagged',r.status=='flagged' and 'v_spikes' in flag(r))
    r=rec('t10',info['t10'][3]); check('a counter excursion is only flagged',r.status=='flagged' and 'counter_glitch' in flag(r))
    r=rec('t10',info['t10'][5]); check('minute-resolution timestamps are flagged',r.status=='flagged' and 'timestamps_minute_resolution' in flag(r))
    clean=t[t.directory.isin([D['t5a'],D['t5b'],D['t8']])]; check('no clean file gets a flag',clean['flags'].isna().all())
    check('clean streams close the advance balance within 5 %',u[u.stream.isin(['TBM5-2023','TBM5-2024','TBM6-2023','TBM8-2024'])].closure_total_ratio.sub(1).abs().max()<.05)
    mv=m.set_index('stream').verdict; check('machine verdicts: copies, repair, usable',mv['TBM6-2024']=='copy_of_other_stream' and mv['TBM2-2024']=='copy_of_other_stream' and mv['TBM9-2023']=='usable_after_repair' and mv['TBM5-2024']=='usable')
    check('every file receives exactly one status',len(t)==t.source_id.nunique() and t.status.notna().all())
    return ok

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--keep',action='store_true'); a=ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)/'repo'; (root/'analysis').mkdir(parents=True)
        for s in SCRIPTS: shutil.copy(HERE/s,root/'analysis'/s)
        text=(root/'analysis/p0_trust_list.py').read_text(encoding='utf-8'); assert 'MIN_ACTIVE_ROWS=20000' in text
        (root/'analysis/p0_trust_list.py').write_text(text.replace('MIN_ACTIVE_ROWS=20000','MIN_ACTIVE_ROWS=2000'),encoding='utf-8')   # the synthetic streams are small
        D,info=build(root)
        for s in SCRIPTS: run([sys.executable,s],cwd=root/'analysis')
        ok=checks(root,D,info)
        run([sys.executable,'p0_trust_list.py'],cwd=root/'analysis',stdout=subprocess.DEVNULL)      # a second run must reproduce the tables
        first=(root/'analysis/outputs/p0/trusted_sources.csv').read_bytes(); run([sys.executable,'p0_trust_list.py'],cwd=root/'analysis',stdout=subprocess.DEVNULL)
        det=first==(root/'analysis/outputs/p0/trusted_sources.csv').read_bytes(); print(('PASS ' if det else 'FAIL ')+'trusted_sources.csv is reproducible'); ok.append(det)
        if a.keep: shutil.copytree(root,HERE.parent/'p0_smoke_repo_kept')
    print(f'{sum(ok)}/{len(ok)} checks passed'); sys.exit(0 if all(ok) else 1)

if __name__=='__main__': main()
