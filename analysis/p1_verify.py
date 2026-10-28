"""P1 verification (read-only): the standard time series against its manifest, the P0 list, the raw exports and the audit's canonical caches.
Every check prints PASS or FAIL; the exit status is 1 if any fails. Results go to outputs/standard_v1/verification.json."""
import json, sys
import numpy as np
import pandas as pd
from audit_data import ALIASES
from p1_standardize import (DEST, DATA, OUT, P0, VALUE_COLUMNS, N_RPM_LIMIT, V_IMPLAUSIBLE, T_IMPLAUSIBLE, F_IMPLAUSIBLE, checksums, sha256, STATE_ALIASES)

SPOT_FILES=60; SPOT_ROWS=25; SEED=20240101

def same(a,b):
    a,b=np.asarray(a,float),np.asarray(b,float); return bool(np.all((a==b)|(np.isnan(a)&np.isnan(b))))

def main():
    results=[]
    def check(name,cond,detail=''):
        print(('PASS ' if cond else 'FAIL ')+name+(f'  [{detail}]' if detail else ''),flush=True); results.append(dict(check=name,passed=bool(cond),detail=str(detail)))
    manifest=json.loads((DEST/'manifest.json').read_text(encoding='utf-8')); mach=manifest['machines']
    frames={m:pd.read_parquet(DEST/v['file']) for m,v in mach.items()}
    t=pd.read_csv(P0/'trusted_sources.csv'); excl=pd.read_csv(P0/'row_exclusions.csv'); acct=pd.read_csv(DEST/'row_accounting.csv')
    check('P0 inputs are the ones the data was built from',all(sha256(P0/n)==h for n,h in manifest['p0_inputs_sha256'].items()))
    for m,d in frames.items():        # the column checksums decide; the bytes of a parquet file can differ between pyarrow versions
        v=mach[m]; same_bytes=sha256(DEST/v['file'])==v['sha256']
        check(f'{m}: row count and column checksums match the manifest',len(d)==v['rows'] and
              all(abs(c.get('sum',0)-v['checksums'][k].get('sum',0))<=1e-9*max(1,abs(c.get('sum',0))) and c.get('nan',0)==v['checksums'][k].get('nan',0) and c.get('nunique',0)==v['checksums'][k].get('nunique',0) for k,c in checksums(d).items()),
              'file bytes identical' if same_bytes else 'file bytes differ from the manifest')
    all_rows=sum(len(d) for d in frames.values())
    check('every file reconciles: raw = P0-excluded + invalid time + exact duplicates + kept',(acct.rows_raw==acct.rows_p0_excluded+acct.rows_invalid_time+acct.rows_exact_duplicate_dropped+acct.rows_kept).all(),f'raw {acct.rows_raw.sum()}')
    kept=pd.concat([d[['source_id']] for d in frames.values()]).groupby('source_id').size()
    check('kept rows per file equal the rows found in the data',(acct.set_index('source_id').rows_kept.reindex(kept.index)==kept).all() and acct.rows_kept.sum()==all_rows,f'{all_rows} rows')
    present=set().union(*[set(d.source_id.unique()) for d in frames.values()])
    check('no row comes from a file P0 excluded',present.isdisjoint(set(t[t.status=='exclude'].source_id)))
    check('files absent from the data are exactly the excluded ones',set(t.source_id)-present==set(t[t.status=='exclude'].source_id))
    cut_ok=all(frames[t.machine[t.source_id==r.source_id].iloc[0]].query('source_id==@r.source_id').row.max()<r.row_start for r in excl.itertuples() if r.rows_excluded<0.99*r.file_rows)
    check('truncated files keep only the rows before the cut',cut_ok)
    for m,d in frames.items():
        key=np.lexsort((d.row.to_numpy(),d.source_id.to_numpy(),d.time.to_numpy())); check(f'{m}: sorted by time, then file, then row',(key==np.arange(len(d))).all())
        dt=d.time.diff().dt.total_seconds().astype('float32'); check(f'{m}: dt_prev_s is the time difference to the previous row',same(dt,d.dt_prev_s))
        check(f'{m}: q_time_conflict marks exactly the repeated timestamps',(d.time.duplicated(keep=False)==d.q_time_conflict).all())
        check(f'{m}: no row repeats another one completely',not d.duplicated(['time']+VALUE_COLUMNS).any())
    for m,d in frames.items():
        ok=((d.q_neg==(d[['F','T','n_raw','v','p']]<0).any(axis=1)).all() and (d.q_core_missing==d[['F','T','n','v']].isna().any(axis=1)).all() and (d.q_v_high==(d.v>V_IMPLAUSIBLE)).all()
            and (d.q_T_high==(d['T']>T_IMPLAUSIBLE)).all() and (d.q_F_high==(d.F>F_IMPLAUSIBLE)).all() and (d.active==(d[['F','T','n','v']]>0).all(axis=1)).all()
            and (d.valid_core==~d[['q_neg','q_core_missing','q_v_high','q_T_high','q_F_high','q_v_invalid_file','q_time_conflict']].any(axis=1)).all())
        check(f'{m}: value flags, active and valid_core follow from the stored columns',ok)
        src=d.n_source.astype(str); rep=src.isin(['repaired_v_over_p','repaired_drive_speed','unrecoverable'])
        vp=(src=='repaired_v_over_p'); ok=(d.q_n_invalid==rep).all() and d.n[src.isin(['exported','missing'])].equals(d.n_raw[src.isin(['exported','missing'])]) and d.n[src=='unrecoverable'].isna().all()
        ok=ok and (d.n[vp]-d.v[vp]/d.p[vp]).abs().max()<=1e-9 if vp.any() else ok
        ok=ok and (d.n[src=='exported'].dropna()<=N_RPM_LIMIT).all() and (d.n[src=='repaired_drive_speed']<=N_RPM_LIMIT).all()
        check(f'{m}: cutterhead speed repair is consistent with n_source',ok,f"{int(rep.sum())} rows repaired or unrecoverable")
        check(f'{m}: rows of signal_invalid files carry q_v_invalid_file',(d.q_v_invalid_file==(d.file_status.astype(str)=='signal_invalid')).all())
    def active_keys(d):
        a=d[d.active]; k=pd.DataFrame({'t':a.time.astype('int64').to_numpy(),**{c:a[c].astype('float64').to_numpy() for c in ('F','T','n_raw','v')}})
        return set(pd.util.hash_pandas_object(k,index=False).tolist())
    keys={m:active_keys(d) for m,d in frames.items()}
    overlap=sum(len(keys[a]&keys[b]) for i,a in enumerate(keys) for b in list(keys)[i+1:])
    check('no active reading (time, F, T, n, v) appears in the data of two machines',overlap==0,f'{overlap} shared')
    # spot check against the raw exports with a separate read of each file
    rng=np.random.default_rng(SEED); allsrc=pd.concat([d[['source_id']].assign(machine=m) for m,d in frames.items()]); files=rng.choice(allsrc.source_id.unique(),size=min(SPOT_FILES,allsrc.source_id.nunique()),replace=False)
    fa=pd.read_csv(OUT/'file_audit.csv'); fa['file']=fa.file.str.replace('\\','/'); bad=0; compared=0
    for sid in files:
        d=frames[allsrc.machine[allsrc.source_id==sid].iloc[0]]; d=d[d.source_id==sid]; pick=d.iloc[rng.choice(len(d),size=min(SPOT_ROWS,len(d)),replace=False)]
        header=pd.read_csv(DATA/fa.file[sid],encoding=fa.encoding[sid],nrows=0).columns
        raw=pd.read_csv(DATA/fa.file[sid],encoding=fa.encoding[sid],low_memory=False); r=raw.iloc[pick.row.to_numpy()]
        for k,col in {**{k:next((c for c in ALIASES[k] if c in header),None) for k in ('F','T','v','p')},'n_raw':next((c for c in ALIASES['n'] if c in header),None),
                      'state_boring':next((c for c in STATE_ALIASES['state_boring'] if c in header),None)}.items():
            x=pd.to_numeric(r[col],errors='coerce').to_numpy(float) if col else np.full(len(r),np.nan); compared+=len(x)
            if not same(x,pick[k].astype('float64').to_numpy()): bad+=1
        tcol=next(c for c in ALIASES['time'] if c in header); bad+=int(not (pd.to_datetime(r[tcol],errors='coerce').to_numpy()==pick.time.to_numpy()).all())
    check('rows re-read from the raw exports match (time, F, T, v, p, exported n, boring state)',bad==0,f'{len(files)} files, {compared} values')
    # the audit's canonical caches hold the same values for the rows that were kept
    canon=OUT/'canonical'; mism=0; checked=0
    if canon.exists():
        for m,d in frames.items():
            for p in sorted(canon.glob(f'{m}-*.parquet')):
                c=pd.read_parquet(p,columns=['time','F','T','n','v','p','source_id']); c['row']=c.groupby('source_id').cumcount().astype('int32')
                j=d.merge(c,on=['source_id','row'],suffixes=('','_c')); checked+=len(j)
                mism+=int(not (same(j.F,j.F_c) and same(j['T'],j.T_c) and same(j.v,j.v_c) and same(j.p,j.p_c) and same(j.n_raw,j.n_c) and (j.time==j.time_c).all()))
        check('values equal the audit\'s canonical caches for every kept row',mism==0 and checked>0,f'{checked} rows compared')
    else: print('SKIP canonical caches not found')
    n_ok=sum(r['passed'] for r in results); (DEST/'verification.json').write_text(json.dumps(dict(passed=n_ok,total=len(results),all_passed=n_ok==len(results),checks=results),ensure_ascii=False,indent=1),encoding='utf-8')
    print(f'{n_ok}/{len(results)} checks passed'); sys.exit(0 if n_ok==len(results) else 1)

if __name__=='__main__': main()
