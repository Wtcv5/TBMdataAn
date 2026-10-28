"""Self-test for the P1 standardisation. Uses the SYNTHETIC raw exports of smoke_test_p0 (a copied stream, another machine's rows inside a
file, a cutterhead-speed column that holds the motor frequency, garbage and spiky advance speed, counter excursions, minute timestamps),
runs the real audit, the P0 scripts, p1_standardize.py and p1_verify.py on them, and checks that every planted problem ends up in the
right flag. Nothing here is a research result and nothing is written to the real analysis/outputs."""
import argparse, shutil, subprocess, sys, tempfile
from pathlib import Path
import pandas as pd
import smoke_test_p0 as p0

HERE=Path(__file__).resolve().parent
SCRIPTS=p0.SCRIPTS+['p1_standardize.py','p1_verify.py']

def checks(root,D,info):
    P=root/'analysis/outputs/standard_v1'; ok=[]
    def check(name,cond):
        print(('PASS ' if cond else 'FAIL ')+name,flush=True); ok.append(bool(cond))
    t=pd.read_csv(root/'analysis/outputs/p0/trusted_sources.csv'); frames={m:pd.read_parquet(P/f'{m}.parquet') for m in ['TBM5','TBM6','TBM8','TBM9','TBM10']}
    sid=lambda d,name: int(t[(t.directory==D[d])&t.file.str.endswith('/'+name)].source_id.iloc[0]); rows=lambda m,s: frames[m][frames[m].source_id==s].sort_values('row')
    check('copies are absent: no TBM2 file, no TBM6-2024 stream',not (P/'TBM2.parquet').exists() and set(frames['TBM6'].stream.astype(str))=={'TBM6-2023'})
    check('TBM5 keeps both years in one time-sorted series',set(frames['TBM5'].stream.astype(str))=={'TBM5-2023','TBM5-2024'} and frames['TBM5'].time.is_monotonic_increasing)
    cut=rows('TBM9',sid('t9',info['t9'][4])); check('truncated TBM9 file stops before the foreign rows',len(cut)==info['cut'] and cut.row.max()==info['cut']-1)
    check('the TBM9 file that is entirely TBM10 data is gone',sid('t9',info['t9'][7]) not in set(frames['TBM9'].source_id))
    sh=rows('TBM9',sid('t9',info['t9'][9])); act=sh[sh.n_raw>12]
    check('shifted file: n is repaired from v/p, n_raw keeps the exported value',len(act)>500 and (act.n_source.astype(str)=='repaired_v_over_p').mean()>.99 and act.n.between(5.1,5.7).all() and (act.n_raw>50).all())
    check('shifted file: repaired rows are active and valid',bool(act.active.mean()>.99) and bool(act.valid_core.mean()>.99))
    idle=sh[(sh.n_raw==0)]; check('shifted file: idle rows keep n = 0 as exported',len(idle)>0 and (idle.n==0).all() and (idle.n_source.astype(str)=='exported').all())
    mixed=rows('TBM9',sid('t9',info['t9'][11])); boring=mixed[mixed.v>0]
    check('mixed file: only the planted alternate rows are marked',bool(((boring.row%2==1)==boring.q_n_invalid).all()) and boring.q_n_invalid.any())
    ok_file=rows('TBM9',sid('t9',info['t9'][0])); check('healthy TBM9 file: nothing repaired, no flag',(ok_file.n_source.astype(str)=='exported').all() and not ok_file[[f for f in ['q_n_invalid','q_v_high','q_neg','q_counter_glitch']]].any().any() and ok_file.valid_core.all())
    bad=rows('TBM10',sid('t10',info['t10'][1])); check('garbage advance speed: whole file carries q_v_invalid_file and is not valid_core',bad.q_v_invalid_file.all() and not bad.valid_core.any() and (bad.file_status.astype(str)=='signal_invalid').all())
    sp=rows('TBM10',sid('t10',info['t10'][2])); check('spiky file: exactly the planted rows have q_v_high',int(sp.q_v_high.sum())==int((sp.v==1500).sum())>=3 and bool((sp.v[sp.q_v_high]==1500).all()))
    ex=rows('TBM10',sid('t10',info['t10'][3])); g=ex[ex.q_counter_glitch]
    check('counter excursion: exactly rows 1000 to 1299 are marked',len(g)==300 and g.row.min()==1000 and g.row.max()==1299)
    mi=rows('TBM10',sid('t10',info['t10'][5])); check('minute timestamps: whole file is q_time_coarse and repeated stamps are conflicts',mi.q_time_coarse.all() and mi.q_time_conflict.any())
    check('every machine series: first dt_prev_s empty, the others not negative',all(d.dt_prev_s.iloc[0] != d.dt_prev_s.iloc[0] and (d.dt_prev_s.iloc[1:]>=0).all() for d in frames.values()))
    check('TBM8 (falling chainage, TBM8 state column name) is clean and uses its own boring flag',frames['TBM8'].valid_core.all() and frames['TBM8'].state_boring.notna().all() and not frames['TBM8'][['q_counter_glitch','q_n_invalid']].any().any())
    check('field dictionary names every column of the data',set(pd.read_csv(P/'field_dictionary.csv').name)>=set(frames['TBM9'].columns))
    return ok

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--keep',action='store_true'); a=ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)/'repo'; (root/'analysis').mkdir(parents=True)
        for s in SCRIPTS: shutil.copy(HERE/s,root/'analysis'/s)
        text=(root/'analysis/p0_trust_list.py').read_text(encoding='utf-8'); assert 'MIN_ACTIVE_ROWS=20000' in text
        (root/'analysis/p0_trust_list.py').write_text(text.replace('MIN_ACTIVE_ROWS=20000','MIN_ACTIVE_ROWS=2000'),encoding='utf-8')
        D,info=p0.build(root)
        for s in SCRIPTS[:-1]: p0.run([sys.executable,s],cwd=root/'analysis')
        verified=subprocess.run([sys.executable,'p1_verify.py'],cwd=root/'analysis').returncode==0
        ok=checks(root,D,info); print(('PASS ' if verified else 'FAIL ')+'p1_verify.py passes on the synthetic data'); ok.append(verified)
        first=(root/'analysis/outputs/standard_v1/manifest.json').read_text(encoding='utf-8'); p0.run([sys.executable,'p1_standardize.py'],cwd=root/'analysis',stdout=subprocess.DEVNULL)
        again=(root/'analysis/outputs/standard_v1/manifest.json').read_text(encoding='utf-8'); import json
        same=all(json.loads(first)['machines'][m]['checksums']==json.loads(again)['machines'][m]['checksums'] for m in json.loads(first)['machines'])
        print(('PASS ' if same else 'FAIL ')+'a second build reproduces every column checksum'); ok.append(same)
        if a.keep: shutil.copytree(root,HERE.parent/'p1_smoke_repo_kept')
    print(f'{sum(ok)}/{len(ok)} checks passed'); sys.exit(0 if all(ok) else 1)

if __name__=='__main__': main()
