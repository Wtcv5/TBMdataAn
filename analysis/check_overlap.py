"""Check active time-aligned overlaps across machine-labelled exports and workbook."""
from pathlib import Path
import json,re
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'analysis/outputs'
core=['F','T','n','v']
def run():
    rows=[]; overlaps=[]; hashes=[]; wb=pd.read_parquet(OUT/'canonical/unassigned_workbook.parquet')
    sample=wb[(wb[core]>0).all(axis=1)].iloc[::10].copy()
    wh=pd.util.hash_pandas_object(sample[['time']+core].astype({c:'float64' for c in core}),index=False)
    for p in (OUT/'canonical').glob('TBM*.parquet'):
        mid=re.match(r'TBM\d+',p.name)[0]; df=pd.read_parquet(p)
        active=df[(df[core]>0).all(axis=1)].copy()
        h=pd.util.hash_pandas_object(active[['time']+core].astype({c:'float64' for c in core}),index=False)
        hashes.append(pd.DataFrame({'hash':h.to_numpy(),'machine':mid,'directory':p.stem}))
        rows.append(dict(directory=p.stem,active_rows=len(active),workbook_sample_matches=int(wh.isin(h).sum()),workbook_sample_rows=len(wh)))
    allhash=pd.concat(hashes,ignore_index=True).drop_duplicates(['hash','machine'])
    dup=allhash[allhash.hash.duplicated(keep=False)]
    for (a,b),g in dup.merge(dup,on='hash',suffixes=('_a','_b')).query('machine_a < machine_b').groupby(['machine_a','machine_b']):
        overlaps.append(dict(machine_a=a,machine_b=b,matching_active_timestamp_signal_rows=len(g)))
    pd.DataFrame(rows).to_csv(OUT/'workbook_overlap.csv',index=False,encoding='utf-8-sig')
    (OUT/'cross_machine_overlap.json').write_text(json.dumps(overlaps,indent=2),encoding='utf-8')
    print('cross_machine_overlap',overlaps)
    print(pd.DataFrame(rows).query('workbook_sample_matches>0').to_string(index=False))

if __name__=='__main__': run()
