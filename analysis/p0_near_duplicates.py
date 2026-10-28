"""P0 step 6: screen for copies that are not byte-identical. Active rows of every pair of directories that belong to different machines
are joined on the exact timestamp; the share of rows whose thrust, torque and advance speed agree within 0.1 % and the correlation of
each signal are reported. Copies that were shifted in time are not covered. Needs the canonical caches written by audit_data.py."""
import itertools, re
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
CANONICAL=ROOT/'analysis/outputs/canonical'; DEST=ROOT/'analysis/outputs/p0'
CORE=['F','T','n','v']
MIN_JOINED=100          # fewer common timestamps say nothing
TOLERANCE=1e-3          # relative difference counted as equal

def main():
    series={}
    for p in sorted(CANONICAL.glob('TBM*.parquet')):
        df=pd.read_parquet(p,columns=['time']+CORE); df=df[(df[CORE]>0).all(axis=1)&df.time.notna()].drop_duplicates('time').set_index('time')
        series[re.match(r'TBM\d+-\d{4}(?:-part\d)?',p.stem)[0]]=df
    rows=[]
    for a,b in itertools.combinations(sorted(series),2):
        if a.split('-')[0]==b.split('-')[0]: continue                       # same machine
        j=series[a].join(series[b],how='inner',lsuffix='_a',rsuffix='_b')
        if len(j)<MIN_JOINED: continue
        r=dict(stream_a=a,stream_b=b,common_timestamps=len(j))
        for c in ['F','T','v']:
            x,y=j[c+'_a'].to_numpy(float),j[c+'_b'].to_numpy(float)
            r['share_equal_'+c]=float((np.abs(x-y)<=TOLERANCE*np.maximum(np.abs(x),np.abs(y))).mean()); r['corr_'+c]=float(np.corrcoef(x,y)[0,1])
        rows.append(r)
    res=pd.DataFrame(rows).sort_values('share_equal_F',ascending=False); res.to_csv(DEST/'near_duplicate_screen.csv',index=False,encoding='utf-8-sig')
    print(res.round(3).head(12).to_string(index=False))

if __name__=='__main__': main()
