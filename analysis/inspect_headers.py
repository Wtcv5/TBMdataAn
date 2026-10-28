from pathlib import Path
import pandas as pd
from collections import Counter
root=Path(__file__).resolve().parents[1]/'TBMData/TBMData'
for d in sorted(root.iterdir()):
    if not d.is_dir(): continue
    variants=Counter()
    for p in d.rglob('tbm_data*.csv'):
        if 'codes' in p.parts or 'common_using' in p.parts: continue
        try: cs=pd.read_csv(p,nrows=0).columns
        except UnicodeError: cs=pd.read_csv(p,nrows=0,encoding='gb18030').columns
        selected=tuple(c for c in cs if any(s in c for s in ['总推','推力','刀盘转','刀盘扭','贯入','推进速度','刀盘实际','刀盘速度','推进压力']))
        variants[selected]+=1
    print(d.name,flush=True)
    for cs,n in variants.items(): print(n,cs,flush=True)
