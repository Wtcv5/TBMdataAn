from pathlib import Path
import pandas as pd,numpy as np,re,json
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'analysis/outputs'
core=['F','T','n','v']; chunks=[]
for p in (OUT/'canonical').glob('TBM*.parquet'):
    df=pd.read_parquet(p); a=df[(df[core]>0).all(axis=1)]
    h=pd.util.hash_pandas_object(a[['time']+core].astype({c:'float64' for c in core}),index=False)
    z=pd.DataFrame({'hash':h.to_numpy(),'directory':p.stem,'source_id':a.source_id.to_numpy()})
    chunks.append(z)
allhash=pd.concat(chunks,ignore_index=True)
dups=allhash[allhash.hash.duplicated(keep=False)]
pairs=dups.merge(dups,on='hash',suffixes=('_a','_b'))
pairs=pairs[pairs.directory_a<pairs.directory_b]
pairs=pairs[[re.match(r'TBM\d+',a)[0]!=re.match(r'TBM\d+',b)[0] for a,b in zip(pairs.directory_a,pairs.directory_b)]]
counts=allhash.groupby('directory').size().to_dict()
records=[]
for (a,b),g in pairs.groupby(['directory_a','directory_b']):
    records.append(dict(a=a,b=b,matches=g.hash.nunique(),a_active=counts[a],b_active=counts[b],
                        fraction_a=g.hash.nunique()/counts[a],fraction_b=g.hash.nunique()/counts[b]))
pd.DataFrame(records).to_csv(OUT/'directory_overlap.csv',index=False,encoding='utf-8-sig')
files=pd.read_csv(OUT/'file_audit.csv'); filematches=[]
for (a,b),g in pairs.groupby(['source_id_a','source_id_b']):
    fa=files.iloc[a]; fb=files.iloc[b]
    filematches.append(dict(file_a=fa.file,file_b=fb.file,matches=g.hash.nunique(),source_id_a=a,source_id_b=b))
pd.DataFrame(filematches).to_csv(OUT/'file_overlap.csv',index=False,encoding='utf-8-sig')
print(pd.DataFrame(records).to_string(index=False))
print('matching source file pairs',len(filematches))
