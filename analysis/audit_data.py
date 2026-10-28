"""Read-only source audit; outputs are confined to analysis/outputs."""
from pathlib import Path
import json, re, sys, platform, time
import numpy as np
import pandas as pd
import openpyxl

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'TBMData/TBMData'
OUT = ROOT / 'analysis/outputs'
CACHE = OUT/'canonical'
FIELDS = ['F','T','n','v','p']
ALIASES = {'time':['运行时间-time','时间'], 'F':['总推进力','总推力','推力'],
           'T':['刀盘转矩','刀盘扭矩','扭矩'], 'n':['刀盘速度','刀盘实际转速','转速'],
           'v':['推进速度','推进速度(mm/M)'], 'p':['贯入度','刀盘贯入度'],
           'n_aux':['刀盘转速']}

def dump(obj,p):
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str),encoding='utf-8')

def summary(df,label):
    t=df.time.dropna(); dt=t.sort_values().drop_duplicates().diff().dt.total_seconds()
    row=dict(source=label,rows=len(df),invalid_time=int(df.time.isna().sum()),
             time_min=str(t.min()),time_max=str(t.max()),duplicate_time_rows=int(t.duplicated().sum()),
             interval_mode_s=float(dt.mode().iloc[0]) if len(dt.mode()) else None,
             interval_median_s=float(dt.median()),gaps_over_60s=int((dt>60).sum()))
    active=(df[['F','T','n','v']]>0).all(axis=1)
    row['positive_four_rows']=int(active.sum())
    for c in FIELDS:
        x=df[c]
        row[c+'_missing']=int(x.isna().sum()); row[c+'_zero']=int(x.eq(0).sum())
        row[c+'_negative']=int(x.lt(0).sum())
        for q in [0,.01,.5,.99,1]: row[f'{c}_q{q:g}']=float(x.quantile(q))
    for c in ['n','n_aux']:
        if c not in df: continue
        ok=(df.v>1)&(df.p>0)&(df[c]>0)
        ratios=(df.loc[ok,'p']*df.loc[ok,c]/df.loc[ok,'v']).replace([np.inf,-np.inf],np.nan).dropna()
        row[c+'_pn_over_v_count']=len(ratios)
        row[c+'_pn_over_v_median']=float(ratios.median())
        row[c+'_pn_over_v_within_10pct']=float(ratios.between(.9,1.1).mean())
    return row

def run():
    CACHE.mkdir(parents=True,exist_ok=True)
    files=[]; dirs=[]; schemas={}; failures=[]
    for d in sorted(DATA.iterdir()):
        if not d.is_dir(): continue
        sources=sorted(p for p in d.rglob('tbm_data*.csv') if 'common_using' not in p.parts and 'codes' not in p.parts)
        frames=[]
        for i,p in enumerate(sources):
            try:
                enc='utf-8-sig'
                try: header=pd.read_csv(p,nrows=0,encoding=enc).columns.tolist()
                except UnicodeError:
                    enc='gb18030'
                    header=pd.read_csv(p,nrows=0,encoding=enc).columns.tolist()
                mapping={k:next((c for c in names if c in header),None) for k,names in ALIASES.items()}
                schema_key=json.dumps(header,ensure_ascii=False)
                if schema_key not in schemas: schemas[schema_key]=dict(id=len(schemas),columns=header,mapping=mapping)
                sid=schemas[schema_key]['id']
                raw=pd.read_csv(p,usecols=[c for c in mapping.values() if c],encoding=enc,low_memory=False)
                df=pd.DataFrame(index=raw.index)
                df['time']=pd.to_datetime(raw[mapping['time']],errors='coerce')
                for k,c in mapping.items():
                    if k!='time': df[k]=pd.to_numeric(raw[c],errors='coerce') if c else np.nan
                df.replace([np.inf,-np.inf],np.nan,inplace=True)
                stat=dict(file=str(p.relative_to(DATA)),bytes=p.stat().st_size,encoding=enc,schema=sid,rows=len(df),
                          time_min=str(df.time.min()),time_max=str(df.time.max()),invalid_time=int(df.time.isna().sum()),
                          duplicate_time_rows=int(df.time.duplicated().sum()),backward_steps=int(df.time.diff().dt.total_seconds().lt(0).sum()),
                          missing_fields=','.join(k for k in FIELDS if mapping[k] is None))
                for c in FIELDS: stat[c+'_missing']=int(df[c].isna().sum())
                df['source_id']=len(files); files.append(stat); frames.append(df)
            except Exception as e:
                failures.append(dict(file=str(p.relative_to(DATA)),error=str(e)[:1000]))
            if i%50==0: print(d.name,i+1,'/',len(sources),flush=True)
        if frames:
            df=pd.concat(frames,ignore_index=True)
            df['directory']=d.name
            df.to_parquet(CACHE/(d.name+'.parquet'),index=False)
            dirs.append(summary(df,d.name))
            print('DONE',d.name,len(df),flush=True)
            pd.DataFrame(files).to_csv(OUT/'file_audit.csv',index=False,encoding='utf-8-sig')
            pd.DataFrame(dirs).to_csv(OUT/'directory_audit.csv',index=False,encoding='utf-8-sig')
    dump(list(schemas.values()),OUT/'schemas.json'); dump(failures,OUT/'read_failures.json')
    # Workbook has no explicit machine identifier: audit it but do not train on it.
    for p in DATA.glob('*.xlsx'):
        if (CACHE/'unassigned_workbook.parquet').exists() and (OUT/'workbook_audit.json').exists(): continue
        print('Reading workbook',p.name,flush=True)
        w=openpyxl.load_workbook(p,read_only=True,data_only=True)
        s=w.worksheets[0]; it=s.iter_rows(values_only=True); header=next(it)
        rows=list(it); w.close()
        raw=pd.DataFrame(rows,columns=header); del rows
        df=pd.DataFrame({'time':pd.to_datetime(raw['时间'],errors='coerce')})
        for k,c in dict(F='推力',T='扭矩',n='转速',v='推进速度',p='贯入度').items():
            df[k]=pd.to_numeric(raw[c],errors='coerce')
        df.to_parquet(CACHE/'unassigned_workbook.parquet',index=False)
        dump(summary(df,p.name),OUT/'workbook_audit.json')
    dump(dict(python=sys.version,platform=platform.platform(),pandas=pd.__version__,numpy=np.__version__,
              aliases=ALIASES,raw_units='unconfirmed',machine_identity='TBM number confirmed by user',
              source_policy='tbm_data*.csv only, excluding codes and common_using; workbook audited separately',
              csv_files=len(files),csv_bytes=sum(x['bytes'] for x in files)),OUT/'audit_config.json')

if __name__=='__main__': run()
