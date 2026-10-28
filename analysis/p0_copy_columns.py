"""P0 step 5: are the copied files copies in every column, not only in the columns used for hashing? (read-only)
For each pair flagged full_copy in duplicate_pairs.csv, both raw files are read as text and every column they share is compared row by row."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'TBMData/TBMData'; OUT=ROOT/'analysis/outputs'; DEST=OUT/'p0'

def compare(args):
    a,b,file_a,file_b,enc_a,enc_b=args
    A=pd.read_csv(DATA/file_a,encoding=enc_a,low_memory=False,dtype=str); B=pd.read_csv(DATA/file_b,encoding=enc_b,low_memory=False,dtype=str)
    common=[c for c in A.columns if c in set(B.columns)]; differing=[]
    if len(A)==len(B):
        for c in common:
            x,y=A[c].reset_index(drop=True),B[c].reset_index(drop=True)
            if not ((x==y)|(x.isna()&y.isna())).all(): differing.append(c)
    return dict(source_id_a=a,source_id_b=b,file_a=file_a,file_b=file_b,rows_a=len(A),rows_b=len(B),columns_a=A.shape[1],columns_b=B.shape[1],shared_columns=len(common),
                only_in_a=A.shape[1]-len(common),only_in_b=B.shape[1]-len(common),shared_columns_differing=len(differing),differing_columns='|'.join(differing))

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--workers',type=int,default=4); a=ap.parse_args()
    fa=pd.read_csv(OUT/'file_audit.csv'); fa['file']=fa.file.str.replace('\\','/'); d=pd.read_csv(DEST/'duplicate_pairs.csv'); d=d[d.relation=='full_copy']
    jobs=[(r.source_id_a,r.source_id_b,r.file_a,r.file_b,fa.encoding[r.source_id_a],fa.encoding[r.source_id_b]) for r in d.itertuples()]
    with ProcessPoolExecutor(a.workers) as ex: res=pd.DataFrame(list(ex.map(compare,jobs,chunksize=2)))
    res.to_csv(DEST/'copy_column_check.csv',index=False,encoding='utf-8-sig')
    same=(res.rows_a==res.rows_b)&(res.shared_columns_differing==0)
    print(f'{len(res)} copied pairs; same row count and every shared column identical in {int(same.sum())}; shared columns {res.shared_columns.min()}..{res.shared_columns.max()}; columns only on one side up to {int(max(res.only_in_a.max(),res.only_in_b.max()))}')

if __name__=='__main__': main()
