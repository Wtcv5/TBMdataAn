"""Scoring of the P2 review sheet: reads one or more sheets filled in by reviewers (outputs/segments_v1/review/sample.csv with the answer columns
q1_boring ... q5_end_start filled) and reports, overall, by machine and by sampling stratum, how the answers fall (Wilson 95 % intervals), plus a
population-weighted point estimate (every sampled segment stands for its stratum of the machine) and, for two or more sheets, the agreement between
reviewers. Answers: q1 y/p/n; q2 to q5 ok/early/late; unclear and na are counted separately and left out of the shares."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from p2_review import ANSWERS, stratum
from p2_segment import DEST, finish

ALLOWED=dict(q1_boring={'y','p','n','unclear'},**{q:{'ok','early','late','unclear','na'} for q in ANSWERS[1:]})
Z=1.96

def wilson(k,n):
    if n==0: return np.nan,np.nan
    p=k/n; d=1+Z*Z/n; c=(p+Z*Z/(2*n))/d; h=Z*np.sqrt(p*(1-p)/n+Z*Z/(4*n*n))/d; return max(0.,c-h),min(1.,c+h)

def load(path):
    s=pd.read_csv(path,dtype=str).fillna('')
    for q in ANSWERS: s[q]=s[q].str.strip().str.lower()
    bad=[(r.sample_id,q,r[q]) for _,r in s.iterrows() for q in ANSWERS if r[q] not in ALLOWED[q]|{''}]
    if bad: raise SystemExit(f'{path}: answers outside the allowed words (sample, question, answer): {bad[:10]}')
    return s

def table(s,weights):
    rows=[]
    groups=[('overall','all',s)]+[('machine',m,g) for m,g in s.groupby('machine',sort=False)]+[('stratum',k,g) for k,g in s.groupby('stratum',sort=False)]
    for gt,gn,g in groups:
        for q in ANSWERS:
            a=g[q]; asked=int((a!='').sum()); good='y' if q=='q1_boring' else 'ok'; judged=a[a.isin(['y','p','n'] if q=='q1_boring' else ['ok','early','late'])]; k=int((judged==good).sum()); lo,hi=wilson(k,len(judged))
            w=g.loc[judged.index,'weight'] if len(judged) else pd.Series(dtype=float)
            rows.append(dict(group_type=gt,group=gn,question=q,answered=asked,judged=len(judged),good=k,share_good=k/len(judged) if len(judged) else np.nan,ci_low=lo,ci_high=hi,
                             weighted_share_good=float(w[judged==good].sum()/w.sum()) if len(judged) and w.sum()>0 else np.nan,**{v:int((a==v).sum()) for v in ('y','p','n','ok','early','late','unclear','na')}))
    return pd.DataFrame(rows)

def agreement(sheets):
    rows=[]
    for i in range(len(sheets)):
        for j in range(i+1,len(sheets)):
            m=sheets[i].merge(sheets[j],on='sample_id',suffixes=('_a','_b'))
            for q in ANSWERS:
                ok=m[(m[q+'_a'].isin(['y','p','n','ok','early','late']))&(m[q+'_b'].isin(['y','p','n','ok','early','late']))]
                rows.append(dict(reviewer_a=i+1,reviewer_b=j+1,question=q,both_judged=len(ok),same_answer=float((ok[q+'_a']==ok[q+'_b']).mean()) if len(ok) else np.nan))
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--sheet',nargs='+',required=True); ap.add_argument('--dest',default=str(DEST)); ap.add_argument('--out',default=''); a=ap.parse_args()
    seg=pd.read_csv(Path(a.dest)/'segments.csv'); seg['stratum']=seg.apply(stratum,axis=1); pop=seg.groupby(['machine','stratum']).size()
    sheets=[load(p) for p in a.sheet]; outs=[]
    for k,s in enumerate(sheets):
        n=s.groupby(['machine','stratum']).size(); s['weight']=[pop.get((r.machine,r.stratum),np.nan)/n[(r.machine,r.stratum)] for r in s.itertuples()]
        t=table(s,None); t.insert(0,'sheet',a.sheet[k]); outs.append(t)
    res=pd.concat(outs,ignore_index=True); out=Path(a.out) if a.out else Path(a.sheet[0]).with_name('review_scores.csv'); res.round(4).to_csv(out,index=False,encoding='utf-8-sig')
    pd.set_option('display.width',250); pd.set_option('display.max_columns',30)
    print(res[res.group_type!='machine'].round(3).drop(columns=['sheet','group_type']).to_string(index=False))
    if len(sheets)>1: ag=agreement(sheets); ag.round(3).to_csv(out.with_name('review_agreement.csv'),index=False,encoding='utf-8-sig'); print(ag.round(3).to_string(index=False))
    print('written',out)

if __name__=='__main__': main(); finish()
