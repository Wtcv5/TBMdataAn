"""Self-test for the P2 segmentation. Builds a SYNTHETIC standard time series (two machines, files in the layout of standard_v1) with planted cases: a clean stroke,
a stroke cut by a data hole, a stroke with a pause and a short dip, a stalled stretch, a short stroke, a flicker, a change of level, a slow thrust decay, a
terminal decline, invalid advance speed, a stroke that starts after a hole, strokes at the edges of the series and a machine whose boring flag comes from the thrust state
and the signals. Runs the real p2_segment.py, p2_verify.py, p2_checks.py, p2_review.py and p2_review_score.py on it, checks that every planted case ends up
where it belongs, that the verification fails when the labels or the table are tampered with, and that a second build reproduces every checksum.
Nothing here is a research result and nothing is written to the real analysis/outputs."""
import argparse, json, shutil, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from p2_segment import finish
STEP=10.

def stroke(ramp,steady,end,F0=25000.,v0=40.,T0=3000.,n0=5.5,decay=0.,state=1):
    """Rows of one stroke: linear ramp, noisy plateau (optionally with a slow decay of the thrust), linear terminal decline; the boring flag is on throughout."""
    k=np.arange(1,ramp+1)/max(ramp,1); i=np.arange(steady); j=np.arange(end-1,-1,-1)/max(end,1)
    F=np.r_[F0*k[:ramp],F0*(1-decay*i/max(steady,1))*(1+.01*np.sin(.9*i)),F0*j[:end]]; v=np.r_[v0*k[:ramp],v0*(1+.02*np.sin(1.3*i)),v0*j[:end]]; T=np.r_[T0*k[:ramp],T0*(1+.03*np.sin(.7*i)),T0*j[:end]]
    m=len(F); return pd.DataFrame(dict(F=F,T=T,n=np.full(m,n0),v=v,sb=state,st=pd.NA,valid=True))

def block(m,F=0.,T=0.,n=0.,v=0.,state=0,valid=True):
    return pd.DataFrame(dict(F=np.full(m,float(F)),T=np.full(m,float(T)),n=np.full(m,float(n)),v=np.full(m,float(v)),sb=state,st=pd.NA,valid=valid))

def assemble(parts,machine,flag_columns=('sb',)):
    """parts: list of (name, frame, gap_before_s). Returns the standard-series frame and {name: (first row, last row)}."""
    frames=[]; where={}; row=0; gaps=[]
    for name,f,gap in parts:
        f=f.copy(); where[name]=(row,row+len(f)-1); frames.append(f); gaps+=[gap if gap else STEP]+[STEP]*(len(f)-1); row+=len(f)
    d=pd.concat(frames,ignore_index=True); dt=np.array(gaps,float); dt[0]=np.nan
    t=pd.Timestamp('2024-01-01 00:00:00')+pd.to_timedelta(np.cumsum(np.nan_to_num(dt)),unit='s')
    out=pd.DataFrame(dict(time=t,machine=machine,stream=pd.Categorical([f'{machine}-2024']*len(d)),source_id=np.int16(1),row=np.arange(len(d),dtype=np.int32),F=d.F,T=d['T'],n=d.n,v=d.v,
                          state_boring=pd.array(d.sb,dtype='Int16'),state_thrust=pd.array(d.st,dtype='Int16'),dt_prev_s=dt.astype('float32')))
    out['active']=(out.F>0)&(out['T']>0)&(out.n>0)&(out.v>0); out['valid_core']=d.valid.to_numpy(bool)
    out['chainage']=np.round(np.cumsum(np.clip(d.v,0,None)*np.nan_to_num(dt,nan=0)/60/1000),1); out['ring']=np.nan; out['cutter_mileage']=np.nan; out['q_counter_glitch']=False
    return out,where

def cases():
    """The synthetic K2-like machine (boring flag) with its planted cases, and the K1-like machine (thrust state, boring flag, signals)."""
    pause_a=stroke(12,60,0); pause_a.loc[12+20:12+23,['F','v']]=0.        # a 40 s drop of the thrust inside the stroke
    invalid=stroke(12,100,0); bad=np.arange(len(invalid))%5==0; invalid.loc[bad,'v']=3000.; invalid.loc[bad,'valid']=False
    stalled=block(60,F=30000,T=400,n=2,v=0,state=1); stalled.loc[stalled.index%4==0,'v']=.5          # thrust on, the advance speed reads 0.5 mm/min on every fourth reading
    p8=[('series_start',block(30,F=25000,T=3000,n=5.5,v=40,state=1),0),('i0',block(40),0),
        ('clean',stroke(12,150,0),0),('i1',block(10,T=300,n=5.5),0),('i1b',block(60),0),
        ('hole_a',stroke(12,100,0),0),('hole_b',block(100,F=25000,T=3000,n=5.5,v=40,state=1),300),('i2',block(60),0),
        ('pause_a',pause_a,0),('pause',block(30,T=300,n=5.5,state=1),0),('pause_b',stroke(2,60,6),0),('i3',block(60),0),
        ('stalled',stalled,0),('i4',block(60),0),
        ('short',stroke(5,15,0),0),('i5',block(60),0),('blip',block(3,F=20000,T=3000,n=5,v=30,state=1),0),('i6',block(60),0),
        ('regime_a',stroke(5,40,0,v0=20.,T0=2500.),0),('regime_b',block(100,F=25000,T=2800,n=5.5,v=40,state=1),0),('i7',block(60),0),
        ('decay',stroke(12,200,0,decay=.3),0),('i8',block(60),0),
        ('terminal',stroke(12,150,12),0),('i9',block(60),0),
        ('invalid',invalid,0),('i10',block(60),0),
        ('after_hole',stroke(12,100,0),300),('i11',block(60),0),
        ('series_end',block(30,F=25000,T=3000,n=5.5,v=40,state=1),0)]
    k1_a=stroke(12,100,0); k1_a['sb']=pd.NA; k1_a['st']=3; k1_idle=block(60); k1_idle['sb']=pd.NA; k1_idle['st']=0
    k1_sig=block(5,F=20000,T=3000,n=5,v=30); k1_sig['sb']=pd.NA; k1_sig['st']=pd.NA
    p5=[('idle0',k1_idle,0),('a',k1_a,0),('idle1',k1_idle,0),('b',stroke(12,100,0),0),('idle2',block(60),0),('sig',k1_sig,0),('idle3',block(60),0)]
    return p8,p5

def run(cmd,**kw): return subprocess.run([sys.executable]+cmd,cwd=HERE,**kw)

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--keep',action='store_true'); a=ap.parse_args(); ok=[]
    def check(name,cond):
        print(('PASS ' if cond else 'FAIL ')+name,flush=True); ok.append(bool(cond))
    with tempfile.TemporaryDirectory() as tmp:
        tmp=Path(tmp); src=tmp/'src'; out=tmp/'out'; src.mkdir(); (out/'labels').mkdir(parents=True); p8,p5=cases(); d8,w8=assemble(p8,'TBM8'); d5,w5=assemble(p5,'TBM5')
        d8.to_parquet(src/'TBM8.parquet'); d5.to_parquet(src/'TBM5.parquet'); (src/'manifest.json').write_text(json.dumps(dict(synthetic=True)),encoding='utf-8')
        r=run(['p2_segment.py','--src',str(src),'--dest',str(out),'--machines','TBM5,TBM8','--workers','2'],stdout=subprocess.DEVNULL); check('p2_segment.py runs',r.returncode==0)
        seg=pd.read_csv(out/'segments.csv'); lab8=pd.read_parquet(out/'labels/TBM8.parquet'); s8=seg[seg.machine=='TBM8']; s5=seg[seg.machine=='TBM5']
        t8=d8.time.dt.strftime('%Y-%m-%d %H:%M:%S').to_numpy(); at=lambda name,k=0: s8[s8.start==t8[w8[name][k]]].iloc[0]
        def seg_of(name,machine='TBM8'):
            s=s8 if machine=='TBM8' else s5; tt=t8 if machine=='TBM8' else d5.time.dt.strftime('%Y-%m-%d %H:%M:%S').to_numpy(); w=w8 if machine=='TBM8' else w5
            return s[(s.start>=tt[w[name][0]])&(s.start<=tt[w[name][1]])]
        c=at('clean'); check('clean stroke: one long segment of all its rows, both edges state changes, ramp about 0.9 of the ramp rows, no end phase',
                             c.seg_class=='long' and c.rows==162 and c.left_edge==c.right_edge=='state_change' and abs(c.ramp_s-.9*12*STEP)<=4*STEP and c.end_s==0 and abs(c.steady_s-(161*STEP-c.ramp_s))<=1)
        labc=lab8.iloc[w8['clean'][0]:w8['clean'][1]+1]; check('clean stroke: rows are coded ramp, then steady, nothing else',set(labc.phase)=={1,2} and (np.diff(labc.phase.to_numpy())>=0).all() and (labc.seg_no==c.seg_no).all() and (labc.op_state==0).all())
        check('the rotating idle rows before the stroke are coded rotating, the rest stopped',(lab8.op_state.iloc[w8['i1'][0]:w8['i1'][1]+1]==1).all() and (lab8.op_state.iloc[w8['i1b'][0]:w8['i1b'][1]+1]==2).all())
        ha=seg_of('hole_a'); hb=seg_of('hole_b'); check('hole inside a stroke: two segments, the cut edges are hole_boring and censored',len(ha)==1 and len(hb)==1 and ha.iloc[0].right_edge=='hole_boring' and hb.iloc[0].left_edge=='hole_boring'
              and bool(ha.iloc[0].right_censored) and bool(hb.iloc[0].left_censored) and abs(ha.iloc[0].gap_after_s-300)<1 and abs(hb.iloc[0].gap_before_s-300)<1)
        check('segment after a hole starts at its plateau: no ramp is named, steady from the first row',pd.isna(hb.iloc[0].ramp_s) and hb.iloc[0].steady_start==hb.iloc[0].start and (lab8.phase.iloc[w8['hole_b'][0]:w8['hole_b'][1]+1]==2).all())
        pa=seg_of('pause_a'); pb=seg_of('pause_b'); check('pause of 300 s with the flag on: two segments with pause edges, the paused rows are outside every segment',len(pa)==1 and len(pb)==1 and pa.iloc[0].right_edge=='pause' and pb.iloc[0].left_edge=='pause'
              and not pa.iloc[0].right_censored and (lab8.seg_no.iloc[w8['pause'][0]:w8['pause'][1]+1]==-1).all() and (lab8.phase.iloc[w8['pause'][0]:w8['pause'][1]+1]==0).all())
        check('the 40 s dip of the thrust stays inside its segment',pa.iloc[0].rows==72 and (lab8.seg_no.iloc[w8['pause_a'][0]:w8['pause_a'][1]+1]==pa.iloc[0].seg_no).all())
        st=seg_of('stalled').iloc[0]; check('stalled stretch (flag and thrust on, no advance): long, flagged stalled, no steady state',st.seg_class=='long' and bool(st.stalled) and st.phase_status=='no_steady' and (lab8.phase.iloc[w8['stalled'][0]:w8['stalled'][1]+1]==4).all())
        sh=seg_of('short').iloc[0]; check('short stroke: class short, no phases, code 4',sh.seg_class=='short' and sh.phase_status=='not_long' and (lab8.phase.iloc[w8['short'][0]:w8['short'][1]+1]==4).all())
        bl=seg_of('blip').iloc[0]; check('three-row flicker is a blip',bl.seg_class=='blip' and bl.rows==3)
        rg=seg_of('regime_a').iloc[0]; check('change of level: the stretch before it is named ramp (steady starts at the step)',abs(rg.ramp_s-45*STEP)<=4*STEP and rg.rows==145)
        dc=seg_of('decay').iloc[0]; check('slow decay of the thrust (30 % over the stroke): no end phase, steady to the last row',dc.end_s==0 and dc.tail_s==0 and dc.steady_end==dc.end)
        tm=seg_of('terminal').iloc[0]; check('terminal decline over 12 rows: end phase of about 0.9 of it',abs(tm.end_s-.9*12*STEP)<=4*STEP and tm.tail_s==tm.end_s)
        iv=seg_of('invalid').iloc[0]; check('20 % invalid advance speed: status few_valid, no phases',iv.phase_status=='few_valid' and abs(iv.valid_share-.8)<.02 and (lab8.phase.iloc[w8['invalid'][0]:w8['invalid'][1]+1]==4).all())
        ah=seg_of('after_hole').iloc[0]; check('stroke after a hole: left edge hole_other, censored, but its ramp is named',ah.left_edge=='hole_other' and bool(ah.left_censored) and ah.ramp_s>0 and ah.steady_start>ah.start)
        ss=s8.iloc[0]; se=s8.iloc[-1]; check('series start and end: edges series_edge, censored',ss.left_edge=='series_edge' and se.right_edge=='series_edge' and bool(ss.left_censored) and bool(se.right_censored))
        check('every planted segment of TBM8 is accounted for (no stray segments)',len(s8)==14+1 and seg_of('i0').empty and seg_of('i5').empty)
        check('K1-like machine: flag from the thrust state, then the boring flag, then the signals',list(s5.sort_values('seg_no').ref_source)==['state_thrust','state_boring','signal'] and list(s5.sort_values('seg_no').seg_class)==['long','long','blip'])
        cons=pd.read_csv(out/'state_consistency.csv'); k=cons[(cons.machine=='TBM5')]; check('state consistency: three sources for TBM5, all signal rows with F>0 are boring',set(k.ref_source)=={'state_thrust','state_boring','signal'} and (cons.thrust_on_not_boring[cons.machine=='TBM5']==0).all())
        r=run(['p2_verify.py','--src',str(src),'--dest',str(out)],stdout=subprocess.DEVNULL); check('p2_verify.py passes on the synthetic data',r.returncode==0)
        # tampering must be caught
        bad=tmp/'bad1'; shutil.copytree(out,bad); l=pd.read_parquet(bad/'labels/TBM8.parquet'); i=w8['clean'][0]+30; l.loc[i,'phase']=3; l.to_parquet(bad/'labels/TBM8.parquet'); r=run(['p2_verify.py','--src',str(src),'--dest',str(bad)],stdout=subprocess.DEVNULL)
        check('verification fails when one phase code of the labels is changed',r.returncode==1)
        bad=tmp/'bad2'; shutil.copytree(out,bad); sg=pd.read_csv(bad/'segments.csv'); sg.loc[sg.segment_id==c.segment_id,'steady_end']=sg.loc[sg.segment_id==c.segment_id,'start']; sg.to_csv(bad/'segments.csv',index=False,encoding='utf-8-sig'); r=run(['p2_verify.py','--src',str(src),'--dest',str(bad)],stdout=subprocess.DEVNULL)
        check('verification fails when the table is changed',r.returncode==1)
        bad=tmp/'bad3'; shutil.copytree(out,bad); l=pd.read_parquet(bad/'labels/TBM8.parquet'); i=w8['pause'][0]+5; l.loc[i,'seg_no']=pa.iloc[0].seg_no; l.loc[i,'op_state']=0; l.to_parquet(bad/'labels/TBM8.parquet'); r=run(['p2_verify.py','--src',str(src),'--dest',str(bad)],stdout=subprocess.DEVNULL)
        check('verification fails when a paused row is put into a segment',r.returncode==1)
        # second build reproduces every checksum
        out2=tmp/'out2'; (out2/'labels').mkdir(parents=True); run(['p2_segment.py','--src',str(src),'--dest',str(out2),'--machines','TBM5,TBM8','--workers','1'],stdout=subprocess.DEVNULL)
        m1=json.loads((out/'manifest.json').read_text(encoding='utf-8')); m2=json.loads((out2/'manifest.json').read_text(encoding='utf-8'))
        check('a second build reproduces the segment table and every column checksum',m1['segments']['sha256']==m2['segments']['sha256'] and all(m1['machines'][m]['checksums']==m2['machines'][m]['checksums'] for m in m1['machines']))
        # checks, review package, scoring
        r=run(['p2_checks.py','--src',str(src),'--dest',str(out),'--machines','TBM5,TBM8','--workers','2'],stdout=subprocess.DEVNULL); sens=pd.read_csv(out/'sensitivity_segmentation.csv') if r.returncode==0 else None
        check('p2_checks.py runs: the 60 s and 300 s gap variants differ on the hole, 120 s does not split the pause',r.returncode==0 and sens is not None
              and int(sens[(sens.machine=='TBM8')&(sens.parameter=='max_gap_s')&(sens.value==60)].segments.iloc[0])>int(sens[(sens.machine=='TBM8')&(sens.parameter=='max_gap_s')&(sens.value==300)].segments.iloc[0])
              and int(sens[(sens.machine=='TBM8')&(sens.parameter=='interrupt_s')&(sens.value==1e9)].segments.iloc[0])<int(sens[(sens.machine=='TBM8')&(sens.parameter=='interrupt_s')&(sens.value==120)].segments.iloc[0]))
        r=run(['p2_review.py','--src',str(src),'--dest',str(out)],stdout=subprocess.DEVNULL); rv=out/'review'
        check('p2_review.py writes the sheet, the figures and the instructions',r.returncode==0 and (rv/'sample.csv').exists() and (rv/'figures.pdf').stat().st_size>10000 and (rv/'README.md').exists())
        sheet=pd.read_csv(rv/'sample.csv',dtype=str).fillna(''); sheet['q1_boring']='y'; sheet['q2_start']='ok'; sheet['q3_end']='ok'; sheet['q4_ramp_end']='ok'; sheet['q5_end_start']=np.where(np.arange(len(sheet))%2==0,'ok','early'); sheet.to_csv(rv/'filled.csv',index=False)
        r=run(['p2_review_score.py','--sheet',str(rv/'filled.csv'),'--dest',str(out),'--out',str(rv/'scores.csv')],stdout=subprocess.DEVNULL); sc=pd.read_csv(rv/'scores.csv') if r.returncode==0 else None
        o=sc[(sc.group_type=='overall')&(sc.question=='q5_end_start')].iloc[0] if sc is not None else None
        check('p2_review_score.py: answers are tallied with the right shares',r.returncode==0 and o is not None and o.judged==len(sheet) and abs(o.share_good-(np.arange(len(sheet))%2==0).mean())<1e-9
              and sc[(sc.group_type=='overall')&(sc.question=='q1_boring')].iloc[0].share_good==1.)
        sheet.loc[0,'q1_boring']='maybe'; sheet.to_csv(rv/'filled_bad.csv',index=False); r=run(['p2_review_score.py','--sheet',str(rv/'filled_bad.csv'),'--dest',str(out)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        check('p2_review_score.py refuses an answer outside the allowed words',r.returncode!=0)
        if a.keep: shutil.copytree(tmp,HERE.parent/'p2_smoke_kept')
    print(f'{sum(ok)}/{len(ok)} checks passed'); finish(0 if all(ok) else 1)

if __name__=='__main__': main()
