"""P2 review package: a stratified sample of segments with one figure each, for a person to judge. Writes outputs/segments_v1/review/sample.csv (the
sheet with empty answer columns), figures.pdf (one page per sampled segment: thrust, torque, cutterhead speed and advance speed with the segment, the
phases and the data holes marked) and README.md (what to answer). The sample is drawn with a fixed seed from segments.csv, so it is the same on every run.
Scoring of the returned sheet: p2_review_score.py."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from p2_segment import DEST, SRC, PHASE, finish

SEED=20241101
QUOTA=[('clean_long',7),('censored_long',4),('irregular_long',3),('invalid_long',1),('short',3),('blip',2)]   # per machine
SURFACE='#fcfcfb'; INK=dict(primary='#0b0b0b',secondary='#52514e',muted='#898781',grid='#e1e0d9',axis='#c3c2b7',band='#efeee9')
SERIES=dict(F='#2a78d6',T='#eb6834',n='#1baf7a',v='#6250d6')      # one colour per signal in every figure of the project
PHASE_STYLE={PHASE['ramp']:('#b7d3f6','ramp',INK['primary']),PHASE['steady']:('#2a78d6','steady','#ffffff'),PHASE['end']:('#eb6834','end','#ffffff'),PHASE['unassigned']:('#e1e0d9','no phase',INK['secondary'])}
AXES=[('F','F (thrust)'),('T','T (torque)'),('n','n (rpm)'),('v','v (mm/min)')]
ANSWERS=['q1_boring','q2_start','q3_end','q4_ramp_end','q5_end_start']
HOLE_S=60.

def stratum(r):
    if r.seg_class in ('blip','short'): return r.seg_class
    if r.phase_status in ('few_valid','few_rows'): return 'invalid_long'
    if r.phase_status!='ok' or r.max_inactive_s>=60: return 'irregular_long'
    return 'censored_long' if (r.left_censored or r.right_censored) else 'clean_long'

def draw_sample(seg,seed=SEED):
    """Quota per machine and stratum; every draw has its own seed, so the sample of one stratum does not move when another one changes."""
    seg=seg.assign(stratum=seg.apply(stratum,axis=1)); picks=[]
    for m in sorted(seg.machine.unique(),key=lambda x:int(x[3:])):
        for q,(name,k) in enumerate(QUOTA):
            pool=seg[(seg.machine==m)&(seg.stratum==name)]; rng=np.random.default_rng([seed,int(m[3:]),q])
            picks.append(pool.iloc[np.sort(rng.choice(len(pool),size=min(k,len(pool)),replace=False))])
    s=pd.concat(picks); s=s.iloc[np.random.default_rng(seed).permutation(len(s))].reset_index(drop=True)      # shuffled, so the order tells nothing about machine or stratum
    s.insert(0,'sample_id',[f'R{i+1:03d}' for i in range(len(s))]); return s

def runs(code):
    """(first,last,code) of every run of equal codes."""
    cut=np.r_[0,np.flatnonzero(np.diff(code))+1,len(code)]; return [(a,b-1,int(code[a])) for a,b in zip(cut[:-1],cut[1:])]

def broken(x,y,dt):
    """Line data with a NaN wherever the readings are more than HOLE_S apart, so a data hole shows as a gap."""
    cut=np.flatnonzero(dt>HOLE_S); return np.insert(x,cut,np.nan),np.insert(y,cut,np.nan)

def draw(row,d,lab):
    """One figure: the segment of `row` with its context. d: signals of the machine, lab: its labels (same rows)."""
    t0=pd.Timestamp(row.start); t1=pd.Timestamp(row.end); dur=(t1-t0).total_seconds(); pad=pd.Timedelta(seconds=max(300.,.2*dur)); times=d.time.to_numpy()
    i0,i1=np.searchsorted(times,(t0-pad).to_datetime64()),np.searchsorted(times,(t1+pad).to_datetime64(),side='right'); w=d.iloc[i0:i1]; lw=lab.iloc[i0:i1]
    x=(w.time-t0).dt.total_seconds().to_numpy()/60; dt=w.dt_prev_s.to_numpy(float); inside=(lw.seg_no.to_numpy()==row.seg_no); ph=lw.phase.to_numpy()
    fig,axs=plt.subplots(5,1,figsize=(10,6.6),sharex=True,gridspec_kw=dict(height_ratios=[.42,1,1,1,1]),facecolor=SURFACE)
    st=axs[0]; st.set_facecolor(SURFACE); st.set_yticks([]); st.set_ylim(0,1)
    for a,b,c in runs(np.where(inside,ph,0)):
        if c==0: continue
        fill,name,ink=PHASE_STYLE[c]; xa=x[a]; xb=x[b+1] if b+1<len(x) and inside[b+1] else x[b]; xb=max(xb,xa+.004*(x[-1]-x[0]))     # a run ends where the next one starts; a one-row run is drawn as a sliver
        st.axvspan(xa,xb,color=fill,lw=0,hatch='////' if c==PHASE['unassigned'] else None,ec=INK['axis'] if c==PHASE['unassigned'] else fill)
        if (xb-xa)>=.07*(x[-1]-x[0]): st.text((xa+xb)/2,.5,name,ha='center',va='center',fontsize=8,color=ink)
    st.set_ylabel('phase',fontsize=8,color=INK['secondary'],rotation=0,ha='right',va='center')
    for ax,(c,label) in zip(axs[1:],AXES):
        ax.set_facecolor(SURFACE); y=w[c].to_numpy(float); xo,yo=broken(x,np.where(inside,np.nan,y),dt); xi,yi=broken(x,np.where(inside,y,np.nan),dt)
        ax.axvspan(0,dur/60,color=INK['band'],lw=0); ax.plot(xo,yo,color=INK['muted'],lw=1.,alpha=.8); ax.plot(xi,yi,color=SERIES[c],lw=1.5,solid_capstyle='round')
        for tt in (row.steady_start,row.steady_end):
            if isinstance(tt,str) or pd.notna(tt): ax.axvline((pd.Timestamp(tt)-t0).total_seconds()/60,color=INK['secondary'],lw=.9,ls=(0,(4,3)))
        fin=y[np.isfinite(y)]; top=float(np.quantile(fin,.995)*1.08) if len(fin) else 1.; ax.set_ylim(min(0,float(fin.min()) if len(fin) else 0),max(top,1e-6)) if len(fin) else None
        ax.set_ylabel(label,fontsize=8,color=INK['secondary']); ax.grid(axis='y',color=INK['grid'],lw=.6); ax.set_axisbelow(True)
    for ax in axs:
        for s in ('top','right'): ax.spines[s].set_visible(False)
        for s in ('left','bottom'): ax.spines[s].set_color(INK['axis'])
        ax.tick_params(colors=INK['muted'],labelsize=8)
    axs[-1].set_xlabel('minutes from segment start',fontsize=8,color=INK['secondary']); axs[-1].set_xlim(x[0]-(x[-1]-x[0]<1e-6),x[-1]+(x[-1]-x[0]<1e-6))
    fig.suptitle(f'{row.sample_id}   {row.machine}   {row.start}  to  {row.end}   ({dur/60:.1f} min)',fontsize=10,color=INK['primary'],x=.01,ha='left',y=.995)
    fig.text(.01,.955,f'{row.seg_class}  |  left edge: {row.left_edge}  |  right edge: {row.right_edge}  |  phases: {row.phase_status}',fontsize=7.5,color=INK['secondary'])
    fig.text(.01,.927,'grey band = segment, dashed lines = steady start and end, grey line = rows outside the segment, gap in a line = readings more than 60 s apart.  Phase strip: light blue ramp, blue steady, orange end, hatched = inside the segment without a phase.',fontsize=6.5,color=INK['muted'])
    fig.tight_layout(rect=(0,0,1,.915)); return fig

README='''# P2 抽样验证包

每个样本是一个连续掘进段,在 `figures.pdf` 里占一页(页眉的 R001 等就是 `sample.csv` 里的 sample_id)。请对照图回答 `sample.csv` 右侧的五列。

**怎么看图**:灰色底带是算法划出的掘进段;最上面一条是阶段(ramp 上升段、steady 稳定段、end 下降段、no phase 未命名);虚线是稳定段的起点和终点;
灰线是段外的读数;线断开的地方是超过60秒的读数缺口(数据洞)。四个面板依次是推力F、扭矩T、刀盘转速n、推进速度v。

**算法怎么划**(供你判断时参考,不必照它的规则想):机器自己的掘进状态为开,且推力不是连续2分钟以上为0,读数间隔不超过60秒,就是一个连续掘进段;
段内推力和推进速度都达到本段稳定水平(段中间一半的中位数)的90%和80%并保持,才算进入稳定段(扭矩只要求为正);
推力或推进速度相对此前5分钟的水平降到90%或80%以下之后到段末是下降段(超过5分钟的不算)。

**怎么答**(每格只填下面的词,小写英文;拿不准填 unclear;不适用填 na。q1 填 n 时,q2 到 q5 填 na):

| 列 | 问题 | 可填 |
|---|---|---|
| q1_boring | 灰带整体是不是一段真实的连续掘进(推力和推进在、刀盘在转)? | `y` 是;`n` 不是;`p` 部分是(带里夹着明显不是掘进的部分,或漏了明显的掘进部分) |
| q2_start | 灰带的起点对不对? | `ok`;`early` 起点早于实际开始掘进;`late` 起点晚于实际开始(漏掉了开头);`unclear` |
| q3_end | 灰带的终点对不对? | `ok`;`early` 终点早于实际结束(漏掉了结尾);`late` 终点晚于实际结束;`unclear` |
| q4_ramp_end | 上升段结束的位置(稳定段起点的虚线)对不对? | `ok`;`early` 画早了(图上推力或推进速度还在明显爬升);`late` 画晚了(此前已有一段平稳的掘进,例如先在较低水平稳定、后来整体抬高;请在 comment 写 regime);`unclear`;`na` 该段没有稳定段 |
| q5_end_start | 下降段开始的位置(稳定段终点的虚线)对不对? | `ok`;`early`(终点前面还有一段平稳的掘进被划进了下降段);`late`(平稳段已经结束仍被划进稳定段);`unclear`;`na` |

“对”的标准:边界落在你自己会画的位置前后约1分钟(或5到6个读数)以内。起点或终点贴着数据洞(页眉里写 hole_boring 或 hole_other)的,边界本来就不可见,q2 或 q3 填 `unclear`。
`comment` 列随意写。

填完后把 `sample.csv` 另存发回,或自己运行:

```
python analysis/p2_review_score.py --sheet <填好的文件.csv>
```

它会按机器和抽样层给出各问题的“对”的比例及95%区间。**这些比例是对这 {n} 个样本的判断,不是全部 {N:,} 个段的正确率**;抽样层的含义见 P2 报告。
'''

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--src',default=str(SRC)); ap.add_argument('--dest',default=str(DEST)); ap.add_argument('--png',default='',help='also write one PNG per sample here'); ap.add_argument('--dpi',type=int,default=75); a=ap.parse_args()
    dest=Path(a.dest); out=dest/'review'; out.mkdir(exist_ok=True); seg=pd.read_csv(dest/'segments.csv'); s=draw_sample(seg)
    cols=['sample_id','machine','segment_id','start','end','duration_s','seg_class','stratum','left_edge','right_edge','phase_status','ramp_s','steady_s','end_s']
    sheet=s[cols].copy(); sheet['duration_min']=(sheet.pop('duration_s')/60).round(2)
    for c in ('ramp_s','steady_s','end_s'): sheet[c.replace('_s','_min')]=(sheet.pop(c)/60).round(2)
    for c in ANSWERS+['comment']: sheet[c]=''
    sheet.to_csv(out/'sample.csv',index=False,encoding='utf-8-sig',lineterminator='\n'); (out/'README.md').write_text(README.format(n=len(s),N=len(seg)),encoding='utf-8')
    if a.png: Path(a.png).mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'pdf.fonttype':42,'font.family':'DejaVu Sans'})
    cache={m:(pd.read_parquet(Path(a.src)/f'{m}.parquet',columns=['time','F','T','n','v','dt_prev_s']),pd.read_parquet(dest/'labels'/f'{m}.parquet',columns=['seg_no','phase'])) for m in s.machine.unique()}
    with PdfPages(out/'figures.pdf',metadata=dict(Title='P2 review sample',CreationDate=None)) as pdf:
        for r in s.itertuples():
            d,lab=cache[r.machine]; fig=draw(r,d,lab); pdf.savefig(fig,facecolor=SURFACE)
            if a.png: fig.savefig(Path(a.png)/f'{r.sample_id}.png',dpi=a.dpi,facecolor=SURFACE)
            plt.close(fig)
    print(len(s),'samples;',s.stratum.value_counts().to_dict())

if __name__=='__main__': main(); finish()
