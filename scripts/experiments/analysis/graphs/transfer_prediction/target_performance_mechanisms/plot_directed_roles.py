"""Private figure for the completed predeclared directed-context experiment."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--summary',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    data=json.loads(args.summary.read_text())
    summaries=[r for r in data['summaries'] if r['decoder']=='full_model']
    cells=[r for r in data['source_cells'] if r['decoder']=='full_model']
    if len(summaries)!=2 or len(cells)!=18: raise ValueError('Complete native panel required')
    fig,axes=plt.subplots(1,3,figsize=(12,4.4),constrained_layout=True)
    conditions=('intact','support','query','both')
    labels=('Intact','Support\nreversed','Query\nreversed','Both\nreversed')
    colors={'original':'#326b9d','fresh':'#c88239'}
    all_contrasts=[]
    for row in summaries:
        stream=row['stream']; marker='o' if stream=='original' else '^'; ls='-' if stream=='original' else '--'
        for ax,step in zip(axes[:2],('100','2500')):
            vals=[row['endpoint'][step][c] for c in conditions]
            ax.plot(range(4),vals,marker=marker,ls=ls,color=colors[stream],label=stream.capitalize())
            if step=='100':
                threshold=.5+.5*(vals[0]-.5)
                ax.plot([2.75,3.25],[threshold,threshold],color=colors[stream],lw=1.2,ls=':')
        for cell in [r for r in cells if r['stream']==stream]:
            a,b=[100*cell['matching_contrast'][step] for step in ('100','2500')]
            all_contrasts.extend([a,b]); axes[2].scatter(a,b,color=colors[stream],marker=marker,s=35,alpha=.85)
    for ax,step,letter in zip(axes[:2],(100,2500),('A','B')):
        ax.axhline(.5,lw=.7,color='#777',ls=':')
        ax.set_xticks(range(4),labels,fontsize=8)
        ax.set_ylabel('Full-model mean episode AUC')
        ax.set_title(f'{letter}  {step:,} pretraining updates',loc='left',fontsize=11)
    low=min(ax.get_ylim()[0] for ax in axes[:2]); high=max(ax.get_ylim()[1] for ax in axes[:2])
    for ax in axes[:2]: ax.set_ylim(low,high)
    axes[0].legend(frameon=False,fontsize=8)
    lo=min(0,min(all_contrasts))-2; hi=max(0,max(all_contrasts))+2
    axes[2].plot([lo,hi],[lo,hi],color='#888',ls=':',lw=1)
    axes[2].axhline(0,color='#bbb',lw=.6); axes[2].axvline(0,color='#bbb',lw=.6)
    axes[2].set(xlim=(lo,hi),ylim=(lo,hi),xlabel='Early matching contrast (AUC points)',ylabel='Late matching contrast (AUC points)')
    axes[2].set_title('C  Does matching weaken with training?',loc='left',fontsize=10)
    for ax in axes: ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Directed context: reverse supports, queries, or both',fontsize=14)
    fig.supxlabel('Same sampled nodes, biographies and undirected edge slots. Nine sources; two episode streams, not training seeds.\nShort dotted marks in A: predeclared half-advantage recovery threshold. C = (intact + both − support − query)/2.',fontsize=8)
    fig.savefig(args.output,dpi=180)
    print(args.output)


if __name__=='__main__': main()
