"""Three-panel scientific figure for the bounded mechanism decision page."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


SOURCES=['covid','covid_political','cp_hk','election2020','facebook_page_reference','midterm','twibot20','ukr_rus','ukr_rus_suspended']
TARGETS=['covid_political','election2020','facebook_page_reference','twibot20','ukr_rus_suspended']
SOURCE_LABELS=['covid','covid-political','hongkong','election2020','facebook-page','midterm','twibot20','ukraine','ukraine-suspended']
TARGET_LABELS=['CP','E20','FB','BOT','SUSP']


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--swap-targets',nargs='+',default=['covid_political','facebook_page_reference','twibot20'])
    args=p.parse_args()
    cells=pd.read_csv(args.input/'cells.csv'); g=pd.read_csv(args.input/'geometry.csv')
    grid=pd.read_csv(args.input/'nine_source_map.csv')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.labelsize':8,'axes.titlesize':9,
                         'axes.titleweight':'bold','xtick.labelsize':7,'ytick.labelsize':7})
    fig=plt.figure(figsize=(8,5.05),facecolor='white')
    ax=fig.add_axes([.075,.645,.34,.285]); geo=fig.add_axes([.075,.36,.34,.205])
    heat=fig.add_axes([.615,.405,.35,.525]); bar=fig.add_axes([.095,.09,.86,.15])
    cp=cells[(cells.target=='covid_political') & cells.condition.str.startswith('scale_')].copy()
    cp['alpha']=cp.condition.str.removeprefix('scale_').astype(float)
    colors={'cp_hk':'#c45827','ukr_rus':'#27768b'}
    for source,label in [('cp_hk','Hong Kong'),('ukr_rus','Ukraine')]:
        stats=cp[cp.source==source].groupby('alpha').delta_roc_auc.agg(['mean','min','max']).sort_index()
        x=np.maximum(stats.index.to_numpy(),1e-7)
        ax.fill_between(x,100*stats['min'],100*stats['max'],color=colors[source],alpha=.12,lw=0)
        ax.plot(x,100*stats['mean'],marker='o',ms=3,lw=1.6,color=colors[source],label=label)
    for a in (ax,geo):
        a.set_xscale('log');a.set_xlim(.7e-7,1.4);a.set_xticks([1e-7,1e-4,.01,1])
        a.set_xticklabels(['0',r'$10^{-4}$','0.01','1']);a.grid(axis='y',alpha=.18)
    ax.axhline(0,color='#747b85',lw=.5);ax.set_ylabel('Change in AUC (points)')
    ax.set_title('A  Scale response: political target',loc='left',pad=8)
    ax.legend(frameon=False,fontsize=7,loc='upper right');ax.tick_params(labelbottom=False)
    cg=g[(g.target=='covid_political') & (g.source=='cp_hk') & (g.stage=='U1_pre_meta') &
         (g.cohort=='support') & g.condition.str.startswith('scale_')].copy()
    cg['alpha']=cg.condition.str.removeprefix('scale_').astype(float)
    st=cg.groupby('alpha')[['direction_change_intact','norm_ratio_intact']].mean().sort_index()
    x=np.maximum(st.index.to_numpy(),1e-7)
    geo.plot(x,st.direction_change_intact,lw=1.5,marker='o',ms=2.7,color='#5d4890',label='direction: 1 - cosine')
    geo.plot(x,st.norm_ratio_intact-1,lw=1.3,ls='--',marker='s',ms=2.2,color='#688238',label='norm ratio - 1')
    geo.axhline(0,lw=.5,color='#747b85');geo.set_ylabel('Change before metagraph')
    geo.set_xlabel('Support-message multiplier (0 shown separately)')
    geo.legend(frameon=True,facecolor='white',edgecolor='none',framealpha=.85,fontsize=6.5,loc='upper right')
    heat.set_title('B  Nine-source support-removal map',loc='left',pad=8)
    mean=grid.groupby(['source','target']).support_delta_points.mean().unstack().reindex(index=SOURCES,columns=TARGETS)
    image=heat.imshow(mean,aspect='auto',cmap='RdBu',norm=TwoSlopeNorm(vmin=-11,vcenter=0,vmax=11))
    for yi,source in enumerate(SOURCES):
        for xi,target in enumerate(TARGETS):
            v=mean.loc[source,target]
            pair=grid[(grid.source==source)&(grid.target==target)].support_delta_points
            mixed=pair.min()<=0<=pair.max() and pair.min()!=pair.max()
            label=f'{v:+.1f}'+('*' if mixed else '')
            heat.text(xi,yi,label,ha='center',va='center',fontsize=7.1,color='white' if abs(v)>6.3 else '#20232b')
            if source==target:heat.add_patch(Rectangle((xi-.48,yi-.48),.96,.96,fill=False,edgecolor='#252a34',lw=1))
    heat.set_xticks(range(5),TARGET_LABELS);heat.set_yticks(range(9),SOURCE_LABELS)
    heat.tick_params(length=0);heat.set_ylabel('Pretraining source',labelpad=5)
    for spine in heat.spines.values():spine.set_visible(False)
    cb=fig.add_axes([.615,.34,.35,.014]);colorbar=fig.colorbar(image,cax=cb,orientation='horizontal',ticks=[-10,0,10])
    colorbar.ax.tick_params(labelsize=6,length=2,pad=2)
    fig.text(.615,.276,'AUC points: blue helps / red hurts. * Signs differ.\nOutline: source = target; 2-stream means, 1 seed/source.',fontsize=6.3,linespacing=1.3)
    bar.set_title('C  Hong Kong: pre-metagraph swaps',loc='left',pad=7)
    selected=args.swap_targets
    if not set(selected)<=set(TARGETS):raise ValueError('unknown swap target')
    conditions=[('pre_meta_norm_only','Change norms only','#688238',-.2),
                ('pre_meta_direction_only','Change directions only','#5d4890',0.),
                ('scale_0','Remove messages','#7a7c85',.2)]
    for condition,label,color,offset in conditions:
        means=[];mins=[];maxs=[]
        for target in selected:
            values=100*cells[(cells.source=='cp_hk')&(cells.target==target)&(cells.condition==condition)].delta_roc_auc
            if len(values)!=6:raise ValueError('complete six seed/stream cells required')
            means.append(values.mean());mins.append(values.min());maxs.append(values.max())
        means=np.asarray(means)
        bar.errorbar(np.arange(len(selected))+offset,means,yerr=[means-np.asarray(mins),np.asarray(maxs)-means],
                     fmt='o',color=color,ms=4,elinewidth=1,capsize=2,label=label)
    bar.axhline(0,color='#555e6a',lw=.7);bar.grid(axis='y',alpha=.18)
    labels={'covid_political':'covid-political','facebook_page_reference':'facebook-page-reference','twibot20':'twibot20'}
    bar.set_xticks(range(len(selected)),[labels.get(t,t) for t in selected])
    bar.set_ylabel('AUC points');bar.set_xlim(-.6,len(selected)-.4)
    bar.legend(frameon=False,ncol=3,fontsize=6.5,loc='upper center',bbox_to_anchor=(.5,-.20),columnspacing=1.5)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,dpi=240,facecolor='white')
    plt.close(fig)


if __name__=='__main__':main()
