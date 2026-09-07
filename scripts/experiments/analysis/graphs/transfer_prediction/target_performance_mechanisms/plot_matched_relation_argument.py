"""Figure for the completed matched-account relation-setting test."""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def main():
    root=Path(__file__).parent
    follow=json.loads((root/'data/matched_follow_endpoints_20260907.json').read_text())
    old=json.loads((root/'data/degree_pair_accounting_20260907.json').read_text())['rows']
    coverage=json.loads((root/'data/matched_follow_coverage_20260907.json').read_text())
    fig,axes=plt.subplots(1,3,figsize=(12,4.1),constrained_layout=True)
    colors={'retweet':'#b54b45','follow':'#326b9d'}
    for stream,ls,marker in (('original','-','o'),('fresh','--','^')):
        c={r['graph']:r['degree_cue_mean_episode_auc'] for r in coverage if r['stream']==stream}
        axes[0].plot([0,1],[c['retweet'],c['follow']],ls,marker=marker,color='#555',label=stream.capitalize())
        a={r['source']:r for r in old if r['stream']==stream and r['decoder']=='full_model' and r['group']=='equal_degree'}
        b={(r['source'],r['step']):r for r in follow if r['stream']==stream and r['decoder']=='full_model'}
        rv=[np.mean([r[k] for r in a.values()]) for k in ('early_auc','late_auc')]
        fv=[np.mean([b[s,step]['mean_episode_auc'] for s in a]) for step in (100,2500)]
        for name,values in (('retweet',rv),('follow',fv)):
            axes[1].plot([0,1],values,ls,marker=marker,color=colors[name],label=name.capitalize() if stream=='original' else None)
        for source,r in a.items():
            dx=100*(r['late_auc']-r['early_auc'])
            dy=100*(b[source,2500]['mean_episode_auc']-b[source,100]['mean_episode_auc'])
            axes[2].scatter(dx,dy,marker=marker,color=colors['follow'],alpha=.8,s=38)
    axes[0].set_xticks([0,1],['Retweet','Follow'])
    axes[0].set_ylabel('Support-fitted degree cue AUC')
    axes[0].set_title('A  Degree probe falls to chance',fontsize=11,loc='left')
    axes[0].legend(frameon=False,fontsize=8)
    axes[1].set_xticks([0,1],['100 updates','2,500 updates'])
    axes[1].set_ylabel('Full-model AUC, mean of nine sources')
    axes[1].set_title('B  No decline, but ranking stays weak',fontsize=10,loc='left')
    axes[1].legend(frameon=False,fontsize=8)
    for ax in axes[:2]:
        ax.axhline(.5,color='#888',lw=.8,linestyle=':')
        ax.set_ylim(.47,.75)
    axes[2].plot([-20,10],[-20,10],color='#888',ls=':',lw=1)
    axes[2].axhline(0,color='#aaa',lw=.7)
    axes[2].set_xlim(-19,0); axes[2].set_ylim(-8,8)
    axes[2].set_xlabel('Retweet training change (AUC points)')
    axes[2].set_ylabel('Follow training change (AUC points)')
    axes[2].set_title('C  All nine sources shift upward',fontsize=11,loc='left')
    for ax in axes: ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Same accounts, different relations, different training trajectories',fontsize=14)
    path=root/'figures/matched_relation_argument.png'
    fig.savefig(path,dpi=180)
    print(path)


if __name__=='__main__': main()
