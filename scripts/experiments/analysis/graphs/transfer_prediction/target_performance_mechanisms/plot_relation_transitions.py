"""Private figure: distinguish flatter trajectories from preserved rankings."""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def main():
    root = Path(__file__).parent
    data = json.loads((root/'data/relation_transitions_20260907.json').read_text())['rows']
    fig, axes = plt.subplots(1, 2, figsize=(10.8,4.6), constrained_layout=True)
    colors = {'original':'#326b9d', 'fresh':'#cf843a'}
    groups = [('correct','correct'),('wrong','wrong'),('correct','wrong'),('wrong','correct')]
    for i,stream in enumerate(('original','fresh')):
        rows = [r for r in data if r['stream']==stream and r['decoder']=='full_model']
        selected = [r for r in rows if r['retweet_early_state']==r['follow_early_state']=='correct']
        mass = np.mean([r['mass'] for r in selected])
        vals = [np.mean([r[k+'_contribution'] for r in selected])/mass for k in ('retweet_late','follow_late')]
        axes[0].plot([0,1],vals,marker='o' if i==0 else '^',ls='-' if i==0 else '--',color=colors[stream],label=stream.capitalize())
        for x,y in enumerate(vals):
            axes[0].annotate(f'{100*y:.1f}%',(x,y),xytext=(6,5 if i==0 else -14),textcoords='offset points',fontsize=9,color=colors[stream])
        values = [100*np.mean([r['difference_in_changes_contribution'] for r in rows
                              if (r['retweet_early_state'],r['follow_early_state'])==g]) for g in groups]
        axes[1].bar(np.arange(4)+(i-.5)*.34,values,width=.32,color=colors[stream],label=stream.capitalize())
        same = sum(r['difference_in_changes_contribution'] for r in rows if r['retweet_early_state']==r['follow_early_state'])/9*100
        print(stream,'joint-correct mass',mass,'late conditional',vals,'same-start contribution',same)
    axes[0].set_xticks([0,1],['Retweet','Follow'])
    axes[0].set_xlim(-.2,1.3); axes[0].set_ylim(.53,.73)
    axes[0].set_ylabel('Late correct-ranking credit on jointly correct pairs')
    axes[0].set_title('A  Retweets retain more shared early successes',loc='left',fontsize=11)
    axes[0].legend(frameon=False,loc='lower left')
    axes[1].set_xticks(range(4),['Both\ncorrect','Both\nwrong','Retweet only\ncorrect','Follow only\ncorrect'])
    axes[1].set_ylabel('Contribution to follow − retweet training change\n(AUC points; positive favors follow)')
    axes[1].set_title('B  Initial disagreements drive the interaction',loc='left',fontsize=11)
    axes[1].axhline(0,color='#777',lw=.8)
    axes[1].set_xlabel('Ranking state at 100 updates')
    for ax in axes: ax.spines[['top','right']].set_visible(False)
    fig.suptitle('A flatter learning curve is not better preservation',fontsize=15)
    fig.supxlabel('Nine source checkpoints; two episode streams, not training seeds. Equal episode/source weights.\nPanel A: pooled weighted credit / mass. Panel B: four main strata; small tie strata retained in exact accounting.',fontsize=8)
    out=root/'figures/relation_transition_argument.png'
    fig.savefig(out,dpi=180)
    print(out)


if __name__=='__main__': main()
