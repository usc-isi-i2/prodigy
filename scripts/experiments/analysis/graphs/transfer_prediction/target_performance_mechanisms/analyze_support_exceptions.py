"""Frozen support-exception discovery decision, no outcome-selected thresholds."""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np


def summarize(rows):
    keys=('stream','step','rule','polarity','decoder')
    expected=set(itertools.product(('original','fresh'),(100,2500),
        ('receive_edge','exceptions20','exceptions40'),(0,1),
        ('full_model','S0_pool/ridge','scalar_rule/ridge')))
    indexed={tuple(r[k] for k in keys):r for r in rows}
    if len(rows)!=72 or len(indexed)!=72 or set(indexed)!=expected:
        raise ValueError('Incomplete or duplicate grid')
    means=[]
    for stream,rule,decoder in itertools.product(('original','fresh'),
            ('receive_edge','exceptions20','exceptions40'),
            ('full_model','S0_pool/ridge','scalar_rule/ridge')):
        r=dict(stream=stream,rule=rule,decoder=decoder)
        for step in (100,2500):
            r[str(step)]={metric:float(np.mean([indexed[(stream,step,rule,p,decoder)][metric]
                         for p in (0,1)])) for metric in ('mean_episode_auc','accuracy')}
        r['auc_change']=r['2500']['mean_episode_auc']-r['100']['mean_episode_auc']
        means.append(r)
    primary={r['stream']:r['auc_change'] for r in means if r['rule']=='exceptions40' and r['decoder']=='full_model'}
    return dict(means=means,primary_changes=primary,passes=all(v<=-.03 for v in primary.values()))


def main():
    import matplotlib.pyplot as plt
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('input',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--figure',type=Path,required=True)
    a=p.parse_args()
    result=summarize(json.loads(a.input.read_text()))
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(8.5,4),sharey=True)
    for ax,stream in zip(axes,('original','fresh')):
        for step,color in (('100','#3269a8'),('2500','#bd6531')):
            values=[next(r for r in result['means'] if r['stream']==stream and r['rule']==rule and r['decoder']=='full_model')[step]['mean_episode_auc']
                    for rule in ('receive_edge','exceptions20','exceptions40')]
            ax.plot([0,20,40],values,'o-',color=color,label=f'Step {step}')
        ax.axhline(.5,color='gray',lw=.7)
        ax.set(title=stream.capitalize(),xlabel='Support exceptions (%)',xticks=[0,20,40],ylim=(.4,1))
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Native mean episode AUC')
    axes[1].legend(frameon=False)
    fig.suptitle('Fixed queries, balanced support exceptions — Hong Kong')
    fig.text(.5,.01,'Synthetic exceptions; clean queries; both label polarities averaged. Primary test fixed at40%.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.05,1,.94))
    fig.savefig(a.figure,dpi=180)
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
