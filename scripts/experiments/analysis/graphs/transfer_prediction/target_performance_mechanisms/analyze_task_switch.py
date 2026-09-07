"""Apply frozen discovery gates and plot paired task trajectories; private."""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np


def summarize(rows):
    keys = ['stream','step','rule','polarity','decoder']
    expected = set(itertools.product(('original','fresh'),(100,2500),
        ('receive_edge','content_feature'),(0,1),('full_model','S0_pool/ridge','scalar_rule/ridge')))
    indexed = {tuple(r[k] for k in keys):r for r in rows}
    if len(rows)!=48 or len(indexed)!=48 or set(indexed)!=expected:
        raise ValueError('Incomplete or duplicate discovery grid')
    if any(r['source']!='cp_hk' for r in rows):
        raise ValueError('Discovery source changed')
    means = []
    for stream,rule,decoder in itertools.product(('original','fresh'),
            ('receive_edge','content_feature'),('full_model','S0_pool/ridge','scalar_rule/ridge')):
        values = {step:{metric:float(np.mean([indexed[(stream,step,rule,p,decoder)][metric]
                    for p in (0,1)])) for metric in ('mean_episode_auc','accuracy')}
                  for step in (100,2500)}
        means.append(dict(stream=stream,rule=rule,decoder=decoder,
                          early=values[100],late=values[2500],
                          auc_change=values[2500]['mean_episode_auc']-values[100]['mean_episode_auc']))
    gates = {}
    for stream in ('original','fresh'):
        def get(rule,decoder):
            return next(r for r in means if (r['stream'],r['rule'],r['decoder'])==(stream,rule,decoder))
        structural=get('receive_edge','full_model')
        content=get('content_feature','full_model')
        pool=get('receive_edge','S0_pool/ridge')
        gates[stream] = dict(structural_decline=structural['auc_change']<=-.03,
            content_gain=content['auc_change']>=.03,
            early_structural_useful=structural['early']['mean_episode_auc']>=.60,
            late_content_useful=content['late']['mean_episode_auc']>=.60,
            late_structural_decodable=pool['late']['mean_episode_auc']>=.70,
            structural_decodability_retained=pool['auc_change']>=-.02)
    return dict(means=means,gates=gates,passes=all(all(v.values()) for v in gates.values()))


def main():
    import matplotlib.pyplot as plt
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('input',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--figure',type=Path,required=True)
    args=p.parse_args()
    result=summarize(json.loads(args.input.read_text()))
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(8.5,4.3),sharey=True)
    for ax,stream in zip(axes,('original','fresh')):
        for rule,color in (('receive_edge','#3269a8'),('content_feature','#bd6531')):
            for decoder,style in (('full_model','-'),('S0_pool/ridge','--')):
                r=next(r for r in result['means'] if (r['stream'],r['rule'],r['decoder'])==(stream,rule,decoder))
                ax.plot([100,2500],[r['early']['mean_episode_auc'],r['late']['mean_episode_auc']],
                        style+'o',color=color,label=f'{rule.replace("_"," ")} / {"native" if decoder=="full_model" else "pool ridge"}')
        ax.axhline(.5,color='gray',lw=.7)
        ax.set(title=stream.capitalize(),xlabel='Pretraining updates',xticks=[100,2500],ylim=(.35,1.01))
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Mean episode AUC')
    handles,labels=axes[1].get_legend_handles_labels()
    fig.legend(handles,labels,frameon=False,fontsize=8,loc='lower center',ncol=2,bbox_to_anchor=(.5,.04))
    fig.suptitle('Same graph inputs, two class rules — Hong Kong discovery')
    fig.text(.5,.01,'Both label polarities averaged; one training seed; synthetic diagnostic tasks, not bot labels.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.19,1,.94))
    fig.savefig(args.figure,dpi=180)
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
