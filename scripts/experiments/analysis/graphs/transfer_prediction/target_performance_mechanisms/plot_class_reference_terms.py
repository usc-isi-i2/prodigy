"""Show score accounting and an actual fixed-episode ranking/accuracy contrast."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).parent/'data/classrefterms_20260907')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    rows = json.loads((args.root/'summary.json').read_text())
    cases = json.loads((args.root/'numeric_examples.json').read_text())
    conditions = ['intact','values_only','removed']
    components = ['support_values','bn_offset','label_self_values','label_residual','output_bias']
    colors = ['#207F83','#B9783A','#7461A3','#70808D','#B5BDC5']
    names = ['Support values','Shared BN offset','Label self-message','Label residual','Output bias']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,
        'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold',
        'axes.titlesize':11,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8})
    fig, axes = plt.subplots(1,2,figsize=(10,4.3),gridspec_kw={'width_ratios':[1.25,1]})
    fig.subplots_adjust(left=.08,right=.98,bottom=.29,top=.86,wspace=.32)
    ax = axes[0]; positive=np.zeros(3);negative=np.zeros(3)
    for comp,color,label in zip(components,colors,names):
        values=[]
        for condition in conditions:
            found=[r for r in rows if r['phase']=='long' and r['condition']==condition and r['component']==comp]
            assert len(found)==2
            values.append(np.mean([r['mean_margin'] for r in found]))
        values=np.array(values); bottoms=np.where(values>=0,positive,negative)
        ax.bar(range(3),values,bottom=bottoms,color=color,width=.52,label=label)
        positive+=np.maximum(values,0);negative+=np.minimum(values,0)
    totals=positive+negative
    ax.scatter(range(3),totals,c='#192B3C',s=29,marker='D',zorder=5,label='Total native margin')
    for i,total in enumerate(totals):ax.text(i+.30,total,f'{total:+.3f}',va='center',fontsize=8)
    ax.axhline(0,color='#7D8892',lw=.8);ax.grid(axis='y',alpha=.12);ax.set_axisbelow(True)
    ax.set_xticks(range(3),['Intact','Values only','Suppressed'])
    ax.set_ylabel('Mean global-class logit margin')
    ax.set_ylim(-.39,.53)
    ax.set_title('A  Score bias survives a much weaker contrast',loc='left',pad=14)
    ax.legend(frameon=False,ncol=2,loc='upper center',bbox_to_anchor=(.5,-.14),fontsize=7.5,columnspacing=.8)
    ax.text(.5,1.01,'50k political; mean over 2 x 128 episodes',transform=ax.transAxes,ha='center',fontsize=8,color='#596775')
    ax=axes[1]
    selected={r['condition']:r for r in cases if r['phase']=='long' and r['stream']=='original'}
    for qi,label,color,marker in [(3,'Conservative-labeled query','#207F83','o'),
                                  (21,'Non-conservative-labeled query','#B9783A','s')]:
        scores=[selected[c]['native_margins'][qi] for c in conditions]
        expected=1 if qi==3 else 0
        assert all(selected[c]['true_labels'][qi]==expected for c in conditions)
        ax.plot(range(3),scores,marker=marker,color=color,lw=1.7,ms=5,label=label)
        for i,score in enumerate(scores):
            offset=.023 if qi==3 and i>0 else -.034
            if qi==21:offset=-.036 if i>0 else .021
            ax.text(i,score+offset,f'{score:+.3f}',ha='center',va='center',fontsize=8,color=color)
    ax.axhline(0,color='#596775',lw=.9,ls='--')
    ax.text(.02,.55,'Conservative prediction above zero',transform=ax.transAxes,fontsize=7.5,color='#596775')
    ax.set_xticks(range(3),['Intact','Values only','Suppressed'])
    ax.set_ylabel('Global-class logit margin')
    ax.set_ylim(-.36,.32);ax.set_xlim(-.2,2.3)
    ax.grid(axis='y',alpha=.12)
    ax.set_title('B  Better ranking, worse decisions',loc='left',pad=14)
    ax.text(.5,1.01,'Two actual queries; first saved original episode',transform=ax.transAxes,ha='center',fontsize=8,color='#596775')
    ax.legend(frameon=False,loc='upper center',bbox_to_anchor=(.5,-.14),fontsize=7.5)
    fig.text(.08,.055,'A: additive accounting with observed label norms, not causal component-removal effects. B: an illustrative pair chosen within the fixed first episode.',fontsize=7.5)
    fig.text(.08,.02,'Query representations stay fixed in every condition. Positive is the dataset conservative label (25% prevalence); annotations are not independently verified.',fontsize=7.5)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,dpi=220,facecolor='white');plt.close(fig)
    print(args.output)


if __name__=='__main__':main()
