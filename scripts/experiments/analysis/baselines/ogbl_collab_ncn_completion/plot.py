"""Full frozen-budget curves and all selected seeds, no independent-data CI."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--runtime',type=Path,required=True);p.add_argument('--audit',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
r=json.loads(a.audit.read_text());fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
colors={'control':'#637282','observed':'#147c90','completion':'#d6662e'}
labels={'control':'Joint control','observed':'Observed neighbors','completion':'With completion'}
for arm in colors:
    curves=[]
    for seed in range(3):
        h=json.loads((a.runtime/f'{arm}{seed}/history.json').read_text());steps=[v['step'] for v in h];curves.append([100*v['validation_hits'] for v in h])
    curves=np.array(curves);axes[0].plot(steps,curves.mean(0),color=colors[arm],label=labels[arm]);axes[0].fill_between(steps,curves.min(0),curves.max(0),color=colors[arm],alpha=.12)
for seed in range(3):
    values=[100*next(x['hits'] for x in r['rows'] if x['arm']==arm and x['seed']==seed) for arm in colors]
    axes[1].plot(range(3),values,marker='o',label=f'Seed {seed}')
axes[0].set(xlabel='Training updates',ylabel='2018 Hits@50 (%)',title='Mean curve; shading spans three seeds')
axes[1].set(xticks=range(3),xticklabels=['Control','Observed','Completion'],ylabel='Selected Hits@50 (%)',title='Earliest best checkpoint, equal updates')
for ax in axes:ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2);ax.legend(frameon=False,fontsize=8)
fig.suptitle('Compact NCN and one-step completion',fontsize=14)
a.out.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.out,dpi=180)
