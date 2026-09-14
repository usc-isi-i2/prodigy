"""Plot all predeclared seeds and matched update budgets; no confidence intervals."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--runtime',type=Path,required=True);p.add_argument('--audit',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
r=json.loads(a.audit.read_text());fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
colors={'bce':'#637282','warm':'#d6662e'}
for arm,label in [('bce','All historical positives'),('warm','Warm-endpoint positives')]:
    curves=[]
    for seed in range(3):
        h=json.loads((a.runtime/f'{arm}{seed}/history.json').read_text());steps=[v['step'] for v in h];curves.append([100*v['validation_hits'] for v in h])
    curves=np.array(curves)
    axes[0].plot(steps,curves.mean(0),color=colors[arm],label=label)
    axes[0].fill_between(steps,curves.min(0),curves.max(0),color=colors[arm],alpha=.13)
for seed in range(3):
    pair=[next(x for x in r['rows'] if x['arm']==arm and x['seed']==seed) for arm in ['bce','warm']]
    axes[1].plot([0,1],[100*x['hits'] for x in pair],marker='o',label=f'Seed {seed}')
axes[0].set(xlabel='Training updates',ylabel='2018 validation Hits@50 (%)',title='Mean curve; shading spans three seeds')
axes[1].set(xticks=[0,1],xticklabels=['All positives','Warm positives'],ylabel='Selected Hits@50 (%)',title='Earliest best checkpoint, equal budgets')
for ax in axes: ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2);ax.legend(frameon=False,fontsize=8)
fig.suptitle('Warm-endpoint training: matched joint-model test',fontsize=14)
a.out.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.out,dpi=180)
