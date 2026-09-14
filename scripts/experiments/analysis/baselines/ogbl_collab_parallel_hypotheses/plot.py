"""Plot completed matched experiments; seed ranges are not confidence intervals."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--renewal',type=Path,required=True);p.add_argument('--tail',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
recipes=[('BCE control',a.tail,'bce','validation_hits','#555c65'),('Renewed negatives',a.renewal,'renew','hits','#1971b8'),('BCE + tail loss',a.tail,'tail','validation_hits','#c86c26')]
fig,axes=plt.subplots(1,2,figsize=(12,4.7),layout='constrained');peaks=[]
for label,root,arm,key,color in recipes:
    hs=[json.loads((root/f'{arm}{seed}/history.json').read_text()) for seed in range(3)]
    steps=np.array([x['step'] for x in hs[0]])
    values=np.array([[x[key]*100 for x in h] for h in hs]);assert values.shape==(3,40)
    axes[0].plot(steps,values.mean(0),label=label,color=color,lw=2)
    axes[0].fill_between(steps,values.min(0),values.max(0),color=color,alpha=.1)
    selected=values.max(1);peaks.append(selected)
    i=len(peaks)-1;axes[1].scatter(i,selected.mean(),color=color,marker='_',s=900,linewidths=3)
    axes[1].scatter(np.full(3,i),selected,color='black',s=16,zorder=3)
    axes[1].text(i,selected.max()+.1,f'{selected.mean():.3f}%',ha='center',fontsize=10)
axes[0].set(title='Training curves: mean and range across three seeds',xlabel='Training updates',ylabel='2018 validation Hits@50 (%)')
axes[0].legend(frameon=False,fontsize=9)
flat=np.concatenate(peaks);lo=min(65,float(flat.min())-1);hi=float(flat.max())+1
axes[1].set(title='Best standalone checkpoint selected for each seed',xticks=range(3),xticklabels=['BCE\ncontrol','Renewed\nnegatives','BCE +\ntail loss'],ylabel='Selected 2018 validation Hits@50 (%)',ylim=(lo,hi))
for ax in axes:
    ax.grid(axis='y',alpha=.15);ax.spines[['top','right']].set_visible(False)
fig.suptitle('Matched historical experiments — no new 2019 evaluation',fontsize=14,fontweight='bold')
a.out.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.out,dpi=160)
