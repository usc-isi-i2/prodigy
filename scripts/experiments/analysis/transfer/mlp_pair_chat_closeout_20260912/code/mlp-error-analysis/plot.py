from pathlib import Path
from io import StringIO
import pandas as pd,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path('/tmp/mlp-error-analysis');m=pd.read_csv(P/'metrics.csv');m['delta']=100*(m.auc-m.baseline_auc)
s=pd.read_csv(StringIO((P/'aggregate_output.txt').read_text().split('AGGREGATE_STRATA\n')[1]));s.to_csv(P/'aggregate_strata.csv',index=False)
pairs=['Ukraine + Facebook','COVID political + suspended'];fig,axs=plt.subplots(2,2,figsize=(13,9.5))
colors=['#798b9e','#798b9e','#237b9f','#ba863f'];names=['A→B','B→A','Interleaved','Ensemble']
for j,pair in enumerate(pairs):
 ax=axs[0,j];g=m[m.pair==pair].groupby('model').delta.mean().reindex(names)
 ax.bar(names,g,color=colors,width=.65)
 for i,v in enumerate(g):ax.text(i,v+(.1 if v>=0 else -.1),f'{v:+.2f}',ha='center',va='bottom' if v>=0 else 'top',fontsize=11)
 ax.axhline(0,color='#777',lw=.8);ax.set_ylim(-2.7,4);ax.set_title(pair,fontsize=14,fontweight='bold');ax.set_ylabel('Mean AUC change (percentage points)');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
 ax=axs[1,j]
 for label,color,legend in [(1,'#2479a4','True edges'),(0,'#d17b32','Non-edges')]:
  z=s[(s.pair==pair)&(s.feature=='node_cosine')&(s.label==label)].set_index('bin').reindex(['Q1','Q2','Q3','Q4']);v=100*z.rank_delta
  ax.plot(range(4),v,marker='o',color=color,lw=2,label=legend)
 ax.axhline(0,color='#777',lw=.8);ax.set_ylim(-3.4,5.8);ax.set_xticks(range(4),['Lowest','Q2','Q3','Highest']);ax.set_xlabel('Endpoint feature cosine similarity\nWithin-target, within-class validation quartiles');ax.set_ylabel('Interleaved ranking-credit change (pp)');ax.grid(alpha=.15);ax.legend(frameon=False)
 for ax in axs[:,j]:ax.spines[['top','right']].set_visible(False)
fig.suptitle('Where combining two graphs helps—and hurts',fontsize=20,fontweight='bold',y=.985)
fig.text(.5,.937,'Baseline: singleton chosen by target-validation AUC · evaluated on six other graphs per pair · Election excluded',ha='center',fontsize=10)
fig.text(.5,.021,'Ranking credit = fraction of opposite-class test pairs ranked correctly by an edge. Lower panels show interleaving only.\nSingle seed; descriptive associations. Ensemble averages validation-calibrated singleton probabilities.',ha='center',fontsize=9,color='#555')
fig.subplots_adjust(top=.88,bottom=.12,hspace=.4,wspace=.26)
fig.savefig(P/'error_diagnostics.png',dpi=180,facecolor='white');fig.savefig(P/'error_diagnostics.svg',facecolor='white')
