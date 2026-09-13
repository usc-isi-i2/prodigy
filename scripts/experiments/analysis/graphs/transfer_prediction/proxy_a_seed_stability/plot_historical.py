import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
HERE=Path(__file__).resolve().parent
root=HERE.parents[1]
d=json.loads((root/'structure/graph_divergence/data/graph_divergence_data.json').read_text())
names=d['graphs']; x=np.array(d['pairwise']['proxy_a_distance'])
f=pd.read_csv(root/'transfer_prediction/similarity_vs_transfer_v2/data/final_core_auc/transfer_matrix_three_seed_mean_long.csv')
print(f.metric.unique())
y=f[f.metric=='roc_auc_ovr_macro'].pivot(index='train',columns='test',values='value').reindex(index=names,columns=names).to_numpy()
assert y.shape==(9,9) and np.isfinite(y).all()
labels=['COVID','Ukraine','Midterm','Hong Kong','TwiBot20','Election2020','COVID Political','Suspension','Facebook']
colors=plt.get_cmap('tab10').colors[:9]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(3,3,figsize=(13,10),sharex=True,sharey=True)
rhos=[]
for j,ax in enumerate(axs.flat):
 keep=np.arange(9)!=j
 r=spearmanr(x[keep,j],y[keep,j]).statistic;rhos.append(r)
 for i in np.flatnonzero(keep):
  ax.scatter(x[i,j],y[i,j],s=70,color=colors[i],edgecolor='white',linewidth=.7,zorder=3)
 ax.set_title(f'Target: {labels[j]}   |   ρ = {r:.2f}',fontsize=11,loc='left',pad=10)
 ax.grid(alpha=.17);ax.set_xlim(0,2);ax.set_ylim(.45,1.0)
 ax.set_xticks([0,.5,1,1.5,2]);ax.set_yticks([.5,.6,.7,.8,.9,1])
fig.suptitle('More distinguishable graphs generally transfer worse',fontsize=20,x=.075,ha='left',y=.98)
fig.text(.075,.938,f'Raw-feature proxy-A vs. transfer ROC-AUC • mean within-target Spearman ρ = {np.mean(rhos):.3f}',fontsize=12,color='#444444')
fig.supxlabel('Proxy-A distance  (0 = chance graph identification; 2 = perfect identification)',y=.125)
fig.supylabel('Transfer ROC-AUC · mean of 3 training seeds',x=.02)
handles=[plt.Line2D([],[],marker='o',linestyle='',color=colors[i],label=labels[i],markersize=7) for i in range(9)]
fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.51,.035),ncol=5,frameon=False,title='Pretraining source')
fig.text(.075,.012,'8 foreign sources per target; self-transfer excluded. Fixed step 2,500; 512 shared test episodes per target.\nUses stored source→target proxy-A entries; matrix cells are dependent. Correlation does not establish causation.',fontsize=9,color='#555555')
fig.subplots_adjust(left=.075,right=.98,top=.89,bottom=.19,hspace=.35,wspace=.16)
for ext in ['png','pdf']:fig.savefig(HERE / ('figures/historical_proxy_a_vs_transfer.'+ext),dpi=180,facecolor='white')
print('mean rho',np.mean(rhos))
