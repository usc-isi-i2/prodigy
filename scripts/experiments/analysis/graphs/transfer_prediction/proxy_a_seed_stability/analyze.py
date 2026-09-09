"""Validate three-seed proxy-A outputs and compare to fixed NM transfer AUC."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(__doc__)
parser.add_argument('--run-dir',type=Path,default=BASE,help='Folder containing data/results.json and data/protocol.json')
HERE=parser.parse_args().run_dir
protocol=json.loads((HERE/'data/protocol.json').read_text())
nsamples=int(protocol['samples'])
d=json.loads((HERE/'data/results.json').read_text()); names=d['graphs']
assert len(names)==9 and len(d['runs'])==6
assert {(r['policy'],r['seed']) for r in d['runs']}=={(p,s) for p in ['legacy_sorted','uniform'] for s in range(3)}
f=pd.read_csv(BASE.parent/'similarity_vs_transfer_v2/data/final_core_auc/transfer_matrix_three_seed_mean_long.csv')
y=f[f.metric=='roc_auc_ovr_macro'].pivot(index='train',columns='test',values='value').reindex(index=names,columns=names).to_numpy()
assert np.isfinite(y).all()
mask=~np.eye(9,dtype=bool); upper=np.triu_indices(9,1)
rows=[];summary=[];matrices={}
for policy in ['legacy_sorted','uniform']:
 a=np.array([r['proxy_a_distance'] for r in sorted(d['runs'],key=lambda r:r['seed']) if r['policy']==policy])
 assert a.shape==(3,9,9) and np.isfinite(a).all() and np.allclose(a,a.transpose(0,2,1))
 matrices[policy]=a
 for seed in range(3):
  rhos=[spearmanr(a[seed,mask[:,j],j],y[mask[:,j],j]).statistic for j in range(9)]
  summary.append(dict(policy=policy,seed=seed,mean_target_spearman=float(np.mean(rhos))))
  for j,rho in enumerate(rhos): rows.append(dict(policy=policy,seed=seed,target=names[j],spearman=rho))
 pairs=[]
 for i,j in zip(*upper):
  pairs.append(dict(source=names[i],target=names[j],proxy_a_mean=a[:,i,j].mean(),proxy_a_sd=a[:,i,j].std(ddof=1),accuracy_mean=(a[:,i,j].mean()+2)/4,accuracy_sd=a[:,i,j].std(ddof=1)/4))
 pd.DataFrame(pairs).to_csv(HERE/f'data/{policy}_pairs.csv',index=False)
pd.DataFrame(summary).to_csv(HERE/'data/correlation_by_seed.csv',index=False)
pd.DataFrame(rows).to_csv(HERE/'data/correlation_by_target.csv',index=False)
labels=['COVID','Ukraine','Midterm','Hong Kong','TwiBot20','Election2020','COVID Political','Suspension','Facebook']
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
(HERE/'figures').mkdir(exist_ok=True)
fig,axs=plt.subplots(3,3,figsize=(13,10),sharex=True,sharey=True)
colors=plt.get_cmap('tab10').colors
for j,ax in enumerate(axs.flat):
 for i in range(9):
  if i==j:continue
  a=matrices['uniform'][:,i,j]
  ax.errorbar(a.mean(),y[i,j],xerr=a.std(ddof=1),fmt='o',color=colors[i],capsize=3,ms=6)
 ax.set_title('Target: '+labels[j],loc='left');ax.grid(alpha=.18);ax.set_xlim(0,2);ax.set_ylim(.45,1)
fig.suptitle('Proxy-A repeatability with uniform node sampling',fontsize=19,y=.98)
fig.text(.07,.934,'Dots: three-estimator-seed means; horizontal bars: ±1 sample SD. NM AUC fixed at three-training-seed means.',fontsize=11)
fig.supxlabel('Raw-feature proxy-A distance',y=.115);fig.supylabel('Target NM ROC-AUC',x=.015)
fig.legend(handles=[plt.Line2D([],[],marker='o',linestyle='',color=colors[i],label=labels[i]) for i in range(9)],loc='lower center',bbox_to_anchor=(.5,.025),ncol=5,frameon=False,title='Pretraining source')
fig.text(.07,.008,f'{nsamples:,} nonzero-feature nodes/graph. Self-transfer excluded. Three seeds measure repeatability, not sample-size convergence.',fontsize=9)
fig.subplots_adjust(left=.07,right=.98,top=.88,bottom=.18,hspace=.3)
for ext in ['png','pdf']:fig.savefig(HERE/f'figures/proxy_a_seed_stability.{ext}',dpi=180)
report=['# Proxy-A: three-seed stability','', f'Raw features, {nsamples:,} nonzero nodes per graph. Joint node-sampling and classifier-split variation. Fixed NM transfer outcomes.','']
for policy,a in matrices.items():
 vals=[r['mean_target_spearman'] for r in summary if r['policy']==policy]
 sd=a.std(axis=0,ddof=1)[upper]
 report.extend([f'## {policy}',f'- Within-target mean Spearman for seeds 0/1/2: '+', '.join(f'{v:.4f}' for v in vals),f'- Mean / maximum pairwise accuracy SD: {sd.mean()*25:.3f} / {sd.max()*25:.3f} percentage points.',''])
shift=(matrices['uniform'].mean(0)-matrices['legacy_sorted'].mean(0))[upper]
report.extend([f'Mean absolute sampling-policy shift in domain accuracy: {np.abs(shift).mean()*25:.3f} percentage points.', '', 'Historical sorted-candidate sampling favors low node IDs. Both policies use new random streams; legacy seed 0 is not an exact historical replay. Three estimator seeds are not training replications or a confidence interval. No inference of causal transfer effects.'])
(HERE/'FINDINGS.md').write_text('\n'.join(report)+'\n')
print('\n'.join(report))
