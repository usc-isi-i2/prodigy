"""Compare completed 4k and 40k proxy-A repeats without changing NM outcomes."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

base=Path(__file__).resolve().parent
large=base/'sample_40000'
a=json.loads((base/'data/results.json').read_text());b=json.loads((large/'data/results.json').read_text())
assert a['graphs']==b['graphs']
assert json.loads((base/'data/protocol.json').read_text())['samples']==4000
assert json.loads((large/'data/protocol.json').read_text())['samples']==40000
upper=np.triu_indices(9,1)
rows=[]
fig,axs=plt.subplots(1,2,figsize=(11,5),sharex=True,sharey=True)
for ax,policy in zip(axs,['legacy_sorted','uniform']):
 x=pd.read_csv(base/f'data/{policy}_pairs.csv');y=pd.read_csv(large/f'data/{policy}_pairs.csv')
 assert x[['source','target']].equals(y[['source','target']])
 delta=100*(y.accuracy_mean-x.accuracy_mean)
 ax.errorbar(x.accuracy_mean*100,y.accuracy_mean*100,xerr=x.accuracy_sd*100,yerr=y.accuracy_sd*100,fmt='o',ms=4,alpha=.8,capsize=2)
 ax.plot([50,100],[50,100],color='gray',ls='--');ax.set_xlim(50,100);ax.set_ylim(50,100);ax.grid(alpha=.2)
 ax.set_title(policy.replace('_',' '));ax.set_xlabel('4,000 nodes: graph-classifier accuracy (%)')
 rows.append(dict(policy=policy,mean_absolute_accuracy_shift_pp=float(abs(delta).mean()),max_absolute_accuracy_shift_pp=float(abs(delta).max()),mean_accuracy_sd_4k_pp=float(x.accuracy_sd.mean()*100),mean_accuracy_sd_40k_pp=float(y.accuracy_sd.mean()*100),pair_rank_spearman=float(spearmanr(x.accuracy_mean,y.accuracy_mean).statistic)))
 x['accuracy_40k']=y.accuracy_mean;x['accuracy_shift_pp']=delta
 x.to_csv(large/f'data/{policy}_sample_size_comparison.csv',index=False)
axs[0].set_ylabel('40,000 nodes: graph-classifier accuracy (%)')
fig.suptitle('Does proxy-A change with ten times as many sampled nodes?',fontsize=15)
fig.text(.5,.015,'36 graph pairs; means ±1 SD across three estimator seeds. Dashed line: unchanged accuracy.',ha='center',fontsize=9)
fig.tight_layout(rect=[0,.04,1,.94])
for ext in ['png','pdf']:fig.savefig(large/f'figures/sample_size_comparison.{ext}',dpi=180)
pd.DataFrame(rows).to_csv(large/'data/sample_size_summary.csv',index=False)
print(pd.DataFrame(rows).to_string(index=False))
