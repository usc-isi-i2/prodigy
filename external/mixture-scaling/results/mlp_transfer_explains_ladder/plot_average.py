from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).parent
s=pd.read_csv(root/'data/single_source.tsv',sep='\t');l=pd.read_csv(root/'data/ladder.csv')
order=l.sort_values('n_sources').iloc[-1].sources.split(',');targets=list(s.target.drop_duplicates())
S=s.pivot(index='source',columns='target',values='value').loc[order,targets].to_numpy()
L=l.pivot(index='n_sources',columns='target',values='roc_auc').loc[range(1,10),targets].to_numpy()
mean=np.cumsum(S,axis=0)/np.arange(1,10)[:,None];best=np.maximum.accumulate(S,axis=0)
curves={'Observed ladder':L.mean(1)*100,'Mean specialist gain':(L[0]+mean-mean[0]).mean(1)*100,'Best specialist gain':(L[0]+best-best[0]).mean(1)*100}
names=['UKR','+COVID','+Midterm','+CovPol','+Election','+Suspended','+TwiBot','+HK','+Facebook'];x=np.arange(1,10)
fig,ax=plt.subplots(figsize=(10,5.8));ax.axvspan(5.5,9.3,color='#fff1d6')
for (label,values),style,color in zip(curves.items(),['o-','s--','D:'],['#202020','#2879bd','#ab488c']):ax.plot(x,values,style,color=color,label=label,lw=2.3,ms=5)
ax.set_xticks(x,names,rotation=30,ha='right');ax.set_xlim(.8,9.2);ax.set_ylabel('Mean cosine ROC-AUC (%)');ax.set_xlabel('Dataset added to cumulative mixture');ax.grid(alpha=.2);ax.legend(frameon=False,loc='lower left');ax.set_title('MLP ladder vs specialist predictions: average over six targets',loc='left',pad=15,weight='bold')
fig.text(.1,.02,'Equal target weights; predictions anchored at observed rung 1.\nShading: rungs trained with corrupted Suspended data. Specialist training budgets differ.',fontsize=10,color='#555555')
fig.tight_layout(rect=(0,.1,1,1));fig.savefig(root/'average.png',dpi=190);fig.savefig(root/'average.pdf')
pd.DataFrame({'rung':x,'added':order,**curves}).to_csv(root/'data/average.csv',index=False)
print(pd.DataFrame(curves).round(2).to_string(index=False))
