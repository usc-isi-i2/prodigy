import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).parent
data=json.loads((root/'trajectories.json').read_text())
singles={r['sources'][0]:r for r in data['singleton']}
names={'ukr_rus_twitter':'Ukraine','facebook_page_reference':'Facebook','covid19_twitter':'COVID','midterm':'Midterm','covid_political':'Political','ukr_rus_suspended':'Suspended','cp_hk_twitter':'HK','twibot20':'TwiBot20'}
cases=[('ukr_rus_twitter','facebook_page_reference'),('covid19_twitter','midterm'),('covid_political','ukr_rus_suspended')]
def positives(run,source,updates):
 n=singles[source]['edge_counts']['supervision']
 batches=(n+1023)//1024
 epochs,rem=divmod(updates,batches)
 return epochs*n+min(rem*1024,n)
def reference(source):
 s=singles[source]
 return next(v['per_source'][source] for v in s['validation'] if v['step']==s['best_step'])
fig,axes=plt.subplots(3,2,figsize=(12,10),layout='constrained')
for row,pair in enumerate(cases):
 r=next(r for r in data['interleaved'] if tuple(r['sources'])==pair)
 for col,source in enumerate(pair):
  ax=axes[row,col];s=singles[source]
  ax.plot([v['step']/1000 for v in s['validation']],[v['per_source'][source]['roc_auc']*100 for v in s['validation']],color='#57606a',label='Singleton',lw=2)
  ax.plot([v['step']/2000 for v in r['validation']],[v['per_source'][source]['roc_auc']*100 for v in r['validation']],color='#1b86a6',label='Interleaved',lw=2)
  sel=next(v for v in r['validation'] if v['step']==r['best_step'])
  ax.scatter([r['best_step']/2000],[sel['per_source'][source]['roc_auc']*100],s=130,marker='*',color='#d75e26',zorder=5,label='Selected joint checkpoint')
  ax.axhline(reference(source)['roc_auc']*100,color='#57606a',ls=':',label='Selected singleton AUC')
  ax.set_title(f"{names[pair[0]]} + {names[pair[1]]}: evaluate {names[source]}",fontsize=11)
  ax.set_xlabel('Updates using this source (thousands)')
  ax.set_ylabel('Source validation AUC (%)')
  ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
handles,labels=axes[0,0].get_legend_handles_labels()
fig.legend(handles,labels,loc='outside lower center',ncol=4,frameon=False)
fig.suptitle('Does joint training preserve source performance?\nFixed validation pairs; x-axis matches source exposure, not total optimizer steps',fontsize=14)
fig.savefig(root/'source_trajectories.png',dpi=180);fig.savefig(root/'source_trajectories.pdf')
rows=[];pairrows=[]
for r in data['interleaved']:
 src=r['sources'];full={s:reference(s)['roc_auc'] for s in src}
 eligible={tol:[v['step'] for v in r['validation'] if all(100*(v['per_source'][s]['roc_auc']-full[s])>=-tol for s in src)] for tol in [0,.05,.1]}
 pairrows.append(dict(pair=' + '.join(src),selected_step=r['best_step'],both_retain_selected=all(100*(next(v for v in r['validation'] if v['step']==r['best_step'])['per_source'][s]['roc_auc']-full[s])>=0 for s in src),exact_steps=eligible[0],tol_005_steps=eligible[.05],tol_01_steps=eligible[.1]))
 for source in src:
  s=singles[source];singlevals={v['step']:v['per_source'][source] for v in s['validation']}
  for v in r['validation']:
   updates=v['step']//2
   match=singlevals.get(updates);budget=singlevals.get(v['step'])
   rows.append(dict(pair=' + '.join(src),source=source,total_step=v['step'],source_updates=updates,positive_examples=positives(r,source,updates),selected=v['step']==r['best_step'],auc=100*v['per_source'][source]['roc_auc'],bce=v['per_source'][source]['bce'],delta_to_selected_single_pp=100*(v['per_source'][source]['roc_auc']-full[source]),matched_exposure_delta_pp=100*(v['per_source'][source]['roc_auc']-match['roc_auc']) if match else None,matched_budget_delta_pp=100*(v['per_source'][source]['roc_auc']-budget['roc_auc']) if budget else None))
pd.DataFrame(rows).to_csv(root/'trajectory_comparison.csv',index=False)
pd.DataFrame(pairrows).to_csv(root/'pair_retention.csv',index=False)
print('pair counts',len(pairrows),'selected preserve both',sum(r['both_retain_selected'] for r in pairrows),'any logged',sum(bool(r['exact_steps']) for r in pairrows),'within .1',sum(bool(r['tol_01_steps']) for r in pairrows))
t=pd.DataFrame(rows)
for pair in cases:
 g=t[t.pair==' + '.join(pair)]
 print(pair)
 print(g[g.total_step==max(g.total_step)//4000*4000][['source','total_step','auc','matched_exposure_delta_pp','matched_budget_delta_pp']].round(4).to_string(index=False))
