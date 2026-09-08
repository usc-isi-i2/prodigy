"""Display archived SAMGPT native GraphCL ladders, highlighting held-out targets."""
from pathlib import Path
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
HERE=Path(__file__).resolve().parent
REPO=next(p for p in HERE.parents if (p/'docs/graph_catalog.json').exists())
raw=REPO/'scripts/experiments/analysis/transfer/ladders/prodigy_nm/robustness/nm_ladder_order_robustness/data/samgpt_graphcl_9x3_carc_v100'
d=pd.read_csv(raw/'metrics_long.csv')
orders=json.loads((raw/'manifest.json').read_text())['orders']
assert len(d)==243 and not d.duplicated(['order','rung','target']).any()
assert d.groupby(['order','target']).rung.nunique().eq(9).all()
targets=['covid_political','election2020','ukr_rus_suspended','twibot20','facebook_page_reference','ukr_rus_twitter','covid19_twitter','midterm','cp_hk_twitter']
colors=dict(zip(targets,['#4477AA','#EE7733','#AA3377','#228833','#8866BB','#0099AA','#CC6677','#997744','#888811']))
names={g['dataset_key']:g['canonical_name'] for g in json.loads((REPO/'docs/graph_catalog.json').read_text())['graphs']}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,3,figsize=(15,7.2),sharey=True)
for ax,order in zip(axes,'ABC'):
 for target in targets:
  part=d[d.order.eq(order)&d.target.eq(target)].sort_values('rung')
  entry=orders[order].index(target)+1
  assert (part.in_training == (part.rung>=entry)).all()
  ax.plot(part.rung,part.probability_margin,color=colors[target],alpha=.22,ls='--',lw=1)
  held=part[part.rung<entry]
  ax.plot(held.rung,held.probability_margin,color=colors[target],lw=1.8,marker='o',ms=3.5)
  row=part[part.rung==entry].iloc[0]
  ax.scatter([entry],[row.probability_margin],marker='s',facecolors='white',edgecolors=colors[target],s=28,zorder=4)
 ax.set_title(f'Source order {order}',fontsize=13)
 ax.set_xticks(range(1,10));ax.set_xlabel('Number of pretraining graphs')
 ax.set_ylim(-.02,1.03);ax.grid(alpha=.17)
 ax.set_axisbelow(True)
axes[0].set_ylabel('Native GraphCL probability margin (higher is better)')
fig.suptitle('SAMGPT · native GraphCL ladder',x=.065,ha='left',fontsize=19,y=.97)
fig.text(.065,.91,'Solid: target held out   ·   □: target enters training   ·   Faint dashed: target included',color='#56606B',fontsize=11)
handles=[Line2D([],[],color=colors[t],marker='o',label=names[t],lw=1.5,ms=4) for t in targets]
fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.09),ncol=5,frameon=False,fontsize=9)
fig.text(.065,.065,'Archived August 2026 ladder: 27 models × 9 targets; one seed and one fixed unseen GraphCL view per target.',fontsize=9,color='#56606B')
fig.text(.065,.03,'At rung 9 every target is in training. This is a separate experiment from the recent 500-update pair sweep.',fontsize=9,color='#56606B')
fig.subplots_adjust(left=.065,right=.985,top=.81,bottom=.24,wspace=.13)
(HERE/'figures').mkdir(exist_ok=True)
for ext in ['png','pdf']:fig.savefig(HERE/f'figures/native_ladder_overview.{ext}',dpi=180,bbox_inches='tight')
