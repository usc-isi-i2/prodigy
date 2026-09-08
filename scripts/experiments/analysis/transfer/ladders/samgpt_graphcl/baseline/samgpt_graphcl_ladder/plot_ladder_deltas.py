"""Adjacent-rung native GraphCL margin changes for the archived SAMGPT ladder."""
from pathlib import Path
import argparse
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
parser=argparse.ArgumentParser(__doc__)
parser.add_argument('--zoom',type=float,default=None,help='Symmetric y-axis limit in margin units')
args=parser.parse_args()
if args.zoom is not None and args.zoom<=0: parser.error('--zoom must be positive')
HERE=Path(__file__).resolve().parent
REPO=next(p for p in HERE.parents if (p/'docs/graph_catalog.json').exists())
raw=REPO/'scripts/experiments/analysis/transfer/ladders/prodigy_nm/robustness/nm_ladder_order_robustness/data/samgpt_graphcl_9x3_carc_v100'
d=pd.read_csv(raw/'metrics_long.csv').sort_values(['order','target','rung'])
orders=json.loads((raw/'manifest.json').read_text())['orders']
assert len(d)==243 and not d.duplicated(['order','target','rung']).any()
d['delta_margin']=d.groupby(['order','target']).probability_margin.diff()
d=d[d.rung.gt(1)].copy()
d['entry_rung']=[orders[o].index(t)+1 for o,t in zip(d.order,d.target)]
d['target_role']=['held_out' if r<e else 'enters' if r==e else 'already_included' for r,e in zip(d.rung,d.entry_rung)]
assert len(d)==216 and d.delta_margin.notna().all()
d.to_csv(HERE/'data/native_ladder_adjacent_deltas.csv',index=False)
targets=['covid_political','election2020','ukr_rus_suspended','twibot20','facebook_page_reference','ukr_rus_twitter','covid19_twitter','midterm','cp_hk_twitter']
colors=dict(zip(targets,['#4477AA','#EE7733','#AA3377','#228833','#8866BB','#0099AA','#CC6677','#997744','#888811']))
names={g['dataset_key']:g['canonical_name'] for g in json.loads((REPO/'docs/graph_catalog.json').read_text())['graphs']}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,3,figsize=(15,7.2),sharey=True)
for ax,order in zip(axes,'ABC'):
 for target in targets:
  p=d[d.order.eq(order)&d.target.eq(target)].sort_values('rung')
  ax.plot(p.rung,p.delta_margin,color=colors[target],alpha=.22,ls='--',lw=1)
  held=p[p.target_role.eq('held_out')]
  ax.plot(held.rung,held.delta_margin,color=colors[target],lw=1.7,marker='o',ms=4)
  entry=p[p.target_role.eq('enters')]
  ax.scatter(entry.rung,entry.delta_margin,marker='s',facecolors='white',edgecolors=colors[target],s=35,zorder=5)
 if args.zoom is not None:
  ax.set_ylim(-args.zoom,args.zoom)
  for target in targets:
   clipped=d[d.order.eq(order)&d.target.eq(target)&d.delta_margin.abs().gt(args.zoom)]
   for _,row in clipped.iterrows():
    sign=1 if row.delta_margin>0 else -1
    ax.scatter(row.rung,sign*args.zoom*.97,marker='^' if sign>0 else 'v',s=45,color=colors[target],zorder=6)
 ax.axhline(0,color='#505964',ls='--',lw=1)
 ax.set_title(f'Source order {order}',fontsize=13)
 ax.set_xticks(range(2,10),[f'{i-1}→{i}' for i in range(2,10)])
 ax.set_xlabel('Pretraining graphs: previous → current rung')
 ax.grid(axis='y',alpha=.17);ax.set_axisbelow(True)
axes[0].set_ylabel('GraphCL probability-margin change vs previous rung')
fig.suptitle('SAMGPT · adjacent-rung changes, zoomed' if args.zoom is not None else 'SAMGPT · change from adding the next graph',x=.065,ha='left',fontsize=19,y=.97)
fig.text(.065,.91,'Positive = improvement · solid: still held out · □: target enters · faint dashed: target included',fontsize=11,color='#56606B')
handles=[Line2D([],[],color=colors[t],marker='o',label=names[t],lw=1.5,ms=4) for t in targets]
fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.09),ncol=5,frameon=False,fontsize=9)
fig.text(.065,.065,(f'Zoom: ±{args.zoom:g} margin units. Triangles at the boundary mark off-scale deltas. ' if args.zoom is not None else '')+'Δ = margin at rung k − margin at rung k−1, for the same target and source order. Raw margin units; common y-scale.',fontsize=9,color='#56606B')
fig.text(.065,.03,'Archived August 2026 ladder · one training seed · each rung is a separate mixture model. At rung 9 no target remains held out.',fontsize=9,color='#56606B')
fig.subplots_adjust(left=.065,right=.985,top=.81,bottom=.24,wspace=.13)
suffix=f'_zoom_{args.zoom:g}' if args.zoom is not None else ''
for ext in ['png','pdf']:fig.savefig(HERE/f'figures/native_ladder_adjacent_deltas{suffix}.{ext}',dpi=180,bbox_inches='tight')
print(d.loc[d.delta_margin.abs().nlargest(4).index,['order','rung','target','delta_margin','target_role']].to_string(index=False))
