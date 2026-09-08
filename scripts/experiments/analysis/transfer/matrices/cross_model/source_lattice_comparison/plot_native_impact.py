"""Within-model native-task pair effects; metrics retain separate scales."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser(__doc__)
parser.add_argument('--heldout-only',action='store_true')
parser.add_argument('--include-mae-lp',action='store_true',help='Include GraphMAE downstream LP results')
args=parser.parse_args()
prefix='native_heldout' if args.heldout_only else 'native'
if args.include_mae_lp: prefix += '_with_mae_lp'
HERE=Path(__file__).resolve().parent
REPO=next(p for p in HERE.parents if (p/'docs/graph_catalog.json').exists())
GFM=REPO.parent
paths={
 'PRODIGY / NM':REPO/'scripts/experiments/analysis/transfer/matrices/prodigy_nm/source_additions/nm_pairwise_source_additions/data/nm_directional_intervention_deltas_all_targets_long.csv',
 'GraphSAGE / LP':GFM/'mixture-scaling-graphmae/results/social9_source_lattice/aggregated/lp_static_link_prediction.tsv',
 'SAMGPT / GraphCL':GFM/'samgpt-social-lattice/analysis/source_lattice/data/native/cells.csv',
}
if args.include_mae_lp:
 paths={**dict(list(paths.items())[:2]),'GraphSAGE / GraphMAE':GFM/'mixture-scaling-graphmae/results/social9_source_lattice/aggregated/graphmae_static_link_prediction.tsv',**dict(list(paths.items())[2:])}
aliases={'ukr_rus':'ukr_rus_twitter','covid':'covid19_twitter','cp_hk':'cp_hk_twitter'}
frames=[]
for model,path in paths.items():
 d=pd.read_csv(path,sep='\t' if path.suffix=='.tsv' else ',')
 if model.startswith('PRODIGY'):
  d=d.rename(columns={'base_source':'baseline_source','base_specialist_auc':'singleton_value','pair_auc':'pair_value'})
  d['model']=model
  d['impact']=d.delta_auc
  for col in ['target','baseline_source','added_source']:d[col]=d[col].map(lambda s:aliases.get(s,s))
  d['metric']='macro ROC-AUC'
  frames.append(d[['model','pair_model_id','target','baseline_source','added_source','singleton_value','pair_value','impact','metric']])
  continue
 if model.startswith('SAMGPT'):
  d=d[d.checkpoint_update.eq(500)].copy()
  assert set(d.seed)=={39} and len(d)==486
  metric='probability margin'
  d=d.rename(columns={'probability_margin':'value'})
 else:
  assert len(d)==324 and set(d.objective)==({'graphmae'} if model.endswith('GraphMAE') else {'lp'})
  metric='link-prediction ROC-AUC'
  d=d.rename(columns={'auc':'value','run_id':'model_id'})
 d['members']=d.sources.str.split(',').map(tuple)
 d['k']=d.members.map(len)
 assert not d.duplicated(['sources','target']).any()
 assert np.isfinite(d.value).all()
 assert d.groupby('target').size().eq(54).all()
 records=[]
 for _,pair in d[d.k.eq(2)].iterrows():
  for base in pair.members:
   match=d[d.k.eq(1)&d.target.eq(pair.target)&d.sources.eq(base)]
   assert len(match)==1
   single=match.iloc[0]
   records.append(dict(model=model,pair_model_id=pair.model_id,target=pair.target,baseline_source=base,
      added_source=next(s for s in pair.members if s!=base),singleton_value=single.value,pair_value=pair.value,
      impact=pair.value-single.value,metric=metric))
 frames.append(pd.DataFrame(records))
all_data=pd.concat(frames,ignore_index=True)
assert all_data.groupby('model').size().to_dict()=={'PRODIGY / NM':648,'GraphSAGE / LP':432,'SAMGPT / GraphCL':648,**({'GraphSAGE / GraphMAE':432} if args.include_mae_lp else {})}
assert not all_data.duplicated(['model','baseline_source','added_source','target']).any()
if args.heldout_only:
 all_data=all_data[all_data.target.ne(all_data.baseline_source)&all_data.target.ne(all_data.added_source)].copy()
 assert all_data.groupby('model').size().to_dict()=={'PRODIGY / NM':504,'GraphSAGE / LP':336,'SAMGPT / GraphCL':504,**({'GraphSAGE / GraphMAE':336} if args.include_mae_lp else {})}
best=all_data.sort_values('singleton_value').groupby(['model','pair_model_id','target'],as_index=False).tail(1)
all_data.to_csv(HERE/f'data/{prefix}_each_singleton_contrasts.csv',index=False)
best.to_csv(HERE/f'data/{prefix}_best_singleton_contrasts.csv',index=False)
summary=best.groupby('model').impact.agg(mean='mean',median='median',win_fraction=lambda x:(x>0).mean(),n='size').reindex(paths)
summary.to_csv(HERE/f'data/{prefix}_summary.csv')
(HERE/f'data/{prefix}_provenance.json').write_text(json.dumps({m:{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for m,p in paths.items()},indent=2)+'\n')
targets=['covid_political','election2020','ukr_rus_suspended','twibot20','facebook_page_reference','ukr_rus_twitter','covid19_twitter','midterm','cp_hk_twitter']
colors=dict(zip(targets,['#4477AA','#EE7733','#AA3377','#228833','#8866BB','#0099AA','#CC6677','#997744','#888811']))
names={g['dataset_key']:g['canonical_name'] for g in json.loads((REPO/'docs/graph_catalog.json').read_text())['graphs']}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,len(paths),figsize=(18 if args.include_mae_lp else 15,7.4))
rng=np.random.default_rng(8)
for ax,model in zip(axes,paths):
 scale=1 if model.startswith('SAMGPT') else 100
 for i,view in enumerate([all_data,best]):
  view=view[view.model.eq(model)]
  present_targets=[target for target in targets if target in set(view.target)]
  offsets=np.linspace(-.32,.32,len(present_targets))
  half_width=.24/max(len(present_targets),1)
  for target,offset in zip(present_targets,offsets):
   vals=view.loc[view.target.eq(target),'impact'].to_numpy()*scale
   ax.scatter(i+offset+rng.uniform(-half_width,half_width,len(vals)),vals,s=14,alpha=.6,linewidths=0,color=colors[target],zorder=3)
  median=view.impact.median()*scale
  ax.plot([i-.37,i+.37],[median]*2,color='#15202B',lw=2.2,zorder=4)
 ax.axhline(0,color='#637082',lw=1,ls='--')
 ax.set_xticks([0,1],['Pair − each\nsingleton','Pair − better\nsingleton'])
 ax.set_xlim(-.55,1.55)
 model_best=best[best.model.eq(model)]
 n=model_best.target.nunique()
 task_label='Downstream link prediction' if model.endswith('GraphMAE') else ('Neighbor matching' if model.startswith('PRODIGY') else ('Link prediction' if model.startswith('GraphSAGE') else 'GraphCL discrimination'))
 ax.set_title(f'{model}\n{task_label}\n{n} targets · {len(model_best)} pair–target cells',fontsize=11,pad=12)
 ax.set_ylabel('Probability-margin change' if scale==1 else 'ROC-AUC change (percentage points)')
 ax.grid(axis='y',alpha=.17);ax.set_axisbelow(True)
fig.suptitle(('Pair impact · held-out targets only' if args.include_mae_lp else 'Native-objective pair impact · held-out targets only') if args.heldout_only else 'Native-objective pair impact · targets shown by color',x=.06,ha='left',fontsize=18,y=.97)
fig.text(.06,.915,'Targets grouped in legend order · separate metric scales · black bars = medians',color='#56606B',fontsize=11)
handles=[plt.Line2D([],[],marker='o',linestyle='',color=colors[t],label=names[t],markersize=6) for t in targets]
fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.115),ncol=9,frameon=False,fontsize=8)
fig.text(.06,.09,'GraphMAE panel shows available downstream link-prediction AUC, not its masked-feature reconstruction objective.' if args.include_mae_lp else 'GraphMAE reconstruction: no per-target native evaluation grid found. Its static-link results are a downstream task and are excluded.',fontsize=9,color='#56606B')
fig.text(.06,.058,'PRODIGY: 2,500 updates / seed 0. GraphSAGE: SSL-validation-selected checkpoints / seed 0. SAMGPT: 500 updates / seed 39.',fontsize=9,color='#56606B')
fig.text(.06,.026,'Held-out = target absent from both pair sources. One seed per model; native tasks, protocols, and scales differ; dots are dependent.' if args.heldout_only else 'One training seed per model; includes source-seen and held-out targets. Native tasks, evaluation protocols, and scales differ; dots are dependent.',fontsize=9,color='#56606B')
fig.subplots_adjust(left=.065,right=.985,top=.79,bottom=.29,wspace=.34)
for ext in ['png','pdf']:fig.savefig(HERE/f'figures/{prefix}_pair_impact_by_model.{ext}',dpi=180,bbox_inches='tight')
print(summary.to_string())
