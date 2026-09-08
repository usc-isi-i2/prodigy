"""Compare within-model source-pair effects across the four completed lattices."""
from pathlib import Path
import ast
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / 'docs/graph_catalog.json').exists())
GFM = REPO.parent
sources = {
    'PRODIGY / NM': REPO / 'scripts/experiments/analysis/graphs/transfer_prediction/target_performance_mechanisms/data/raw/classification_long.tsv',
    'GraphSAGE / LP': GFM / 'mixture-scaling-graphmae/results/social9_source_lattice/aggregated/lp_classification.tsv',
    'GraphSAGE / GraphMAE': GFM / 'mixture-scaling-graphmae/results/social9_source_lattice/aggregated/graphmae_classification.tsv',
    'SAMGPT / GraphCL': GFM / 'samgpt-social-lattice/analysis/source_lattice/data/cls/cells.csv',
}
aliases={'ukr_rus':'ukr_rus_twitter','covid':'covid19_twitter','cp_hk':'cp_hk_twitter'}
records=[]
for model, path in sources.items():
    d=pd.read_csv(path,sep='\t' if path.suffix=='.tsv' else ',')
    if model.startswith('PRODIGY'):
        d=d.rename(columns={'dataset':'target','roc_auc':'auc'})
        d['members']=d.sources.map(ast.literal_eval)
        assert set(d.checkpoint_step)=={2500} and set(d.training_seed)=={0}
    else:
        d['members']=d.sources.str.split(',')
        d=d.rename(columns={'run_id':'model_id','roc_auc_mean':'auc'})
    d['members']=d.members.map(lambda ss: tuple(sorted(aliases.get(s,s) for s in ss)))
    d['k']=d.members.map(len)
    assert len(d)==270 and not d.duplicated(['members','target']).any()
    assert d.groupby('k').size().to_dict()=={1:45,2:180,8:45}
    if 'episode_fingerprint' in d:
        assert d.groupby('target').episode_fingerprint.nunique().eq(1).all()
    assert np.isfinite(d.auc).all() and d.auc.between(0,1).all()
    for _, pair in d[d.k.eq(2)].iterrows():
        for base in pair.members:
            single=d[(d.k==1)&d.target.eq(pair.target)&d.members.map(lambda ss:ss==(base,))].iloc[0]
            records.append(dict(model=model,pair_model_id=pair.model_id,sources=','.join(pair.members),target=pair.target,
                baseline_source=base,added_source=next(s for s in pair.members if s!=base),pair_auc=pair.auc,
                singleton_auc=single.auc,impact_pp=100*(pair.auc-single.auc),target_in_pair=pair.target in pair.members))
contrasts=pd.DataFrame(records)
assert len(contrasts)==1440
best=contrasts.sort_values('singleton_auc').groupby(['model','pair_model_id','target'],as_index=False).tail(1)
assert len(best)==720
contrasts.to_csv(HERE/'data/each_singleton_contrasts.csv',index=False)
best.to_csv(HERE/'data/best_singleton_contrasts.csv',index=False)
(HERE/'data/provenance.json').write_text(json.dumps({m:{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for m,p in sources.items()},indent=2)+'\n')
targets=['covid_political','election2020','ukr_rus_suspended','twibot20','facebook_page_reference']
catalog=json.loads((REPO/'docs/graph_catalog.json').read_text())
names={g['dataset_key']:g['canonical_name'] for g in catalog['graphs']}
colors=dict(zip(targets,['#4477AA','#EE7733','#AA3377','#228833','#8866BB']))
models=list(sources)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(14,7.2),sharey=True)
rng=np.random.default_rng(7)
for ax,data,title in zip(axes,[contrasts,best],['Pair − each constituent singleton','Pair − better constituent singleton']):
    for i,model in enumerate(models):
        for j,target in enumerate(targets):
            vals=data.loc[data.model.eq(model)&data.target.eq(target),'impact_pp'].to_numpy()
            ax.scatter(i+(j-2)*.135+rng.uniform(-.055,.055,len(vals)),vals,s=12,color=colors[target],alpha=.6,linewidths=0,zorder=3)
        median=data.loc[data.model.eq(model),'impact_pp'].median()
        ax.plot([i-.37,i+.37],[median,median],color='#15202B',lw=2.2,zorder=4)
    ax.axhline(0,color='#637082',lw=1,ls='--')
    ax.set_xticks(range(4),['PRODIGY\nNM','GraphSAGE\nLP','GraphSAGE\nGraphMAE','SAMGPT\nGraphCL'])
    ax.set_xlim(-.6,3.6)
    ax.set_title(title,fontsize=13,pad=12)
    ax.grid(axis='y',alpha=.17);ax.set_axisbelow(True)
axes[0].set_ylabel('Classification ROC-AUC change (percentage points)')
fig.suptitle('Recent pair experiments · impact by model and objective',fontsize=18,x=.065,ha='left',y=.965)
fig.text(.065,.91,'36 source pairs × 5 targets per model · target colors · black bars = medians',color='#56606B')
handles=[plt.Line2D([],[],marker='o',linestyle='',color=colors[t],label=names[t],markersize=6) for t in targets]
fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.11),ncol=5,frameon=False,fontsize=9)
fig.text(.065,.078,'360 dots/model on the left; 180 on the right. One training seed per model; contrasts share models and are not independent.',fontsize=9,color='#56606B')
fig.text(.065,.048,'PRODIGY: 2,500 updates, seed 0. GraphSAGE: SSL-validation-selected checkpoints, seed 0. SAMGPT: 500 updates, seed 39.',fontsize=9,color='#56606B')
fig.text(.065,.018,'Within-model effects: GraphSAGE uses frozen linear probes; PRODIGY and SAMGPT use episodic few-shot evaluation. Budgets/readouts differ.',fontsize=9,color='#56606B')
fig.subplots_adjust(left=.07,right=.985,top=.79,bottom=.245,wspace=.1)
for ext in ['png','pdf']:
    fig.savefig(HERE/f'figures/pair_impact_by_model.{ext}',dpi=180,bbox_inches='tight')
summary=best.groupby('model').impact_pp.agg(mean_pp='mean',median_pp='median',win_fraction=lambda x:(x>0).mean(),n='size').reindex(models)
summary.to_csv(HERE/'data/summary.csv')
print(summary.to_string())
