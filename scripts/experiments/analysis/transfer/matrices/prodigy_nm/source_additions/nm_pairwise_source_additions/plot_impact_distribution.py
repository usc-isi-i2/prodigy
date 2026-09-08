"""Plot the completed 2500-update PRODIGY all-pairs sweep."""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / 'docs/graph_catalog.json').exists())
MECH = REPO / 'scripts/experiments/analysis/graphs/transfer_prediction/target_performance_mechanisms'
sys.path.insert(0, str(MECH))
from analyze_lattice import validate_cls
cls = validate_cls(pd.read_csv(MECH / 'data/raw/classification_long.tsv', sep='\t'))
singles = cls[cls.k.eq(1)].set_index(['target', 'sources'])
records = []
for _, pair in cls[cls.k.eq(2)].iterrows():
    for base in pair.source_set:
        single = cls[(cls.k == 1) & (cls.target == pair.target) & cls.source_set.map(lambda s: s == (base,))]
        assert len(single) == 1
        single = single.iloc[0]
        assert single.episode_fingerprint == pair.episode_fingerprint
        records.append(dict(base_source=base, added_source=next(s for s in pair.source_set if s != base),
                            pair_model_id=pair.model_id, target=pair.target,
                            base_specialist_auc=single.roc_auc, pair_auc=pair.roc_auc,
                            delta_auc=pair.roc_auc-single.roc_auc,
                            target_heldout_from_pair=int(pair.target not in pair.source_set)))
classification = pd.DataFrame(records)
assert len(classification) == 360
nm = pd.read_csv(HERE / 'data/nm_directional_intervention_deltas_all_targets_long.csv')
assert len(nm) == 648 and not nm.duplicated(['base_source','added_source','target']).any()
assert np.allclose(nm.delta_auc, nm.pair_auc-nm.base_specialist_auc)
classification.to_csv(HERE / 'data/classification_impact_contrasts.csv', index=False)
catalog = json.loads((REPO / 'docs/graph_catalog.json').read_text())
names = {g['dataset_key']: g['canonical_name'] for g in catalog['graphs']}
names.update(ukr_rus=names['ukr_rus_twitter'], covid=names['covid19_twitter'], cp_hk=names['cp_hk_twitter'])
targets = list(nm.target.unique())
colors = dict(zip(targets, plt.get_cmap('tab10').colors))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig, axes = plt.subplots(1, 2, figsize=(13,7), sharey=True)
rng = np.random.default_rng(0)
summaries=[]
for ax, data, task in zip(axes, [nm, classification], ['Neighbor matching', 'Node classification']):
    best = data.sort_values('base_specialist_auc').groupby(['pair_model_id','target'],as_index=False).tail(1)
    for x, (view, label) in enumerate([(data,'Each singleton'),(best,'Better singleton')]):
        for target in targets:
            vals = view.loc[view.target.eq(target),'delta_auc'].to_numpy()*100
            ax.scatter(x+rng.uniform(-.32,.32,len(vals)), vals, color=colors[target], s=18, alpha=.58, linewidths=0, zorder=3)
        vals=view.delta_auc*100
        ax.plot([x-.36,x+.36],[vals.median()]*2,color='#15202B',lw=2.3,zorder=4)
        summaries.append(dict(task=task,baseline=label,n=len(view),median_pp=vals.median(),mean_pp=vals.mean(),positive_fraction=(vals>0).mean()))
    ax.axhline(0,color='#647080',linestyle='--',lw=1)
    ax.set_xticks([0,1],['Pair − each singleton','Pair − better singleton'])
    ax.set_xlim(-.55,1.55)
    ax.set_title(f'{task}\n{data.target.nunique()} targets · {len(data)} / {len(best)} dots',fontsize=13,pad=12)
    ax.grid(axis='y',alpha=.17);ax.set_axisbelow(True)
axes[0].set_ylabel('Change in macro ROC-AUC (percentage points)')
fig.suptitle('PRODIGY · recent all-pairs impact distribution',x=.07,ha='left',fontsize=18,y=.965)
fig.text(.07,.91,'36 source pairs · 2,500 updates · training seed 0 · colors = evaluation targets',color='#56606B')
handles=[plt.Line2D([],[],marker='o',linestyle='',color=colors[t],label=names[t],markersize=6) for t in targets]
fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.065),ncol=5,frameon=False,fontsize=9)
fig.text(.07,.035,'Black bars: medians. Each pair contributes two singleton contrasts or one best-singleton contrast per target.',fontsize=9,color='#56606B')
fig.text(.07,.012,'Includes both source-seen and held-out targets; dots share models and are not independent replicates.',fontsize=9,color='#56606B')
fig.subplots_adjust(left=.075,right=.985,bottom=.24,top=.79,wspace=.12)
for ext in ['png','pdf']:
    fig.savefig(HERE / f'figures/recent_pair_impact_distribution.{ext}',dpi=180,bbox_inches='tight')
summary=pd.DataFrame(summaries)
summary.to_csv(HERE / 'data/recent_pair_impact_summary.csv',index=False)
print(summary.to_string(index=False))
