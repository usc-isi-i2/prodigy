"""Plot matched pair-minus-singleton ROC-AUC distributions at update 100."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
frame = pd.read_csv(ROOT / 'data/raw_aggregate/summary/classification_long.csv')
keys = ['architecture', 'seed', 'checkpoint_step', 'task', 'dataset']
assert not frame.duplicated(keys + ['sources']).any()
assert set(frame.checkpoint_step) == {100}
singles = frame[~frame.sources.str.contains(',')]
pairs = frame[frame.sources.str.count(',') == 1]
rows = []
for _, pair in pairs.iterrows():
    members = pair.sources.split(',')
    for baseline in members:
        match = singles[singles.sources.eq(baseline)]
        for key in keys:
            match = match[match[key].eq(pair[key])]
        assert len(match) == 1
        single = match.iloc[0]
        for key in ['episode_fingerprint', 'episodes', 'n_way', 'n_shot', 'n_query']:
            assert pair[key] == single[key], key
        rows.append({**{k: pair[k] for k in keys}, 'pair': pair.sources,
                     'baseline_source': baseline, 'added_source': next(s for s in members if s != baseline),
                     'pair_auc': pair.roc_auc, 'singleton_auc': single.roc_auc,
                     'impact_pp': 100 * (pair.roc_auc - single.roc_auc)})
contrasts = pd.DataFrame(rows)
assert len(contrasts) == 72
best = contrasts.sort_values('singleton_auc').groupby(keys + ['pair'], as_index=False).tail(1)
assert len(best) == 36
contrasts.to_csv(ROOT / 'data/pair_impact_contrasts.csv', index=False)
summary = contrasts.groupby('architecture').impact_pp.agg(['count', 'mean', 'median', 'min', 'max'])
summary.to_csv(ROOT / 'data/pair_impact_summary.csv')
catalog = json.loads((ROOT.parents[6] / 'docs/graph_catalog.json').read_text())
names = {g['dataset_key']: g['canonical_name'] for g in catalog['graphs']}
targets = list(frame.dataset.unique())
colors = dict(zip(targets, ['#4477AA', '#EE7733', '#AA3377', '#228833']))
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(1, 2, figsize=(12, 6.6), sharey=True)
for ax, data, title in zip(axes, [contrasts, best], ['Relative to each constituent singleton', 'Relative to the better constituent singleton']):
    ax.axhline(0, color='#56606B', lw=1, linestyle='--', zorder=1)
    for i, arch in enumerate(['prodigy', 'vision', 'gilt']):
        for j, target in enumerate(targets):
            values = data[(data.architecture == arch) & (data.dataset == target)].impact_pp.to_numpy()
            x = i + (j - 1.5) * .13 + np.linspace(-.04, .04, len(values))
            ax.scatter(x, values, s=47, color=colors[target], alpha=.83, edgecolors='white', linewidth=.5, zorder=3,
                       label=names[target] if i == 0 else None)
        values = data[data.architecture == arch].impact_pp
        ax.plot([i-.33, i+.33], [values.median()]*2, color='#202833', lw=2.1, zorder=4)
    ax.set_xticks(range(3), ['PRODIGY', 'VISION', 'GILT'])
    ax.set_xlim(-.6, 2.6)
    ax.set_title(title, fontsize=12, pad=15)
    ax.grid(axis='y', alpha=.16)
    ax.set_axisbelow(True)
axes[0].set_ylabel('Pair − singleton ROC-AUC (percentage points)')
fig.suptitle('Impact of two-source pretraining, by architecture', fontsize=18, x=.07, ha='left', y=.97)
fig.text(.07, .91, '3 source pairs × 4 targets · matched 100-update models · training seed 0', color='#56606B')
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .075), ncol=4, frameon=False, fontsize=10)
fig.text(.07, .04, 'Each dot is one pair–target contrast; black bars mark medians. Positive = pair improves ROC-AUC.', fontsize=10, color='#56606B')
fig.text(.07, .012, 'Left: 24 dots/architecture (two baselines per pair–target). Right: 12 dots/architecture. Dots are not independent replicates.', fontsize=9, color='#56606B')
fig.tight_layout(rect=[.045, .15, .99, .865])
(ROOT / 'figures').mkdir(exist_ok=True)
for ext in ['png', 'pdf']:
    fig.savefig(ROOT / f'figures/pair_impact_distribution.{ext}', dpi=190, bbox_inches='tight')
print(summary.to_string())
