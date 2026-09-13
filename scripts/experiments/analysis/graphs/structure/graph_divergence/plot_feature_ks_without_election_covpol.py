"""Plot the repaired feature KS results with Election and COVID Political excluded (no resampling)."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
r = json.loads((ROOT / 'data/feature_ks_political_updated.json').read_text())
names = [n for n in r['graphs'] if n not in {'election2020', 'covid_political'}]
labels = ['UKR/RUS', 'COVID', 'Midterm', 'Suspended', 'TwiBot20', 'CP/HK', 'Facebook']
fig, axs = plt.subplots(1, 2, figsize=(14, 6.8))
for ax, mode, title in zip(axs, ['all', 'nonzero'], ['Including all-zero rows', 'Excluding all-zero rows']):
    matrix = np.zeros((len(names), len(names)))
    for pair in r['pairs']:
        if pair['a'] in names and pair['b'] in names:
            i, j = names.index(pair['a']), names.index(pair['b'])
            matrix[i, j] = matrix[j, i] = pair[mode]['mean']
    im = ax.imshow(matrix, cmap='viridis', vmin=0, vmax=.3)
    ax.set_xticks(range(len(names)), labels, rotation=50, ha='right')
    ax.set_yticks(range(len(names)), labels)
    ax.set_title(title)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f'{matrix[i,j]:.3f}', ha='center', va='center', fontsize=9,
                    color='white' if matrix[i,j] < .15 else 'black')
fig.suptitle('Mean per-coordinate feature KS · Election and COVID Political excluded', fontsize=16, y=.97)
fig.text(.5, .915, '10,000 nodes per graph · 768 dimensions · repaired Suspended', ha='center', fontsize=11)
fig.subplots_adjust(left=.09, right=.91, bottom=.22, top=.85, wspace=.35)
cax = fig.add_axes([.935, .24, .015, .58])
fig.colorbar(im, cax=cax, label='Mean KS (lower = more similar)')
for extension in ['png', 'pdf']:
    fig.savefig(ROOT / f'figures/feature_ks_matrix_without_election_covpol.{extension}', dpi=180)
