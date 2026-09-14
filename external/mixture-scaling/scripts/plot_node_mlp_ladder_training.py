"""Plot the recorded single-batch losses; do not imply window averages."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/node_mlp_ladder_s0_2500'
SOURCES = ['Ukraine', 'COVID', 'Midterm', 'COVID political', 'Election', 'Ukraine suspended', 'TwiBot-20', 'Hong Kong', 'Facebook']
rows = []
for line in (OUT / 'data/training.jsonl').read_text().splitlines():
    row = json.loads(line)
    if 'loss' not in row:
        continue
    k = int(row['run_id'].split('_r')[1])
    rows.append(dict(rung=k, step=row['step'], loss=row['loss'],
                     sampled_source=SOURCES[(row['step']-1) % k]))
rows.sort(key=lambda r: (r['rung'], r['step']))
assert len(rows) == 90
assert all([r['step'] for r in rows if r['rung']==k] == list(range(250,2501,250)) for k in range(1,10))
with (OUT/'data/training_curves.csv').open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
colors = plt.get_cmap('tab10').colors
fig, axes = plt.subplots(3, 3, figsize=(13, 9), sharex=True, sharey=True)
for k, ax in enumerate(axes.flat, 1):
    subset = [r for r in rows if r['rung']==k]
    ax.plot([r['step'] for r in subset], [r['loss'] for r in subset], color='#aeb6c2', lw=1.4, zorder=1)
    for r in subset:
        ax.scatter(r['step'], r['loss'], color=colors[SOURCES.index(r['sampled_source'])], s=30, zorder=2)
    ax.set_title(f'{k} source'+('s' if k>1 else ''), loc='left', fontweight='bold')
    ax.set_xlim(180,2570); ax.set_ylim(.655,.701)
    ax.set_xticks([250,1000,1750,2500]); ax.set_yticks([.66,.67,.68,.69,.70])
    ax.grid(axis='y',color='#e4e8ed',lw=.7); ax.set_axisbelow(True)
    if k>6: ax.set_xlabel('Optimizer update')
    if k in (1,4,7): ax.set_ylabel('Training BCE loss')
fig.suptitle('Pure MLP ladder · training loss', x=.07, ha='left', fontsize=21, fontweight='bold', y=.98)
fig.text(.07,.934,'LP pretraining · seed 0 · 2,500 updates per rung · shared axis limits',fontsize=12,color='#4c5666')
handles = [Line2D([0],[0],marker='o',ls='',color=colors[i],label=s) for i,s in enumerate(SOURCES)]
fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5,.025), ncol=5, frameon=False, title='Source of the logged batch')
fig.text(.07,.012,'Each point is ONE batch logged every 250 updates, not a 250-update mean. Source rotation contributes to the zigzags.', fontsize=10,color='#4c5666')
fig.subplots_adjust(left=.07,right=.98,top=.885,bottom=.165,hspace=.32,wspace=.15)
for ext in ['png','pdf']:
    fig.savefig(OUT/f'figures/training_curves.{ext}',dpi=180,facecolor='white')
print(OUT/'figures/training_curves.png')
