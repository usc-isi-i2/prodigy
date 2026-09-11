"""Plot the JSONL mirror of the completed offline W&B ladder histories."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/node_mlp_ladder_s0_2500_wandb'
rows = []
k = 1
for line in (OUT/'data/metrics_all.jsonl').read_text().splitlines():
    r = json.loads(line)
    if 'train/loss' in r:
        rows.append({'rung': k, 'step': r['optimizer_step'], 'training_loss': r['train/loss']})
    elif 'validation/loss' in r:
        k += 1
assert k == 10 and len(rows) == 900
for k in range(1,10):
    assert [r['step'] for r in rows if r['rung']==k] == list(range(25,2501,25))
with (OUT/'data/training_curves.csv').open('w', newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(3,3,figsize=(12.5,8.6),sharex=True,sharey=True)
lo=min(r['training_loss'] for r in rows);hi=max(r['training_loss'] for r in rows)
for k,ax in enumerate(axs.flat,1):
    rs=[r for r in rows if r['rung']==k]
    ax.plot([r['step'] for r in rs],[r['training_loss'] for r in rs],color='#2563a6',lw=1.8)
    ax.scatter(rs[-1]['step'],rs[-1]['training_loss'],s=23,color='#2563a6',zorder=3)
    ax.set_title(f'{k} source'+('s' if k>1 else ''),loc='left',fontweight='bold')
    ax.set_xlim(0,2550);ax.set_ylim(lo-.002,hi+.002)
    ax.set_xticks([0,500,1000,1500,2000,2500]);ax.tick_params(axis='x',labelsize=9)
    ax.grid(axis='y',color='#e5e9ef',lw=.7);ax.set_axisbelow(True)
    if k>6:ax.set_xlabel('Optimizer update')
    if k in (1,4,7):ax.set_ylabel('Training BCE loss')
fig.suptitle('Pure MLP ladder · averaged training curves',x=.075,ha='left',y=.98,fontsize=20,fontweight='bold')
fig.text(.075,.929,'Offline W&B rerun · seed 0 · 2,500 updates per rung · shared axis limits',fontsize=11.5,color='#4c5666')
fig.text(.075,.035,'Each point averages all examples in 25 updates (100 points per rung). No additional smoothing.',fontsize=10.5,color='#4c5666')
fig.text(.075,.012,'Source mixtures differ between panels; absolute losses are not a matched-distribution performance comparison.',fontsize=10,color='#4c5666')
fig.subplots_adjust(left=.075,right=.985,top=.88,bottom=.11,hspace=.3,wspace=.15)
for ext in ('png','pdf'):fig.savefig(OUT/f'figures/training_curves.{ext}',dpi=180,facecolor='white')
print(f'Validated {len(rows)} points. Loss range: {lo:.5f}–{hi:.5f}')
