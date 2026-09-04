"""Render training-only snapshots for manual curve review, including live jobs."""
import argparse
import json
import math
from pathlib import Path
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('snapshot',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--rung',type=int,default=8)
    args=p.parse_args();snapshot=json.loads(args.snapshot.read_text())
    models=[m for m in snapshot['models'] if f'_r{args.rung}_' in m['model'] and m.get('training_curve')]
    if not models:raise ValueError('No curves for requested rung')
    fig,axes=plt.subplots(math.ceil(len(models)/4),4,figsize=(18,3.4*math.ceil(len(models)/4)),squeeze=False)
    for ax,model in zip(axes.flat,models):
        curve=model['training_curve'];history=model.get('validation_history',[])
        ax.plot([r['step'] for r in curve],[r['loss'] for r in curve],color='tab:blue',lw=1.4)
        ax.set_title(model['model'].removeprefix('nmi_'),fontsize=10)
        ax.set(xlabel='Episodes',ylabel='NM training loss');ax.grid(alpha=.15)
        if history:
            right=ax.twinx()
            right.plot([r['step'] for r in history],[r['macro_roc_auc'] for r in history],
                       color='tab:orange',marker='o',lw=1.5)
            right.set_ylabel('Source validation AUC',color='tab:orange',fontsize=8)
            right.set_ylim(.5,1)
    for ax in list(axes.flat)[len(models):]:ax.set_visible(False)
    fig.suptitle(f'Rung {args.rung}: blue = training NM loss; orange = training-source validation only')
    fig.tight_layout();args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,dpi=140);plt.close(fig)

if __name__=='__main__':main()
