"""Plot the frozen six-cell training history; never reads assessment predictions."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=root/'data/joint_v1/selection_frozen.json')
    parser.add_argument('--out', type=Path, default=root/'figures/training_curves')
    args = parser.parse_args()
    selections = json.loads(args.input.read_text())['selections']
    assert len(selections) == 6
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.titleweight': 'bold', 'axes.labelcolor': '#243445',
                         'text.color': '#172b3a', 'axes.edgecolor': '#c2cbd3',
                         'xtick.color': '#526476', 'ytick.color': '#526476'})
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 7.5), sharex=True)
    fig.patch.set_facecolor('#fafbfd')
    colors = ['#167d9a', '#d7882c', '#8464b5']
    labels = {'structure': 'Structure-only · 36,909 fitted scalars',
              'joint': 'Joint graph + features + structure · 496,429 fitted scalars'}
    all_losses = [h['train_loss'] for s in selections for h in s['history']]
    all_hits = [100*h['selection_hits_at_50'] for s in selections for h in s['history']]
    for row, arm in enumerate(('structure', 'joint')):
        for s in sorted((s for s in selections if s['arm']==arm), key=lambda s:s['seed']):
            steps = [h['step'] for h in s['history']]
            loss = [h['train_loss'] for h in s['history']]
            hits = [100*h['selection_hits_at_50'] for h in s['history']]
            color = colors[s['seed']]
            for col, values in enumerate((loss, hits)):
                axes[row, col].plot(steps, values, color=color, lw=1.6, alpha=.9, label=f"Seed {s['seed']}")
            selected = next(h for h in s['history'] if h['step']==s['selected_step'])
            axes[row, 1].scatter([s['selected_step']], [100*selected['selection_hits_at_50']],
                                s=70, color=color, edgecolors='white', linewidths=1.4, zorder=5)
        axes[row, 0].set_title(labels[arm], loc='left', fontsize=11, pad=12)
        axes[row, 1].set_title('2017 validation · dots mark selected checkpoints', loc='left', fontsize=11, pad=12)
        axes[row, 0].set_yscale('log')
        axes[row, 0].set_ylim(min(all_losses)*.75, max(all_losses)*1.2)
        axes[row, 0].set_ylabel('Training BCE loss (log scale)')
        axes[row, 1].set_ylim(min(all_hits)-2, max(all_hits)+2)
        axes[row, 1].set_ylabel('Hits@50 (%)')
        for ax in axes[row]:
            ax.set_xlim(0, 2050)
            ax.set_facecolor('white')
            ax.grid(axis='y', color='#e3e8ed', lw=.65)
            ax.set_axisbelow(True)
    axes[0, 0].legend(frameon=False, loc='upper right', ncol=3, fontsize=9)
    for ax in axes[-1]:
        ax.set_xlabel('Optimizer updates')
    fig.suptitle('Compact Collab experiment: training and validation curves',
                 x=.07, y=.98, ha='left', fontsize=18, fontweight='bold')
    fig.text(.07, .928, 'Train on 2016 links; select on 2017 links. All six runs completed 2,000 updates.', fontsize=11)
    fig.text(.07, .025, 'Raw checkpoint samples every 50 updates; no smoothing. Training loss is a sampled minibatch, not a full-panel average.\n'
             'These are historical selection curves, not the official 2018 assessment or 2019 test.', fontsize=9, color='#526476')
    fig.subplots_adjust(left=.07, right=.975, top=.855, bottom=.14, hspace=.38, wspace=.23)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ('.png', '.pdf'):
        fig.savefig(args.out.with_suffix(suffix), dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(args.out.with_suffix('.png'))


if __name__ == '__main__':
    main()
