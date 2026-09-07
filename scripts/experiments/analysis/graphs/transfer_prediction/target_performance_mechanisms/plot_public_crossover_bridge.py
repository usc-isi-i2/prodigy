"""Figure from verified aggregate exports; no fitting or model evaluation."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--input', type=Path, default=root / 'data/public_crossover_bridge_20260907.json')
    parser.add_argument('--output', type=Path, default=root / 'figures/public_crossover_bridge.png')
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    assert data['sources']['publickg_crossover_20260907']['rows'] == 512
    assert data['sources']['publickg_normalization_20260907']['rows'] == 256
    norm, cross = data['normalization'], data['crossover']
    for step in (2000, 8000):
        assert abs(norm[str(step)]['batch/native']['accuracy'] - cross[f'E{step}I{step}']['accuracy']) < 1e-12
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.7), gridspec_kw={'width_ratios': [1, 1.12, 1]})
    colors = {'native': '#b14d3d', 'centered': '#197b85'}
    for ax, mode, title in ((axes[0], 'batch', 'A  Native evaluation protocol'),
                            (axes[2], 'running', 'C  Frozen checkpoint BN statistics')):
        for rule, label in [('native', 'Learned inference'), ('centered', 'Support-fitted ridge')]:
            values = [100 * norm[str(s)][f'{mode}/{rule}']['accuracy'] for s in (2000, 8000)]
            ax.plot([0, 1], values, 'o-', color=colors[rule], lw=2.3, label=label)
            for x, y in enumerate(values):
                ax.annotate(f'{y:.2f}', (x, y), xytext=(0, 9), textcoords='offset points', ha='center', color=colors[rule])
            ax.text(.5, (values[0] + values[1])/2 - 1.5, f'{values[1]-values[0]:+.2f} points', ha='center', color=colors[rule], fontsize=10)
        ax.set(xticks=[0, 1], xticklabels=['2,001', '8,001'], ylim=(68, 85), xlim=(-.2, 1.2), xlabel='Completed training updates', ylabel='Target accuracy (%)')
        ax.set_title(title, loc='left', fontsize=11, fontweight='bold', pad=15)
        ax.grid(axis='y', alpha=.15)
    axes[0].legend(loc='lower left', frameon=False, fontsize=9)
    ax = axes[1]
    matrix = np.array([[100 * cross[f'E{e}I{i}']['accuracy'] for i in (2000, 8000)] for e in (2000, 8000)])
    ax.imshow(matrix, cmap='Blues', vmin=68, vmax=85, aspect='auto')
    for row in range(2):
        for col in range(2):
            ax.text(col, row, f'{matrix[row,col]:.2f}%', ha='center', va='center', fontsize=17, color='#102b46')
    ax.set(xticks=[0, 1], xticklabels=['Early inference', 'Late inference'], yticks=[0, 1], yticklabels=['Early encoder', 'Late encoder'])
    ax.tick_params(length=0, pad=8)
    ax.set_title('B  Temporal component crossover', loc='left', fontsize=11, fontweight='bold', pad=15)
    ax.set_xlabel('Later inference helps both encoders\nLater encoder hurts both inference modules', fontsize=9, labelpad=13)
    fig.suptitle('Native transfer can deteriorate while support-fitted classification remains useful', fontsize=14, fontweight='bold', y=.99)
    fig.text(.5, .045, 'Wiki → FB15K-237 · one training seed · identical 128 episodes · fixed support-centered ridge, λ=1\nB uses native batch statistics; C is not a replication of the crossover. Accuracy is not probability quality.', ha='center', fontsize=9, color='#444444')
    fig.subplots_adjust(left=.06, right=.985, top=.81, bottom=.27, wspace=.49)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=190, facecolor='white')
    plt.close(fig)
    print(args.output)


if __name__ == '__main__':
    main()
