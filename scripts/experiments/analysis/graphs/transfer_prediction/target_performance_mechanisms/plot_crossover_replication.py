"""Plot two completed crossover summaries; no fitting or model execution."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def paired_sensitivity(data, draws=20000):
    """Empirical episode reweighting, not a population or training-seed CI."""
    conditions = ['E2000I2000', 'E2000I8000', 'E8000I2000', 'E8000I8000']
    records = [{r['ordinal']: r['metrics']['accuracy'] for r in data['rows']
                if r['condition'] == c} for c in conditions]
    ordinals = sorted(records[0])
    assert len(ordinals) == 128 and all(sorted(r) == ordinals for r in records)
    a, b, c, d = np.array([[r[o] for o in ordinals] for r in records]) * 100
    effects = np.array([c-a, d-b, b-a, d-c, d-a])
    indices = np.random.default_rng(20260907).integers(0, len(ordinals), (draws, len(ordinals)))
    # The same sampled episode indices apply to all four conditions and contrasts.
    bounds = np.quantile(effects[:, indices].mean(axis=2), [.025, .975], axis=1)
    leave_one_out = (effects.sum(axis=1, keepdims=True) - effects) / (len(ordinals)-1)
    return effects.mean(axis=1), bounds, np.array([leave_one_out.min(axis=1), leave_one_out.max(axis=1)])


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--seed0', type=Path, required=True)
    parser.add_argument('--seed1', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--roles', type=Path, help='Optional verified role-crossover aggregate export')
    args = parser.parse_args()
    matrices = []
    sensitivity = []
    expected = ['507e1169eb782be76223643ec3b81eb2ba8ca5a3faccf3d7cfb3805519f5d672',
                '22c68738f422d2815f8661f5ef2039ce6004dbd79c3edb138673654d07ecfdb9']
    for path, digest in zip([args.seed0, args.seed1], expected):
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == digest
        data = json.loads(raw)
        assert len(data['rows']) == 512
        values = []
        for e in [2000, 8000]:
            row = []
            for i in [2000, 8000]:
                condition = f'E{e}I{i}'
                records = [r for r in data['rows'] if r['condition'] == condition]
                assert len(records) == 128 and len({r['ordinal'] for r in records}) == 128
                accuracy = sum(r['metrics']['accuracy'] for r in records) / 128
                assert abs(accuracy - data['means'][condition]['accuracy']) < 1e-12
                row.append(100 * accuracy)
            values.append(row)
        matrices.append(np.array(values))
        sensitivity.append(paired_sensitivity(data))
    if args.roles:
        roles = json.loads(args.roles.read_text())
        fig, axes = plt.subplots(2, 2, figsize=(8.2, 7.8))
        for seed in [0, 1]:
            assert roles[str(seed)]['rows'] == 512
            role = np.array([[100 * roles[str(seed)]['means'][f'S{s}Q{q}I8000']['accuracy']
                              for q in [2000, 8000]] for s in [2000, 8000]])
            assert np.allclose(role.diagonal(), matrices[seed][:, 1], atol=1e-10, rtol=0)
            for row, matrix in enumerate([matrices[seed], role]):
                ax = axes[row, seed]
                ax.imshow(matrix, cmap='Blues', vmin=35, vmax=85)
                for r in [0, 1]:
                    for c in [0, 1]:
                        ax.text(c, r, f'{matrix[r,c]:.2f}%', ha='center', va='center', fontsize=17,
                                color='white' if matrix[r,c] > 70 else '#152b40')
                ax.set(xticks=[0, 1], xticklabels=['Early', 'Late'], yticks=[0, 1],
                       yticklabels=['Early', 'Late'], xlabel='Inference module' if row == 0 else 'Query encoder',
                       ylabel='Encoder' if row == 0 else 'Support encoder')
                ax.tick_params(length=0)
                ax.set_title(f'Seed {seed}: '+('whole-module crossover' if row == 0 else 'roles at fixed late inference'), fontsize=11)
        fig.suptitle('Component effects replicate; support-specific attribution does not', fontsize=13, weight='bold')
        fig.text(.5, .045, 'Top: later inference helps either encoder; later encoder hurts either inference module.\n'
                 'Bottom: mixed-age support/query inputs fail in both directions.\n'
                 'Wiki → FB15K-237 · 128 paired episodes · accuracy · native batch statistics\n'
                 'Early / late = 2,001 / 8,001 updates. No fitted alignment; no uncertainty intervals shown.',
                 ha='center', fontsize=9)
        fig.subplots_adjust(left=.12, right=.96, top=.91, bottom=.22, hspace=.47, wspace=.45)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=180, facecolor='white')
        plt.close(fig)
        return
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4), gridspec_kw={'width_ratios': [1, 1, 1.3]})
    for seed, (ax, matrix) in enumerate(zip(axes[:2], matrices)):
        ax.imshow(matrix, cmap='Blues', vmin=68, vmax=82)
        for r in range(2):
            for c in range(2):
                ax.text(c, r, f'{matrix[r,c]:.2f}%', ha='center', va='center', fontsize=17)
        ax.set(xticks=[0, 1], xticklabels=['Early', 'Late'], yticks=[0, 1],
               yticklabels=['Early', 'Late'], xlabel='Inference module', ylabel='Encoder')
        ax.set_title(f'Seed {seed}: ' + ('discovery' if seed == 0 else 'replication'))
        ax.tick_params(length=0)
    for seed, matrix in enumerate(matrices):
        deltas = [matrix[1, 0]-matrix[0, 0], matrix[1, 1]-matrix[0, 1],
                  matrix[0, 1]-matrix[0, 0], matrix[1, 1]-matrix[1, 0],
                  matrix[1, 1]-matrix[0, 0]]
        means, bounds, loo = sensitivity[seed]
        assert np.allclose(means, deltas, atol=1e-10, rtol=0)
        print(json.dumps({'seed': seed, 'contrast_order': ['encoder_earlyI', 'encoder_lateI',
                         'inference_earlyE', 'inference_lateE', 'end_to_end'],
                         'mean_pp': means.tolist(), 'empirical_95_bounds_pp': bounds.tolist(),
                         'leave_one_episode_out_range_pp': loo.tolist()}))
        axes[2].errorbar(means, np.arange(5)+(seed-.5)*.16,
                        xerr=np.array([means-bounds[0], bounds[1]-means]), fmt='o', capsize=3,
                        label=f'Seed {seed}', color=['#197b85', '#b14d3d'][seed])
    axes[2].axvline(0, color='#777', lw=.8)
    axes[2].set(yticks=range(5), yticklabels=['Encoder | early inference', 'Encoder | late inference',
                 'Inference | early encoder', 'Inference | late encoder', 'End-to-end'],
                 xlabel='Late − early accuracy (points)', xlim=(-6, 3.5))
    axes[2].invert_yaxis()
    axes[2].legend(frameon=False, loc='upper right', fontsize=9)
    axes[2].set_title('Component changes partly cancel', fontsize=11)
    fig.suptitle('A nearly flat transfer score can conceal opposing component changes', fontsize=14, weight='bold')
    fig.text(.5, .03, 'Wiki → FB15K-237 · 128 episodes per condition · 2,001 vs 8,001 updates · native batch statistics\n'
             'Bars: 95% paired episode-reweighting ranges (20,000 draws), conditional on saved episodes and checkpoints.\n'
             'Not entity-, domain-, or training-seed confidence intervals. Fixed swaps; no fitted alignment.',
             ha='center', fontsize=9, color='#444')
    fig.subplots_adjust(left=.055, right=.985, top=.80, bottom=.25, wspace=1.05)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    main()
