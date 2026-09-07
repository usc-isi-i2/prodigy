"""Bounded synthesis of completed ranking, contrast, and nominated 50k tests.

No model forwards, checkpoint selection, inference tests, or private predictions.
All error bars summarize training-seed means after averaging the two streams;
they are descriptive standard deviations, not confidence intervals.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ['source', 'seed', 'step', 'target', 'stream']
RANKS = ['pooled_margin_auc', 'within_episode_auc']


def paired(frame, baseline, extra=()):
    keys = KEYS + list(extra)
    base = frame.loc[frame.condition.eq(baseline), keys + RANKS]
    if base.duplicated(keys).any():
        raise ValueError('nonunique baseline')
    out = frame.merge(base, on=keys, suffixes=('', '_baseline'), validate='many_to_one')
    if len(out) != len(frame):
        raise ValueError('missing paired baseline')
    for metric in RANKS:
        out['delta_' + metric + '_points'] = 100 * (out[metric] - out[metric + '_baseline'])
    return out


def summarize(frame, extra=()):
    group = ['source', 'step', 'target', 'condition'] + list(extra)
    columns = [c for c in frame if c.startswith('delta_')]
    seed_means = frame.groupby(group + ['seed'], as_index=False)[columns].mean()
    rows = []
    for key, sub in frame.groupby(group):
        row = dict(zip(group, key))
        seed = seed_means
        for k, v in row.items():
            seed = seed[seed[k].eq(v)]
        row.update(cells=len(sub), seeds=len(seed), streams=sub.stream.nunique())
        for col in columns:
            row.update({col + '_mean': sub[col].mean(), col + '_min': sub[col].min(),
                        col + '_max': sub[col].max(), col + '_positive': int(sub[col].gt(0).sum()),
                        col + '_negative': int(sub[col].lt(0).sum()),
                        col + '_seed_sd': seed[col].std(ddof=1) if len(seed) > 1 else np.nan})
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--data-root', type=Path, default=Path(__file__).parent / 'data')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a new output directory; preserve earlier analyses.')
    folders = {'ranking': 'classref_ranking_20260907', 'initial': 'classref_contrast_20260907',
               'long': 'classref50k_20260907'}
    inputs = {}; completion = {}; frames = {}
    for name, folder in folders.items():
        root = args.data_root / folder
        completion[name] = json.loads((root / 'DONE.json').read_text())
        for path in sorted(root.glob('*.json')):
            inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        frames[name] = pd.read_json(root / 'metrics.json')
    if [len(frames[k]) for k in folders] != [1500, 252, 84]:
        raise ValueError('Incomplete declared evidence.')
    if any(completion[k].get('smoke_only', False) for k in folders):
        raise ValueError('Smoke results are not evidence.')
    if not completion['initial']['all_endpoint_logits_bit_exact'] or not completion['long']['all_weights_unchanged']:
        raise ValueError('Verification failed.')
    initial = paired(frames['initial'], 'intact')
    long = paired(frames['long'][frames['long'].family.eq('contrast')], 'intact')
    readouts = paired(frames['long'][frames['long'].family.eq('readout')], 'native_intact')
    ranking = frames['ranking']
    steps = paired(ranking[ranking['mode'].eq('training_step')], 'intact')
    scale = paired(ranking[ranking['mode'].eq('scale')], 'scale_1')
    for data in [initial, long]:
        pivot = data.pivot(index=KEYS, columns='condition', values='within_episode_auc')
        for a, b in [('orientation_only', 'removed'), ('strength_only', 'intact'),
                     ('unit_strength_intact', 'intact'), ('unit_strength_removed', 'removed')]:
            if not np.array_equal(pivot[a].values, pivot[b].values):
                raise ValueError('Positive-scaling ranking invariant failed.')
    endpoints = frames['long'][frames['long'].family.eq('readout') &
        frames['long'].condition.isin(['native_intact', 'native_removed'])].copy()
    endpoints['condition'] = endpoints.condition.str.removeprefix('native_')
    endpoints = endpoints.merge(frames['long'][frames['long'].family.eq('contrast')],
                               on=KEYS + ['condition'], suffixes=('_native', '_contrast'))
    rounding = {metric: float((endpoints[metric + '_native'] - endpoints[metric + '_contrast']).abs().max())
                for metric in RANKS}
    if rounding['pooled_margin_auc'] > 1e-4 or rounding['within_episode_auc'] > 1e-4:
        raise ValueError('Final contrast and native endpoints disagree beyond rounding.')
    args.output.mkdir(parents=True)
    tables = {'initial_contrast_cells': initial, 'initial_contrast_summary': summarize(initial),
              'long_contrast_cells': long, 'long_contrast_summary': summarize(long),
              'long_readout_cells': readouts, 'long_readout_summary': summarize(readouts),
              'training_ranking_cells': steps, 'training_ranking_summary': summarize(steps),
              'scale_ranking_summary': summarize(scale)}
    baselines = readouts.groupby(['target', 'condition'], as_index=False)[
        ['roc_auc', 'pooled_margin_auc', 'within_episode_auc', 'accuracy', 'f1', 'nll']].mean()
    tables['long_readout_means'] = baselines
    for name, table in tables.items():
        table.to_csv(args.output / (name + '.csv'), index=False)
    validation = {'inputs_sha256': inputs, 'completion': completion,
                  'native_vs_reconstructed_max_auc_difference': rounding,
                  'positive_scaling_ranking_invariants': True,
                  'new_model_forwards': 0,
                  'aggregation': 'Equal stream means; seed SD computed after averaging streams within each seed.',
                  'inference': 'Descriptive only. Streams are not training seeds; repeated accounts preclude IID interpretation.',
                  'units': 'AUC differences in percentage points; baselines in 0-1 units.',
                  'counterfactuals': 'Final contrast swaps are score counterfactuals, not necessarily realizable latent label pairs.'}
    (args.output / 'validation.json').write_text(json.dumps(validation, indent=2) + '\n')
    print(json.dumps({'tables': len(tables), 'validation': validation}, indent=2))


if __name__ == '__main__':
    main()
