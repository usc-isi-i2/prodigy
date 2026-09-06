"""One-source/one-seed execution-repeat stress test, not a transfer remedy."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .analyze_member_initial_reference import METRICS, validate_inputs
from .analyze_member_intervention import validate_replay
from .analyze_trajectories import DECODERS, PANEL, read_jsonl

MODES = ('default', 'deterministic')
INITIAL = 'c350d50c3a2883b1e4f8fc62f7b3bff81880691db12024469aa94231386f2723'


def validate_training(receipt, arms, comparisons):
    if (not all(receipt.get(k) is True for k in ('valid', 'exact_historical_inputs', 'same_initialization'))
            or receipt.get('smoke') is not False or receipt.get('models') != 4
            or receipt.get('steps_per_model') != 2500 or receipt.get('independent_seeds') != 1):
        raise ValueError('complete full-length execution controls required')
    if (len(arms) != 4 or arms.model_id.duplicated().any()
            or set(map(tuple, arms[['mode', 'repeat']].to_numpy())) != {(m, r) for m in MODES for r in (0, 1)}
            or set(arms.seed) != {2} or set(arms.source) != {'cp_hk'}
            or set(arms.initial_sha256) != {INITIAL} or set(arms.inputs_matched) != {2500}
            or (arms.initial_sha256 == arms.final_sha256).any()):
        raise ValueError('control grid, initialization or input audit differs')
    states = pd.DataFrame(comparisons)
    if (len(states) != 10 or states.duplicated(['mode', 'step']).any()
            or set(map(tuple, states[['mode', 'step']].to_numpy())) != {
                (m, s) for m in MODES for s in (0, 100, 300, 900, 2500)}
            or not states[['rng_bit_exact', 'train_batch_sampler_bit_exact']].all().all()
            or not states.loc[states.step == 0, ['model_bit_exact', 'optimizer_bit_exact']].all().all()):
        raise ValueError('saved state grid or random-stream contract differs')
    exact = bool(states.loc[states['mode'] == 'deterministic', ['model_bit_exact', 'optimizer_bit_exact']].all().all())
    if receipt.get('deterministic_states_bit_exact') != exact:
        raise ValueError('deterministic outcome differs from saved-state evidence')
    for mode in MODES:
        final = states[(states['mode'] == mode) & (states.step == 2500)].iloc[0]
        if (arms[arms['mode'] == mode].final_sha256.nunique() == 1) != bool(final.model_bit_exact):
            raise ValueError('final state hash contradicts exact-repeat result')
    return exact


def validate_logits(receipt, comparisons):
    expected = {(s, t, m) for s in ('original', 'fresh') for t in PANEL for m in MODES}
    if (len(comparisons) != 20 or {(r['stream'], r['target'], r['mode']) for r in comparisons} != expected
            or receipt.get('complete') is not True or receipt.get('same_target_inputs') is not True
            or (receipt.get('models'), receipt.get('targets'), receipt.get('streams')) != (4, 5, 2)):
        raise ValueError('complete prediction comparison grid required')
    for row in comparisons:
        values = row['maximum_absolute_difference_by_decoder']
        if (set(values) != DECODERS or row['comparisons'] != 32 * 17
                or not all(np.isfinite(v) and v >= 0 for v in values.values())
                or row['all_logits_bit_exact'] != all(v == 0 for v in values.values())
                or any(v != 0 for k, v in values.items() if k.startswith('raw_'))):
            raise ValueError('prediction difference evidence or raw-input controls differ')
    exact = all(r['all_logits_bit_exact'] for r in comparisons if r['mode'] == 'deterministic')
    if receipt.get('deterministic_logits_bit_exact') != exact:
        raise ValueError('deterministic prediction outcome disagrees')
    return exact


def repeat_differences(cells):
    keys = ['stream', 'dataset', 'decoder', 'mode']
    a, b = [cells[cells['repeat'] == r] for r in (0, 1)]
    paired = b.merge(a, on=keys, suffixes=('_r1', '_r0'), validate='one_to_one')
    if (len(paired) != 340 or len(a) != len(b)
            or not (paired.episode_fingerprint_r1 == paired.episode_fingerprint_r0).all()):
        raise ValueError('complete input-matched repeat grid required')
    for metric in METRICS:
        paired[f'delta_{metric}'] = paired[f'{metric}_r1'] - paired[f'{metric}_r0']
    if (paired.loc[paired.decoder.str.startswith('raw_'), [f'delta_{m}' for m in METRICS]].abs() > 1e-8).any().any():
        raise ValueError('raw input metrics differ across repeats')
    return paired


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--data', type=Path, default=Path(__file__).parent / 'data')
    args = parser.parse_args()
    root = args.data / 'numerical_controls_verified'
    receipt = json.loads((root / 'DONE.json').read_text())
    arms = pd.read_json(root / 'arms.json')
    exact_states = validate_training(receipt, arms, json.loads((root / 'state_comparisons.json').read_text()))
    exact_logits = validate_logits(json.loads((root / 'evaluation_DONE.json').read_text()),
                                   json.loads((root / 'logit_comparisons.json').read_text()))
    validate_inputs(json.loads((root / 'input_validation.json').read_text()),
                    json.loads((args.data / 'member_training_verified/input_validation.json').read_text()))
    reference = pd.read_csv(args.data / 'member_replay_cells.csv')
    streams = []
    for stream in ('original', 'fresh'):
        rows = validate_replay(read_jsonl(args.data / f'numerical_controls_{stream}'),
                               arms.assign(policy='lowest_sorted'), reference[reference.stream == stream], stream)
        streams.append(rows.merge(arms[['model_id', 'mode', 'repeat']], on='model_id', validate='many_to_one'))
    cells = pd.concat(streams, ignore_index=True)
    differences = repeat_differences(cells)
    if exact_logits and not differences.loc[differences['mode'] == 'deterministic', [f'delta_{m}' for m in METRICS]].eq(0).all().all():
        raise ValueError('exact logits but different reported metrics')
    older = reference[(reference.source == 'cp_hk') & (reference.seed == 2) & (reference.policy == 'lowest_sorted')].copy()
    newer = pd.read_csv(args.data / 'readout_training_cells.csv')
    newer = newer[(newer.source == 'cp_hk') & (newer.seed == 2) & (newer.condition == 'free')].copy()
    for historical in (older, newer):
        if len(historical) != 170:
            raise ValueError('historical control grid incomplete')
        joined = cells.merge(historical, on=['stream', 'dataset', 'decoder'], suffixes=('', '_historical'), validate='many_to_one')
        if not (joined.episode_fingerprint == joined.episode_fingerprint_historical).all():
            raise ValueError('historical controls used different target inputs')
    full = cells[cells.decoder == 'full_model'].assign(origin='new_control', full_training_inputs_verified=True)
    older = older[older.decoder == 'full_model'].assign(origin='earlier_factorial', mode='default', full_training_inputs_verified=False)
    newer = newer[newer.decoder == 'full_model'].assign(origin='earlier_readout_free', mode='default', full_training_inputs_verified=True)
    columns = ['origin', 'model_id', 'stream', 'dataset', 'mode', 'full_training_inputs_verified', *METRICS]
    historical_table = pd.concat([f[columns] for f in (full, older, newer)], ignore_index=True)
    cells.assign(sources=cells.sources.map(json.dumps)).to_csv(args.data / 'numerical_controls_cells.csv', index=False)
    differences.to_csv(args.data / 'numerical_controls_repeat_differences.csv', index=False)
    historical_table.to_csv(args.data / 'numerical_controls_full_with_historical.csv', index=False)
    summary = dict(valid=True, cells=len(cells), full_model_cells=len(full), comparisons=len(differences),
                   deterministic_saved_states_bit_exact=exact_states, deterministic_target_logits_bit_exact=exact_logits,
                   reproducibility_prediction_supported=exact_states and exact_logits, independent_seeds=1,
                   selected_stress_case=True, general_variability_estimate=False,
                   default_maximum_absolute_full_auc_difference=float(differences.loc[
                       (differences['mode'] == 'default') & (differences.decoder == 'full_model'), 'delta_roc_auc'].abs().max()))
    (args.data / 'numerical_controls_validation.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    print(historical_table[['origin', 'model_id', 'stream', 'dataset', 'roc_auc']].to_string(index=False))


if __name__ == '__main__':
    main()
