"""Validate fixed-input CPU numerical audits; these are not target-performance results."""
import argparse
import csv
import json
import math
from pathlib import Path

MODES = ('default', 'deterministic', 'decoder_index_select')
CHANNELS = ('logits', 'gradients', 'weights')
INITIAL = '45bb682d264595f7cc71903e33f8abc9b91954c596e5a0cdce3defb9bba84904'
HERE = Path(__file__).parent


def validate(receipt):
    if (receipt.get('complete') is not True or receipt.get('research_model_result') is not False
            or receipt.get('torch_version') != '2.0.1+cu118' or receipt.get('threads') != 8
            or not receipt.get('actual_initial_checkpoint')):
        raise ValueError('incomplete or incompatible numerical audit')
    if receipt.get('target_evaluation_episodes_used', receipt.get('target_data_used')) is not False:
        raise ValueError('target evaluation must not enter the diagnostic')
    if receipt.get('synthetic_workload') is True:
        if receipt.get('synthetic_input_dimension') != 768:
            raise ValueError('full input dimension required')
        workload = 'synthetic'
    elif (receipt.get('synthetic_workload') is False and receipt.get('real_source') == 'ukr_rus'
          and receipt.get('reference_inputs', {}).get('all_inputs_bit_exact') is True
          and receipt['reference_inputs']['steps_checked'] == 4):
        workload = 'real_ukraine'
    else:
        raise ValueError('unverified real input prefix')
    wanted = {f'{mode}_{kind}_{repeat}': (mode, kind, repeat)
              for mode in MODES for kind in ('plain', 'hook') for repeat in (0, 1)}
    replays = {r['name']: r for r in receipt['replays']}
    if len(receipt['replays']) != 12 or replays.keys() != wanted.keys():
        raise ValueError('incomplete replay grid')
    inputs = None
    for name, row in replays.items():
        mode, kind, _ = wanted[name]
        if (row['initial_sha256'] != INITIAL or row['hook'] != (kind == 'hook')
                or row['deterministic'] != (mode == 'deterministic')
                or row['decoder_index_select'] != (mode == 'decoder_index_select')
                or row['decoder_forward_parity'] != ([True] * 4 if row['decoder_index_select'] else [])
                or [r['step'] for r in row['steps']] != [1, 2, 3, 4]):
            raise ValueError('replay initialization, mode or forward-parity contract failed')
        hashes = [r['input_sha256'] for r in row['steps']]
        if inputs is None:
            inputs = hashes
        if inputs != hashes or not all(math.isfinite(r['loss']) for r in row['steps']):
            raise ValueError('input mismatch or non-finite loss')
    wanted_comparisons = {name for name in wanted if not name.endswith('plain_0')}
    comparisons = {r['name']: r for r in receipt['comparisons']}
    if len(receipt['comparisons']) != 9 or comparisons.keys() != wanted_comparisons:
        raise ValueError('incomplete comparison grid')
    rows = []
    for name, comparison in comparisons.items():
        mode = wanted[name][0]
        ref = f'{mode}_plain_0'
        if comparison['against'] != ref or [r['step'] for r in comparison['steps']] != [1, 2, 3, 4]:
            raise ValueError('comparison reference or steps differ')
        for step, diff in enumerate(comparison['steps']):
            for channel in CHANNELS:
                value = diff[channel]
                maximum = value['maximum_absolute_difference']
                hash_key = f'{channel}_sha256'
                exact = replays[name]['steps'][step][hash_key] == replays[ref]['steps'][step][hash_key]
                largest = value['largest_differences']
                if (not math.isfinite(maximum) or maximum < 0 or value['bit_exact'] != exact
                        or exact != (maximum == 0) or exact != (value['changed_keys'] == 0)
                        or len(largest) != min(value['changed_keys'], 5)
                        or maximum != max(largest.values(), default=0)
                        or not all(math.isfinite(v) and v > 0 for v in largest.values())):
                    raise ValueError('tensor difference and digest evidence disagree')
                rows.append(dict(workload=workload, mode=mode, comparison=name, step=step + 1,
                                 channel=channel, bit_exact=exact, maximum_absolute_difference=maximum,
                                 changed_keys=value['changed_keys']))
    summary = []
    for mode in MODES:
        selected = [r for r in rows if r['mode'] == mode]
        summary.append(dict(workload=workload, mode=mode,
                            comparisons=3, all_four_updates_bit_exact=all(r['bit_exact'] for r in selected),
                            **{f'step{step}_{channel}_maximum_difference': max(
                                r['maximum_absolute_difference'] for r in selected
                                if r['step'] == step and r['channel'] == channel)
                               for step in (1, 4) for channel in CHANNELS}))
    return rows, summary, dict(valid=True, workload=workload, replays=12, updates_per_replay=4,
                              tensor_comparisons=len(rows), decoder_forward_parity_checks=16,
                              inputs_match_completed_training=workload == 'real_ukraine',
                              historical_auc_contribution_quantified=False)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--receipts', type=Path, nargs='+', required=True)
    parser.add_argument('--output-dir', type=Path, default=HERE / 'data')
    args = parser.parse_args()
    rows, summary, validations = [], [], []
    for path in args.receipts:
        r, s, v = validate(json.loads(path.read_text()))
        if any(v['workload'] == old['workload'] for old in validations):
            raise ValueError('duplicate workload')
        v['receipt'] = str(path)
        rows.extend(r)
        summary.extend(s)
        validations.append(v)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, values in (('steps', rows), ('summary', summary)):
        with (args.output_dir / f'control_numerics_{name}.csv').open('w') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    (args.output_dir / 'control_numerics_validation.json').write_text(json.dumps(validations, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
