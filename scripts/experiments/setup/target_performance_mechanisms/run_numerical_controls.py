"""Four full CPU reproducibility controls on one prespecified source/seed."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import time

import torch

from experiments.run_shared_graph import validate_configs, write_json
from .audit_readout_control_states import state_equal
from .readout_training_constraint import configure_trainer
from .replay import batch_hash
from .run_cpu_member_training import execute_cpu_plans
from .verify_member_training import model_digest
from .verify_readout_training import input_hashes

REFERENCE = Path('/dataMeR1/phil/gfm/prodigy-mechanisms-freeze/log/target_mechanisms/readout_constraint_training_20260906')
REFERENCE_ID = 'readouttrain_cp_hk_free_s2'
MODES = ('default', 'deterministic')


def reference_params(root):
    matches = [p for p in json.loads((root / 'manifest.json').read_text())['jobs'] if p['prefix'] == REFERENCE_ID]
    if len(matches) != 1:
        raise ValueError('unique prespecified Hong Kong seed-2 reference required')
    p = matches[0]
    if (p['seed'] != 2 or p['neighbor_sampling_source_subset'] != 'cp_hk'
            or p['neighbor_matching_member_policy'] != 'lowest_sorted' or p['dataset_len_cap'] != 2500
            or p['workers'] != 2 or p['readout_training_condition'] != 'free'):
        raise ValueError('reference recipe differs')
    return p


def make_plans(reference, run_dir, smoke_steps):
    stamp = time.strftime('%Y%m%d_%H%M%S')
    plans = []
    for mode in MODES:
        for repeat in (0, 1):
            p = copy.deepcopy(reference)
            prefix = f'numctl_cp_hk_s2_{mode}_r{repeat}'
            p.update(device=torch.device('cpu'), prefix=prefix, exp_name=f'{prefix}_{stamp}',
                     state_dir=str(run_dir / 'state'), log_dir=str(run_dir / 'log'))
            if smoke_steps:
                p.update(dataset_len_cap=smoke_steps, epochs=1, checkpoint_steps=f'0,{smoke_steps}')
            plans.append(p)
    validate_configs(plans)
    for p, mode, repeat in zip(plans, ('default', 'default', 'deterministic', 'deterministic'), (0, 1, 0, 1)):
        p.update(numerical_mode=mode, numerical_repeat=repeat,
                 numerical_reference_checkpoint=str(Path(reference['state_dir']) / reference['exp_name'] / 'checkpoint/training_state_0.ckpt'),
                 numerical_reference_inputs=str(Path(reference['log_dir']) / reference['exp_name'] / 'data/constraint_inputs.jsonl.gz'))
    return plans


def configure_numerics(trainer):
    p = trainer.parameter
    torch.use_deterministic_algorithms(p['numerical_mode'] == 'deterministic')
    reference = torch.load(p['numerical_reference_checkpoint'], map_location='cpu', weights_only=False)
    state = reference['_training_checkpoint']
    if (not state_equal(reference['model'], trainer.model.state_dict())
            or not state_equal(state['optimizer'], trainer.optimizer.state_dict())
            or not state_equal(state['rng'], trainer._rng_state_dict())):
        raise ValueError('initial model/optimizer/RNG differs from prespecified control')
    expected_contract = dict(state['parameter_contract'], dataset_len_cap=p['dataset_len_cap'])
    if trainer._resume_parameter_contract() != expected_contract:
        raise ValueError('resume contract differs beyond bounded smoke length')
    expected = input_hashes(Path(p['numerical_reference_inputs']), 2500)
    receipt = configure_trainer(trainer)
    receipt.update(numerical_mode=p['numerical_mode'], historical_inputs_checked=0)

    def check_input(module, batch):
        if not module.training:
            return
        step = receipt['historical_inputs_checked']
        if batch_hash(batch) != expected[step]:
            raise ValueError(f'full historical training input mismatch at step {step + 1}')
        receipt['historical_inputs_checked'] += 1
    trainer.model.register_forward_pre_hook(check_input)


def verify(training, steps):
    manifest = json.loads((training / 'manifest.json').read_text())
    jobs = manifest['jobs']
    if len(jobs) != 4 or {(p['numerical_mode'], p['numerical_repeat']) for p in jobs} != {
            (mode, repeat) for mode in MODES for repeat in (0, 1)}:
        raise ValueError('complete four-control grid required')
    records, states = [], {}
    saved_steps = [0, steps] if steps != 2500 else [0, 100, 300, 900, 2500]
    for i, p in enumerate(jobs):
        result = json.loads((training / f'job_{i:03d}/result.json').read_text())
        audit = result.get('training_constraint', {})
        if (result['status'] != 'complete' or result['steps_observed'] != steps
                or audit.get('historical_inputs_checked') != steps
                or audit.get('inputs_hashed') != steps or audit.get('steps_checked') != steps
                or audit.get('numerical_mode') != p['numerical_mode']):
            raise ValueError('incomplete control or per-update audit')
        inputs = input_hashes(Path(p['log_dir']) / p['exp_name'] / 'data/constraint_inputs.jsonl.gz', steps)
        if inputs != input_hashes(Path(p['numerical_reference_inputs']), 2500)[:steps]:
            raise ValueError('independent historical input audit failed')
        folder = Path(result['checkpoint_dir'])
        saved = {s: torch.load(folder / f'training_state_{s}.ckpt', map_location='cpu', weights_only=False) for s in saved_steps}
        initial = torch.load(p['numerical_reference_checkpoint'], map_location='cpu', weights_only=False)
        if not state_equal(saved[0]['model'], initial['model']):
            raise ValueError('saved initialization differs')
        for s, checkpoint in saved.items():
            if checkpoint['_training_checkpoint']['completed_steps'] != s or not all(
                    torch.isfinite(t).all() for t in checkpoint['model'].values()):
                raise ValueError('wrong checkpoint step or non-finite state')
        states[(p['numerical_mode'], p['numerical_repeat'])] = saved
        records.append(dict(model_id=p['prefix'], mode=p['numerical_mode'], repeat=p['numerical_repeat'],
                            seed=2, source='cp_hk', inputs_matched=steps, initial_sha256=model_digest(saved[0]['model']),
                            final_sha256=model_digest(saved[steps]['model']), checkpoint=str(folder / f'state_dict_{steps}.ckpt')))
    comparisons = []
    for mode in MODES:
        for step in saved_steps:
            a, b = [states[(mode, r)][step] for r in (0, 1)]
            comparisons.append(dict(mode=mode, step=step, model_bit_exact=state_equal(a['model'], b['model']),
                                    **{f'{k}_bit_exact': state_equal(a['_training_checkpoint'][k], b['_training_checkpoint'][k])
                                       for k in ('optimizer', 'rng', 'train_batch_sampler')}))
    output = training / 'verified'
    output.mkdir()
    write_json(output / 'arms.json', records)
    write_json(output / 'state_comparisons.json', comparisons)
    (output / 'model_list.tsv').write_text('model_id\tcheckpoint\tsources\n' + ''.join(
        f"{r['model_id']}\t{r['checkpoint']}\tcp_hk\n" for r in records))
    write_json(output / 'DONE.json', dict(valid=True, models=4, steps_per_model=steps, exact_historical_inputs=True,
                                         same_initialization=True, smoke=steps != 2500, independent_seeds=1,
                                         deterministic_states_bit_exact=all(r['model_bit_exact'] and r['optimizer_bit_exact']
                                             for r in comparisons if r['mode'] == 'deterministic')))


def evaluate(training):
    from .finish_member_pipeline import TARGETS, compare_cached_inputs
    output = training / 'evaluation'
    output.mkdir()
    outputs = {}
    for stream, offset in (('original', 0), ('fresh', 100003)):
        outputs[stream] = output / stream
        command = [sys.executable, '-u', '-m', 'scripts.experiments.setup.target_performance_mechanisms.replay',
                   '--model-list', str(training / 'verified/model_list.tsv'), '--output', str(outputs[stream]),
                   '--datasets', ','.join(TARGETS), '--variants', 'baseline', '--device', '123',
                   '--threads', '4', '--eval-episode-seed-offset', str(offset)]
        with (output / f'{stream}.log').open('w') as handle:
            subprocess.run(command, check=True, stdout=handle, stderr=subprocess.STDOUT)
    inputs = compare_cached_inputs(outputs, Path('/dataMeR1/phil/gfm/prodigy-mechanisms/log/target_mechanisms'))
    write_json(output / 'input_validation.json', inputs)
    records = json.loads((training / 'verified/arms.json').read_text())
    comparisons = []
    for stream, destination in outputs.items():
        for target in TARGETS:
            for mode in MODES:
                models = sorted((r for r in records if r['mode'] == mode), key=lambda r: r['repeat'])
                left, right = [torch.load(destination / target / f"{r['model_id']}__baseline.pt",
                                         map_location='cpu', weights_only=False) for r in models]
                comparisons.append(dict(stream=stream, target=target, mode=mode,
                                        **compare_logits(left, right)))
    write_json(output / 'logit_comparisons.json', comparisons)
    write_json(output / 'DONE.json', dict(complete=True, models=4, targets=5, streams=2, same_target_inputs=True,
        deterministic_logits_bit_exact=all(r['all_logits_bit_exact'] for r in comparisons if r['mode'] == 'deterministic')))


def compare_logits(left, right, expected_batches=32, expected_decoders=17):
    if len(left) != expected_batches or len(right) != expected_batches:
        raise ValueError('incomplete saved prediction stream')
    maxima = {}
    for i, (a, b) in enumerate(zip(left, right, strict=True)):
        if (a['batch'] != i or b['batch'] != i or a['batch_sha256'] != b['batch_sha256']
                or a['logits'].keys() != b['logits'].keys() or len(a['logits']) != expected_decoders):
            raise ValueError('prediction decoder or input identity differs')
        for decoder in a['logits']:
            x, y = a['logits'][decoder], b['logits'][decoder]
            if x.shape != y.shape or not torch.isfinite(x).all() or not torch.isfinite(y).all():
                raise ValueError('prediction shape differs or values non-finite')
            maxima[decoder] = max(maxima.get(decoder, 0), float((x.double() - y.double()).abs().max()))
    return dict(all_logits_bit_exact=all(v == 0 for v in maxima.values()),
                comparisons=expected_batches * expected_decoders, maximum_absolute_difference_by_decoder=maxima)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--reference', type=Path, default=REFERENCE)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--smoke-steps', type=int, default=0)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    args = parser.parse_args()
    if args.smoke_steps not in (0, 8) or (args.smoke_steps and args.evaluate):
        raise ValueError('only eight-step smoke or full 2500-update controls allowed')
    args.run_dir = args.run_dir.resolve()
    if args.verify_only:
        verify(args.run_dir, args.smoke_steps or 2500)
        return
    plans = make_plans(reference_params(args.reference), args.run_dir, args.smoke_steps)
    print(json.dumps({'controls': 4, 'source': 'cp_hk', 'seed': 2, 'steps': args.smoke_steps or 2500,
                      'modes': MODES, 'cpu_threads_and_workers': 40, 'target_selected_checkpoint': False}), flush=True)
    if args.dry_run:
        return
    args.models, args.threads_per_model, args.workers_per_model = 4, 8, 2
    command = [sys.executable, '-m', __spec__.name, '--run-dir', str(args.run_dir), '--verify-only']
    if args.smoke_steps:
        command += ['--smoke-steps', '8']
    execute_cpu_plans(args, plans, [Path(p['config']) for p in plans], command, trainer_setup=configure_numerics)
    if args.evaluate:
        evaluate(args.run_dir)


if __name__ == '__main__':
    main()
