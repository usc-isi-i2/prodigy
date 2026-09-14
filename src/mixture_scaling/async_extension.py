"""Fixed-horizon KD extension and Ukraine-only controls from one full state."""
import argparse
import copy
import hashlib
import json
import math
import shutil
import subprocess
import time
from pathlib import Path

import torch

from . import async_convergence as ac
from .async_singleton_reference import model_match
from .evaluation_artifacts import atomic_json

ARMS = ('kd_extended', 'ukraine_only')


def read_history(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def on_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, list):
        return [on_device(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(on_device(v, device) for v in value)
    return value


def source_index(arm, additional_step):
    return additional_step % 2 if arm == 'kd_extended' else 0


def weighted_task_loss(logits, y, teacher_logits=None, kd_weight=1.):
    loss = ac.task_loss(logits, y, teacher_logits)
    return loss * kd_weight if teacher_logits is not None and kd_weight != 1. else loss


def increment(counts, index, npos, kd):
    item = counts[index]
    prefix = 'kd' if kd else 'supervised'
    item[prefix + '_updates'] += 1
    item['positive_examples'] += npos
    item['negative_examples'] += 5 * npos
    item[prefix + '_positive_examples'] += npos
    item[prefix + '_negative_examples'] += 5 * npos


def load_start(parent):
    run = Path(parent) / 'node_neighbors/lp/async_kd'
    summary = json.loads((run / 'summary.json').read_text())
    if summary.get('stop_reason') not in (None, 'all_tasks_converged'):
        raise ValueError('continuation requires an all-converged parent, not a capped run')
    selected = torch.load(summary['events'][-1]['teacher_checkpoint'], map_location='cpu', weights_only=False)
    path = run / 'endpoint.pt'
    ck = torch.load(path, map_location='cpu', weights_only=False)
    # The terminal state retains both teacher declarations even if their order
    # differs from seed0; a raw historical best snapshot need not contain them.
    if not model_match(selected['model'], ck['model'], tolerance=0)['matches']:
        raise ValueError('start does not match the declared parent endpoint')
    if ck['runtime']['counts'] != summary['surviving_counts'] or ck['step'] != summary['logical_step']:
        raise ValueError('start exposure differs from parent endpoint')
    if ck['step'] % 2 or summary['sources'] != list(ac.SOURCES):
        raise ValueError('expected an even alternating-source checkpoint in Ukraine/Facebook order')
    if not ck.get('exact_sampling_resume') or not ck['runtime']['teacher_paths'][1]:
        raise ValueError('exact resume state and original Facebook teacher required')
    return run, summary, path, ck


def train(args, device):
    run_id = getattr(args, 'run_id', None) or args.arm
    kd_weight = getattr(args, 'kd_weight', 1.)
    if not math.isfinite(kd_weight) or kd_weight <= 0:
        raise ValueError('KD loss weight must be finite and positive')
    run_dir = Path(args.root) / 'node_neighbors/lp' / run_id
    if run_dir.exists():
        raise FileExistsError(run_dir)
    parent, original, start_path, ck = load_start(args.parent_root)
    config = ac.load_config(args.config)
    ac.base.preflight(config)
    data_seed = ac.optional_seed(args, 'data_seed', original.get('data_seed', original['seed']))
    graphs = [ac.base.load_graph(s, config, data_seed, device) for s in ac.SOURCES]
    if args.seed != original['seed']:
        raise ValueError('continuation seed must match parent')
    for source, graph in zip(ac.SOURCES, graphs):
        if graph['receipt'] != original['graph_split_receipts'][source]:
            raise ValueError('graph/split receipt changed')
    ac.initialize_sampling(graphs, args.seed, device)
    validation = [ac.diagnostic.validation_pairs(g) for g in graphs]
    probe_paths = [parent / f'probe_{s}.pt' for s in ac.SOURCES]
    probes = [on_device(torch.load(p, map_location='cpu', weights_only=False), device) for p in probe_paths]
    model = ac.base.BiasMLP('node_neighbors').to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=.0005, weight_decay=1e-5)
    runtime = ac.restore(ck, model, opt, graphs)
    # Never inherit the terminal declaration that Ukraine has also converged.
    facebook_teacher = runtime['teacher_paths'][1]
    runtime['converged'] = [False, args.arm == 'kd_extended']
    runtime['teacher_paths'] = [None, facebook_teacher if args.arm == 'kd_extended' else None]
    teacher = ac.frozen_teacher(facebook_teacher, device) if args.arm == 'kd_extended' else None
    metadata = dict(run_id=run_id, sources=list(ac.SOURCES), seed=args.seed, training_seed=args.seed,
                    data_seed=data_seed, probe_seed=original.get('probe_seed',original['seed']), view='node_neighbors',
                    architecture='node_mlp', objective='lp',
                    code_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    start_checkpoint=str(start_path), start_sha256=sha256(start_path),
                    parent_root=str(args.parent_root), start_counts=copy.deepcopy(runtime['counts']),
                    graph_split_receipts=original['graph_split_receipts'],
                    probe_sha256={s:sha256(p) for s,p in zip(ac.SOURCES,probe_paths)},
                    teacher_checkpoint=facebook_teacher if teacher is not None else None,
                    teacher_sha256=sha256(facebook_teacher) if teacher is not None else None,
                    protocol=dict(additional_optimizer_steps=args.additional_steps, validation_interval=args.validation_interval,
                                  schedule='alternating_1_to_1' if teacher is not None else 'ukraine_only',
                                  stopping='fixed added-update horizon; no early stopping or rewinds',
                                  learning_rate=opt.param_groups[0]['lr'], weight_decay=opt.param_groups[0]['weight_decay'],
                                  optimizer='restored AdamW; no reset', kd_temperature=1., kd_weight=kd_weight if teacher is not None else 0.,
                                  selection='fixed endpoints plus source-validation constrained selection on common Ukraine exposure grid'),
                    prefix_physical_updates=original['physical_steps'], prefix_compute_shared=True)
    run_dir.mkdir(parents=True)
    atomic_json(run_dir / 'metadata.json', metadata)
    physical_counts = [ac.empty_counts() for _ in ac.SOURCES]
    original_rows = {r['logical_step']: r for r in read_history(parent / 'history.jsonl')}
    replay_checks = []
    replay_not_applicable = []
    additional = 0
    started = time.monotonic()
    with ac.tracked_run(run_dir, metadata, args) as (wandb, log):
        def validate_save():
            report = ac.measure(model, graphs, validation, probes)
            if additional == 0 and report != original['endpoint']:
                raise ValueError('initial validation/probe measurements do not reproduce parent endpoint')
            path = run_dir / 'checkpoints' / f'additional_{additional:06d}.pt'
            saved = ac.snapshot(model, opt, graphs, runtime, metadata)
            ac.save(path, saved)
            row = dict(additional_step=additional, logical_step=runtime['logical_step'],
                       counts=copy.deepcopy(runtime['counts']), checkpoint=str(path), **report)
            with (run_dir / 'history.jsonl').open('a') as handle:
                handle.write(json.dumps(row) + '\n')
            if args.arm == 'kd_extended' and kd_weight == 1. and additional and runtime['logical_step'] in original_rows:
                old = original_rows[runtime['logical_step']]
                prior = torch.load(old['checkpoint'], map_location='cpu', weights_only=False)
                prior_state = prior['runtime']
                same_objective = (prior_state['converged'] == [False, True]
                                  and prior_state['teacher_paths'][1] == facebook_teacher)
                if same_objective:
                    check = model_match(saved['model'], prior['model'], tolerance=0)
                    check.update(additional_step=additional, logical_step=runtime['logical_step'],
                                 measurements_identical=all(report[k] == old[k] for k in report))
                    replay_checks.append(check)
                    if not check['matches'] or not check['measurements_identical']:
                        raise ValueError(f'prior KD replay failed: {check}')
                else:
                    replay_not_applicable.append(dict(additional_step=additional,
                        reason='parent trajectory has different active objectives or Facebook teacher'))
            fields = {f'validation/{s}/{k}': v for s, measurement in zip(ac.SOURCES, report['validation']) for k,v in measurement.items()}
            log(additional, fields)
            print(json.dumps(dict(arm=run_id, additional_step=additional, counts=runtime['counts'], **report)), flush=True)
            return report

        last = validate_save()
        for additional_before in range(args.additional_steps):
            index = source_index(args.arm, additional_before)
            pairs, y, npos = ac.next_batch(graphs[index])
            opt.zero_grad(set_to_none=True)
            logits = ac.base.score(model, graphs[index]['x'], pairs)
            teacher_logits = None
            if index == 1 and teacher is not None:
                with torch.no_grad():
                    teacher_logits = ac.base.score(teacher, graphs[index]['x'], pairs)
            loss = weighted_task_loss(logits, y, teacher_logits, kd_weight)
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite continuation loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
            additional = additional_before + 1
            runtime['logical_step'] += 1
            for counts in (runtime['counts'], physical_counts):
                increment(counts, index, npos, teacher_logits is not None)
            if additional % args.log_interval == 0:
                fields = {f'train/{ac.SOURCES[index]}/objective':float(loss.detach())}
                if teacher_logits is not None:
                    fields[f'train/{ac.SOURCES[index]}/unweighted_kd_bce'] = float(loss.detach()) / kd_weight
                log(additional, fields)
            if additional % args.validation_interval == 0 or additional in (args.additional_steps // 2, args.additional_steps):
                last = validate_save()
        summary = dict(metadata, status='complete', stop_reason='fixed_horizon', budget_steps=args.additional_steps,
                       additional_steps=additional, logical_step=runtime['logical_step'],
                       surviving_counts=runtime['counts'], physical_counts=physical_counts, endpoint=last,
                       teacher_forward_batches=physical_counts[1]['kd_updates'],
                       teacher_forward_pairs=physical_counts[1]['kd_positive_examples'] + physical_counts[1]['kd_negative_examples'],
                       elapsed_seconds=time.monotonic()-started, replay_checks=replay_checks,
                       replay_not_applicable=replay_not_applicable)
        atomic_json(run_dir / 'summary.json', summary)
        wandb.summary.update(dict(additional_steps=additional, stop_reason='fixed_horizon'))


def reference_metrics(parent):
    references = {}
    thresholds = []
    for source in ac.SOURCES:
        run = Path(parent) / 'singleton_references' / source
        summary = json.loads((run / 'summary.json').read_text())
        if not summary['historical_model_identity_verified']:
            raise ValueError('singleton identity not verified')
        history = read_history(run / 'history.jsonl')
        selected = next(r for r in history if r['step'] == summary['historical_best_step'])
        references[source] = dict(summary=summary, history=history)
        thresholds.append(selected['validation']['auc'])
    return references, thresholds


def select_preserving(rows, facebook_auc, start_updates, max_added_ukraine, exposure_grid):
    candidates = [r for r in rows if
                  0 <= r['counts'][0]['supervised_updates'] - start_updates <= max_added_ukraine
                  and (r['counts'][0]['supervised_updates'] - start_updates) % exposure_grid == 0
                  and r['validation'][1]['auc'] >= facebook_auc]
    return max(candidates, key=lambda r:(r['validation'][0]['auc'], -r['additional_step'])) if candidates else None


def prepare_eval(args):
    root = Path(args.root)
    summaries = {a:json.loads((root / 'node_neighbors/lp' / a / 'summary.json').read_text()) for a in ARMS}
    histories = {a:read_history(root / 'node_neighbors/lp' / a / 'history.jsonl') for a in ARMS}
    references, thresholds = reference_metrics(args.parent_root)
    first, second = [summaries[a] for a in ARMS]
    for key in ('start_sha256', 'start_counts', 'probe_sha256', 'graph_split_receipts', 'budget_steps'):
        if first[key] != second[key]:
            raise ValueError(f'arms disagree on {key}')
    budget = first['budget_steps']
    def at(arm, step):
        return next(r for r in histories[arm] if r['additional_step'] == step)
    kd, exposure, updates = at('kd_extended', budget), at('ukraine_only', budget//2), at('ukraine_only', budget)
    if kd['counts'][0] != exposure['counts'][0]:
        raise ValueError('Ukraine exposure counters differ')
    left, right = [torch.load(r['checkpoint'], map_location='cpu', weights_only=False) for r in (kd,exposure)]
    for key in ('generator', 'order_generator', 'order'):
        if not torch.equal(left['samplers'][0][key], right['samplers'][0][key]):
            raise ValueError(f'Ukraine sampler differs at exposure match: {key}')
    if left['samplers'][0]['offset'] != right['samplers'][0]['offset']:
        raise ValueError('Ukraine sampler offsets differ')
    models, unavailable = [], []
    def export(alias, arm, row, rule):
        dest = root / 'node_neighbors/lp' / alias
        dest.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(row['checkpoint'], dest / 'best.pt')
        models.append(dict(run_id=alias, arm=arm, selection=rule, original_checkpoint=row['checkpoint'],
                           additional_step=row['additional_step'], counts=row['counts'],
                           validation=row['validation'], start_fallback=row['additional_step']==0))
    export('kd_fixed', 'kd_extended', kd, f'fixed {budget} added updates; {kd["counts"][0]["supervised_updates"]} cumulative Ukraine updates')
    export('ukraine_exposure', 'ukraine_only', exposure, 'same added Ukraine exposure as KD endpoint')
    export('ukraine_updates', 'ukraine_only', updates, 'same added optimizer updates as KD endpoint; not equal compute')
    for arm in ARMS:
        row = select_preserving(histories[arm], thresholds[1], first['start_counts'][0]['supervised_updates'],
                                budget//2, args.selection_grid)
        if row is None:
            unavailable.append(dict(arm=arm, reason='no source-validation candidate preserves Facebook'))
        else:
            export(arm + '_selected', arm, row,
                   f'max Ukraine validation AUC subject to Facebook singleton AUC; common {args.selection_grid} Ukraine grid and <= {budget//2} added Ukraine updates')
    audit = dict(ukraine_exposure_and_sampler_match=True, start_state_identical=True,
                 kd_parent_replay_checks=first['replay_checks'])
    manifest = dict(models=models, unavailable=unavailable, replay_check=audit,
                    singleton_source_auc=dict(zip(ac.SOURCES, thresholds)),
                    selection_grid_ukraine_updates=args.selection_grid,
                    selection_data='source validation only; frozen before downstream evaluation')
    atomic_json(root / 'evaluation_manifest.json', manifest)
    aggregate = dict(history=histories, summary=summaries, manifest=manifest, references=references)
    atomic_json(root / 'analysis_data.json', aggregate)
    print(json.dumps(manifest, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['plan','train','prepare_eval','eval','aggregate'])
    parser.add_argument('--root', required=True)
    parser.add_argument('--parent-root', required=True)
    parser.add_argument('--arm', choices=ARMS, default='kd_extended')
    parser.add_argument('--run-id')
    parser.add_argument('--kd-weight', type=float, default=1.)
    parser.add_argument('--additional-steps', type=int, default=60000)
    parser.add_argument('--validation-interval', type=int, default=2000)
    parser.add_argument('--selection-grid', type=int, default=2000)
    parser.add_argument('--log-interval', type=int, default=100)
    parser.add_argument('--device', type=int, choices=range(4), default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--data-seed', type=int)
    parser.add_argument('--config', default='configs/nonzero_mini_transfer.yaml')
    parser.add_argument('--wandb-mode', default='offline')
    parser.add_argument('--wandb-project', default='nonzero-mini-transfer')
    parser.add_argument('--wandb-group', default='async-extension')
    parser.add_argument('--prodigy-root', default='/dataMeR1/phil/gfm/prodigy-walk-mini-pilot')
    args = parser.parse_args()
    if any(n <= 0 or n % 2 for n in (args.additional_steps,args.validation_interval,args.log_interval,args.selection_grid)):
        parser.error('positive even budgets/intervals required')
    if args.additional_steps % (2 * args.selection_grid) or args.additional_steps % (2 * args.validation_interval):
        parser.error('budget must divide both exposure and validation grids')
    if args.phase == 'plan':
        _, summary, path, _ = load_start(args.parent_root)
        print(json.dumps(dict(start=str(path), start_counts=summary['surviving_counts'], added_updates=args.additional_steps,
                              arms=ARMS, inputs=ac.base.preflight(ac.load_config(args.config))), indent=2))
        return
    if args.phase == 'prepare_eval':
        prepare_eval(args)
        return
    if args.phase == 'aggregate':
        ac.aggregate(args)
        return
    torch.set_num_threads(4)
    torch.cuda.set_device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(f'cuda:{args.device}')
    if args.phase == 'train':
        train(args, device)
    else:
        args.views = ('node_neighbors',)
        args.targets = ac.base.SOURCES
        args.model_rows = [(m['run_id'],m['run_id']) for m in json.loads((Path(args.root)/'evaluation_manifest.json').read_text())['models']]
        ac.base.evaluate(args, device)


if __name__ == '__main__':
    main()
