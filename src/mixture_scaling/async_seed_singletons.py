"""New-seed singleton baselines with fixed seed-zero data and validation.

Checkpoint selection and stopping reproduce mini_lp.train's policy, while
retaining every validation checkpoint and a fixed hard-label training probe.
The seed changes initialization, training edge order, and sampled negatives;
it does not resample graph splits, context features, validation, or probes.
"""
import argparse
import json
import math
import subprocess
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from . import async_convergence as ac
from .config import load_config
from .evaluation_artifacts import atomic_json


def update_selection(state, report, step):
    """Original strict-best and 1e-4 absolute-improvement stopping policy."""
    value = report['bce']
    if not math.isfinite(value):
        raise FloatingPointError('nonfinite singleton source validation BCE')
    improved = value < state['best_bce']
    if improved:
        state.update(best_bce=value, selected_step=step, selected_validation=dict(report))
    if value < state['reference_bce'] - 1e-4:
        state.update(reference_bce=value, stale=0)
    elif step >= 2500:
        state['stale'] += 1
    return improved


def measure(model, graph, probe):
    mode = model.training
    try:
        # This retains the original float64 metric calculation, rather than
        # changing the early-stopping threshold via float32 batch reductions.
        report = ac.base.validate(model, graph, 0)
        model.eval()
        training_probe = ac.diagnostic.evaluate(model, graph, probe)[0]
        return dict(validation=dict(bce=report['bce'], auc=report['roc_auc']),
                    training_probe=training_probe)
    finally:
        model.train(mode)


def train(source, args, device):
    if args.data_seed != 0:
        raise ValueError('data, context, validation, and probes must remain seed zero')
    if args.seed not in (1, 2):
        raise ValueError('this locked replication uses training seeds 1 and 2')
    if not 0 < args.max_steps <= 100000 or args.validation_interval < 1 or args.patience < 1:
        raise ValueError('invalid singleton stopping configuration')
    run_dir = Path(args.root) / 'singletons' / source
    if run_dir.exists():
        raise FileExistsError(f'singleton output already exists: {run_dir}')
    source_index = ac.SOURCES.index(source)
    graph = ac.base.load_graph(source, load_config(args.config), args.data_seed, device)
    ac.initialize_sampling([graph], args.seed, device)
    probe_seed = 813719 + 1009 * source_index
    probe = ac.fixed_probe(graph, probe_seed)
    ac.seed_everything(args.seed)
    model = ac.base.BiasMLP('node_neighbors').to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0005, weight_decay=1e-5)
    metadata = dict(run_id=f'singleton_{source}_training_seed{args.seed}', sources=[source],
                    seed=args.seed, data_seed=args.data_seed, view='node_neighbors',
                    architecture='node_mlp', objective='lp',
                    code_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    graph_split_receipt=graph['receipt'],
                    checkpoint_selection='minimum source validation BCE; step zero ineligible',
                    protocol=dict(initialization_seed=args.seed, training_order_seed=args.seed + 7919,
                                  training_negative_seed=args.seed + 49979687,
                                  data_seed=0, fixed_probe_seed=probe_seed, validation_seed=32452843,
                                  learning_rate=.0005, weight_decay=1e-5, gradient_clip_norm=1.,
                                  source_schedule='source-only every optimizer update',
                                  stopping=dict(min_delta=1e-4, minimum_steps=2500,
                                                patience=args.patience, validation_interval=args.validation_interval,
                                                safety_cap=args.max_steps),
                                  selection_data='source validation only; no downstream evaluation'))
    runtime = dict(logical_step=0, counts=[ac.empty_counts()],
                   selection=dict(best_bce=float('inf'), reference_bce=float('inf'), stale=0,
                                  selected_step=None, selected_validation=None))
    run_dir.mkdir(parents=True)
    atomic_json(run_dir / 'metadata.json', metadata)
    ac.save(run_dir / 'training_probe.pt', ac.cpu_clone(probe))
    started = time.monotonic()
    window_total = torch.zeros((), device=device)
    window_count = 0
    stop_reason = 'safety_cap'
    with ac.tracked_run(run_dir, metadata, args) as (wandb, log):
        def validate_save(step, initial=False):
            reports = measure(model, graph, probe)
            improved = False if initial else update_selection(runtime['selection'], reports['validation'], step)
            path = run_dir / 'checkpoints' / f'step_{step}.pt'
            checkpoint = ac.snapshot(model, optimizer, [graph], runtime, metadata)
            ac.save(path, checkpoint)
            if not initial:
                ac.save(run_dir / 'latest.pt', checkpoint)
            if improved:
                ac.save(run_dir / 'best.pt', checkpoint)
            row = dict(step=step, source=source, checkpoint=str(path),
                       counts=ac.cpu_clone(runtime['counts'][0]),
                       validation_stale_checks=runtime['selection']['stale'],
                       eligible_for_selection=not initial, **reports)
            atomic_json(run_dir / 'validation' / f'step_{step}.json', row)
            with (run_dir / 'history.jsonl').open('a') as handle:
                handle.write(json.dumps(row) + '\n')
            log(step, {'validation/auc': reports['validation']['auc'],
                       'validation/loss': reports['validation']['bce'],
                       'validation/bce': reports['validation']['bce'],
                       'validation/stale_checks': runtime['selection']['stale'],
                       'check/hard_label_probe_bce': reports['training_probe']['bce']})
            print(json.dumps(dict(source=source, seed=args.seed, step=step,
                                  validation=reports['validation'], stale=runtime['selection']['stale'])), flush=True)
            return reports

        validate_save(0, initial=True)
        for step in range(1, args.max_steps + 1):
            pairs, labels, npositive = ac.next_batch(graph)
            optimizer.zero_grad(set_to_none=True)
            logits = ac.base.score(model, graph['x'], pairs)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            runtime['logical_step'] = step
            counts = runtime['counts'][0]
            counts['supervised_updates'] += 1
            for key, amount in (('positive_examples', npositive), ('negative_examples', 5 * npositive),
                                ('supervised_positive_examples', npositive), ('supervised_negative_examples', 5 * npositive)):
                counts[key] += amount
            window_total += loss.detach()
            window_count += 1
            if step % args.log_interval == 0:
                value = float(window_total / window_count)
                if not math.isfinite(value):
                    raise FloatingPointError('nonfinite singleton training loss')
                log(step, {'train/loss': value, 'train/elapsed_seconds': time.monotonic() - started})
                window_total.zero_()
                window_count = 0
            if step % args.validation_interval == 0 or step == args.max_steps:
                final_reports = validate_save(step)
                if runtime['selection']['stale'] >= args.patience:
                    stop_reason = 'validation_plateau'
                    break
        selected = runtime['selection']
        summary = dict(metadata, status='complete', selected_step=selected['selected_step'],
                       best_step=selected['selected_step'], final_step=step,
                       selected_validation=selected['selected_validation'],
                       best_validation_bce=selected['best_bce'], final_validation=final_reports['validation'],
                       final_training_probe=final_reports['training_probe'],
                       counts=runtime['counts'][0], stop_reason=stop_reason,
                       converged=stop_reason == 'validation_plateau',
                       elapsed_seconds=time.monotonic() - started)
        atomic_json(run_dir / 'summary.json', summary)
        wandb.summary.update(dict(final_step=step, selected_step=selected['selected_step'], stop_reason=stop_reason))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--source', choices=ac.SOURCES)
    parser.add_argument('--device', type=int, choices=range(4), default=3)
    parser.add_argument('--seed', type=int, choices=(1, 2), required=True)
    parser.add_argument('--data-seed', type=int, choices=(0,), default=0)
    parser.add_argument('--config', default='configs/nonzero_mini_transfer.yaml')
    parser.add_argument('--max-steps', type=int, default=100000)
    parser.add_argument('--validation-interval', type=int, default=2000)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--log-interval', type=int, default=100)
    parser.add_argument('--wandb-mode', default='offline')
    parser.add_argument('--wandb-project', default='nonzero-mini-transfer')
    parser.add_argument('--wandb-group', default='async-convergence-seed-singletons')
    args = parser.parse_args()
    if min(args.max_steps, args.validation_interval, args.patience, args.log_interval) < 1 or args.max_steps > 100000:
        parser.error('positive limits and max-steps <= 100000 required')
    torch.set_num_threads(4)
    torch.cuda.set_device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(f'cuda:{args.device}')
    for source in [args.source] if args.source else ac.SOURCES:
        train(source, args, device)
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
