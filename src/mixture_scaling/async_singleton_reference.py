"""Bounded singleton replays for fixed training-loss/exposure comparisons.

The horizon comes from the completed historical run, never a new stopping rule.
Historical identity is checked at its selected and terminal checkpoints; a
mismatch is reported explicitly and does not turn the replay into a reproduction.
Only training probes and source validation are evaluated here.
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from . import async_convergence as ac
from .config import load_config
from .evaluation_artifacts import atomic_json
from .mini_transfer import identity
from .ranking_preservation import SOURCE_ROOT


def model_match(actual, expected, tolerance=1e-6):
    """Compare all saved model tensors, treating nonfinite values as a mismatch."""
    if set(actual) != set(expected):
        return dict(matches=False, tolerance=tolerance, max_abs_difference=None,
                    reason='model state keys differ')
    maximum = 0.
    for name in actual:
        left, right = actual[name].detach().cpu(), expected[name].detach().cpu()
        if left.shape != right.shape:
            return dict(matches=False, tolerance=tolerance, max_abs_difference=None,
                        reason=f'tensor shape differs: {name}')
        difference = (left.double() - right.double()).abs()
        if not torch.isfinite(difference).all():
            return dict(matches=False, tolerance=tolerance, max_abs_difference=None,
                        reason=f'nonfinite difference: {name}')
        if difference.numel():
            maximum = max(maximum, float(difference.max()))
    return dict(matches=maximum <= tolerance, tolerance=tolerance,
                max_abs_difference=maximum)


def reference_spec(source):
    run = SOURCE_ROOT / f'ss_{source}'
    summary_path = run / 'summary.json'
    summary = json.loads(summary_path.read_text())
    if (summary['status'] != 'complete' or summary['seed'] != 0 or
            summary['sources'] != [source] or summary['view'] != 'node_neighbors'):
        raise ValueError(f'unsupported singleton reference: {summary_path}')
    best, final = int(summary['best_step']), int(summary['final_step'])
    if not 0 < best <= final:
        raise ValueError(f'invalid historical horizon: {summary_path}')
    return run, summary, best, final


def train_reference(source, args, device):
    original, original_summary, best_step, final_step = reference_spec(source)
    root = Path(args.root) / 'singleton_references' / source
    if root.exists():
        raise FileExistsError(f'reference output already exists: {root}')
    source_index = ac.SOURCES.index(source)
    graph = ac.base.load_graph(source, load_config(args.config), 0, device)
    if graph['receipt'] != original_summary['graph_split_receipt']:
        raise ValueError('historical and replay graph/split identities differ')
    ac.initialize_sampling([graph], 0, device)
    validation = ac.diagnostic.validation_pairs(graph)
    probe_seed = 813719 + 1009 * source_index
    probe = ac.fixed_probe(graph, probe_seed)
    ac.seed_everything(0)
    model = ac.base.BiasMLP('node_neighbors').to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0005, weight_decay=1e-5)
    metadata = dict(run_id=f'singleton_reference_{source}', sources=[source], seed=0,
                    view='node_neighbors', architecture='node_mlp', objective='lp',
                    code_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    historical_summary=identity(original / 'summary.json'),
                    historical_code_revision=original_summary['code_revision'],
                    graph_split_receipt=graph['receipt'],
                    protocol=dict(learning_rate=.0005, weight_decay=1e-5, gradient_clip_norm=1.,
                                  schedule='source-only every optimizer update',
                                  stopping='fixed historical final_step; no new early stopping',
                                  final_step=final_step, historical_best_step=best_step,
                                  validation_interval=2000, fixed_probe_seed=probe_seed,
                                  validation_seed=32452843,
                                  training_negative_seed=49979687, training_order_seed=7919,
                                  selection_data='source validation only; no downstream evaluation'))
    root.mkdir(parents=True)
    atomic_json(root / 'metadata.json', metadata)
    ac.save(root / 'training_probe.pt', ac.cpu_clone(probe))
    runtime = dict(logical_step=0, counts=[ac.empty_counts()])
    checks = []
    started = time.monotonic()
    window_total = 0.
    window_count = 0
    with ac.tracked_run(root, metadata, args) as (wandb, log):
        def measure_save(step):
            reports = ac.measure(model, [graph], [validation], [probe])
            checkpoint_path = root / 'checkpoints' / f'step_{step:06d}.pt'
            checkpoint = ac.snapshot(model, optimizer, [graph], runtime, metadata)
            ac.save(checkpoint_path, checkpoint)
            row = dict(step=step, source=source, checkpoint=str(checkpoint_path),
                       counts=ac.cpu_clone(runtime['counts'][0]),
                       validation=reports['validation'][0], training_probe=reports['training_probe'][0])
            for name, expected_step in (('best.pt', best_step), ('latest.pt', final_step)):
                if step != expected_step:
                    continue
                reference_path = original / name
                historical = torch.load(reference_path, map_location='cpu', weights_only=False)
                if historical['step'] != expected_step:
                    raise ValueError(f'historical checkpoint step disagrees with summary: {reference_path}')
                comparison = dict(checkpoint=identity(reference_path), step=step,
                                  **model_match(checkpoint['model'], historical['model']))
                checks.append(comparison)
                print(json.dumps(dict(event='historical_identity_check', source=source, **comparison)), flush=True)
                atomic_json(root / 'identity_checks.json', checks)
            atomic_json(root / 'validation' / f'step_{step:06d}.json', row)
            with (root / 'history.jsonl').open('a') as handle:
                handle.write(json.dumps(row) + '\n')
            log(step, {'validation/auc': row['validation']['auc'],
                       'validation/bce': row['validation']['bce'],
                       'check/hard_label_probe_bce': row['training_probe']['bce']})
            print(json.dumps(dict(source=source, step=step, validation=row['validation'],
                                  training_probe=row['training_probe'])), flush=True)

        measure_save(0)
        for step in range(1, final_step + 1):
            pairs, labels, npositive = ac.next_batch(graph)
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(ac.base.score(model, graph['x'], pairs), labels)
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite singleton reference loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            runtime['logical_step'] = step
            counts = runtime['counts'][0]
            counts['supervised_updates'] += 1
            for key, amount in (('positive_examples', npositive), ('negative_examples', 5 * npositive),
                                ('supervised_positive_examples', npositive), ('supervised_negative_examples', 5 * npositive)):
                counts[key] += amount
            window_total += float(loss.detach())
            window_count += 1
            if step % args.log_interval == 0:
                log(step, {'train/loss': window_total / window_count,
                           'train/elapsed_seconds': time.monotonic() - started})
                window_total = 0.
                window_count = 0
            if step % 2000 == 0 or step in (best_step, final_step):
                measure_save(step)
        reproduced = len(checks) == 2 and all(check['matches'] for check in checks)
        summary = dict(metadata, status='complete', final_step=final_step,
                       historical_best_step=best_step, counts=runtime['counts'][0],
                       historical_model_identity_verified=reproduced,
                       identity_checks=checks,
                       interpretation='historical selected and terminal models reproduced within tolerance' if reproduced else
                           'fresh protocol-matched reference; historical model identity NOT reproduced',
                       elapsed_seconds=time.monotonic() - started)
        atomic_json(root / 'summary.json', summary)
        wandb.summary.update(dict(final_step=final_step, historical_model_identity_verified=reproduced))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--source', choices=ac.SOURCES)
    parser.add_argument('--device', type=int, choices=range(4), default=3)
    parser.add_argument('--config', default='configs/nonzero_mini_transfer.yaml')
    parser.add_argument('--log-interval', type=int, default=100)
    parser.add_argument('--wandb-mode', default='offline')
    parser.add_argument('--wandb-project', default='nonzero-mini-transfer')
    parser.add_argument('--wandb-group', default='async-convergence-singleton-references')
    args = parser.parse_args()
    if args.log_interval < 1:
        parser.error('positive log interval required')
    torch.set_num_threads(4)
    torch.cuda.set_device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(f'cuda:{args.device}')
    for source in [args.source] if args.source else ac.SOURCES:
        train_reference(source, args, device)
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
