"""Replay supplied tensor inputs, or run synthetic contracts; no repository needed."""
import argparse
import json
from pathlib import Path
import platform
import sys
import time

import numpy
import sklearn
import torch
import torch_geometric
import torch_scatter

from .io import checked_path, labels_for, load_batch, sha, write_json
from .model import make_model
from .digest import model_digest
from .metrics import evaluate_logits
from .predict import expected_conditions, predict_batch


def verify_code():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / 'manifest.json').read_text())
    for name, digest in manifest['sha256'].items():
        checked_path(root, {'path': name, 'sha256': digest})
    return sha(root / 'manifest.json')


def run(args):
    code_hash = verify_code()
    if args.output.exists() or not 1 <= args.threads <= 8:
        raise ValueError('New output directory and 1-8 CPU threads required')
    torch.set_num_threads(args.threads)
    root = args.inputs.resolve()
    manifest = json.loads((root / 'inputs.json').read_text())
    if manifest['format'] != 1:
        raise ValueError('Unsupported input manifest')
    models, datasets = manifest['models'], manifest['datasets']
    if not models or not datasets or len({m['id'] for m in models}) != len(models) or len({d['id'] for d in datasets}) != len(datasets):
        raise ValueError('Empty or duplicate model/input panel')
    if args.model_ids:
        if not set(args.model_ids) <= {m['id'] for m in models}:
            raise ValueError('Unknown requested model')
        models = [m for m in models if m['id'] in args.model_ids]
    if args.dataset_ids:
        if not set(args.dataset_ids) <= {d['id'] for d in datasets}:
            raise ValueError('Unknown requested dataset')
        datasets = [d for d in datasets if d['id'] in args.dataset_ids]
    partial = args.max_batches is not None or len(models) != len(manifest['models']) or len(datasets) != len(manifest['datasets'])
    if args.max_batches is not None and args.max_batches < 1:
        raise ValueError('Positive batch limit required')
    args.output.mkdir(parents=True)
    env = {'python': platform.python_version(), 'torch': torch.__version__, 'torch_geometric': torch_geometric.__version__,
           'torch_scatter': torch_scatter.__version__, 'numpy': numpy.__version__, 'sklearn': sklearn.__version__,
           'platform': platform.platform(), 'threads': args.threads, 'device': 'cpu'}
    write_json(args.output / 'protocol.json', {'code_manifest_sha256': code_hash, 'input_manifest_sha256': sha(root/'inputs.json'),
        'environment': env, 'conditions': sorted(expected_conditions()), 'partial_replay': partial,
        'model_ids': [m['id'] for m in models], 'dataset_ids': [d['id'] for d in datasets], 'new_training': False})
    start = time.monotonic()
    rows, receipts = [], []
    for dataset in datasets:
        records = dataset['batches'][:args.max_batches] if args.max_batches else dataset['batches']
        if not records:
            raise ValueError('No input batches')
        # Load one target at a time, not the source graphs or the full corpus.
        batches = [load_batch(root, r) for r in records]
        parts = [labels_for(b, dataset['use_global']) for b in batches]
        labels = {k: torch.cat([p[k] for p in parts]) for k in ('local_y', 'mapping')}
        labels['use_global'] = dataset['use_global']
        for record in models:
            state = torch.load(checked_path(root, record), map_location='cpu', weights_only=True)
            if model_digest(state) != record['tensor_sha256']:
                raise ValueError('Checkpoint tensor digest mismatch')
            model = make_model(record['config'], state)
            values = {k: [] for k in expected_conditions()}
            total_checks = {'query_pre_exact': 0, 'query_post_exact': 0, 'direct_role_checks': 0}
            for bi, batch in enumerate(batches):
                predictions, checks = predict_batch(model, batch, target=dataset['target'], stream=dataset['stream'],
                                                    batch_index=bi, check_direct=bi == 0)
                for k, v in predictions.items():
                    values[k].append(v)
                for k, v in checks.items():
                    total_checks[k] += v
            values = {k: torch.cat(v) for k, v in values.items()}
            scores = {k: evaluate_logits(v, labels) for k, v in values.items()}
            comparison = {'reference_conditions': 0, 'max_logit_error': None, 'max_metric_error': None}
            ref = dataset.get('references', {}).get(record['id'])
            if ref:
                reference = torch.load(checked_path(root, ref), map_location='cpu', weights_only=True)
                expected = expected_conditions() - {'prototype/raw', 'prototype/encoder'}
                if set(reference['logits']) != expected or set(reference['metrics']) != expected:
                    raise ValueError('Incomplete historical reference panel')
                max_logit = max_metric = 0.
                for k in expected:
                    target_logits = reference['logits'][k][:len(labels['local_y'])]
                    torch.testing.assert_close(values[k], target_logits, rtol=0, atol=1e-5)
                    max_logit = max(max_logit, float((values[k]-target_logits).abs().max()))
                    if not args.max_batches:
                        delta = max(abs(scores[k][m]-reference['metrics'][k][m]) for m in scores[k])
                        if delta > 1e-6:
                            raise ValueError('Historical metric mismatch: '+k)
                        max_metric = max(max_metric, delta)
                comparison = {'reference_conditions': len(expected), 'max_logit_error': max_logit,
                              'max_metric_error': max_metric if not args.max_batches else None}
            for k in sorted(values):
                rows.append({'dataset_id': dataset['id'], 'target': dataset['target'], 'stream': dataset['stream'],
                             'model_id': record['id'], 'source': record['source'], 'seed': record['seed'],
                             'step': record['step'], 'condition': k, **scores[k]})
            dest = args.output / 'predictions' / dataset['id']
            dest.mkdir(parents=True, exist_ok=True)
            torch.save({'logits': values, 'labels': labels}, dest / (record['id']+'.pt'))
            receipts.append({'dataset_id': dataset['id'], 'model_id': record['id'], 'batches': len(batches),
                             'query_occurrences': len(labels['local_y']), **total_checks, **comparison,
                             'input_and_model_unchanged': True, 'operator_restored': True})
            write_json(args.output / 'metrics.json', rows)
            write_json(args.output / 'receipts.json', receipts)
            print(json.dumps({'dataset': dataset['id'], 'model': record['id'], 'cells': len(rows),
                              'elapsed_seconds': round(time.monotonic()-start, 1), **comparison}), flush=True)
    imported = {name: str(Path(mod.__file__).resolve()) for name, mod in sys.modules.items()
                if name.startswith('graph_role') and getattr(mod, '__file__', None)}
    code_root = Path(__file__).resolve().parents[1]
    if not all(Path(p).is_relative_to(code_root) for p in imported.values()):
        raise ValueError('Imported inference code outside extracted package')
    if any(name == 'experiments' or name.startswith('scripts.experiments') for name in sys.modules):
        raise ValueError('Research checkout imported during portable replay')
    if len(rows) != len(models)*len(datasets)*len(expected_conditions()):
        raise ValueError('Incomplete requested grid')
    write_json(args.output / 'DONE.json', {'cells': len(rows), 'model_input_cells': len(receipts),
        'partial_replay': partial, 'reference_conditions': sum(r['reference_conditions'] for r in receipts),
        'all_inputs_weights_unchanged': True, 'research_checkout_imported': False,
        'elapsed_seconds': time.monotonic()-start})


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--max-batches', type=int)
    p.add_argument('--model-ids', nargs='+')
    p.add_argument('--dataset-ids', nargs='+')
    run(p.parse_args())
