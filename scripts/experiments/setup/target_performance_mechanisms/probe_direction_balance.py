"""Disambiguate bounded sampled degree, full degree, direction balance and reciprocity."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import subprocess

import torch

from experiments.trainer import TrainerFS, _concat_global_eval_parts
from .probe_cached_inputs import standardized_probe
from .replay import batch_hash


def directed_statistics(edge_index, num_nodes):
    src, dst = edge_index
    keys = src * num_nodes + dst
    sorted_keys = keys.sort().values
    if len(sorted_keys) != len(torch.unique_consecutive(sorted_keys)) or (src == dst).any():
        raise ValueError('expected a simple directed graph without self-loops')
    outgoing = torch.bincount(src, minlength=num_nodes).float()
    incoming = torch.bincount(dst, minlength=num_nodes).float()
    total = incoming + outgoing
    reverse = dst * num_nodes + src
    pos = torch.searchsorted(sorted_keys, reverse)
    valid = pos < len(sorted_keys)
    reciprocal = torch.zeros(len(keys), dtype=torch.bool)
    reciprocal[valid] = sorted_keys[pos[valid]] == reverse[valid]
    mutual = torch.bincount(src[reciprocal], minlength=num_nodes).float()
    return dict(in_count=incoming.log1p(), out_count=outgoing.log1p(), incident_count=total.log1p(),
                in_fraction=incoming / total.clamp_min(1),
                reciprocal_fraction=mutual / (total - mutual).clamp_min(1),
                has_incoming=(incoming > 0).float()), sorted_keys


def verify_sample(batch, full, full_keys):
    graph = batch[0]
    ids = graph.global_node_ids
    real = ids >= 0
    if not torch.equal(graph.x[real], full.x[ids[real]]):
        raise ValueError('cached node features differ from the original graph')
    src, dst = ids[graph.edge_index]
    if (src < 0).any() or (dst < 0).any():
        raise ValueError('synthetic pooling edges entered background graph')
    keys = src * full.num_nodes + dst
    pos = torch.searchsorted(full_keys, keys)
    if (pos >= len(full_keys)).any() or not torch.equal(full_keys[pos], keys):
        raise ValueError('sampled directed edge absent from original graph')


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--original-root', type=Path, required=True)
    parser.add_argument('--fresh-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.threads <= 4:
        raise ValueError('existing output or unbounded CPU threads')
    args.output.mkdir(parents=True)
    torch.set_num_threads(args.threads)
    metric = TrainerFS.__new__(TrainerFS)
    metric.parameter = {'task_name': 'classification'}
    metric.is_feature_prediction = metric.is_regression = False
    rows, inputs = [], []
    graph_path, full, full_stats, full_keys = None, None, None, None
    with torch.no_grad():
        for stream, parent in (('original', args.original_root), ('fresh', args.fresh_root)):
            root = parent / 'twibot20'
            cache = json.loads((root / 'cache.json').read_text())
            if cache['episodes'] != 128 or len(cache['batch_sha256']) != 32:
                raise ValueError('complete fixed target input stream required')
            if graph_path is None:
                graph_path = cache['graph_path']
                loaded = torch.load(graph_path, map_location='cpu', weights_only=False)
                full = loaded['data'] if isinstance(loaded, dict) else loaded
                full_stats, full_keys = directed_statistics(full.edge_index, full.num_nodes)
            elif graph_path != cache['graph_path']:
                raise ValueError('target graph differs across streams')
            collected = defaultdict(lambda: {'yt': [], 'yp': [], 'global': []})
            exports = []
            for i in range(32):
                batch = torch.load(root / 'batches' / f'batch_{i:03d}.pt', map_location='cpu', weights_only=False)
                if batch_hash(batch) != cache['batch_sha256'][i]:
                    raise ValueError('cached input digest differs')
                verify_sample(batch, full, full_keys)
                centers = batch[0].ptr[:-1]
                ids = batch[0].global_node_ids[centers]
                sampled, _ = directed_statistics(batch[0].edge_index, batch[0].num_nodes)
                features = {f'sample_{k}': v[centers] for k, v in sampled.items()}
                features.update({f'full_{k}': v[ids] for k, v in full_stats.items()})
                query = batch[5].reshape(-1, 2)[:, 0].bool()
                truth = batch[2][query]
                predictions = {key: standardized_probe(value[:, None], batch) for key, value in features.items()}
                for scope in ('sample', 'full'):
                    predictions[f'{scope}_fraction_and_incident'] = standardized_probe(torch.stack(
                        [features[f'{scope}_in_fraction'], features[f'{scope}_incident_count']], dim=1), batch)
                for name, logits in predictions.items():
                    part = collected[name]
                    part['yt'].append(truth)
                    part['yp'].append(logits)
                    part['global'].append(metric._extract_global_classification_eval(batch, truth, logits))
                exports.append(dict(batch=i, batch_sha256=cache['batch_sha256'][i],
                                    query_center_ids=ids[query], graph_labels=full.y[ids[query]],
                                    features={k: v[query] for k, v in features.items()}, logits=predictions))
            for name, part in collected.items():
                truth, logits = torch.cat(part['yt']), torch.cat(part['yp'])
                scores = metric._compute_eval_metrics(truth, logits, global_eval=_concat_global_eval_parts(part['global']))
                rows.append(dict(stream=stream, dataset='twibot20', probe=name, episodes=128, queries=len(truth),
                                 episode_fingerprint=cache['episode_fingerprint'], **scores))
            torch.save(exports, args.output / f'{stream}.pt')
            inputs.append(dict(stream=stream, episode_fingerprint=cache['episode_fingerprint'],
                               batch_sha256=cache['batch_sha256'], features_and_directed_edges_verified=True))
    if len(rows) != 28 or len({r['episode_fingerprint'] for r in rows}) != 2:
        raise ValueError('incomplete two-stream probe grid')
    (args.output / 'metrics.json').write_text(json.dumps(rows, indent=2) + '\n')
    receipt = dict(complete=True, diagnostic_only=True, source_training_changed=False, query_labels_fitted=False,
                   graph_path=graph_path, full_graph_nodes=full.num_nodes, full_graph_edges=full.edge_index.shape[1],
                   streams=inputs, probes=14, revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
    (args.output / 'DONE.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
