"""Structural screening only: short-walk selections, complete induced topology.

No training, no full graph serialization, no mutation of parent graph artifacts.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.stats import wasserstein_distance
import torch


def histogram(values):
    degrees, counts = np.unique(values, return_counts=True)
    return {'values': degrees.tolist(), 'counts': counts.tolist()}


def compare_histograms(a, b):
    ax, bx = np.asarray(a['values']), np.asarray(b['values'])
    aw, bw = np.asarray(a['counts'], dtype=float), np.asarray(b['counts'], dtype=float)
    support = np.union1d(ax, bx)
    acdf = np.r_[0., np.cumsum(aw) / aw.sum()]
    bcdf = np.r_[0., np.cumsum(bw) / bw.sum()]
    ks = np.max(np.abs(acdf[np.searchsorted(ax, support, side='right')] - bcdf[np.searchsorted(bx, support, side='right')]))
    return {'degree_ks': float(ks), 'log_degree_wasserstein': float(wasserstein_distance(np.log1p(ax), np.log1p(bx), aw, bw))}


def topology_metrics(adjacency, stored_edges):
    n = adjacency.shape[0]
    degree = np.diff(adjacency.indptr)
    n_components, labels = connected_components(adjacency, directed=False)
    sizes = np.bincount(labels, minlength=n_components)
    degree_hist = histogram(degree)
    return {'nodes': n, 'stored_edges': int(stored_edges), 'edges_per_node': float(stored_edges / n),
            'isolate_fraction': float(np.mean(degree == 0)),
            'mean_unique_undirected_degree': float(degree.mean()),
            'degree_quantiles': {str(q): float(np.quantile(degree, q)) for q in [0.5, 0.9, 0.95, 0.99, 0.999]},
            'degree_histogram': degree_hist, 'components': int(n_components),
            'largest_component_fraction': float(sizes.max() / n),
            'component_node_fractions': {name: float(sizes[(sizes >= lo) & (sizes <= hi)].sum() / n)
                                       for name, lo, hi in [('1', 1, 1), ('2-10', 2, 10), ('11-100', 11, 100), ('101-1000', 101, 1000), ('1001+', 1001, n)]},
            'top_component_fractions': (np.sort(sizes)[-5:][::-1] / n).tolist()}


def short_walk_nodes(adjacency, cap, length, seed, batch_size=2048):
    """Uniform restarts; non-directed neighbor steps; retain walk-prefix nodes."""
    n = adjacency.shape[0]
    if cap > n or cap < 1 or length < 1:
        raise ValueError('Invalid cap or walk length')
    rng = np.random.default_rng(seed)
    degree = np.diff(adjacency.indptr)
    selected = np.zeros(n, dtype=bool)
    total = 0
    restarts = 0
    while total < cap:
        roots = rng.integers(n, size=batch_size)
        walks = np.empty((batch_size, length + 1), dtype=adjacency.indices.dtype)
        walks[:, 0] = roots
        current = roots
        for step in range(length):
            d = degree[current]
            active = d > 0
            nxt = current.copy()
            offsets = (rng.random(np.count_nonzero(active)) * d[active]).astype(np.int64)
            nxt[active] = adjacency.indices[adjacency.indptr[current[active]] + offsets]
            walks[:, step + 1] = nxt
            current = nxt
        # Walk-major order means stopping at the cap can break at most one walk.
        stream = walks.reshape(-1)
        novel = stream[~selected[stream]]
        _, first = np.unique(novel, return_index=True)
        novel = novel[np.sort(first)]
        novel = novel[:cap - total]
        selected[novel] = True
        total += len(novel)
        restarts += batch_size
    return np.flatnonzero(selected), {'generated_restarts': restarts, 'batch_size': batch_size, 'walk_length': length}


def count_induced_edges(edge_index, ids, n):
    selected = np.zeros(n, dtype=bool)
    selected[ids] = True
    count = 0
    for start in range(0, edge_index.shape[1], 1000000):
        block = edge_index[:, start:start + 1000000]
        count += np.count_nonzero(selected[block[0]] & selected[block[1]])
    return int(count)


def rank_metrics(candidate, parent):
    # A transparent diagnostic distance, not a claim of statistical unbiasedness.
    ratio_error = abs(np.log(candidate['edges_per_node'] / parent['edges_per_node']))
    isolate_error = abs(candidate['isolate_fraction'] - parent['isolate_fraction'])
    component_error = abs(candidate['largest_component_fraction'] - parent['largest_component_fraction'])
    return {'density_log_error': float(ratio_error), 'isolate_absolute_error': float(isolate_error),
            'largest_component_absolute_error': float(component_error),
            'screening_score': float(ratio_error + candidate['degree_ks'] + isolate_error + component_error)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['ukr_rus_twitter', 'covid19_twitter'], required=True)
    parser.add_argument('--catalog', type=Path, default=Path('docs/graph_catalog.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--lengths', nargs='+', type=int, default=[1, 2, 4, 8])
    parser.add_argument('--cap', type=int, default=500000)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    start = time.monotonic()
    catalog = json.loads(args.catalog.read_text())
    entry = next(e for e in catalog['graphs'] if e['dataset_key'] == args.dataset + '_nonzero_features_v1')
    path = Path(catalog['data_root']) / entry['relative_path']
    stat = path.stat()
    identity = {'path': str(path), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
    print('Loading parent topology', flush=True)
    graph = torch.load(path, mmap=True, weights_only=False)
    n = len(graph['x'])
    tensor_edges = graph['edge_index']
    original_ids = graph['original_node_ids'].numpy()
    edges = tensor_edges.numpy()
    # Keep mapped topology and row mapping; release large Python metadata.
    del graph
    gc.collect()
    print(f'Building undirected CSR for {n} nodes and {edges.shape[1]} stored edges', flush=True)
    nonself = edges[0] != edges[1]
    adjacency = sparse.csr_matrix((np.ones(np.count_nonzero(nonself), dtype=bool),
                                   (edges[0, nonself], edges[1, nonself])), shape=(n, n))
    adjacency = adjacency.maximum(adjacency.T).tocsr()
    adjacency.sort_indices()
    del nonself
    print('Measuring full parent degree and components', flush=True)
    parent = topology_metrics(adjacency, edges.shape[1])
    parent_degrees = np.diff(adjacency.indptr)
    records = []
    report = {'dataset': args.dataset, 'parent_identity': identity,
              'code_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'protocol': {'cap': args.cap, 'screening_seed': 0, 'confirmation_seed': 1,
                           'lengths': args.lengths, 'walk_graph': 'unique undirected non-self adjacency',
                           'features': 'not loaded into candidate artifacts',
                           'ranking': 'abs(log density ratio) + degree KS + absolute isolate-fraction error + absolute largest-component-fraction error',
                           'degree_definition': 'unique undirected non-self neighbors; edge count retains original stored directed rows',
                           'component_definition': 'weak components including isolates'},
              'parent': parent, 'candidates': records}

    def evaluate(name, ids, details):
        tick = time.monotonic()
        assert len(ids) == args.cap and np.all(ids[1:] > ids[:-1])
        assert ids[0] >= 0 and ids[-1] < n
        sub = adjacency[ids][:, ids].tocsr()
        stored = count_induced_edges(edges, ids, n)
        metrics = topology_metrics(sub, stored)
        metrics.update(compare_histograms(metrics['degree_histogram'], parent['degree_histogram']))
        selection_bias = compare_histograms(histogram(parent_degrees[ids]), parent['degree_histogram'])
        metrics['selected_parent_degree_ks'] = selection_bias['degree_ks']
        metrics.update(rank_metrics(metrics, parent))
        torch.save({'parent_node_ids': torch.from_numpy(ids.copy()),
                    'original_node_ids': torch.from_numpy(original_ids[ids].copy())}, args.output / (name + '.pt'))
        record = {'name': name, 'sampling': details, 'metrics': metrics,
                  'selection_sha256': hashlib.sha256(ids.astype('<i8', copy=False).tobytes()).hexdigest(),
                  'measurement_seconds': time.monotonic() - tick}
        records.append(record)
        (args.output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({k: v for k, v in record.items() if k != 'metrics'} | {'density': metrics['edges_per_node'], 'isolates': metrics['isolate_fraction'], 'degree_ks': metrics['degree_ks'], 'largest_component': metrics['largest_component_fraction'], 'score': metrics['screening_score']}), flush=True)
        return record

    print('Measuring existing baselines', flush=True)
    for name, suffix in [('uniform', '_mini_500k'), ('edge', '_mini_500k_edge_sampled_s0')]:
        p = path.parent.parent / (args.dataset + suffix) / 'graph.pt'
        baseline = torch.load(p, mmap=True, weights_only=False)
        if 'parent_node_ids' in baseline:
            ids = baseline['parent_node_ids'].numpy().copy()
        else:
            raw = baseline['original_node_ids'].numpy()
            ids = np.searchsorted(original_ids, raw)
            assert np.array_equal(original_ids[ids], raw)
        del baseline
        gc.collect()
        evaluate(name, ids, {'method': name, 'existing_artifact': str(p), 'seed': 0})
    for length in args.lengths:
        ids, detail = short_walk_nodes(adjacency, args.cap, length, 0)
        evaluate(f'walk{length}_s0', ids, detail | {'seed': 0, 'method': 'uniform-restart short walk'})
    screened = sorted([r for r in records if r['name'].startswith('walk')], key=lambda r: r['metrics']['screening_score'])
    for choice in screened[:2]:
        length = choice['sampling']['walk_length']
        ids, detail = short_walk_nodes(adjacency, args.cap, length, 1)
        evaluate(f'walk{length}_s1', ids, detail | {'seed': 1, 'method': 'uniform-restart short walk'})
    assert path.stat().st_size == stat.st_size and path.stat().st_mtime_ns == stat.st_mtime_ns
    report['completed'] = True
    report['elapsed_seconds'] = time.monotonic() - start
    report['parent_unchanged'] = True
    (args.output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'COMPLETE {report["elapsed_seconds"]:.1f}s', flush=True)


if __name__ == '__main__':
    main()
