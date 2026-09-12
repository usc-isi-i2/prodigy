"""Build lossless induced nonzero-feature views; originals are read-only.

Run with the isolated prodigy-loadbench Python (torch.load mmap support).
Unknown non-node tensors fail closed instead of being copied with stale indices.
"""
import argparse
import copy
import gc
import json
from pathlib import Path
import subprocess
import time

import torch

SOURCES = ('ukr_rus_twitter', 'covid19_twitter', 'midterm', 'covid_political',
           'election2020', 'ukr_rus_suspended', 'twibot20', 'cp_hk_twitter',
           'facebook_page_reference')
CHUNK = 65536
EDGE_CHUNK = 1000000
HISTORICAL = {'static_split_stats', 'benchmark_target_stats', 'temporal_split_stats', 'metadata'}


def nonzero_ids(x):
    parts = []
    for start in range(0, len(x), CHUNK):
        block = x[start:start + CHUNK]
        if not torch.isfinite(block).all():
            raise ValueError('Nonfinite features: resolve explicitly before building views')
        parts.append(torch.where((block != 0).any(dim=1))[0] + start)
    return torch.cat(parts)


def select_rows(value, ids):
    result = torch.empty((len(ids), *value.shape[1:]), dtype=value.dtype)
    for start in range(0, len(ids), CHUNK):
        result[start:start + CHUNK] = value[ids[start:start + CHUNK]]
    return result


def induced(graph, ids):
    n = len(graph['x'])
    assert ids.dtype == torch.long and ids.ndim == 1
    assert len(ids) == 0 or (int(ids[0]) >= 0 and int(ids[-1]) < n)
    assert len(ids) < 2 or bool((ids[1:] > ids[:-1]).all())
    remap = torch.full((n,), -1, dtype=torch.long)
    remap[ids] = torch.arange(len(ids))
    memo = {}
    selections = {}

    def edge(value):
        key = id(value)
        if key not in memo:
            keep = []
            for start in range(0, value.shape[1], EDGE_CHUNK):
                block = value[:, start:start + EDGE_CHUNK]
                if block.numel() and (int(block.min()) < 0 or int(block.max()) >= n):
                    raise ValueError('Source contains out-of-range edge endpoint')
                keep.append(torch.where((remap[block] >= 0).all(dim=0))[0] + start)
            selected = torch.cat(keep) if keep else torch.empty(0, dtype=torch.long)
            result = torch.empty((2, len(selected)), dtype=value.dtype)
            for start in range(0, len(selected), EDGE_CHUNK):
                result[:, start:start + EDGE_CHUNK] = remap[value[:, selected[start:start + EDGE_CHUNK]]]
            memo[key] = result
            selections[key] = selected
        return memo[key]

    def walk(obj, key='', parent=None):
        if key in HISTORICAL:
            return copy.deepcopy(obj)
        if key == 'u2i':
            users = parent['user_ids']
            if len(users) != n or len(obj) != n:
                raise ValueError('Inconsistent user index')
            for old in ids.tolist():
                if obj[users[old]] != old:
                    raise ValueError('Inconsistent user index mapping')
            return {users[old]: new for new, old in enumerate(ids.tolist())}
        if torch.is_tensor(obj):
            if id(obj) in memo:
                return memo[id(obj)]
            if 'edge_index' in key or (parent is not None and key in parent.get('_edge_names', ())):
                return edge(obj)
            if key == 'edge_attr':
                src = parent['edge_index']
                edge(src)
                assert obj.shape[0] == src.shape[1]
                result = select_rows(obj, selections[id(src)])
            elif obj.ndim and obj.shape[0] == n:
                result = select_rows(obj, ids)
            else:
                raise ValueError(f'Unknown tensor alignment: {key} {obj.shape}')
            memo[id(obj)] = result
            return result
        if isinstance(obj, dict):
            result = {}
            for k, v in obj.items():
                if k in ('edge_index_views', 'target_edge_index_views'):
                    result[k] = {name: edge(t) for name, t in v.items()}
                elif k == 'edge_attr_views':
                    result[k] = {}
                    for name, attr in v.items():
                        src = obj['edge_index_views'][name]
                        edge(src)
                        assert attr.shape[0] == src.shape[1]
                        if id(attr) not in memo:
                            memo[id(attr)] = select_rows(attr, selections[id(src)])
                        result[k][name] = memo[id(attr)]
                else:
                    result[k] = walk(v, k, obj)
            return result
        if hasattr(obj, 'to_dict'):
            return type(obj)(**walk(obj.to_dict()))
        if isinstance(obj, (list, tuple)) and key in ('user_ids', 'handles'):
            assert len(obj) == n
            return type(obj)(obj[i] for i in ids.tolist())
        if key == 'num_nodes':
            return len(ids)
        return copy.deepcopy(obj)

    result = walk(graph)
    historical = {k: result.pop(k) for k in HISTORICAL if k in result}
    if historical:
        result['source_graph_metadata'] = historical
    result['original_node_ids'] = ids.clone()
    result['num_nodes'] = len(ids)
    return result


def assert_equal(a, b):
    """Bounded-memory exact payload comparison, including NaNs in missing targets."""
    if torch.is_tensor(a):
        assert a.shape == b.shape and a.dtype == b.dtype
        aa, bb = a.reshape(-1), b.reshape(-1)
        for start in range(0, aa.numel(), EDGE_CHUNK):
            x, y = aa[start:start + EDGE_CHUNK], bb[start:start + EDGE_CHUNK]
            equal = x == y
            if x.is_floating_point():
                equal |= torch.isnan(x) & torch.isnan(y)
            assert equal.all(), 'Payload differs after serialization'
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_equal(a[key], b[key])
    elif hasattr(a, 'to_dict'):
        assert type(a) is type(b)
        assert_equal(a.to_dict(), b.to_dict())
    else:
        assert a == b


def verify_induced(source, result, ids):
    """Independent source-order scan proves no retained edge was lost or invented."""
    keep = torch.zeros(len(source['x']), dtype=torch.bool)
    keep[ids] = True
    for start in range(0, len(ids), CHUNK):
        assert_equal(source['x'][ids[start:start + CHUNK]], result['x'][start:start + CHUNK])
    assert len(nonzero_ids(result['x'])) == len(ids)
    families = [(source['edge_index'], result['edge_index'])]
    for key in ('edge_index_views', 'target_edge_index_views'):
        for name, value in source.get(key, {}).items():
            families.append((value, result[key][name]))
    for old, new in families:
        offset = 0
        for start in range(0, old.shape[1], EDGE_CHUNK):
            block = old[:, start:start + EDGE_CHUNK]
            selected = block[:, keep[block[0]] & keep[block[1]]]
            assert_equal(selected, ids[new[:, offset:offset + selected.shape[1]]])
            offset += selected.shape[1]
        assert offset == new.shape[1]


def statistics(g):
    n = len(g['x'])
    touched = torch.zeros(n, dtype=torch.bool)
    for start in range(0, g['edge_index'].shape[1], EDGE_CHUNK):
        touched[g['edge_index'][:, start:start + EDGE_CHUNK]] = True
    labels = g.get('y')
    return {'nodes': n, 'edges': g['edge_index'].shape[1],
            'isolated_nodes': int((~touched).sum()),
            'labeled_nodes': int((labels >= 0).sum()) if labels is not None else 0,
            'class_counts': {str(int(k)): int(v) for k, v in zip(*torch.unique(labels[labels >= 0], return_counts=True))} if labels is not None else {},
            'edge_views': {k: v.shape[1] for k, v in g.get('edge_index_views', {}).items()},
            'target_edge_views': {k: v.shape[1] for k, v in g.get('target_edge_index_views', {}).items()}}


def identity(path):
    s = path.stat()
    return {'path': str(path.resolve()), 'bytes': s.st_size, 'mtime_ns': s.st_mtime_ns}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--catalog', type=Path, default=Path('docs/graph_catalog.json'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--datasets', nargs='+', default=list(SOURCES), choices=SOURCES)
    p.add_argument('--mini-nodes', type=int, choices=[500000], default=500000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--threads', type=int, default=8)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    catalog = json.loads(args.catalog.read_text())
    entries = {e['dataset_key']: e for e in catalog['graphs']}
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    args.output.mkdir(parents=True, exist_ok=True)
    for key in args.datasets:
        entry = entries[key]
        path = Path(catalog['data_root']) / entry['relative_path']
        before = identity(path)
        start = time.monotonic()
        print(f'{key}: loading', flush=True)
        graph = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
        ids = nonzero_ids(graph['x'])
        print(f'{key}: {len(ids)}/{len(graph["x"])} nonzero nodes', flush=True)
        variants = [(key, ids, None)]
        if key in ('ukr_rus_twitter', 'covid19_twitter'):
            if len(ids) < args.mini_nodes:
                raise ValueError(f'{key}: too few nonzero nodes for requested mini')
            selected = torch.randperm(len(ids), generator=torch.Generator().manual_seed(args.seed))[:args.mini_nodes]
            variants.append((key + '_mini_500k', ids[selected].sort().values, args.seed))
        for name, selected, seed in variants:
            dest = args.output / name
            dest.mkdir(exist_ok=False)
            print(f'{name}: filtering {len(selected)} nodes', flush=True)
            view = induced(graph, selected)
            verify_induced(graph, view, selected)
            stats = statistics(view)
            meta = {'schema_version': 1, 'view': 'nonzero_features_v1',
                    'source_dataset_key': key, 'source_canonical_name': entry['canonical_name'],
                    'source_identity': before, 'code_revision': revision,
                    'filter': 'any(x != 0), exact; finite features required; isolates retained',
                    'sampling': None if seed is None else {'method': 'uniform_without_replacement', 'seed': seed, 'nodes': len(selected)},
                    'original_node_ids': 'view row -> original source graph row, sorted',
                    'source_nodes': len(graph['x']), 'nonzero_source_nodes': len(ids),
                    'statistics': stats, 'torch_version': torch.__version__,
                    'verification': {'source_induced_edges_all_views': True, 'features_exact': True}}
            view['view_metadata'] = meta
            artifact = dest / 'graph.pt'
            torch.save(view, artifact)
            restored = torch.load(artifact, mmap=True, weights_only=False)
            assert_equal(view, restored)
            assert identity(path) == before, 'Source changed during construction'
            meta['artifact_size_bytes'] = artifact.stat().st_size
            meta['verification']['serialization_exact_all_fields'] = True
            meta['elapsed_seconds_from_source_start'] = time.monotonic() - start
            (dest / 'metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
            print(f'{name}: VERIFIED {json.dumps(stats)}', flush=True)
            del view, restored
            gc.collect()
        del graph
        gc.collect()


if __name__ == '__main__':
    main()
