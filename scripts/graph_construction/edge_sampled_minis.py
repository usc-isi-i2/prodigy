"""Select endpoints of uniformly sampled edge rows, then preserve all induced edges."""
import argparse
import json
from pathlib import Path
import subprocess
import time

import torch
from nonzero_feature_views import (induced, verify_induced, assert_equal, identity,
                                   statistics, EDGE_CHUNK)
from verify_nonzero_feature_views import verify_attributes


def sample_endpoints(edges, num_nodes, cap, seed=0, batch_size=65536, max_draws=50000000):
    if not 2 <= cap <= num_nodes or edges.shape[1] == 0:
        raise ValueError('Need at least two available nodes and nonempty edges')
    selected = torch.zeros(num_nodes, dtype=torch.bool).numpy()
    generator = torch.Generator().manual_seed(seed)
    count = 0
    draws = 0
    witnesses = []
    while count < cap and draws < max_draws:
        rows = torch.randint(edges.shape[1], (min(batch_size, max_draws - draws),), generator=generator)
        pairs = edges[:, rows].t().tolist()
        for row, (u, v) in zip(rows.tolist(), pairs):
            draws += 1
            if u == v:
                continue
            extra = int(not selected[u]) + int(not selected[v])
            # Never trim off an endpoint: skip edges that would exceed the cap.
            if extra == 0 or count + extra > cap:
                continue
            selected[u] = selected[v] = True
            count += extra
            witnesses.append(row)
            if count == cap:
                break
    if count != cap:
        raise ValueError(f'Only reached {count}/{cap} connected endpoints after {draws} draws')
    ids = torch.from_numpy(selected).nonzero().flatten()
    witness = torch.tensor(witnesses, dtype=torch.long)
    assert torch.equal(edges[:, witness].unique(sorted=True), ids)
    assert (edges[0, witness] != edges[1, witness]).all()
    return ids, witness, draws


def nonself_isolates(edges, n):
    touched = torch.zeros(n, dtype=torch.bool)
    for start in range(0, edges.shape[1], EDGE_CHUNK):
        block = edges[:, start:start + EDGE_CHUNK]
        touched[block[:, block[0] != block[1]]] = True
    return int((~touched).sum())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True, choices=['ukr_rus_twitter', 'covid19_twitter'])
    p.add_argument('--catalog', type=Path, default=Path('docs/graph_catalog.json'))
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--threads', type=int, default=4)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    c = json.loads(args.catalog.read_text())
    entry = next(e for e in c['graphs'] if e['dataset_key'] == args.dataset + '_nonzero_features_v1')
    path = Path(c['data_root']) / entry['relative_path']
    destination = path.parent.parent / (args.dataset + '_mini_500k_edge_sampled_s0')
    if args.seed != 0:
        raise ValueError('This registered variant uses seed 0')
    destination.mkdir(exist_ok=False)
    before = identity(path)
    started = time.monotonic()
    print('Loading verified nonzero parent with mmap', flush=True)
    source = torch.load(path, mmap=True, map_location='cpu', weights_only=False)
    print('Sampling endpoints', flush=True)
    ids, witness, draws = sample_endpoints(source['edge_index'], len(source['x']), 500000, args.seed)
    # Strip only prior-view provenance; aligned attributes stay in the payload.
    payload = {k: v for k, v in source.items() if k not in
               {'original_node_ids', 'num_nodes', 'source_graph_metadata', 'view_metadata'}}
    print(f'Inducing {len(ids)} nodes from {len(witness)} accepted edges ({draws} draws)', flush=True)
    result = induced(payload, ids)
    print('Checking every induced edge view and aligned attribute', flush=True)
    verify_induced(payload, result, ids)
    verify_attributes(payload, result, ids)
    result['parent_node_ids'] = ids
    result['original_node_ids'] = source['original_node_ids'][ids]
    result['source_graph_metadata'] = source.get('source_graph_metadata', {})
    stats = statistics(result)
    stats['nonself_isolated_nodes'] = nonself_isolates(result['edge_index'], len(ids))
    assert stats['nonself_isolated_nodes'] == 0
    if 'static_background' in result.get('edge_index_views', {}):
        stats['static_background_nonself_isolated_nodes'] = nonself_isolates(result['edge_index_views']['static_background'], len(ids))
    meta = {'schema_version': 1, 'view': 'nonzero_features_v1',
            'source_dataset_key': args.dataset, 'parent_dataset_key': entry['dataset_key'],
            'parent_identity': before, 'code_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            'sampling': {'method': 'uniform edge-row draws with replacement; collect endpoints; complete induced graph',
                         'seed': args.seed, 'cap': 500000, 'skip_self_loops': True,
                         'skip_edges_exceeding_node_cap': True, 'draws': draws,
                         'accepted_witness_edges': len(witness), 'batch_size': 65536},
            'statistics': stats, 'parent_node_ids': 'row in full nonzero parent',
            'original_node_ids': 'row in original unfiltered source graph',
            'torch_version': torch.__version__,
            'verification': {'all_induced_edge_views_exact': True, 'all_attributes_exact': True,
                             'nonzero_features': True, 'no_nonself_isolates': True,
                             'witness_endpoints_equal_selection': True}}
    result['view_metadata'] = meta
    print('Writing and verifying mini only', flush=True)
    artifact = destination / 'graph.pt'
    torch.save(result, artifact)
    torch.save({'parent_node_ids': ids, 'parent_edge_ids': witness}, destination / 'selection.pt')
    restored = torch.load(artifact, mmap=True, weights_only=False)
    assert_equal(result, restored)
    assert identity(path) == before
    meta['artifact_size_bytes'] = artifact.stat().st_size
    meta['elapsed_seconds'] = time.monotonic() - started
    meta['verification']['serialization_exact_all_fields'] = True
    (destination / 'metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    print('VERIFIED ' + json.dumps(meta), flush=True)


if __name__ == '__main__':
    main()
