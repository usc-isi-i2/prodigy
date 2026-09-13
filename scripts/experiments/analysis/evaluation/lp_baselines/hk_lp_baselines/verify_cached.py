"""Read-only verification of an existing pilot: no sampling or model encoding."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[6]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts/experiments/setup/final_core')]
import numpy as np
import torch
from scripts.experiments.setup.hk_lp_baselines.protocol import edge_keys, pair_keys, evaluate_score
from scripts.experiments.setup.nm_source_stage_audit.run import map_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    started = time.monotonic()
    result = json.loads((args.run/'results.json').read_text())
    assert result['complete'] and result['protocol']['status'] == 'pilot'
    cache = torch.load(result['protocol']['views'], map_location='cpu', weights_only=False)
    x = map_features(Path(cache['feature_path']))
    merged = map_features(Path(cache['source_artifact']))
    n = len(x)
    offset = cache['offset']
    # Map only the HK feature slice; do not load the merged graph or other sources.
    for start in range(0, n, 8192):
        assert np.array_equal(x[start:start+8192], merged[offset+start:offset+min(n,start+8192)])
    train = edge_keys(cache['views']['train'].numpy(), n)
    all_positive = np.concatenate([edge_keys(e.numpy(), n) for e in cache['views'].values()])
    splits = {s: np.load(args.run/f'{s}_pairs.npz') for s in ('train','validation','test')}
    pair_sets = {}
    for split, pairs in splits.items():
        key = pair_keys(pairs['u'], pairs['v'], n)
        assert len(np.unique(key)) == len(key)
        positive = pairs['y'] == 1
        assert np.isin(key[positive], edge_keys(cache['views'][split].numpy(), n)).all()
        assert not np.isin(key[~positive], all_positive).any()
        pair_sets[split] = key
    for a, b in [('train','validation'), ('train','test'), ('validation','test')]:
        assert not np.intersect1d(pair_sets[a], pair_sets[b]).size
    emb = np.load(args.run/'endpoint_embeddings.npz')
    nodes = emb['node_ids']
    lookup = np.full(n, -1, dtype=np.int64)
    lookup[nodes] = np.arange(len(nodes))
    batches, edge_occurrences, cursor = 0, 0, 0
    for path in sorted((args.run/'prodigy_contexts').glob('batch_*.pt')):
        batch = torch.load(path, map_location='cpu', weights_only=False)
        roots = batch.global_node_ids[batch.ptr[:-1]].numpy()
        assert np.array_equal(roots, nodes[cursor:cursor+len(roots)])
        u, v = batch.global_node_ids[batch.edge_index].numpy()
        assert ((u >= 0) & (v >= 0)).all()
        assert np.isin(pair_keys(u, v, n), train).all()
        edge_occurrences += len(u)
        cursor += len(roots)
        batches += 1
    assert cursor == len(nodes)
    saved = {s: np.load(args.run/f'{s}_scores.npz') for s in ('validation','test')}
    max_error = 0.
    for name, metrics in result['metrics'].items():
        for split in saved:
            pair, scores = splits[split], saved[split]
            for field in ('u','v','y'):
                assert np.array_equal(pair[field], scores[field])
            if name in emb:
                z = emb[name]
                replay = (z[lookup[pair['u']]]*z[lookup[pair['v']]]).sum(-1)
                error = float(np.max(np.abs(replay-scores[name])))
                max_error = max(max_error, error)
                assert error < 1e-7
        replay = evaluate_score(splits['validation'], splits['test'], saved['validation'][name], saved['test'][name])
        assert replay == metrics
    print(json.dumps(dict(complete=True, standalone_features_equal_original_training_slice=True,
        verified_feature_rows=n, disjoint_pairs=True, all_negative_pairs_absent_from_all_edge_views=True,
        all_saved_context_edges_in_training_view=True, context_batches=batches,
        context_edge_occurrences=edge_occurrences, unique_eval_nodes=len(nodes),
        max_cached_score_replay_error=max_error, all_aggregate_metrics_reproduced=True,
        seconds=time.monotonic()-started), indent=2))


if __name__ == '__main__':
    main()
