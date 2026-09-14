"""Replay selected checkpoints and freeze two-scorer fusion on 2017; no test IO."""
import argparse
import json
from pathlib import Path

import numpy as np

ALPHAS = (0., .01, .025, .05, .1, .2, .3, .5, 1.)


def cutoff(x):
    assert len(x) >= 50 and np.isfinite(x).all()
    return float(np.partition(x, -50)[-50])


def hits(p, n):
    return float(np.mean(p > cutoff(n)))


def transform(panel, scores, scale, offset):
    assert scale > 0
    return {s: np.stack([panel['b'+s] / scale,
                        np.exp(np.clip(scores[s].astype(np.float64)-offset, -30, 30))])
            for s in ('p', 'n')}


def maximum(pair, alpha):
    return {s: np.maximum(v[0], alpha*v[1]) if alpha else v[0]
            for s, v in pair.items()}


def hyper_weights(partitions):
    # Exact released H -> H H^T -> sum construction, generalized from 3 to 2 bases.
    from sklearn.metrics import pairwise_distances
    h = np.zeros((2, len(partitions)))
    for k, partition in enumerate(partitions):
        d = pairwise_distances(partition, metric='cosine')
        for edge in np.argwhere((d > 0) & (d < .1)):
            h[edge, k] = 1
    return (h @ h.T).sum(0), h


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', default='cuda:0', choices=['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'])
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    protocol = dict(name='fusion_frozen_v1', seeds=[0, 1, 2], alphas=ALPHAS,
        selection='one common alpha maximizing mean 2017 Hits@50; smallest alpha breaks ties',
        normalization='AA 50th negative and joint logit 50th negative frozen on 2017 per seed',
        base='2015-calibrated AA-DC bp/bn in both years; official 2018 AA reference separate',
        hyper='two-base released cosine threshold .1, H H^T sum; same frozen transformed scores; no zero-weight fallback',
        hyper_arms=['2017 positive/negative partitions', '2017 and 2018 positive/negative partitions'],
        assessment_previously_inspected=True, test_scored=False, training=False,
        expected_cells=3, total_inference_scalars_upper_bound=496434,
        count='496385 neural +44 existing auxiliary +2 normalization +1 alpha +2 Hyper weights; seeds not ensembled')
    if args.dry_run:
        print(json.dumps(protocol, indent=2))
        return
    import run as joint
    import torch
    import wandb
    torch.set_num_threads(8)
    args.out.mkdir(parents=True, exist_ok=False)
    protocol['revision'] = joint.revision()
    joint.save(args.out/'protocol.json', protocol)
    tracking = wandb.init(project='ogbl-collab-compact-joint', group='fusion_frozen_v1',
                          name=args.out.name, mode='offline', dir=str(args.out), config=protocol)
    root = args.runtime
    prepared = joint.read(root/'prepared.json')
    assert prepared['complete']
    for name in ('year2017.npz', 'standardization.npz', 'node_features.npy'):
        assert joint.shared.sha256_file(root/name) == prepared['files'][name]
    frozen = joint.read(root/'assessment/selection_frozen.json')
    va = np.load(root/'year2017.npz')
    ms = np.load(root/'standardization.npz')
    x = torch.as_tensor(np.load(root/'node_features.npy'), device=args.device)
    adj = joint.adjacency(va['graph_edges'], len(x), args.device)
    tensors = joint.panel_tensors(va, ms['mean'], ms['std'], args.device)
    replay, normalized, receipts = {}, [], []
    for seed in range(3):
        selection = next(s for s in frozen['selections'] if s['arm']=='joint' and s['seed']==seed)
        ckpt = root/f'joint_seed{seed}/best.pt'
        assert joint.shared.sha256_file(ckpt) == selection['checkpoint_sha256']
        state = torch.load(ckpt, map_location=args.device, weights_only=False)
        assert state['step'] == selection['selected_step']
        assert state['revision'] == selection['revision'] == prepared['revision']
        model = joint.Model('joint').to(args.device)
        model.load_state_dict(state['model'])
        score = joint.evaluate(model, x, adj, tensors)
        assert hits(**score) == selection['selection_hits_at_50'], 'Replay must match exactly'
        replay.update({f'joint_seed{seed}_{s}': v for s, v in score.items()})
        scale, offset = cutoff(va['bn']), cutoff(score['n'])
        normalized.append(transform(va, score, scale, offset))
        receipts.append(dict(seed=seed, selected_step=state['step'],
                             checkpoint_sha256=selection['checkpoint_sha256'],
                             replay_hits=hits(**score), scale=scale, offset=offset))
    np.savez_compressed(args.out/'scores2017.npz', **replay)
    grid = [dict(alpha=a, hits=[hits(**maximum(v, a)) for v in normalized]) for a in ALPHAS]
    winner = max(grid, key=lambda row: np.mean(row['hits']))
    alpha = winner['alpha']
    selection = dict(alpha=alpha, grid=grid, checkpoints=receipts,
                     source_revision=prepared['revision'],
                     prepared_sha256=joint.shared.sha256_file(root/'prepared.json'),
                     scores2017_sha256=joint.shared.sha256_file(args.out/'scores2017.npz'))
    # Freeze everything before opening the previously inspected assessment arrays.
    joint.save(args.out/'selection.json', selection)
    old = joint.read(root/'assessment/results.json')
    assert joint.shared.sha256_file(root/'assessment/year2018.npz') == old['panel_sha256']
    assert joint.shared.sha256_file(root/'assessment/scores.npz') == old['score_sha256']
    assessment = np.load(root/'assessment/year2018.npz')
    saved = np.load(root/'assessment/scores.npz')
    rows = []
    for seed, (v, r) in enumerate(zip(normalized, receipts)):
        score = {s: saved[f'joint_seed{seed}_{s}'] for s in ('p', 'n')}
        a = transform(assessment, score, r['scale'], r['offset'])
        fused = maximum(a, alpha)
        basehit = assessment['bp'] > cutoff(assessment['bn'])
        fusedhit = fused['p'] > cutoff(fused['n'])
        hyper = []
        for parts in ([v['p'], v['n']], [v['p'], v['n'], a['p'], a['n']]):
            weights, h = hyper_weights(parts)
            hyper.append(dict(weights=weights.tolist(), H=h.tolist(),
                              hits=hits(weights@a['p'], weights@a['n'])))
        rows.append(dict(seed=seed, frozen_max_hits=hits(**fused),
                         recovered=int((fusedhit & ~basehit).sum()),
                         lost=int((basehit & ~fusedhit).sum()),
                         hyper_frozen=hyper[0], hyper_adapted=hyper[1]))
    result = dict(complete=True, test_scored=False, alpha=alpha, rows=rows,
                  baseline2017=hits(va['bp'], va['bn']),
                  baseline2018=hits(assessment['bp'], assessment['bn']),
                  official_reference2018=hits(assessment['official_bp'], assessment['official_bn']),
                  mean_frozen_max_hits=float(np.mean([r['frozen_max_hits'] for r in rows])),
                  panel_sha256=old['panel_sha256'], score_sha256=old['score_sha256'],
                  selection_sha256=joint.shared.sha256_file(args.out/'selection.json'),
                  wandb_directory=tracking.dir)
    joint.save(args.out/'results.json', result)
    tracking.log({'mean_frozen_max_hits': result['mean_frozen_max_hits']})
    tracking.finish()
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
