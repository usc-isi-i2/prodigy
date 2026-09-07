"""Input-only task-rule feasibility; never opens model predictions or weights."""
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np


def fit_rule(features):
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or min(x.shape) < 2:
        raise ValueError('At least two rows and two features required')
    norms = np.linalg.norm(x, axis=1)
    if not np.isfinite(x).all() or np.any(norms <= 0):
        raise ValueError('Finite nonzero feature rows required')
    x = x / norms[:, None]
    mean = x.mean(0)
    centered = x - mean
    values, vectors = np.linalg.eigh(centered.T @ centered / len(x))
    if values[-1] <= 0 or values[-1] - values[-2] <= 1e-8 * values[-1]:
        raise ValueError('Degenerate leading eigenspace')
    direction = vectors[:, -1]
    if direction[np.argmax(np.abs(direction))] < 0:
        direction = -direction
    scores = centered @ direction
    return mean, direction, float(np.median(scores)), values[-2:]


def main():
    import torch
    from .replay import batch_hash
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--reference-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists() or torch.cuda.is_available():
        raise ValueError('New output and hidden GPUs required')
    torch.set_num_threads(2)
    banks = {}
    label_table = None
    for stream in ('original', 'fresh'):
        root = args.reference_root / stream / 'twibot20'
        cache = json.loads((root / 'cache.json').read_text())
        if len(cache['batch_sha256']) != 32:
            raise ValueError('Expected complete 32-batch bank')
        seen, features, degree, identities = set(), [], [], []
        for bi, digest in enumerate(cache['batch_sha256']):
            batch = torch.load(root / 'batches' / f'batch_{bi:03d}.pt',
                               map_location='cpu', weights_only=False)
            if batch_hash(batch) != digest:
                raise ValueError('Cached input changed')
            g = batch[0]
            if label_table is None:
                label_table = batch[1][:2].clone()
            # Existing binary episodes may permute the same two token vectors.
            # The diagnostic will use the first table, fixed across both rules.
            for table in batch[1].reshape(4, 2, -1):
                if not (torch.equal(table, label_table) or
                        torch.equal(table, label_table.flip(0))):
                    raise ValueError('Label-token values differ, not just order')
            centers = g.ptr[:-1]
            ids = g.global_node_ids[centers]
            if not torch.equal(ids, g.center_node_idx):
                raise ValueError('Center identity mismatch')
            # Background edges must not include synthetic pooling nodes.
            synthetic = g.ptr[1:] - 1
            if torch.isin(g.edge_index, synthetic).any():
                raise ValueError('Synthetic endpoint in background edges')
            counts = torch.bincount(g.edge_index[1], minlength=len(g.x))[centers]
            for si, node in enumerate(ids.tolist()):
                if node in seen:
                    continue
                seen.add(node)
                features.append(g.x[centers[si]].numpy().copy())
                degree.append(int(counts[si]))
                identities.append(dict(node=node, batch=bi, subgraph=si, batch_sha256=digest))
        banks[stream] = dict(features=np.stack(features), degree=np.array(degree),
                             identities=identities)
    mean, direction, threshold, eigenvalues = fit_rule(banks['original']['features'])
    reports = {}
    manifests = {}
    for stream, bank in banks.items():
        x = bank['features'].astype(np.float64)
        score = (x / np.linalg.norm(x, axis=1)[:, None] - mean) @ direction
        structural = bank['degree'] > 0
        content = score > threshold
        cell = 2 * structural.astype(int) + content.astype(int)
        counts = np.bincount(cell, minlength=4)
        reports[stream] = dict(unique_centers=len(x), cell_counts=counts.tolist(),
                               feasible=bool((counts >= 11).all()),
                               content_threshold_ties=int((score == threshold).sum()))
        manifests[stream] = [dict(**identity, degree=int(deg), content_score=float(s),
                                  joint_cell=int(c)) for identity, deg, s, c in
                             zip(bank['identities'], bank['degree'], score, cell)]
    args.output.mkdir(parents=True)
    result = dict(reports=reports, leading_eigenvalues=eigenvalues.tolist(),
                  new_model_forwards=0, revision=subprocess.check_output(
                      ['git', 'rev-parse', 'HEAD'], text=True).strip())
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    (args.output / 'private_manifest.json').write_text(json.dumps(manifests) + '\n')
    np.savez(args.output / 'private_rule.npz', mean=mean, direction=direction,
             threshold=threshold, label_table=label_table.numpy())
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
