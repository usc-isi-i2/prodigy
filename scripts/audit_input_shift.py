"""Read-only input/edge distribution audit of exact seed-zero MLP caches."""
import argparse
import itertools
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.stats import ks_2samp
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


def identity(path):
    p = Path(path).resolve()
    s = p.stat()
    return dict(path=str(p), bytes=s.st_size, mtime_ns=s.st_mtime_ns)


def cosine(a, b):
    return (a*b).sum(1) / np.maximum(np.linalg.norm(a, axis=1)*np.linalg.norm(b, axis=1), 1e-12)


def nonedges(n, keys, count, rng):
    p = rng.integers(n, size=(2, count))
    while True:
        lo, hi = p.min(0), p.max(0)
        bad = (lo == hi) | np.isin(lo*n+hi, keys, assume_unique=False)
        if not bad.any():
            return p
        p[:, bad] = rng.integers(n, size=(2, bad.sum()))


def directions(dim, count, seed):
    p = np.random.default_rng(seed).normal(size=(dim, count)).astype('float32')
    return p / np.linalg.norm(p, axis=0)


def pair_projection(x, pairs, projection):
    # Symmetric endpoint representation preserves an unordered pair in each coordinate,
    # but discards cross-coordinate endpoint alignment. Diagnostic, not model features.
    out = []
    d = x.shape[1]
    for start in range(0, pairs.shape[1], 1024):
        u, v = x[pairs[0, start:start+1024]], x[pairs[1, start:start+1024]]
        out.append(((u+v)/2) @ projection[:d] + np.abs(u-v) @ projection[d:])
    return np.concatenate(out)


def domain_auc(a, b, seed):
    # Fixed settings, no test-based tuning. Disjoint sampled center nodes across splits.
    ntrain, ntest = min(3000, len(a)//2), min(2000, len(a)//2)
    x = np.concatenate([a[:ntrain], b[:ntrain]])
    y = np.repeat([0, 1], ntrain)
    test = np.concatenate([a[-ntest:], b[-ntest:]])
    scaler = StandardScaler().fit(x)
    model = SGDClassifier(loss='log_loss', alpha=.001, max_iter=300,
                          tol=1e-4, average=True, random_state=seed)
    model.fit(scaler.transform(x), y)
    return float(roc_auc_score(np.repeat([0, 1], ntest), model.decision_function(scaler.transform(test))))


def swd_table(projected):
    names = list(projected)
    # Common scale for all graph pairs within a view, estimated on equal graph samples.
    scale = np.concatenate([projected[s] for s in names]).std(0).clip(1e-8)
    n = min(len(v) for v in projected.values())//2
    first = {s: np.sort(projected[s][:n], axis=0) for s in names}
    second = {s: np.sort(projected[s][n:2*n], axis=0) for s in names}
    baseline = {s: float(np.mean(np.abs(first[s]-second[s])/scale)) for s in names}
    rows = []
    for a, b in itertools.combinations(names, 2):
        value = float(np.mean(np.abs(first[a]-first[b])/scale))
        noise = (baseline[a]+baseline[b])/2
        rows.append(dict(first=a, second=b, normalized_sliced_w1=value,
                         within_graph_reference=noise, ratio_to_reference=value/noise))
    return dict(pairs=rows, within_graph=baseline)


def run(args):
    out = Path(args.output)
    if (out/'COMPLETE.json').exists():
        print('Already complete', flush=True)
        return
    if out.exists():
        raise FileExistsError('Refusing partial output: '+str(out))
    out.mkdir(parents=True)
    started = time.monotonic()
    torch.set_num_threads(args.threads)
    config = yaml.safe_load(Path(args.config).read_text())
    names = [s for s in config['graphs'] if s != 'election2020']
    rng = np.random.default_rng(args.seed)
    raw, neighbor, projected, metadata, edges = {}, {}, {}, {}, {}
    views = ['node', 'node_neighbors', 'pair_node', 'pair_node_neighbors']
    projected = {v: {} for v in views}
    projections = {v: directions(d, args.projections, args.seed+17*i)
                   for i, (v, d) in enumerate(zip(views, [768, 1536, 1536, 3072]))}
    rawdir = out/'raw'
    rawdir.mkdir()
    for s in names:
        cache = Path(args.cache)
        base, context = cache/(s+'.pt'), cache/(s+'_neighbors10_disjoint50.pt')
        data = torch.load(base, map_location='cpu', weights_only=False)
        ctx = torch.load(context, map_location='cpu', weights_only=False)
        assert data['receipt'] == ctx['receipt']
        assert data['receipt']['graph'] == identity(config['graphs'][s]['path'])
        assert data['receipt']['seed'] == 0
        x, m = data['x'].numpy(), ctx['means'].numpy()
        assert x.shape == m.shape and x.shape[1] == 768
        assert np.isfinite(x).all() and np.isfinite(m).all()
        assert len(x) >= args.nodes
        ids = rng.choice(len(x), args.nodes, replace=False)
        raw[s], neighbor[s] = x[ids].copy(), m[ids].copy()
        pos = data['supervision'].numpy()
        # Independent with-replacement pair draws, matching the task's 1:5 mixture.
        p = pos[:, rng.integers(pos.shape[1], size=args.pairs)]
        neg = nonedges(len(x), data['known_keys'].numpy(), args.pairs*5, rng)
        mix = np.concatenate([p, neg], axis=1)
        mix = mix[:, rng.permutation(mix.shape[1])]
        edge_cos, neg_cos = cosine(x[p[0]], x[p[1]]), cosine(x[neg[0]], x[neg[1]])
        edges[s] = dict(positive=x[p.T].copy())
        # Only sampled inputs persist; full caches never modified.
        np.savez_compressed(rawdir/(s+'.npz'), node_ids=ids, node=raw[s], mean_neighbor=neighbor[s],
                            positive_pairs=p, negative_pairs=neg, edge_cosine=edge_cos, nonedge_cosine=neg_cos)
        for v, features in [('node', x), ('node_neighbors', np.concatenate([x, m], axis=1))]:
            projected[v][s] = features[ids] @ projections[v]
            projected['pair_'+v][s] = pair_projection(features, mix, projections['pair_'+v])
        metadata[s] = dict(cache=identity(base), context_cache=identity(context), receipt=data['receipt'],
                           node_count=len(x), supervision_edges=pos.shape[1],
                           sampled_isolate_fraction=float((np.linalg.norm(neighbor[s], axis=1)==0).mean()),
                           edge_cosine_mean=float(edge_cos.mean()), nonedge_cosine_mean=float(neg_cos.mean()),
                           edge_similarity_auc=float(roc_auc_score(np.r_[np.ones(len(edge_cos)), np.zeros(len(neg_cos))], np.r_[edge_cos, neg_cos])),
                           edge_nonedge_ks=float(ks_2samp(edge_cos, neg_cos).statistic),
                           cosine_histogram=dict(bins=np.linspace(-1, 1, 101).tolist(),
                               edge=np.histogram(edge_cos, bins=np.linspace(-1, 1, 101))[0].tolist(),
                               nonedge=np.histogram(neg_cos, bins=np.linspace(-1, 1, 101))[0].tolist()),
                           node_neighbor_cosine_mean=float(cosine(raw[s], neighbor[s]).mean()))
        del data, ctx, x, m, features
        print(json.dumps(dict(sampled=s, elapsed=time.monotonic()-started)), flush=True)
    result = dict(protocol=vars(args), revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  graphs=metadata, distances={v: swd_table(projected[v]) for v in views}, domain={})
    for v in ['node', 'node_neighbors']:
        features = raw if v == 'node' else {s: np.concatenate([raw[s], neighbor[s]], axis=1) for s in names}
        rows = []
        for a, b in itertools.combinations(names, 2):
            auc = domain_auc(features[a], features[b], args.seed)
            rows.append(dict(first=a, second=b, domain_auc=auc))
            print(json.dumps(dict(view=v, first=a, second=b, domain_auc=auc)), flush=True)
        baseline = {s: domain_auc(features[s][::2], features[s][1::2], args.seed) for s in names}
        result['domain'][v] = dict(pairs=rows, within_graph=baseline)
    # Shared clusters fitted only to sampled raw inputs, no supervised targets.
    unit = lambda z: z/np.maximum(np.linalg.norm(z, axis=-1, keepdims=True), 1e-12)
    cluster = MiniBatchKMeans(n_clusters=24, batch_size=2048, n_init=3, random_state=args.seed)
    cluster.fit(np.concatenate([unit(raw[s][:2000]) for s in names]))
    result['mixing'] = {}
    for s in names:
        counts = np.zeros((24, 24), dtype=np.int64)
        p = edges[s]['positive']
        labels = cluster.predict(unit(p.reshape(-1, 768))).reshape(-1, 2)
        np.add.at(counts, (labels[:, 0], labels[:, 1]), 1)
        np.add.at(counts, (labels[:, 1], labels[:, 0]), 1)
        joint = counts/counts.sum()
        endpoint = joint.sum(0)
        expected = np.outer(endpoint, endpoint)
        result['mixing'][s] = dict(counts=counts.tolist(), configuration_expected=expected.tolist(),
                                   observed_same_cluster=float(np.trace(joint)),
                                   expected_same_cluster=float(np.trace(expected)))
    np.savez_compressed(rawdir/'projections.npz', **projections, cluster_centers=cluster.cluster_centers_)
    (out/'audit.json').write_text(json.dumps(result, indent=2))
    (out/'COMPLETE.json').write_text(json.dumps(dict(elapsed_seconds=time.monotonic()-started, graphs=names)))
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--config', default='configs/nonzero_mini_transfer.yaml')
    p.add_argument('--cache', default='/dataMeR1/phil/gfm/mixture-scaling-disjoint-neighbor/state/disjoint_context_s0/_cache')
    p.add_argument('--seed', type=int, default=20260912)
    p.add_argument('--nodes', type=int, default=20000)
    p.add_argument('--pairs', type=int, default=10000)
    p.add_argument('--projections', type=int, default=128)
    p.add_argument('--threads', type=int, default=4)
    run(p.parse_args())
