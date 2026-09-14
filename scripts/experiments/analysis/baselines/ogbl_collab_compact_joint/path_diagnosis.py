"""Matched, post-hoc path/recency diagnostic. No fitting or test split access."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, maximum_bipartite_matching


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def top(scores):
    m = np.zeros(len(scores), bool)
    m[np.argsort(-scores, kind='stable')[:50]] = True
    return m


def hits(pos, neg):
    return pos > np.partition(neg, -50)[-50]


def match(positive, negative, candidates, targets, positive_model=None, negative_model=None, model_std=None):
    # Exact AA-zero/recurrence strata plus fixed physical-unit calipers.
    cols = [5, 6, 7, 8, 9, 10, 11]
    limits = np.array([.7, .7, .03, .7, 2., .7, .7])
    output = []
    for index in targets:
        diff = np.abs(positive[candidates][:, cols]-negative[index, cols])
        eligible = ((diff <= limits).all(1) &
                    ((positive[candidates, 12] > 0) == (negative[index, 12] > 0)) &
                    ((positive[candidates, 8] > 0) == (negative[index, 8] > 0)))
        distance = np.sqrt(((diff / limits)**2).mean(1))
        extra = None
        if positive_model is not None:
            extra = np.abs(positive_model[candidates]-negative_model[index])/np.maximum(model_std,1e-5)
            eligible &= (extra <= .75).all(1)
            distance = np.sqrt(distance**2+(extra**2).mean(1))
        ids = np.flatnonzero(eligible)
        ids = ids[np.argsort(distance[ids], kind='stable')[:5]]
        output.append(dict(negative_index=int(index), positives=candidates[ids].tolist(),
                           distances=distance[ids].tolist(),
                           max_model_input_standardized_difference=float(extra[ids].max()) if extra is not None and len(ids) else None,
                           max_scaled_covariate_difference=float((diff[ids]/limits).max()) if len(ids) else None))
    return output


def make_graph(edges, years, n):
    assert len(edges) == len(years) and int(years.max()) <= 2017
    key = np.minimum(edges[:, 0], edges[:, 1])*n + np.maximum(edges[:, 0], edges[:, 1])
    order = np.argsort(key, kind='stable')
    keys, first = np.unique(key[order], return_index=True)
    lo, hi = keys//n, keys % n
    adjacency = csr_matrix((np.ones(len(keys)*2), (np.r_[lo, hi], np.r_[hi, lo])), shape=(n, n))
    adjacency.sort_indices()
    earliest = np.minimum.reduceat(years[order], first)
    latest = np.maximum.reduceat(years[order], first)
    return adjacency, keys, earliest, latest


def path_features(u, v, adj, keys, first, last, x, components):
    n = adj.shape[0]
    neighbors = lambda a: adj.indices[adj.indptr[a]:adj.indptr[a+1]]
    nu, nv = neighbors(u), neighbors(v)
    degree = np.diff(adj.indptr)
    paths = [(int(a), int(b)) for a in nu if a != v
             for b in np.intersect1d(neighbors(a), nv, assume_unique=True) if b != u and b != a]
    out = dict(common_neighbors=int(len(np.intersect1d(nu, nv))),
               same_component=int(components[u] == components[v]),
               paths3=len(paths), disjoint_paths3=0, bridge_nodes=0,
               recent_paths3=0, recent_path_fraction=0., independent_recent_paths3=0,
               max_bridge_path_share=0., bridge_degree_median=None,
               freshest_bottleneck_age=None, median_path_year_spread=None,
               median_path_formation_age=None, bridge_cosine_median=None)
    if not paths:
        return out
    a, b = np.asarray(paths).T
    edges = np.stack([np.stack([np.full(len(a), u), a], 1), np.stack([a, b], 1),
                      np.stack([b, np.full(len(b), v)], 1)], axis=1)
    ek = edges.min(2)*n+edges.max(2)
    loc = np.searchsorted(keys, ek)
    assert np.array_equal(keys[loc], ek)
    latest, earliest = last[loc], first[loc]
    recent = (latest >= 2016).all(1)
    bridge, counts = np.unique(np.r_[a, b], return_counts=True)

    def independent(mask):
        if not mask.any():
            return 0
        _, ai = np.unique(a[mask], return_inverse=True)
        _, bi = np.unique(b[mask], return_inverse=True)
        matrix = csr_matrix((np.ones(len(ai)), (ai, bi)))
        return int((maximum_bipartite_matching(matrix, perm_type='column') >= 0).sum())

    # For AA-zero pairs, the intermediate partitions are disjoint; matching is
    # exactly the maximum number of internally vertex-disjoint length-three paths.
    if out['common_neighbors'] == 0:
        out['disjoint_paths3'] = independent(np.ones(len(a), bool))
        out['independent_recent_paths3'] = independent(recent)
    else:
        out['disjoint_paths3'] = None
        out['independent_recent_paths3'] = None
    out.update(bridge_nodes=int(len(bridge)), recent_paths3=int(recent.sum()),
               recent_path_fraction=float(recent.mean()), max_bridge_path_share=float(counts.max()/len(paths)),
               bridge_degree_median=float(np.median(degree[bridge])),
               freshest_bottleneck_age=int(2018-latest.min(1).max()),
               median_path_year_spread=float(np.median(latest.max(1)-latest.min(1))),
               median_path_formation_age=float(np.median(2018-earliest.max(1))),
               bridge_cosine_median=float(np.median((x[a]*x[b]).sum(1))))
    return out


def summarize(matches, computed):
    paired = [m for m in matches if m['positives']]
    metrics = {}
    for key in next(iter(computed.values())):
        negatives, positives = [], []
        for m in paired:
            nv = computed['n'+str(m['negative_index'])][key]
            pv = [computed['p'+str(i)][key] for i in m['positives']]
            pv = [v for v in pv if v is not None]
            if nv is not None and pv:
                negatives.append(nv)
                positives.append(float(np.mean(pv)))
        a, b = np.asarray(negatives), np.asarray(positives)
        metrics[key] = dict(paired_negatives=len(a), negative_median=float(np.median(a)) if len(a) else None,
                            matched_positive_mean_median=float(np.median(b)) if len(a) else None,
                            positive_greater_fraction=float((b>a).mean()) if len(a) else None,
                            positive_less_fraction=float((b<a).mean()) if len(a) else None)
    used = [i for m in paired for i in m['positives']]
    return dict(target_negatives=len(matches), matched_negatives=len(paired),
                unmatched_negative_indices=[m['negative_index'] for m in matches if not m['positives']],
                positive_matches=len(used), distinct_positive_matches=len(set(used)), metrics=metrics)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime', type=Path, required=True)
    p.add_argument('--evidence', type=Path, required=True)
    p.add_argument('--raw', type=Path, default=Path('/dataMeR1/phil/data/ogb/ogbl_collab/raw'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--match-all-inputs', action='store_true')
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    config = dict(classification='post_hoc_matched_case_diagnostic', neighbors=5,
                  calipers={'log_degree_min': .7, 'log_degree_max': .7, 'cosine': .03,
                            'log_pair_count': .7, 'pair_age': 2, 'log_recent_min': .7, 'log_recent_max': .7},
                  exact_strata=['aa_zero', 'previous_pair_seen'], matching_with_reuse_across_negatives=True,
                  all14_additional_caliper_in_training_std=.75 if args.match_all_inputs else None,
                  recent_edge_definition='last event in2016 or2017', test_read_or_scored=False,
                  warning='Selected error cohorts; no classifier fitting, causal estimate, or generalization claim')
    (args.out/'protocol.json').write_text(json.dumps(config, indent=2)+'\n')
    import wandb
    run = wandb.init(project='ogbl-collab-compact-joint', group='failure-paths', name='matched_paths_v1',
                     mode='offline', dir=str(args.out), config=config)
    start = time.monotonic()
    results = json.loads((args.evidence/'results.json').read_text())
    assert sha(args.runtime/'scores.npz') == results['score_sha256']
    assert sha(args.runtime/'year2018.npz') == results['panel_sha256']
    panel, score = np.load(args.runtime/'year2018.npz'), np.load(args.runtime/'scores.npz')
    prepared = json.loads((args.runtime.parent/'prepared.json').read_text())
    model_std = None
    if args.match_all_inputs:
        stdfile = args.runtime.parent/'standardization.npz'
        assert sha(stdfile) == prepared['files']['standardization.npz']
        model_std = np.load(stdfile)['std']
    baseline = hits(panel['official_bp'], panel['official_bn'])
    jh = np.stack([hits(score[f'joint_seed{s}_p'], score[f'joint_seed{s}_n']) for s in range(3)])
    jt = np.stack([top(score[f'joint_seed{s}_n']) for s in range(3)])
    cohorts = {}
    for name, reducer in [('consistent', np.all), ('any_seed', np.any)]:
        positives = np.flatnonzero(~baseline & reducer(jh, axis=0))
        negatives = np.flatnonzero(~top(panel['official_bn']) & reducer(jt, axis=0))
        cohorts[name] = match(panel['pfeatures'], panel['nfeatures'], positives, negatives,
                             panel['psymmetric'] if args.match_all_inputs else None,
                             panel['nsymmetric'] if args.match_all_inputs else None, model_std)
    xfile = args.runtime.parent/'node_features.npy'
    assert sha(xfile) == prepared['files']['node_features.npy']
    x = np.load(xfile)
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    edges = np.loadtxt(args.raw/'edge.csv.gz', delimiter=',', dtype=np.int64)
    years = np.loadtxt(args.raw/'edge_year.csv.gz', delimiter=',', dtype=np.int64).reshape(-1)
    adj, keys, first, last = make_graph(edges, years, len(x))
    assert np.array_equal(np.stack([keys//len(x), keys % len(x)], 1), panel['graph_edges'])
    _, components = connected_components(adj, directed=False)
    cases = {}
    for matches in cohorts.values():
        for m in matches:
            cases['n'+str(m['negative_index'])] = panel['neg'][m['negative_index']]
            for i in m['positives']:
                cases['p'+str(i)] = panel['pos'][i]
    computed = {key: path_features(int(edge[0]), int(edge[1]), adj, keys, first, last, x, components)
                for key, edge in cases.items()}
    output = dict(complete=True, protocol=config, source_revision=results['revision'],
                  scores_sha256=results['score_sha256'], panel_sha256=results['panel_sha256'],
                  graph_edges_sha256=sha(args.raw/'edge.csv.gz'), graph_years_sha256=sha(args.raw/'edge_year.csv.gz'),
                  graph_max_year=int(years.max()), graph_matches_frozen_panel=True,
                  cohorts={name:summarize(matches, computed) for name,matches in cohorts.items()},
                  matching=cohorts, cases=computed, elapsed_seconds=time.monotonic()-start,
                  diagnostic_source_sha256=sha(Path(__file__)), wandb_directory=run.dir, test_read_or_scored=False)
    (args.out/'results.json').write_text(json.dumps(output, indent=2)+'\n')
    run.summary.update({'cases': len(cases), 'elapsed_seconds': output['elapsed_seconds']})
    run.finish()
    print(json.dumps(output['cohorts'], indent=2))


if __name__ == '__main__':
    main()
