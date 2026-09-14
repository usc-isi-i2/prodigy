"""Audit matching and summarize pairwise contrasts without fitting a rule."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from path_diagnosis import match, hits, top


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime', type=Path, required=True)
    p.add_argument('--diagnostic', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    d = json.loads(args.diagnostic.read_text())
    assert sha(args.runtime/'scores.npz') == d['scores_sha256']
    assert sha(args.runtime/'year2018.npz') == d['panel_sha256']
    panel, scores = np.load(args.runtime/'year2018.npz'), np.load(args.runtime/'scores.npz')
    baseline = hits(panel['official_bp'], panel['official_bn'])
    jh = np.stack([hits(scores[f'joint_seed{s}_p'],scores[f'joint_seed{s}_n']) for s in range(3)])
    jt = np.stack([top(scores[f'joint_seed{s}_n']) for s in range(3)])
    all14 = d['protocol'].get('all14_additional_caliper_in_training_std') is not None
    std = np.load(args.runtime/'standardization.npz')['std'] if all14 else None
    contrasts = {}
    for name, reducer in [('consistent',np.all),('any_seed',np.any)]:
        candidates = np.flatnonzero(~baseline & reducer(jh,axis=0))
        targets = np.flatnonzero(~top(panel['official_bn']) & reducer(jt,axis=0))
        expected = match(panel['pfeatures'],panel['nfeatures'],candidates,targets,
                         panel['psymmetric'] if all14 else None,panel['nsymmetric'] if all14 else None,std)
        assert len(expected) == len(d['matching'][name])
        for a,b in zip(expected,d['matching'][name]):
            assert a['negative_index']==b['negative_index'] and a['positives']==b['positives']
            np.testing.assert_allclose(a['distances'],b['distances'],rtol=0,atol=0)
        values = {}
        for feature in ('paths3','disjoint_paths3','recent_path_fraction','max_bridge_path_share',
                        'freshest_bottleneck_age','median_path_formation_age','bridge_cosine_median'):
            wins, negs, poss = [], [], []
            for m in expected:
                a = d['cases']['n'+str(m['negative_index'])][feature]
                b = [d['cases']['p'+str(i)][feature] for i in m['positives']]
                b = np.array([v for v in b if v is not None])
                if a is None or not len(b):
                    continue
                wins.append(float(((b>a)+.5*(b==a)).mean()))
                negs.append(a); poss.append(float(b.mean()))
            values[feature] = dict(paired_groups=len(wins),
                 positive_higher_pairwise_probability=float(np.mean(wins)) if wins else None,
                 negative_mean=float(np.mean(negs)) if wins else None,
                 matched_positive_mean=float(np.mean(poss)) if wins else None)
        contrasts[name] = values
    # Independent simple length-three enumeration for a fixed reproducible subset.
    edges = panel['graph_edges']
    assert not (edges[:,0]==edges[:,1]).any()
    graph = csr_matrix((np.ones(2*len(edges)),(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),
                       shape=(235868,235868))
    checked = []
    sample = sorted(k for k in d['cases'] if k.startswith('n'))[:10] + sorted(k for k in d['cases'] if k.startswith('p'))[:10]
    for name in sample:
        u,v = panel['neg' if name[0]=='n' else 'pos'][int(name[1:])]
        nu = graph.indices[graph.indptr[u]:graph.indptr[u+1]]
        nv = set(graph.indices[graph.indptr[v]:graph.indptr[v+1]])
        count = sum(len(set(graph.indices[graph.indptr[a]:graph.indptr[a+1]]) & nv - {u,a})
                    for a in nu if a not in (u,v))
        assert count == d['cases'][name]['paths3']
        checked.append(name)
    out = dict(complete=True, matching_recomputed_exactly=True, score_panel_hashes_verified=True,
               independently_recounted_path_cases=checked, contrasts=contrasts,
               diagnostic_sha256=sha(args.diagnostic), audit_source_sha256=sha(Path(__file__)),
               test_read_or_scored=False,
               qualification='Pairwise probabilities are descriptive, not classifier accuracy. Event-date logic has synthetic tests and producer graph checks; dates were not independently reloaded locally.')
    args.out.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))


if __name__ == '__main__':
    main()
