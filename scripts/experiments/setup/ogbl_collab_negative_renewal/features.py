"""Cached graph context; feature arithmetic copied exactly from temporal_gate.make_year."""
import numpy as np

def build(aa, joint, graph, split, year, calibration):
    n = int(graph["num_nodes"])
    train, years, weight, _ = joint.gate.historical_inputs(graph, split, year)
    A = aa.build_weighted_adj(n, train, weight)
    inv = aa.precompute_inv_log_deg(A)
    adj = aa.build_adj(n, train)
    lcc = aa.compute_exact_lcc(A)
    common = dict(A=A,A_invlog=aa.precompute_aa_matrix(A,inv),inv_log_deg=inv,adj=adj,
                  use_gate=True,gate_mode='threshold',ext_threshold=.5,ext_penalty=.5,lcc=lcc,
                  use_l3=True,rescue_mode='anchor',anchor_scale=0.,collect_l3=True,show_progress=False)
    historical = joint.gate.keys(train,n)
    order = np.argsort(historical,kind='stable')
    unique, first, counts = np.unique(historical[order],return_index=True,return_counts=True)
    latest = np.maximum.reduceat(years[order],first)
    degree = np.array([len(a) for a in adj])
    recent = np.bincount(train[years == year-1].flatten(),minlength=n)
    x = np.asarray(graph['node_feat'],dtype=np.float32)
    x = x / np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
    def features(edge,raw):
        u,v = edge.T
        idx = np.searchsorted(unique,joint.gate.keys(edge,n))
        found = (idx<len(unique)) & (unique[np.minimum(idx,len(unique)-1)]==joint.gate.keys(edge,n))
        count, age = np.zeros(len(edge)), np.full(len(edge),20.)
        count[found] = counts[idx[found]]
        age[found] = np.minimum(20,year-latest[idx[found]])
        l3 = np.nan_to_num(raw[4]['l3_values'],nan=0.)
        f = np.stack([np.log1p(raw[1]),np.log1p(l3),np.nan_to_num(raw[3],nan=0.),
                      np.minimum(lcc[u],lcc[v]),np.maximum(lcc[u],lcc[v]),
                      np.log1p(np.minimum(degree[u],degree[v])),np.log1p(np.maximum(degree[u],degree[v])),
                      (x[u]*x[v]).sum(1),np.log1p(count),age,
                      np.log1p(np.minimum(recent[u],recent[v])),np.log1p(np.maximum(recent[u],recent[v])),
                      raw[1]==0],axis=1).astype(np.float32)
        assert f.shape==(len(edge),len(joint.gate.FEATURES)) and np.isfinite(f).all()
        return f
    def score(edge):
        forward = aa.score_edges(edges=edge, **common)
        reverse_edge = edge[:, ::-1].copy()
        reverse = aa.score_edges(edges=reverse_edge, **common)
        def direct(e, raw):
            base = joint.gate.calibrated_scores(aa, raw, e, lcc, calibration['gate'], calibration['anchor_scale'])
            return joint.gate.direct_features(features(e, raw), base)
        return .5 * (direct(edge, forward) + direct(reverse_edge, reverse))
    return score
