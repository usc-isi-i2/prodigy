"""Small, strict paired-LP protocol; no model or graph-framework dependency."""
import hashlib
import numpy as np


def pair_keys(u, v, n):
    return np.minimum(u, v).astype(np.int64) * n + np.maximum(u, v)


def edge_keys(edges, n):
    u, v = np.asarray(edges)
    return np.unique(pair_keys(u[u != v], v[u != v], n))


def degree_bins(degree):
    # Keep true training isolates separate from degree-one nodes.
    return np.where(degree == 0, -1, np.floor(np.log2(np.maximum(degree, 1)))).astype(int)


def hash_arrays(*arrays):
    h = hashlib.sha256()
    for array in arrays:
        x = np.ascontiguousarray(array)
        h.update(str((x.shape, x.dtype.str)).encode())
        h.update(x.tobytes())
    return h.hexdigest()


def make_pairs(positive_keys, all_positive_keys, degree, count, rng, used_negatives):
    """Uniform positive proposals with strict degree-matched unique nonedges.

    Infeasible proposals are recorded and replaced; there is no uniform-negative
    fallback. One endpoint is randomly chosen to remain fixed. All returned
    labels and unordered pairs are unique within this split.
    """
    n = len(degree)
    bins = degree_bins(degree)
    pools = {int(b): np.flatnonzero(bins == b) for b in np.unique(bins)}
    known = set(map(int, all_positive_keys))
    positive, negative, infeasible = [], [], 0
    for key in rng.permutation(positive_keys):
        u, v = divmod(int(key), n)
        if rng.integers(2):
            u, v = v, u
        pool = pools[int(bins[v])]
        candidate = None
        for _ in range(200):
            w = int(pool[rng.integers(len(pool))])
            k = min(u, w) * n + max(u, w)
            if w != u and k not in known and k not in used_negatives:
                candidate = (u, w, k)
                break
        if candidate is None:
            infeasible += 1
            continue
        positive.append((u, v))
        negative.append(candidate[:2])
        used_negatives.add(candidate[2])
        if len(positive) == count:
            break
    if len(positive) != count:
        raise ValueError(f"Requested {count} pairs; only {len(positive)} feasible")
    pairs = np.asarray(positive + negative, dtype=np.int64)
    y = np.r_[np.ones(count, dtype=np.int64), np.zeros(count, dtype=np.int64)]
    keys = pair_keys(pairs[:, 0], pairs[:, 1], n)
    assert len(np.unique(keys)) == len(keys)
    assert np.isin(keys[:count], positive_keys).all()
    assert not np.isin(keys[count:], all_positive_keys).any()
    assert np.array_equal(pairs[:count, 0], pairs[count:, 0])
    assert np.array_equal(bins[pairs[:count, 1]], bins[pairs[count:, 1]])
    return dict(u=pairs[:, 0], v=pairs[:, 1], y=y), dict(
        positive_proposals_rejected=infeasible,
        unordered_pair_hash=hash_arrays(keys, y),
        zero_degree_endpoint_occurrences=int((degree[pairs] == 0).sum()),
        degree_matching="exact log2 bin; isolates separate; no fallback",
    )


def measure(y, score):
    from sklearn.metrics import average_precision_score, roc_auc_score
    assert np.isfinite(score).all()
    return dict(auc=float(roc_auc_score(y, score)),
        average_precision=float(average_precision_score(y, score)))


def evaluate_score(validation, test, val_score, test_score):
    raw_val = measure(validation['y'], val_score)
    sign = 1 if raw_val['auc'] >= .5 else -1
    return dict(orientation=sign, validation=measure(validation['y'], sign * val_score),
        test=measure(test['y'], sign * test_score),
        test_raw_orientation=measure(test['y'], test_score))
