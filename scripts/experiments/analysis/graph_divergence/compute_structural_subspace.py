#!/usr/bin/env python3
"""Is the feature/structure coupling concentrated in a few GTE directions?

``compute_graph_divergence.py`` reports feature homophily as a cosine over all
768 dims: connected users average .568 against a .529 random-pair baseline, a
lift of only +.015..+.070. That average is over every dimension, so a real
coupling confined to a handful of directions would be diluted away.

This script re-asks the question inside a *selected subspace*, and answers three
things the full-space number cannot:

  1. **Within-graph.** Pick the k directions where connected users look most
     alike, then recompute the edge-vs-random-pair table there. Selection is fit
     on a train edge split and reported on a disjoint test split -- picking top-k
     of 768 on the same edges you score would manufacture a lift from noise.
  2. **Ceiling.** GTE's 768 axes are arbitrary, so axis-aligned selection is a
     strictly weaker hypothesis than subspace selection. Take the leading
     eigen-directions of M = E_edge[a b'] - E_nonedge[a b'] and score with the
     rank-r bilinear form. Any per-node linear map z=f(x) compared by inner
     product is a special case, so its held-out AUC bounds what *any* such
     learned projection could extract of the coupling.
  3. **Cross-graph -- the actual question.** Apply graph A's selected dims and
     eigen-directions to graph B's held-out edges. If the coupling lives in the
     *same* directions everywhere, that matrix is flat and there is a shared
     structural subspace worth mapping into. If it is strongly diagonal, each
     graph couples features to structure its own way and no common space exists.

Negatives are degree-matched (sampled from the positive destination's log2
in-degree bucket), matching the LP evaluator, so "hubs have distinctive bios"
cannot masquerade as homophily. Zero-vector (missing-bio) endpoints are dropped
from every pair, as in the parent script.

Runs on Tucker, where the graphs live:

    source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate prodigy
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
    python scripts/experiments/analysis/graph_divergence/compute_structural_subspace.py \
        --data-root /dataMeR1/phil/data \
        --out scripts/experiments/analysis/graph_divergence/data/structural_subspace.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_graph_divergence import (  # noqa: E402
    DEFAULT_GRAPHS,
    as_numpy,
    gather_rows,
    load_graph,
    log,
    unit_norm,
)

TOPK = (8, 16, 32, 64, 128, 256, 768)


# --------------------------------------------------------------------------- #
# Pair sampling
# --------------------------------------------------------------------------- #
def degree_matched_pairs(edge_index: np.ndarray, n_nodes: int, n_pairs: int,
                         rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Positives from real edges; negatives reuse the source, and draw a
    destination from the same log2 in-degree bucket as the true destination."""
    n_edges = edge_index.shape[1]
    sel = rng.choice(n_edges, size=min(n_pairs, n_edges), replace=False)
    u = edge_index[0][sel].astype(np.int64)
    v = edge_index[1][sel].astype(np.int64)

    indeg = np.bincount(edge_index[1], minlength=n_nodes)
    bucket = np.floor(np.log2(indeg + 1.0)).astype(np.int32)
    order = np.argsort(bucket, kind="stable")
    b_sorted = bucket[order]
    starts = np.searchsorted(b_sorted, bucket[v], side="left")
    ends = np.searchsorted(b_sorted, bucket[v], side="right")
    span = np.maximum(ends - starts, 1)
    v_neg = order[starts + (rng.random(len(v)) * span).astype(np.int64)]
    return {"u": u, "v": v, "v_neg": v_neg}


def materialise(x, pairs: dict[str, np.ndarray]) -> dict[str, np.ndarray] | None:
    """Gather features, drop pairs with a missing (all-zero) endpoint."""
    xu = gather_rows(x, pairs["u"])
    xv = gather_rows(x, pairs["v"])
    xn = gather_rows(x, pairs["v_neg"])
    ok = ((np.abs(xu).sum(1) > 0) & (np.abs(xv).sum(1) > 0) & (np.abs(xn).sum(1) > 0))
    if ok.sum() < 2000:
        return None
    return {"a": unit_norm(xu[ok]), "p": unit_norm(xv[ok]), "n": unit_norm(xn[ok]),
            "coverage": float(ok.mean())}


def split(f: dict[str, np.ndarray], frac: float, rng: np.random.Generator):
    m = len(f["a"])
    idx = rng.permutation(m)
    cut = int(m * frac)
    tr, te = idx[:cut], idx[cut:]
    take = lambda s: {k: f[k][s] for k in ("a", "p", "n")}  # noqa: E731
    return take(tr), take(te)


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney ROC-AUC, ties at .5. No sklearn dependency."""
    both = np.concatenate([pos, neg])
    r = np.empty(len(both), dtype=np.float64)
    o = np.argsort(both, kind="mergesort")
    sb = both[o]
    i = 0
    while i < len(sb):
        j = i
        while j + 1 < len(sb) and sb[j + 1] == sb[i]:
            j += 1
        r[o[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    n1 = len(pos)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * len(neg)))


def subspace_table(f: dict[str, np.ndarray], dims: np.ndarray, rng) -> dict[str, float]:
    """Edge cosine vs random-pair cosine vs edge/non-edge AUC, inside `dims`."""
    a, p, n = f["a"][:, dims], f["p"][:, dims], f["n"][:, dims]
    a, p, n = unit_norm(a), unit_norm(p), unit_norm(n)
    cos_e, cos_n = (a * p).sum(1), (a * n).sum(1)
    perm = rng.permutation(len(a))
    cos_r = (a * p[perm]).sum(1)
    return {"edge_cos": float(cos_e.mean()), "random_cos": float(cos_r.mean()),
            "lift": float(cos_e.mean() - cos_r.mean()),
            "auc_vs_degree_matched": auc(cos_e, cos_n), "k": int(len(dims))}


def coupling_matrix(train: dict[str, np.ndarray]) -> np.ndarray:
    """M = E_edge[a b'] - E_nonedge[a b'], symmetrised.

    Cosine is a'b = <a b', I>, so M is exactly the operator whose entries say
    how much more a pair of coordinates agrees across an edge than across a
    degree-matched non-edge. Its diagonal is the per-dim score; its leading
    eigenvectors are the same thing freed from the (arbitrary) GTE axes."""
    n = len(train["a"])
    M = (train["a"].T @ train["p"] - train["a"].T @ train["n"]) / n
    return 0.5 * (M + M.T)


def rank_dims(M: np.ndarray) -> np.ndarray:
    """Coordinates ordered by how much they agree on edges over non-edges."""
    return np.argsort(-np.diag(M))


def spectral_dirs(M: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    """Top-`rank` eigen-directions of M by |eigenvalue|. M is symmetric but not
    PSD -- a negative eigenvalue is a direction where connected users are
    *less* alike than chance, which is just as usable a signal, so keep signs."""
    lam, U = np.linalg.eigh(M)
    keep = np.argsort(-np.abs(lam))[:rank]
    return lam[keep].astype(np.float32), U[:, keep].astype(np.float32)


def spectral_score(f: dict[str, np.ndarray], lam: np.ndarray, U: np.ndarray,
                   key: str) -> np.ndarray:
    """s(a,b) = sum_j lambda_j (a.u_j)(b.u_j) -- a bilinear form of rank len(lam).
    Any per-node linear map z=f(x) scored by inner product is a special case, so
    the held-out AUC of this bounds what such a map could extract."""
    return ((f["a"] @ U) * (f[key] @ U) * lam).sum(1)


def spectral_auc(f: dict[str, np.ndarray], lam: np.ndarray, U: np.ndarray) -> float:
    return auc(spectral_score(f, lam, U, "p"), spectral_score(f, lam, U, "n"))


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="/dataMeR1/phil/data")
    ap.add_argument("--graphs", nargs="*", default=None)
    ap.add_argument("--out", default="structural_subspace.json")
    ap.add_argument("--pairs", type=int, default=50_000, help="sampled edges per graph")
    ap.add_argument("--xgraph-pairs", type=int, default=10_000,
                    help="test pairs cached per graph for the cross-graph matrix")
    ap.add_argument("--rank", type=int, default=64,
                    help="number of eigen-directions kept, and of top-k dims cached "
                         "for the cross-graph matrix")
    ap.add_argument("--train-frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-cross-graph", action="store_true")
    args = ap.parse_args()

    names = args.graphs or list(DEFAULT_GRAPHS)
    root = Path(args.data_root)
    rng = np.random.default_rng(args.seed)
    per_graph: dict[str, Any] = {}
    cache: dict[str, dict[str, np.ndarray]] = {}

    for name in names:
        path = root / DEFAULT_GRAPHS[name]
        if not path.exists():
            log(f"{name}: missing {path}, skipping")
            continue
        t0 = time.time()
        obj = load_graph(path)
        x = obj["x"]
        edge_index = as_numpy(obj["edge_index"]).astype(np.int64)
        n_nodes = int(x.shape[0])
        log(f"{name}: N={n_nodes:,} E={edge_index.shape[1]:,} ({time.time()-t0:.0f}s)")

        f = materialise(x, degree_matched_pairs(edge_index, n_nodes, args.pairs, rng))
        if f is None:
            log(f"{name}: too few pairs with bios on both endpoints, skipping")
            continue
        train, test = split(f, args.train_frac, rng)
        M = coupling_matrix(train)
        order = rank_dims(M)
        lam, U = spectral_dirs(M, args.rank)

        rec: dict[str, Any] = {"bio_coverage_of_pairs": f["coverage"],
                               "n_train": len(train["a"]), "n_test": len(test["a"]),
                               "by_k": {}}
        for k in TOPK:
            if k > order.shape[0]:
                continue
            rec["by_k"][str(k)] = subspace_table(test, np.sort(order[:k]), rng)
        rec["spectral"] = {"rank": args.rank, "test_auc": spectral_auc(test, lam, U),
                           "eigenvalue_mass_in_rank": float(
                               np.abs(lam).sum() / np.abs(np.linalg.eigvalsh(M)).sum())}
        per_graph[name] = rec

        full = rec["by_k"][str(min(768, order.shape[0]))]
        best_k = max(rec["by_k"].values(), key=lambda v: v["auc_vs_degree_matched"])
        log(f"  full-space AUC {full['auc_vs_degree_matched']:.4f} (lift {full['lift']:+.4f})"
            f" | best top-k dims AUC {best_k['auc_vs_degree_matched']:.4f} (k={best_k['k']},"
            f" lift {best_k['lift']:+.4f})"
            f" | rank-{args.rank} spectral {rec['spectral']['test_auc']:.4f}")

        if not args.no_cross_graph:
            m = min(args.xgraph_pairs, len(test["a"]))
            cache[name] = {"test": {k: test[k][:m].astype(np.float16) for k in ("a", "p", "n")},
                           "dims": np.sort(order[:args.rank]), "lam": lam, "U": U}
        del obj, x, f, train, test

    result: dict[str, Any] = {
        "meta": {"pairs": args.pairs, "rank": args.rank,
                 "train_frac": args.train_frac, "seed": args.seed,
                 "negatives": "degree-matched on log2 in-degree of the true destination",
                 "note": "dims/L selected on the train edge split, scored on a disjoint test split"},
        "per_graph": per_graph,
    }

    if cache and not args.no_cross_graph:
        log("cross-graph: applying each source's subspace to every target's held-out edges")
        xg_dims: dict[str, dict[str, float]] = {}
        xg_spec: dict[str, dict[str, float]] = {}
        for src, cs in cache.items():
            xg_dims[src], xg_spec[src] = {}, {}
            for tgt, ct in cache.items():
                te = {k: ct["test"][k].astype(np.float32) for k in ("a", "p", "n")}
                d = cs["dims"]
                a, p, n = (unit_norm(te["a"][:, d]), unit_norm(te["p"][:, d]),
                           unit_norm(te["n"][:, d]))
                xg_dims[src][tgt] = auc((a * p).sum(1), (a * n).sum(1))
                xg_spec[src][tgt] = spectral_auc(te, cs["lam"], cs["U"])
        result["cross_graph_topk_dims_auc"] = xg_dims
        result["cross_graph_spectral_auc"] = xg_spec

        cols = list(cache)
        for title, mat in (("top-k dims", xg_dims), (f"rank-{args.rank} spectral", xg_spec)):
            log("")
            log(f"cross-graph {title} AUC (row = subspace fitted on, col = scored on)")
            log("  " + " " * 22 + "".join(f"{c[:11]:>12s}" for c in cols))
            for src in cols:
                log(f"  {src:22s}" + "".join(f"{mat[src][t]:12.4f}" for t in cols))
            dg = np.mean([mat[g][g] for g in cols])
            od = np.mean([mat[s][t] for s in cols for t in cols if s != t])
            # retention: how much of a graph's own coupling a foreign subspace keeps,
            # measured against the .5 chance floor rather than against 0.
            ret = np.mean([(mat[s][t] - 0.5) / max(mat[t][t] - 0.5, 1e-6)
                           for s in cols for t in cols if s != t])
            log(f"  diagonal {dg:.4f} vs off-diagonal {od:.4f}; foreign subspaces retain "
                f"{100*ret:.0f}% of a graph's own coupling -> "
                f"{'SHARED' if ret > 0.7 else 'partly shared' if ret > 0.4 else 'graph-specific'}"
                f" structural subspace")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, default=float))
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
