#!/usr/bin/env python3
"""Compute per-graph statistics and pairwise divergences across retweet graphs.

Runs on Tucker (graphs live under ``/dataMeR1/phil/data/<name>/graphs``). Every
graph is a ``torch.save`` dict with keys ``x`` (node bio-embedding features,
zero-filled when a user has no bio), ``edge_index`` (directed retweet edges),
``edge_attr``, ``y`` and ``label_names`` (see ``scripts/graph_construction``).

The script emits a single JSON artifact holding, for every graph:
  * topology scalars (n, e, density, reciprocity, degree assortativity,
    approx. clustering, largest WCC/SCC fraction) and in/out degree CCDFs,
  * feature scalars (missing-bio rate, feature norm, effective dimensionality),
  * feature/structure coupling (edge feature homophily vs. a random-pair
    baseline, Dirichlet energy, and label homophily where labels exist),
and, for every ordered pair of graphs:
  * degree-distribution KS distance (in and out),
  * feature centroid cosine distance, Frechet distance, RBF-MMD^2, and
    proxy-A-distance (a domain classifier's separability).

Large graphs (covid ~23M nodes / ~107M edges) are handled by memory-mapping the
feature tensor and subsampling nodes/edges for every feature-based metric; only
``edge_index`` is materialised in full. All sampling is seeded for reproducibility.

Because loading a 75GB graph is heavy, launch this on Tucker yourself, e.g.:

    source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate prodigy
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
    python scripts/experiments/analysis/graph_divergence/compute_graph_divergence.py \
        --data-root /dataMeR1/phil/data \
        --out scripts/experiments/analysis/graph_divergence/data/graph_divergence_data.json

See the companion README.md for the full experiment write-up.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# name -> path relative to --data-root. Single-source retweet graphs only
# (merged graphs are excluded: they are unions of these and not independent
# domains). Override with --graphs / --data-root; missing files are skipped.
DEFAULT_GRAPHS: dict[str, str] = {
    "covid19_twitter": "covid19_twitter/graphs/retweet_graph_parquet.pt",
    "ukr_rus_twitter": "ukr_rus_twitter/graphs/retweet_graph_parquet.pt",
    "midterm": "midterm/graphs/retweet_graph_parquet.pt",
    "cp_hk_twitter": "cp_hk_twitter/graphs/retweet_graph.pt",
    "twibot20": "twibot20/graphs/retweet_graph.pt",
    "election2020": "election2020/graphs/retweet_graph.pt",
    "covid_political": "covid_political/graphs/retweet_graph.pt",
    "ukr_rus_suspended": "ukr_rus_suspended/graphs/retweet_graph.pt",
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Loading helpers
# --------------------------------------------------------------------------- #
def load_graph(path: Path):
    """Memory-map the graph dict; only sampled feature rows are ever read."""
    import torch

    for kwargs in ({"mmap": True, "weights_only": False}, {"weights_only": False}, {}):
        try:
            return torch.load(path, map_location="cpu", **kwargs)
        except TypeError:
            continue
    raise RuntimeError(f"could not load {path}")


def as_numpy(t) -> np.ndarray:
    return t.detach().cpu().numpy()


# --------------------------------------------------------------------------- #
# Feature sampling (mmap-aware, avoids materialising the full X)
# --------------------------------------------------------------------------- #
def sample_feature_rows(
    x, n_nodes: int, n_target: int, rng: np.random.Generator, chunk: int = 200_000
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (non_missing_rows [<=n_target, D], their indices, missing_rate).

    Draws uniform candidate rows and keeps the non-zero (has-bio) ones, so the
    fraction of zero rows among candidates is an unbiased missing-bio estimate.
    """
    import torch

    want_candidates = min(n_nodes, max(n_target * 4, 50_000))
    cand = rng.choice(n_nodes, size=want_candidates, replace=False)
    cand.sort()
    keep_rows: list[np.ndarray] = []
    keep_idx: list[np.ndarray] = []
    n_zero = 0
    n_seen = 0
    for start in range(0, len(cand), chunk):
        idx = cand[start : start + chunk]
        rows = as_numpy(x[torch.from_numpy(idx)]).astype(np.float32)
        norms = np.abs(rows).sum(axis=1)
        nz = norms > 0
        n_zero += int((~nz).sum())
        n_seen += len(idx)
        if nz.any():
            keep_rows.append(rows[nz])
            keep_idx.append(idx[nz])
        if sum(len(r) for r in keep_rows) >= n_target:
            # keep counting zeros only via the estimate already gathered
            break
    missing_rate = n_zero / max(n_seen, 1)
    if keep_rows:
        allrows = np.concatenate(keep_rows)[:n_target]
        allidx = np.concatenate(keep_idx)[:n_target]
    else:
        allrows = np.zeros((0, x.shape[1]), np.float32)
        allidx = np.zeros((0,), np.int64)
    return allrows, allidx, missing_rate


def gather_rows(x, idx: np.ndarray) -> np.ndarray:
    import torch

    return as_numpy(x[torch.from_numpy(idx.astype(np.int64))]).astype(np.float32)


# --------------------------------------------------------------------------- #
# Topology metrics
# --------------------------------------------------------------------------- #
def ccdf(degrees: np.ndarray, max_points: int = 200) -> dict[str, list[float]]:
    vals, counts = np.unique(degrees, return_counts=True)
    ccdf_y = 1.0 - (np.cumsum(counts) - counts) / counts.sum()
    if len(vals) > max_points:
        sel = np.unique(np.linspace(0, len(vals) - 1, max_points).astype(int))
        vals, ccdf_y = vals[sel], ccdf_y[sel]
    return {"deg": vals.astype(float).tolist(), "ccdf": ccdf_y.astype(float).tolist()}


def gini(x: np.ndarray) -> float:
    if len(x) == 0:
        return float("nan")
    xs = np.sort(x.astype(np.float64))
    n = len(xs)
    if xs.sum() == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1).dot(xs) / (n * xs.sum()))


def topology_metrics(
    edge_index: np.ndarray, n_nodes: int, rng: np.random.Generator,
    max_edges_exact: int, n_clustering_nodes: int,
) -> dict[str, Any]:
    src, dst = edge_index[0], edge_index[1]
    n_edges = edge_index.shape[1]
    out_deg = np.bincount(src, minlength=n_nodes)
    in_deg = np.bincount(dst, minlength=n_nodes)
    out: dict[str, Any] = {
        "n_nodes": int(n_nodes),
        "n_edges": int(n_edges),
        "density": float(n_edges / (n_nodes * (n_nodes - 1))) if n_nodes > 1 else float("nan"),
        "in_degree_mean": float(in_deg.mean()),
        "out_degree_mean": float(out_deg.mean()),
        "in_degree_gini": gini(in_deg),
        "out_degree_gini": gini(out_deg),
        "in_degree_max": int(in_deg.max()) if n_nodes else 0,
        "out_degree_max": int(out_deg.max()) if n_nodes else 0,
        "isolated_fraction": float(((in_deg + out_deg) == 0).mean()),
        "indegree_ccdf": ccdf(in_deg[in_deg > 0]),
        "outdegree_ccdf": ccdf(out_deg[out_deg > 0]),
    }

    # edge subsample for O(E) correlation/reciprocity on the biggest graphs
    if n_edges > max_edges_exact:
        esel = rng.choice(n_edges, size=max_edges_exact, replace=False)
        s_src, s_dst = src[esel], dst[esel]
        out["exact_edge_metrics_subsampled"] = True
    else:
        s_src, s_dst = src, dst
        out["exact_edge_metrics_subsampled"] = False

    # degree assortativity: corr(out-deg of source, in-deg of target)
    try:
        a = out_deg[s_src].astype(np.float64)
        b = in_deg[s_dst].astype(np.float64)
        out["degree_assortativity"] = float(np.corrcoef(a, b)[0, 1])
    except Exception as exc:  # noqa: BLE001
        log(f"  assortativity failed: {exc}")
        out["degree_assortativity"] = None

    # reciprocity: fraction of directed edges whose reverse also exists
    try:
        fwd = s_src.astype(np.int64) * n_nodes + s_dst.astype(np.int64)
        rev = s_dst.astype(np.int64) * n_nodes + s_src.astype(np.int64)
        fwd_sorted = np.sort(fwd)
        pos = np.searchsorted(fwd_sorted, rev)
        pos = np.clip(pos, 0, len(fwd_sorted) - 1)
        out["reciprocity"] = float((fwd_sorted[pos] == rev).mean())
    except Exception as exc:  # noqa: BLE001
        log(f"  reciprocity failed: {exc}")
        out["reciprocity"] = None

    # connected components (full graph) via scipy
    try:
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components

        data = np.ones(n_edges, dtype=np.int8)
        adj = csr_matrix((data, (src, dst)), shape=(n_nodes, n_nodes))
        nw, lw = connected_components(adj, directed=False)
        _, counts = np.unique(lw, return_counts=True)
        out["n_weakly_connected"] = int(nw)
        out["largest_wcc_frac"] = float(counts.max() / n_nodes)
        ns, ls = connected_components(adj, directed=True, connection="strong")
        _, countss = np.unique(ls, return_counts=True)
        out["n_strongly_connected"] = int(ns)
        out["largest_scc_frac"] = float(countss.max() / n_nodes)
    except Exception as exc:  # noqa: BLE001
        log(f"  connected_components failed: {exc}")
        out["largest_wcc_frac"] = out["largest_scc_frac"] = None

    # approximate average local clustering (undirected) on a node sample
    try:
        from scipy.sparse import csr_matrix

        undirected = csr_matrix(
            (np.ones(2 * n_edges, np.int8),
             (np.concatenate([src, dst]), np.concatenate([dst, src]))),
            shape=(n_nodes, n_nodes),
        )
        undirected.data[:] = 1
        undirected.sum_duplicates()
        indptr, indices = undirected.indptr, undirected.indices
        sample = rng.choice(n_nodes, size=min(n_clustering_nodes, n_nodes), replace=False)
        coeffs = []
        for u in sample:
            nbrs = indices[indptr[u] : indptr[u + 1]]
            nbrs = nbrs[nbrs != u]
            k = len(nbrs)
            if k < 2:
                coeffs.append(0.0)
                continue
            if k > 500:  # cap hub cost
                nbrs = rng.choice(nbrs, size=500, replace=False)
                k = 500
            nbr_set = set(nbrs.tolist())
            links = 0
            for v in nbrs:
                vn = indices[indptr[v] : indptr[v + 1]]
                links += np.count_nonzero(np.isin(vn, list(nbr_set), assume_unique=False))
            coeffs.append(links / (k * (k - 1)))
        out["avg_clustering_approx"] = float(np.mean(coeffs)) if coeffs else None
    except Exception as exc:  # noqa: BLE001
        log(f"  clustering failed: {exc}")
        out["avg_clustering_approx"] = None

    return out


# --------------------------------------------------------------------------- #
# Feature + coupling metrics
# --------------------------------------------------------------------------- #
def unit_norm(a: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(a, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return a / n


def effective_dim(rows: np.ndarray) -> float:
    """Participation ratio of the covariance spectrum: (Σλ)^2 / Σλ^2."""
    if len(rows) < 2:
        return float("nan")
    c = np.cov(rows, rowvar=False)
    ev = np.clip(np.linalg.eigvalsh(c), 0, None)
    s = ev.sum()
    return float(s * s / (ev @ ev)) if (ev @ ev) > 0 else float("nan")


def coupling_metrics(
    x, edge_index: np.ndarray, n_nodes: int, y: np.ndarray | None,
    has_labels: bool, rng: np.random.Generator, n_edge_sample: int,
    feat_sample: np.ndarray,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    n_edges = edge_index.shape[1]
    esel = rng.choice(n_edges, size=min(n_edge_sample, n_edges), replace=False)
    u, v = edge_index[0][esel].astype(np.int64), edge_index[1][esel].astype(np.int64)
    xu, xv = gather_rows(x, u), gather_rows(x, v)
    both = (np.abs(xu).sum(1) > 0) & (np.abs(xv).sum(1) > 0)
    out["edge_bio_coverage"] = float(both.mean())
    if both.sum() >= 10:
        a, b = unit_norm(xu[both]), unit_norm(xv[both])
        cos = (a * b).sum(1)
        out["feature_homophily"] = float(cos.mean())
        out["dirichlet_energy"] = float((2 * (1 - cos)).mean())
    else:
        out["feature_homophily"] = out["dirichlet_energy"] = None

    # random-pair baseline from the non-missing feature sample
    if len(feat_sample) >= 20:
        m = (len(feat_sample) // 2) * 2
        fs = unit_norm(feat_sample[:m])
        cos_rand = (fs[: m // 2] * fs[m // 2 :]).sum(1)
        out["feature_homophily_random"] = float(cos_rand.mean())
    else:
        out["feature_homophily_random"] = None

    # label homophily over edges where both endpoints carry a valid label
    if has_labels and y is not None:
        yu, yv = y[u], y[v]
        lab = (yu >= 0) & (yv >= 0)
        if lab.sum() >= 10:
            out["label_homophily"] = float((yu[lab] == yv[lab]).mean())
            out["labeled_edge_fraction"] = float(lab.mean())
        else:
            out["label_homophily"] = None
        labeled = y[y >= 0]
        vals, counts = np.unique(labeled, return_counts=True)
        out["class_balance"] = {int(k): int(c) for k, c in zip(vals, counts)}
        out["n_labeled_nodes"] = int(len(labeled))
    else:
        out["label_homophily"] = None
        out["class_balance"] = None
    return out


# --------------------------------------------------------------------------- #
# Pairwise feature divergences
# --------------------------------------------------------------------------- #
def frechet_distance(mu1, c1, mu2, c2) -> float:
    from scipy import linalg

    diff = mu1 - mu2
    covmean = linalg.sqrtm(c1 @ c2)
    if isinstance(covmean, tuple):  # older scipy returns (sqrt, errest)
        covmean = covmean[0]
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(c1 + c2 - 2 * covmean))


def rbf_mmd2(xs: np.ndarray, ys: np.ndarray, rng: np.random.Generator, cap: int = 1000) -> float:
    xs = xs[rng.permutation(len(xs))[:cap]]
    ys = ys[rng.permutation(len(ys))[:cap]]
    z = np.vstack([xs, ys])
    d2 = np.maximum(
        (z * z).sum(1)[:, None] + (z * z).sum(1)[None, :] - 2 * z @ z.T, 0
    )
    med = np.median(d2[d2 > 0]) if (d2 > 0).any() else 1.0
    gamma = 1.0 / med if med > 0 else 1.0
    k = np.exp(-gamma * d2)
    n, m = len(xs), len(ys)
    kxx = (k[:n, :n].sum() - np.trace(k[:n, :n])) / (n * (n - 1))
    kyy = (k[n:, n:].sum() - np.trace(k[n:, n:])) / (m * (m - 1))
    kxy = k[:n, n:].mean()
    return float(max(kxx + kyy - 2 * kxy, 0.0))


def proxy_a_distance(xs: np.ndarray, ys: np.ndarray, rng: np.random.Generator) -> float | None:
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
    except Exception:  # noqa: BLE001
        return None
    x = np.vstack([xs, ys])
    lab = np.concatenate([np.zeros(len(xs)), np.ones(len(ys))])
    xtr, xte, ytr, yte = train_test_split(
        x, lab, test_size=0.3, random_state=int(rng.integers(1 << 31)), stratify=lab
    )
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(xtr, ytr)
    err = 1.0 - clf.score(xte, yte)
    return float(2 * (1 - 2 * err))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent
        ).decode().strip()
    except Exception:  # noqa: BLE001
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default="/dataMeR1/phil/data")
    ap.add_argument("--out", default="scripts/experiments/analysis/graph_divergence/data/graph_divergence_data.json")
    ap.add_argument("--graphs", default="", help="comma list to restrict to a subset of DEFAULT_GRAPHS")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--feat-sample", type=int, default=4000, help="non-missing nodes sampled per graph for feature stats")
    ap.add_argument("--edge-sample", type=int, default=300_000, help="edges sampled for coupling metrics")
    ap.add_argument("--max-edges-exact", type=int, default=40_000_000, help="edge cap for reciprocity/assortativity")
    ap.add_argument("--clustering-nodes", type=int, default=3000)
    ap.add_argument("--mmd-cap", type=int, default=1000)
    args = ap.parse_args()

    graphs = DEFAULT_GRAPHS
    if args.graphs:
        want = [g.strip() for g in args.graphs.split(",") if g.strip()]
        graphs = {k: DEFAULT_GRAPHS[k] for k in want}

    data_root = Path(args.data_root)
    rng_master = np.random.default_rng(args.seed)

    result: dict[str, Any] = {
        "meta": {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "data_root": str(data_root),
            "git_commit": git_commit(),
            "hostname": platform.node(),
            "seed": args.seed,
            "config": {
                "feat_sample": args.feat_sample, "edge_sample": args.edge_sample,
                "max_edges_exact": args.max_edges_exact,
                "clustering_nodes": args.clustering_nodes, "mmd_cap": args.mmd_cap,
            },
        },
        "graphs": [],
        "per_graph": {},
        "feature_moments": {},  # mu / cov kept for pairwise stage
    }
    feat_samples: dict[str, np.ndarray] = {}
    degree_samples: dict[str, dict[str, np.ndarray]] = {}

    for name, rel in graphs.items():
        path = data_root / rel
        if not path.exists():
            log(f"SKIP {name}: {path} not found")
            continue
        log(f"=== {name} ({path}) ===")
        t0 = time.time()
        seed = int(rng_master.integers(1 << 31))
        rng = np.random.default_rng(seed)
        try:
            obj = load_graph(path)
        except Exception as exc:  # noqa: BLE001
            log(f"SKIP {name}: load failed: {exc}")
            continue

        x = obj["x"]
        n_nodes = int(x.shape[0])
        edge_index = as_numpy(obj["edge_index"]).astype(np.int64)
        y = obj.get("y")
        y = as_numpy(y).astype(np.int64) if y is not None else None
        label_names = list(obj.get("label_names") or [])
        has_labels = bool(label_names) and y is not None and int((y >= 0).sum()) > 0

        rows, _idx, missing_rate = sample_feature_rows(x, n_nodes, args.feat_sample, rng)
        feat_samples[name] = rows
        degree_samples[name] = {
            "in": np.bincount(edge_index[1], minlength=n_nodes),
            "out": np.bincount(edge_index[0], minlength=n_nodes),
        }
        log(f"  loaded in {time.time()-t0:.0f}s; N={n_nodes:,} E={edge_index.shape[1]:,} "
            f"feat_sample={len(rows)} missing_rate={missing_rate:.3f}")

        entry: dict[str, Any] = {"label_names": label_names, "has_labels": has_labels}
        entry.update(topology_metrics(
            edge_index, n_nodes, rng, args.max_edges_exact, args.clustering_nodes))
        entry["missing_bio_rate"] = float(missing_rate)
        entry["feature_norm_mean"] = float(np.linalg.norm(rows, axis=1).mean()) if len(rows) else None
        entry["feature_effective_dim"] = effective_dim(rows)
        entry.update(coupling_metrics(
            x, edge_index, n_nodes, y, has_labels, rng, args.edge_sample, rows))

        result["per_graph"][name] = entry
        result["graphs"].append(name)
        result["feature_moments"][name] = {
            "mu": rows.mean(0).tolist() if len(rows) else None,
        }
        log(f"  done {name} in {time.time()-t0:.0f}s")
        del obj, x

    # ------------------------------------------------------------------ #
    # Pairwise stage
    # ------------------------------------------------------------------ #
    names = result["graphs"]
    n = len(names)
    covs = {nm: (np.cov(feat_samples[nm], rowvar=False) if len(feat_samples[nm]) > 1 else None)
            for nm in names}
    mus = {nm: (feat_samples[nm].mean(0) if len(feat_samples[nm]) else None) for nm in names}

    def zeros():
        return [[0.0] * n for _ in range(n)]

    pw: dict[str, list[list[float]]] = {
        k: zeros() for k in
        ["indegree_ks", "outdegree_ks", "feat_centroid_cosdist",
         "feat_frechet", "feat_mmd2", "proxy_a_distance"]
    }
    from scipy.stats import ks_2samp

    log("=== pairwise ===")
    for i in range(n):
        for j in range(n):
            ni, nj = names[i], names[j]
            if i == j:
                continue
            rng = np.random.default_rng(args.seed + i * 1000 + j)
            di = degree_samples[ni]["in"]
            dj = degree_samples[nj]["in"]
            cap = 50_000
            si = di[rng.choice(len(di), size=min(cap, len(di)), replace=False)]
            sj = dj[rng.choice(len(dj), size=min(cap, len(dj)), replace=False)]
            pw["indegree_ks"][i][j] = float(ks_2samp(si, sj).statistic)
            do_i = degree_samples[ni]["out"]; do_j = degree_samples[nj]["out"]
            soi = do_i[rng.choice(len(do_i), size=min(cap, len(do_i)), replace=False)]
            soj = do_j[rng.choice(len(do_j), size=min(cap, len(do_j)), replace=False)]
            pw["outdegree_ks"][i][j] = float(ks_2samp(soi, soj).statistic)

            fi, fj = feat_samples[ni], feat_samples[nj]
            if len(fi) and len(fj):
                if mus[ni] is not None and mus[nj] is not None:
                    a, b = mus[ni], mus[nj]
                    denom = np.linalg.norm(a) * np.linalg.norm(b)
                    pw["feat_centroid_cosdist"][i][j] = float(1 - (a @ b) / denom) if denom else None
                try:
                    pw["feat_frechet"][i][j] = frechet_distance(mus[ni], covs[ni], mus[nj], covs[nj])
                except Exception as exc:  # noqa: BLE001
                    log(f"  frechet {ni},{nj} failed: {exc}")
                    pw["feat_frechet"][i][j] = None
                pw["feat_mmd2"][i][j] = rbf_mmd2(fi, fj, rng, cap=args.mmd_cap)
                pw["proxy_a_distance"][i][j] = proxy_a_distance(fi, fj, rng)
        log(f"  pairwise row {i+1}/{n} ({names[i]}) done")

    result["pairwise"] = pw
    result.pop("feature_moments", None)  # mus already folded into pairwise

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    log(f"WROTE {out_path}  ({len(names)} graphs)")


if __name__ == "__main__":
    sys.exit(main())
