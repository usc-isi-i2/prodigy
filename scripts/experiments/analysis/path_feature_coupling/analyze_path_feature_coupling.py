#!/usr/bin/env python3
"""Measure how within-graph path length relates to node-feature distance.

The social graphs are too large for an all-pairs shortest-path calculation.  This
runner therefore samples *matched anchors*: for each retained anchor it finds one
endpoint at exact undirected shortest-path distance 1, 2, and 3, plus a uniformly
drawn endpoint verified to be farther than 3 hops (or disconnected).  Exact short
distances are obtained by rejection-sampling random-walk prefixes and checking that
no shorter path exists.

For every graph the output reports:

* cosine and Euclidean feature distance by path-length bucket;
* descriptive Spearman trends across the exact 1/2/3-hop buckets and with the
  ``>3_or_disconnected`` bucket treated as ordinal bucket 4;
* held-out probes for distinguishing adjacent from far pairs using feature-pair
  information, including a train-selected single coordinate and sparse/dense
  logistic probes.  This catches a small number of informative feature dimensions
  that an averaged cosine distance can hide;
* directly comparable mean random-pair cosine/Euclidean distances within and
  between graphs.

The graph is treated as undirected because that is the neighborhood used by the
default PRODIGY message-passing setup.  Run this on Tucker, where the graph files
under /dataMeR1 are available.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DEFAULT_DATASET_KEYS = (
    "covid19_twitter",
    "ukr_rus_twitter",
    "midterm",
    "cp_hk_twitter",
    "twibot20",
    "election2020",
    "covid_political",
    "ukr_rus_suspended",
)
DISTANCE_LABELS = ("1", "2", "3", ">3_or_disconnected")


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def json_float(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent
        ).decode().strip()
    except Exception:  # noqa: BLE001
        return None


def load_catalog(path: Path) -> tuple[Path, dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        catalog = json.load(handle)
    paths = {
        entry["dataset_key"]: entry["relative_path"]
        for entry in catalog["graphs"]
        if entry.get("dataset_key") and entry.get("relative_path")
    }
    return Path(catalog["data_root"]), paths


def load_graph(path: Path):
    import torch

    for kwargs in ({"mmap": True, "weights_only": False}, {"weights_only": False}, {}):
        try:
            return torch.load(path, map_location="cpu", **kwargs)
        except TypeError:
            continue
    raise RuntimeError(f"could not load {path}")


def as_numpy(tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def gather_rows(x, indices: np.ndarray, chunk_size: int = 50_000) -> np.ndarray:
    import torch

    chunks = []
    for start in range(0, len(indices), chunk_size):
        idx = torch.from_numpy(indices[start : start + chunk_size].astype(np.int64))
        chunks.append(as_numpy(x[idx]).astype(np.float32))
    if not chunks:
        return np.empty((0, int(x.shape[1])), dtype=np.float32)
    return np.concatenate(chunks)


def build_undirected_csr(edge_index: np.ndarray, n_nodes: int):
    """Build a deduplicated, loop-free, undirected CSR adjacency."""
    from scipy.sparse import csr_matrix

    src = edge_index[0].astype(np.int64, copy=False)
    dst = edge_index[1].astype(np.int64, copy=False)
    rows = np.concatenate((src, dst))
    cols = np.concatenate((dst, src))
    data = np.ones(len(rows), dtype=np.uint8)
    adjacency = csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
    adjacency.sum_duplicates()
    adjacency.data[:] = 1
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    adjacency.sort_indices()
    return adjacency


def neighbors(adjacency, node: int) -> np.ndarray:
    return adjacency.indices[adjacency.indptr[node] : adjacency.indptr[node + 1]]


def contains_sorted(values: np.ndarray, target: int) -> bool:
    if len(values) == 0:
        return False
    pos = int(np.searchsorted(values, target))
    return pos < len(values) and int(values[pos]) == target


def have_common_neighbor(a: np.ndarray, b: np.ndarray) -> bool:
    if len(a) > len(b):
        a, b = b, a
    if len(a) == 0:
        return False
    pos = np.searchsorted(b, a)
    valid = pos < len(b)
    return bool(np.any(b[pos[valid]] == a[valid]))


def farther_than_three(adjacency, u: int, v: int) -> bool:
    """Return true exactly when undirected shortest path(u,v) is >3 or absent."""
    if u == v:
        return False
    nu = neighbors(adjacency, u)
    nv = neighbors(adjacency, v)
    if contains_sorted(nu, v):
        return False
    if have_common_neighbor(nu, nv):
        return False
    # A length-3 path exists iff an edge crosses N(u) and N(v).  Iterate the
    # side with fewer incident neighbor entries to keep hub checks bounded.
    work_u = sum(len(neighbors(adjacency, int(w))) for w in nu)
    work_v = sum(len(neighbors(adjacency, int(w))) for w in nv)
    if work_u <= work_v:
        left, right = nu, nv
    else:
        left, right = nv, nu
    for node in left:
        if have_common_neighbor(neighbors(adjacency, int(node)), right):
            return False
    return True


def exact_three_prefix(adjacency, anchor: int, rng: np.random.Generator) -> tuple[int, int, int] | None:
    """Sample a walk whose prefixes have exact shortest distances 1, 2, and 3."""
    n0 = neighbors(adjacency, anchor)
    if len(n0) == 0:
        return None
    v1 = int(n0[rng.integers(len(n0))])
    n1 = neighbors(adjacency, v1)
    if len(n1) == 0:
        return None
    v2 = int(n1[rng.integers(len(n1))])
    if v2 == anchor or contains_sorted(n0, v2):
        return None
    n2 = neighbors(adjacency, v2)
    if len(n2) == 0:
        return None
    v3 = int(n2[rng.integers(len(n2))])
    if v3 == anchor or contains_sorted(n0, v3):
        return None
    if have_common_neighbor(n0, neighbors(adjacency, v3)):
        return None
    return v1, v2, v3


@dataclass(frozen=True)
class Block:
    anchor: int
    d1: int
    d2: int
    d3: int
    far: int

    def nodes(self) -> tuple[int, int, int, int, int]:
        return self.anchor, self.d1, self.d2, self.d3, self.far


def propose_blocks(
    adjacency,
    n_nodes: int,
    n_blocks: int,
    rng: np.random.Generator,
    attempts_per_block: int,
) -> list[Block]:
    blocks: list[Block] = []
    max_attempts = max(n_blocks * attempts_per_block, 1_000)
    attempts = 0
    while len(blocks) < n_blocks and attempts < max_attempts:
        attempts += 1
        anchor = int(rng.integers(n_nodes))
        prefix = exact_three_prefix(adjacency, anchor, rng)
        if prefix is None:
            continue
        far = int(rng.integers(n_nodes))
        if not farther_than_three(adjacency, anchor, far):
            continue
        blocks.append(Block(anchor, prefix[0], prefix[1], prefix[2], far))
    log(f"  proposed {len(blocks):,}/{n_blocks:,} blocks in {attempts:,} attempts")
    return blocks


def retain_complete_feature_blocks(x, blocks: list[Block], n_keep: int) -> tuple[list[Block], np.ndarray]:
    if not blocks:
        return [], np.empty((0, int(x.shape[1])), dtype=np.float32)
    flat = np.asarray([node for block in blocks for node in block.nodes()], dtype=np.int64)
    unique, inverse = np.unique(flat, return_inverse=True)
    rows = gather_rows(x, unique)
    observed = np.abs(rows).sum(axis=1) > 0
    complete = observed[inverse].reshape(len(blocks), 5).all(axis=1)
    kept_indices = np.flatnonzero(complete)[:n_keep]
    kept = [blocks[int(i)] for i in kept_indices]
    if not kept:
        return [], np.empty((0, int(x.shape[1])), dtype=np.float32)
    kept_flat = np.asarray([node for block in kept for node in block.nodes()], dtype=np.int64)
    pos = np.searchsorted(unique, kept_flat)
    return kept, rows[pos].reshape(len(kept), 5, -1)


def sample_complete_blocks(
    adjacency,
    x,
    n_nodes: int,
    n_blocks: int,
    rng: np.random.Generator,
    attempts_per_block: int,
) -> tuple[list[Block], np.ndarray]:
    kept: list[Block] = []
    feature_chunks: list[np.ndarray] = []
    rounds = 0
    while len(kept) < n_blocks and rounds < 8:
        rounds += 1
        needed = n_blocks - len(kept)
        proposals = propose_blocks(
            adjacency,
            n_nodes,
            max(needed * 2, min(2_000, n_blocks)),
            rng,
            attempts_per_block,
        )
        new_blocks, new_features = retain_complete_feature_blocks(x, proposals, needed)
        kept.extend(new_blocks)
        if len(new_features):
            feature_chunks.append(new_features)
        log(f"  retained {len(kept):,}/{n_blocks:,} complete non-missing blocks")
        if not proposals:
            break
    features = (
        np.concatenate(feature_chunks, axis=0)
        if feature_chunks
        else np.empty((0, 5, int(x.shape[1])), dtype=np.float32)
    )
    return kept, features


def pair_distances(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    anchor = features[:, 0]
    targets = features[:, 1:]
    anorm = np.linalg.norm(anchor, axis=1, keepdims=True)
    tnorm = np.linalg.norm(targets, axis=2)
    denom = anorm * tnorm
    denom[denom == 0] = 1.0
    cosine = 1.0 - (targets * anchor[:, None, :]).sum(axis=2) / denom
    euclidean = np.linalg.norm(targets - anchor[:, None, :], axis=2)
    return cosine, euclidean


def metric_summary(values: np.ndarray) -> dict[str, Any]:
    from scipy.stats import spearmanr

    out: dict[str, Any] = {"by_distance": {}}
    for col, label in enumerate(DISTANCE_LABELS):
        x = values[:, col]
        out["by_distance"][label] = {
            "n": int(len(x)),
            "mean": json_float(np.mean(x)),
            "std": json_float(np.std(x, ddof=1)) if len(x) > 1 else None,
            "median": json_float(np.median(x)),
            "q25": json_float(np.quantile(x, 0.25)),
            "q75": json_float(np.quantile(x, 0.75)),
        }
    exact_y = values[:, :3].reshape(-1)
    exact_d = np.tile(np.arange(1, 4), len(values))
    all_y = values.reshape(-1)
    all_d = np.tile(np.arange(1, 5), len(values))
    out["spearman_exact_1_to_3"] = json_float(spearmanr(exact_d, exact_y).statistic)
    out["spearman_with_far_as_bucket_4"] = json_float(spearmanr(all_d, all_y).statistic)
    far = values[:, 3]
    out["paired_far_minus_d1_mean"] = json_float(np.mean(far - values[:, 0]))
    pooled_sd = np.std(np.concatenate((values[:, 0], far)), ddof=1)
    out["far_vs_d1_standardized_effect"] = json_float(
        np.mean(far - values[:, 0]) / pooled_sd if pooled_sd > 0 else float("nan")
    )
    return out


def node_hash_fold(nodes: np.ndarray, n_folds: int = 4) -> np.ndarray:
    """Deterministic SplitMix64-based fold assignment for node-disjoint probes."""
    z = nodes.astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    z = z ^ (z >> np.uint64(31))
    return (z % np.uint64(n_folds)).astype(np.int8)


def symmetric_pair_features(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.concatenate((np.abs(a - b), (a + b) * 0.5, a * b), axis=1)


def auc_with_orientation(y: np.ndarray, score: np.ndarray, orientation: float = 1.0) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, score * orientation))


def single_coordinate_probe(
    train_abs: np.ndarray,
    y_train: np.ndarray,
    test_abs: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    from sklearn.metrics import roc_auc_score

    train_aucs = np.asarray(
        [roc_auc_score(y_train, train_abs[:, j]) for j in range(train_abs.shape[1])]
    )
    effects = np.abs(train_aucs - 0.5)
    best = int(np.argmax(effects))
    orientation = 1.0 if train_aucs[best] >= 0.5 else -1.0
    test_auc = auc_with_orientation(y_test, test_abs[:, best], orientation)
    order = np.argsort(effects)[::-1][:10]
    return {
        "selected_dimension": best,
        "train_auc_oriented": float(0.5 + effects[best]),
        "test_auc_oriented": test_auc,
        "top10_train_dimensions": [int(v) for v in order],
        "top10_train_auc_oriented": [float(0.5 + effects[v]) for v in order],
    }


def logistic_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    penalty: str,
) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    solver = "liblinear" if penalty == "l1" else "lbfgs"
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            penalty=penalty,
            solver=solver,
            C=0.1 if penalty == "l1" else 1.0,
            max_iter=2_000,
            random_state=0,
        ),
    )
    model.fit(x_train, y_train)
    score = model.predict_proba(x_test)[:, 1]
    pred = score >= 0.5
    coef = model[-1].coef_[0]
    return {
        "test_roc_auc": float(roc_auc_score(y_test, score)),
        "test_balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "nonzero_coefficients": int(np.count_nonzero(np.abs(coef) > 1e-12)),
        "n_coefficients": int(len(coef)),
    }


def adjacent_vs_far_probe(blocks: list[Block], features: np.ndarray) -> dict[str, Any]:
    if len(blocks) < 100:
        return {"error": "fewer than 100 complete blocks"}
    anchors = np.asarray([b.anchor for b in blocks], dtype=np.int64)
    d1 = np.asarray([b.d1 for b in blocks], dtype=np.int64)
    far = np.asarray([b.far for b in blocks], dtype=np.int64)
    anchor_fold = node_hash_fold(anchors)
    d1_fold = node_hash_fold(d1)
    far_fold = node_hash_fold(far)

    train_pos = (anchor_fold != 0) & (d1_fold != 0)
    train_neg = (anchor_fold != 0) & (far_fold != 0)
    test_pos = (anchor_fold == 0) & (d1_fold == 0)
    test_neg = (anchor_fold == 0) & (far_fold == 0)

    a = features[:, 0]
    pos = features[:, 1]
    neg = features[:, 4]
    x_train = np.concatenate(
        (symmetric_pair_features(a[train_pos], pos[train_pos]),
         symmetric_pair_features(a[train_neg], neg[train_neg])),
        axis=0,
    )
    y_train = np.concatenate(
        (np.ones(train_pos.sum(), dtype=np.int8), np.zeros(train_neg.sum(), dtype=np.int8))
    )
    x_test = np.concatenate(
        (symmetric_pair_features(a[test_pos], pos[test_pos]),
         symmetric_pair_features(a[test_neg], neg[test_neg])),
        axis=0,
    )
    y_test = np.concatenate(
        (np.ones(test_pos.sum(), dtype=np.int8), np.zeros(test_neg.sum(), dtype=np.int8))
    )
    if min(np.bincount(y_train, minlength=2)) < 20 or min(np.bincount(y_test, minlength=2)) < 20:
        return {
            "error": "too few node-disjoint train/test pairs",
            "train_class_counts": np.bincount(y_train, minlength=2).astype(int).tolist(),
            "test_class_counts": np.bincount(y_test, minlength=2).astype(int).tolist(),
        }

    dim = int(a.shape[1])
    out: dict[str, Any] = {
        "task": "adjacent (1) versus >3/disconnected",
        "split": "fold 0 test; both endpoints must be in test; all other folds train",
        "train_class_counts_far_adjacent": np.bincount(y_train, minlength=2).astype(int).tolist(),
        "test_class_counts_far_adjacent": np.bincount(y_test, minlength=2).astype(int).tolist(),
    }
    out["single_absdiff_coordinate"] = single_coordinate_probe(
        x_train[:, :dim], y_train, x_test[:, :dim], y_test
    )
    out["sparse_logistic_pair_probe"] = logistic_probe(
        x_train, y_train, x_test, y_test, penalty="l1"
    )
    out["dense_logistic_pair_probe"] = logistic_probe(
        x_train, y_train, x_test, y_test, penalty="l2"
    )
    return out


def uniform_nonmissing_sample(
    x,
    n_nodes: int,
    n_target: int,
    rng: np.random.Generator,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    seen: set[int] = set()
    attempts = 0
    while sum(len(chunk) for chunk in rows) < n_target and attempts < 8:
        attempts += 1
        want = min(n_nodes, max(4 * (n_target - sum(len(c) for c in rows)), 10_000))
        candidates = rng.choice(n_nodes, size=want, replace=False)
        candidates = np.asarray([i for i in candidates if int(i) not in seen], dtype=np.int64)
        seen.update(int(i) for i in candidates)
        sampled = gather_rows(x, np.sort(candidates))
        sampled = sampled[np.abs(sampled).sum(axis=1) > 0]
        if len(sampled):
            rows.append(sampled)
    if not rows:
        return np.empty((0, int(x.shape[1])), dtype=np.float32)
    return np.concatenate(rows)[:n_target]


def random_pair_means(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> dict[str, Any]:
    n = min(len(a), len(b))
    if n < 2:
        return {"n": int(n), "mean_cosine_distance": None, "mean_euclidean_distance": None}
    ia = rng.permutation(len(a))[:n]
    ib = rng.permutation(len(b))[:n]
    xa, xb = a[ia], b[ib]
    denom = np.linalg.norm(xa, axis=1) * np.linalg.norm(xb, axis=1)
    denom[denom == 0] = 1.0
    cosine = 1.0 - (xa * xb).sum(axis=1) / denom
    euclidean = np.linalg.norm(xa - xb, axis=1)
    return {
        "n": int(n),
        "mean_cosine_distance": json_float(np.mean(cosine)),
        "std_cosine_distance": json_float(np.std(cosine, ddof=1)),
        "mean_euclidean_distance": json_float(np.mean(euclidean)),
        "std_euclidean_distance": json_float(np.std(euclidean, ddof=1)),
    }


def graph_result(
    name: str,
    path: Path,
    n_blocks: int,
    cross_graph_sample: int,
    attempts_per_block: int,
    rng: np.random.Generator,
) -> tuple[dict[str, Any], np.ndarray]:
    started = time.time()
    obj = load_graph(path)
    x = obj["x"]
    n_nodes = int(x.shape[0])
    edge_index = as_numpy(obj["edge_index"]).astype(np.int64, copy=False)
    log(f"  loaded N={n_nodes:,} E={edge_index.shape[1]:,} D={int(x.shape[1])}")
    adjacency = build_undirected_csr(edge_index, n_nodes)
    log(f"  built undirected CSR with {adjacency.nnz:,} entries")

    blocks, features = sample_complete_blocks(
        adjacency, x, n_nodes, n_blocks, rng, attempts_per_block
    )
    cosine, euclidean = pair_distances(features)
    cross_sample = uniform_nonmissing_sample(x, n_nodes, cross_graph_sample, rng)

    result = {
        "graph_path": str(path),
        "n_nodes": n_nodes,
        "n_directed_edges": int(edge_index.shape[1]),
        "n_undirected_adjacency_entries": int(adjacency.nnz),
        "feature_dim": int(x.shape[1]),
        "n_complete_blocks": len(blocks),
        "sampling": {
            "unit": "matched anchor block",
            "distances": list(DISTANCE_LABELS),
            "conditioning": "all five nodes have nonzero features",
            "short_path_direction": "undirected",
        },
        "cosine_distance": metric_summary(cosine),
        "euclidean_distance": metric_summary(euclidean),
        "adjacent_vs_far_feature_probe": adjacent_vs_far_probe(blocks, features),
        "elapsed_seconds": float(time.time() - started),
    }
    log(f"  finished {name} in {time.time() - started:.0f}s")
    return result, cross_sample


def parse_overrides(items: Iterable[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--graph-path must be KEY=RELATIVE_PATH, got {item!r}")
        key, value = item.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="docs/graph_catalog.json")
    parser.add_argument("--data-root", default="", help="override catalog data_root")
    parser.add_argument("--graphs", default=",".join(DEFAULT_DATASET_KEYS))
    parser.add_argument(
        "--graph-path", action="append", default=[], metavar="KEY=RELATIVE_PATH",
        help="override a catalog artifact path; repeatable",
    )
    parser.add_argument("--blocks", type=int, default=20_000)
    parser.add_argument("--cross-graph-sample", type=int, default=4_000)
    parser.add_argument("--attempts-per-block", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        default="scripts/experiments/analysis/path_feature_coupling/data/path_feature_coupling.json",
    )
    args = parser.parse_args()

    catalog_root, catalog_paths = load_catalog(Path(args.catalog))
    data_root = Path(args.data_root) if args.data_root else catalog_root
    paths = dict(catalog_paths)
    paths.update(parse_overrides(args.graph_path))
    names = [item.strip() for item in args.graphs.split(",") if item.strip()]

    result: dict[str, Any] = {
        "meta": {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "hostname": platform.node(),
            "git_commit": git_commit(),
            "catalog": str(args.catalog),
            "data_root": str(data_root),
            "seed": args.seed,
            "config": {
                "blocks": args.blocks,
                "cross_graph_sample": args.cross_graph_sample,
                "attempts_per_block": args.attempts_per_block,
                "graphs": names,
            },
        },
        "graphs": [],
        "per_graph": {},
        "random_pair_feature_distance": {},
    }
    samples: dict[str, np.ndarray] = {}
    master = np.random.default_rng(args.seed)

    for name in names:
        if name not in paths:
            log(f"SKIP {name}: no graph-catalog path")
            continue
        path = data_root / paths[name]
        if not path.exists():
            log(f"SKIP {name}: {path} does not exist")
            continue
        log(f"=== {name}: {path} ===")
        rng = np.random.default_rng(int(master.integers(1 << 31)))
        try:
            per_graph, sample = graph_result(
                name,
                path,
                args.blocks,
                args.cross_graph_sample,
                args.attempts_per_block,
                rng,
            )
        except Exception as exc:  # noqa: BLE001
            log(f"FAILED {name}: {type(exc).__name__}: {exc}")
            result["per_graph"][name] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        result["graphs"].append(name)
        result["per_graph"][name] = per_graph
        samples[name] = sample

    for i, left in enumerate(result["graphs"]):
        result["random_pair_feature_distance"][left] = {}
        for j, right in enumerate(result["graphs"]):
            rng = np.random.default_rng(args.seed + i * 10_000 + j)
            result["random_pair_feature_distance"][left][right] = random_pair_means(
                samples[left], samples[right], rng
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    log(f"WROTE {out_path} ({len(result['graphs'])} graphs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
