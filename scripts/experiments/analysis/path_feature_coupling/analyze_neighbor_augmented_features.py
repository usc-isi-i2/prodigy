#!/usr/bin/env python3
"""Compare raw node features with center-plus-sampled-neighbor-mean features.

For each uniformly sampled non-missing center node, construct

    z_v = concat(x_v, mean({x_u : u is a sampled undirected neighbor of v})).

The default fanout is 100 without replacement, matching the historical one-hop
PRODIGY neighborhood sampler. Missing neighbor features remain zero in the mean,
as they do when the model receives the sampled subgraph. The center node is not
included in the neighbor mean.

The runner compares raw, neighbor-mean-only, and concatenated spaces using the
same nodes and pair draws; repeats the matched exact-distance 1/2/3/far analysis;
fits held-out graph-identity probes; and exports separate 3D PCA coordinates for
an interactive raw-versus-augmented visualization.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from analyze_dimension_diagnostics import SAME_PIPELINE_KEYS
from analyze_path_feature_coupling import (
    DEFAULT_DATASET_KEYS,
    as_numpy,
    build_undirected_csr,
    gather_rows,
    git_commit,
    json_float,
    load_catalog,
    load_graph,
    log,
    metric_summary,
    pair_distances,
    parse_overrides,
    sample_complete_blocks,
)


SPACE_NAMES = ("raw_center", "neighbor_mean", "center_plus_neighbor_mean")


def feature_rows(x, indices: np.ndarray) -> np.ndarray:
    """Gather feature rows from either the graph's tensor or a NumPy test fixture."""
    if isinstance(x, np.ndarray):
        return x[indices].astype(np.float32, copy=False)
    return gather_rows(x, indices)


def sample_nonmissing_node_ids(
    x,
    n_nodes: int,
    n_target: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Uniformly sample node ids conditional on a nonzero feature row."""
    retained_ids: list[np.ndarray] = []
    retained_rows: list[np.ndarray] = []
    seen: set[int] = set()
    for _ in range(12):
        have = sum(len(part) for part in retained_ids)
        if have >= n_target:
            break
        want = min(n_nodes, max(10_000, 5 * (n_target - have)))
        candidates = rng.choice(n_nodes, size=want, replace=False).astype(np.int64)
        candidates = np.asarray(
            [node for node in candidates if int(node) not in seen], dtype=np.int64
        )
        seen.update(int(node) for node in candidates)
        rows = feature_rows(x, candidates)
        keep = np.abs(rows).sum(axis=1) > 0
        if np.any(keep):
            retained_ids.append(candidates[keep])
            retained_rows.append(rows[keep])
    if not retained_ids:
        return (
            np.empty(0, dtype=np.int64),
            np.empty((0, int(x.shape[1])), dtype=np.float32),
        )
    ids = np.concatenate(retained_ids)[:n_target]
    rows = np.concatenate(retained_rows, axis=0)[:n_target]
    return ids, rows


def sampled_neighbor_means(
    x,
    adjacency,
    nodes: np.ndarray,
    fanout: int,
    rng: np.random.Generator,
    center_chunk: int = 128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean up to ``fanout`` sampled neighbors for each center.

    Sampling is undirected, loop-free, without replacement. Zero/missing feature
    rows are included rather than filtered, which matches model input semantics.
    """
    if fanout <= 0:
        raise ValueError(f"fanout must be positive, got {fanout}")
    feature_dim = int(x.shape[1])
    means = np.zeros((len(nodes), feature_dim), dtype=np.float32)
    degrees = np.empty(len(nodes), dtype=np.int64)
    sampled_counts = np.empty(len(nodes), dtype=np.int64)

    for start in range(0, len(nodes), center_chunk):
        stop = min(start + center_chunk, len(nodes))
        selected_parts: list[np.ndarray] = []
        lengths: list[int] = []
        for local, node in enumerate(nodes[start:stop]):
            begin, end = adjacency.indptr[node], adjacency.indptr[node + 1]
            nbrs = adjacency.indices[begin:end]
            degrees[start + local] = len(nbrs)
            if len(nbrs) > fanout:
                selected = rng.choice(nbrs, size=fanout, replace=False).astype(np.int64)
            else:
                selected = nbrs.astype(np.int64, copy=False)
            selected_parts.append(selected)
            lengths.append(len(selected))
            sampled_counts[start + local] = len(selected)
        if not any(lengths):
            continue
        flat = np.concatenate([part for part in selected_parts if len(part)])
        rows = feature_rows(x, flat)
        offset = 0
        for local, length in enumerate(lengths):
            if length:
                means[start + local] = rows[offset : offset + length].mean(axis=0)
                offset += length
    return means, degrees, sampled_counts


def pair_indices(n_a: int, n_b: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    n = min(n_a, n_b)
    if n < 2:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    ia = rng.permutation(n_a)[:n]
    ib = rng.permutation(n_b)[:n]
    if n_a == n_b and np.array_equal(ia, ib):
        ib = np.roll(ib, 1)
    return ia.astype(np.int64), ib.astype(np.int64)


def distance_summary(rows_a: np.ndarray, rows_b: np.ndarray, ia: np.ndarray, ib: np.ndarray) -> dict[str, Any]:
    if not len(ia):
        return {"n": 0, "mean_cosine_distance": None, "mean_euclidean_distance": None}
    a, b = rows_a[ia], rows_b[ib]
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    valid = denom > 0
    cosine = np.full(len(a), np.nan, dtype=np.float64)
    cosine[valid] = 1.0 - np.sum(a[valid] * b[valid], axis=1) / denom[valid]
    euclidean = np.linalg.norm(a - b, axis=1)
    return {
        "n": int(len(a)),
        "n_nonzero_cosine": int(valid.sum()),
        "mean_cosine_distance": json_float(np.nanmean(cosine)),
        "std_cosine_distance": (
            json_float(np.nanstd(cosine, ddof=1)) if valid.sum() > 1 else None
        ),
        "mean_euclidean_distance": json_float(euclidean.mean()),
        "std_euclidean_distance": (
            json_float(euclidean.std(ddof=1)) if len(euclidean) > 1 else None
        ),
    }


def spaces(raw: np.ndarray, mean: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "raw_center": raw,
        "neighbor_mean": mean,
        "center_plus_neighbor_mean": np.concatenate((raw, mean), axis=1),
    }


def identity_probe(
    samples: dict[str, dict[str, np.ndarray]],
    graph_names: list[str],
    space_name: str,
    seed: int,
) -> dict[str, Any]:
    """Held-out multinomial linear graph-identity probe."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler, label_binarize

    parts = [samples[name][space_name] for name in graph_names]
    labels = [np.full(len(part), label, dtype=np.int16) for label, part in enumerate(parts)]
    x_all = np.concatenate(parts, axis=0)
    y_all = np.concatenate(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        x_all,
        y_all,
        test_size=0.30,
        random_state=seed,
        stratify=y_all,
    )
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=1_000, solver="lbfgs", random_state=seed),
    )
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    probability = model.predict_proba(x_test)
    binary = label_binarize(y_test, classes=np.arange(len(graph_names)))
    return {
        "graphs": graph_names,
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "chance_balanced_accuracy": float(1.0 / len(graph_names)),
        "test_balanced_accuracy": float(balanced_accuracy_score(y_test, prediction)),
        "test_macro_ovr_auc": float(
            roc_auc_score(binary, probability, average="macro", multi_class="ovr")
        ),
    }


def matched_path_metrics(
    x,
    adjacency,
    n_nodes: int,
    n_blocks: int,
    fanout: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    blocks, raw = sample_complete_blocks(
        adjacency, x, n_nodes, n_blocks, rng, attempts_per_block=200
    )
    if not blocks:
        return {"error": "no complete blocks"}
    flat = np.asarray([node for block in blocks for node in block.nodes()], dtype=np.int64)
    unique, inverse = np.unique(flat, return_inverse=True)
    mean, _, counts = sampled_neighbor_means(x, adjacency, unique, fanout, rng)
    mean = mean[inverse].reshape(len(blocks), 5, -1)
    augmented = np.concatenate((raw, mean), axis=2)
    result: dict[str, Any] = {
        "n_complete_blocks": int(len(blocks)),
        "sampled_neighbor_count_mean": float(counts.mean()),
        "sampled_neighbor_count_median": float(np.median(counts)),
    }
    for name, feature_blocks in (
        ("raw_center", raw),
        ("neighbor_mean", mean),
        ("center_plus_neighbor_mean", augmented),
    ):
        cosine, euclidean = pair_distances(feature_blocks)
        result[name] = {
            "cosine_distance": metric_summary(cosine),
            "euclidean_distance": metric_summary(euclidean),
        }
    return result


def pca_projection(
    samples: dict[str, dict[str, np.ndarray]],
    graph_names: list[str],
    space_name: str,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    from sklearn.decomposition import PCA

    matrix = np.concatenate([samples[name][space_name] for name in graph_names], axis=0)
    pca = PCA(n_components=3, svd_solver="randomized", random_state=seed)
    coordinates = pca.fit_transform(matrix).astype(np.float32)
    return coordinates, {
        "space": space_name,
        "explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_],
        "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
    }


def write_projection_csv(
    path: Path,
    graph_names: list[str],
    node_ids: dict[str, np.ndarray],
    degree: dict[str, np.ndarray],
    sampled_count: dict[str, np.ndarray],
    samples: dict[str, dict[str, np.ndarray]],
    projections: dict[str, np.ndarray],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "graph",
        "node_id",
        "degree",
        "sampled_neighbors",
        *[f"{space}_pc{component}" for space in SPACE_NAMES for component in (1, 2, 3)],
    ]
    offsets: dict[str, tuple[int, int]] = {}
    start = 0
    for graph in graph_names:
        stop = start + len(samples[graph]["raw_center"])
        offsets[graph] = (start, stop)
        start = stop
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for graph in graph_names:
            begin, end = offsets[graph]
            for local in range(end - begin):
                row: dict[str, Any] = {
                    "graph": graph,
                    "node_id": int(node_ids[graph][local]),
                    "degree": int(degree[graph][local]),
                    "sampled_neighbors": int(sampled_count[graph][local]),
                }
                for space in SPACE_NAMES:
                    for component in range(3):
                        row[f"{space}_pc{component + 1}"] = float(
                            projections[space][begin + local, component]
                        )
                writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="docs/graph_catalog.json")
    parser.add_argument("--data-root", default="")
    parser.add_argument("--graphs", default=",".join(DEFAULT_DATASET_KEYS))
    parser.add_argument("--graph-path", action="append", default=[], metavar="KEY=RELATIVE_PATH")
    parser.add_argument("--nodes-per-graph", type=int, default=2_000)
    parser.add_argument("--path-blocks", type=int, default=3_000)
    parser.add_argument("--fanout", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        default="scripts/experiments/analysis/path_feature_coupling/data/neighbor_augmented_features.json",
    )
    parser.add_argument(
        "--projection-out",
        default="scripts/experiments/analysis/path_feature_coupling/data/neighbor_augmented_3d.csv",
    )
    args = parser.parse_args()

    catalog_root, catalog_paths = load_catalog(Path(args.catalog))
    data_root = Path(args.data_root) if args.data_root else catalog_root
    paths = dict(catalog_paths)
    paths.update(parse_overrides(args.graph_path))
    names = [name.strip() for name in args.graphs.split(",") if name.strip()]
    master = np.random.default_rng(args.seed)

    result: dict[str, Any] = {
        "meta": {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "hostname": platform.node(),
            "git_commit": git_commit(),
            "data_root": str(data_root),
            "seed": args.seed,
            "representation": "concat(center feature, mean of sampled undirected neighbor features)",
            "neighbor_sampling": {
                "fanout": args.fanout,
                "replacement": False,
                "center_in_mean": False,
                "missing_neighbor_rows": "included as zero rows",
            },
            "config": vars(args),
        },
        "graphs": [],
        "per_graph": {},
        "random_pair_distances": {},
        "graph_identity": {},
        "projection": {},
    }
    graph_samples: dict[str, dict[str, np.ndarray]] = {}
    node_ids: dict[str, np.ndarray] = {}
    degrees: dict[str, np.ndarray] = {}
    sampled_counts: dict[str, np.ndarray] = {}

    for name in names:
        path = data_root / paths[name]
        if not path.exists():
            log(f"SKIP {name}: {path} missing")
            continue
        started = time.time()
        rng = np.random.default_rng(int(master.integers(1 << 31)))
        log(f"=== {name} ===")
        obj = load_graph(path)
        x = obj["x"]
        n_nodes = int(x.shape[0])
        edge_index = as_numpy(obj["edge_index"]).astype(np.int64, copy=False)
        adjacency = build_undirected_csr(edge_index, n_nodes)
        ids, raw = sample_nonmissing_node_ids(
            x, n_nodes, args.nodes_per_graph, rng
        )
        mean, degree, sampled = sampled_neighbor_means(
            x, adjacency, ids, args.fanout, rng
        )
        graph_samples[name] = spaces(raw, mean)
        node_ids[name] = ids
        degrees[name] = degree
        sampled_counts[name] = sampled
        path_metrics = matched_path_metrics(
            x, adjacency, n_nodes, args.path_blocks, args.fanout, rng
        )
        result["graphs"].append(name)
        result["per_graph"][name] = {
            "graph_path": str(path),
            "n_nodes": n_nodes,
            "n_edges": int(edge_index.shape[1]),
            "n_sampled_centers": int(len(ids)),
            "degree_mean": float(degree.mean()),
            "degree_median": float(np.median(degree)),
            "sampled_neighbor_count_mean": float(sampled.mean()),
            "sampled_neighbor_count_median": float(np.median(sampled)),
            "zero_neighbor_mean_fraction": float((np.abs(mean).sum(axis=1) == 0).mean()),
            "matched_path_distance": path_metrics,
            "elapsed_seconds": float(time.time() - started),
        }
        log(
            f"  sampled {len(ids):,} centers; mean sampled neighbors={sampled.mean():.1f}; "
            f"done in {time.time() - started:.0f}s"
        )
        del obj, x, edge_index, adjacency
        gc.collect()

    graph_names = result["graphs"]
    pair_rng = np.random.default_rng(args.seed + 30_000)
    for i, left in enumerate(graph_names):
        for j in range(i, len(graph_names)):
            right = graph_names[j]
            ia, ib = pair_indices(
                len(graph_samples[left]["raw_center"]),
                len(graph_samples[right]["raw_center"]),
                pair_rng,
            )
            key = f"{left}__{right}"
            result["random_pair_distances"][key] = {
                "pair_type": "within" if left == right else "between",
                **{
                    space: distance_summary(
                        graph_samples[left][space],
                        graph_samples[right][space],
                        ia,
                        ib,
                    )
                    for space in SPACE_NAMES
                },
            }

    same_pipeline = [name for name in SAME_PIPELINE_KEYS if name in graph_names]
    for scope, scoped_names in (
        ("all_graphs", graph_names),
        ("same_pipeline_graphs", same_pipeline),
    ):
        result["graph_identity"][scope] = {
            space: identity_probe(
                graph_samples, scoped_names, space, args.seed + 40_000
            )
            for space in SPACE_NAMES
        }

    projections: dict[str, np.ndarray] = {}
    for offset, space in enumerate(SPACE_NAMES):
        coordinates, metadata = pca_projection(
            graph_samples, graph_names, space, args.seed + 50_000 + offset
        )
        projections[space] = coordinates
        result["projection"][space] = metadata
    write_projection_csv(
        Path(args.projection_out),
        graph_names,
        node_ids,
        degrees,
        sampled_counts,
        graph_samples,
        projections,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    log(f"WROTE {out_path}")
    log(f"WROTE {args.projection_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
