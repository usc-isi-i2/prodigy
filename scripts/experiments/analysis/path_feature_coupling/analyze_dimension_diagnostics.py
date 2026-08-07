#!/usr/bin/env python3
"""Per-dimension diagnostics for graph distance and graph identity.

This companion to ``analyze_path_feature_coupling.py`` answers two questions that
whole-vector cosine distance cannot:

1. For every feature dimension, how strongly do symmetric pair terms correlate
   with exact finite shortest-path distance inside each graph?
2. For every raw feature dimension, how well can that dimension identify which
   graph a node came from?

It also certifies whether an exact 1,000-hop pair can exist.  For every connected
component with at least 1,001 nodes, one exact BFS gives root eccentricity ``e``;
the component diameter is at most ``2e``.  Components smaller than 1,001 nodes
cannot contain a 1,000-hop pair by size alone.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse.csgraph import connected_components, dijkstra

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
    parse_overrides,
    uniform_nonmissing_sample,
)


SAME_PIPELINE_KEYS = (
    "covid19_twitter",
    "ukr_rus_twitter",
    "midterm",
    "cp_hk_twitter",
    "twibot20",
    "ukr_rus_suspended",
)
PAIR_TERMS = ("absdiff", "mean", "product")


def column_pearson(values: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Pearson correlation of each values column with a one-dimensional target."""
    x = values.astype(np.float64, copy=False)
    y = target.astype(np.float64, copy=False)
    yc = y - y.mean()
    xc = x - x.mean(axis=0, keepdims=True)
    numerator = yc @ xc
    denominator = np.sqrt((yc @ yc) * np.sum(xc * xc, axis=0))
    out = np.full(x.shape[1], np.nan, dtype=np.float64)
    np.divide(numerator, denominator, out=out, where=denominator > 0)
    return out


def component_distance_certificate(adjacency) -> dict[str, Any]:
    """Upper-bound component diameters tightly enough to rule on distance 1,000."""
    n_components, labels = connected_components(
        adjacency, directed=False, return_labels=True
    )
    sizes = np.bincount(labels)
    unique_labels, first_indices = np.unique(labels, return_index=True)
    first_by_label = dict(zip(unique_labels.astype(int), first_indices.astype(int)))
    large = np.flatnonzero(sizes >= 1_001)
    large_records: list[dict[str, Any]] = []
    exact_1000_possible = False
    largest_root_distances: np.ndarray | None = None
    largest_label = int(np.argmax(sizes))

    for label in large:
        root = first_by_label[int(label)]
        distances = dijkstra(
            adjacency, directed=False, indices=root, unweighted=True
        )
        finite = np.isfinite(distances)
        eccentricity = int(distances[finite].max())
        upper_bound = min(int(sizes[label]) - 1, 2 * eccentricity)
        contains_observed_1000 = bool(np.any(distances == 1_000))
        if upper_bound >= 1_000:
            exact_1000_possible = True
        large_records.append(
            {
                "component_label": int(label),
                "n_nodes": int(sizes[label]),
                "root": int(root),
                "root_eccentricity": eccentricity,
                "diameter_upper_bound": upper_bound,
                "root_has_node_at_distance_1000": contains_observed_1000,
            }
        )
        if int(label) == largest_label:
            largest_root_distances = distances

    max_small_size = int(sizes[sizes < 1_001].max()) if np.any(sizes < 1_001) else 0
    max_large_upper = max(
        (record["diameter_upper_bound"] for record in large_records), default=0
    )
    global_upper = max(max_small_size - 1, max_large_upper)
    certified_absent = global_upper < 1_000
    return {
        "n_components": int(n_components),
        "largest_component_label": largest_label,
        "largest_component_nodes": int(sizes[largest_label]),
        "n_components_with_at_least_1001_nodes": int(len(large)),
        "largest_small_component_nodes": max_small_size,
        "large_component_certificates": large_records,
        "global_diameter_upper_bound": int(global_upper),
        "exact_distance_1000_certified_absent": bool(certified_absent),
        "exact_distance_1000_not_ruled_out": bool(exact_1000_possible),
        "labels": labels,
        "largest_root_distances": largest_root_distances,
    }


def sample_nonmissing_anchors(
    x,
    labels: np.ndarray,
    component_label: int,
    n_anchors: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n_nodes = len(labels)
    retained: list[int] = []
    seen: set[int] = set()
    for _ in range(12):
        if len(retained) >= n_anchors:
            break
        candidates = rng.choice(
            n_nodes, size=min(n_nodes, max(20_000, 20 * n_anchors)), replace=False
        )
        candidates = np.asarray(
            [int(v) for v in candidates if int(v) not in seen and labels[v] == component_label],
            dtype=np.int64,
        )
        seen.update(int(v) for v in candidates)
        if not len(candidates):
            continue
        rows = gather_rows(x, candidates)
        observed = np.abs(rows).sum(axis=1) > 0
        retained.extend(int(v) for v in candidates[observed])
    return np.asarray(retained[:n_anchors], dtype=np.int64)


def sample_exact_distance_pairs(
    adjacency,
    labels: np.ndarray,
    component_label: int,
    anchors: np.ndarray,
    rng: np.random.Generator,
    targets_per_distance: int,
    candidate_targets: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    n_nodes = adjacency.shape[0]
    pair_anchor: list[np.ndarray] = []
    pair_target: list[np.ndarray] = []
    pair_distance: list[np.ndarray] = []
    eccentricities: list[int] = []
    exact_1000_candidates = 0

    for anchor in anchors:
        distances = dijkstra(
            adjacency, directed=False, indices=int(anchor), unweighted=True
        )
        finite = np.isfinite(distances)
        eccentricities.append(int(distances[finite].max()))

        # Distance 1 is rare under uniform node candidates, so take it directly.
        nbrs = adjacency.indices[
            adjacency.indptr[anchor] : adjacency.indptr[anchor + 1]
        ]
        if len(nbrs):
            selected = rng.choice(
                nbrs, size=min(targets_per_distance, len(nbrs)), replace=False
            ).astype(np.int64)
            pair_anchor.append(np.full(len(selected), anchor, dtype=np.int64))
            pair_target.append(selected)
            pair_distance.append(np.ones(len(selected), dtype=np.int32))

        candidates = rng.choice(
            n_nodes, size=min(n_nodes, candidate_targets), replace=False
        ).astype(np.int64)
        candidates = candidates[
            (labels[candidates] == component_label) & np.isfinite(distances[candidates])
        ]
        candidate_distances = distances[candidates].astype(np.int32)
        for distance in np.unique(candidate_distances):
            if distance < 2:
                continue
            at_distance = candidates[candidate_distances == distance]
            selected = rng.choice(
                at_distance,
                size=min(targets_per_distance, len(at_distance)),
                replace=False,
            ).astype(np.int64)
            pair_anchor.append(np.full(len(selected), anchor, dtype=np.int64))
            pair_target.append(selected)
            pair_distance.append(np.full(len(selected), distance, dtype=np.int32))

        if np.any(distances == 1_000):
            nodes_1000 = np.flatnonzero(distances == 1_000)
            exact_1000_candidates += int(len(nodes_1000))
            selected = rng.choice(
                nodes_1000,
                size=min(targets_per_distance, len(nodes_1000)),
                replace=False,
            ).astype(np.int64)
            pair_anchor.append(np.full(len(selected), anchor, dtype=np.int64))
            pair_target.append(selected)
            pair_distance.append(np.full(len(selected), 1_000, dtype=np.int32))

    if not pair_anchor:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, empty.astype(np.int32), {}
    return (
        np.concatenate(pair_anchor),
        np.concatenate(pair_target),
        np.concatenate(pair_distance),
        {
            "n_anchors": int(len(anchors)),
            "anchor_eccentricity_min": int(min(eccentricities)),
            "anchor_eccentricity_median": float(np.median(eccentricities)),
            "anchor_eccentricity_max": int(max(eccentricities)),
            "exact_distance_1000_candidates": int(exact_1000_candidates),
        },
    )


def pair_feature_diagnostics(
    x,
    anchors: np.ndarray,
    targets: np.ndarray,
    distances: np.ndarray,
) -> dict[str, Any]:
    if not len(anchors):
        return {"error": "no pairs sampled"}
    unique, inverse = np.unique(np.concatenate((anchors, targets)), return_inverse=True)
    rows = gather_rows(x, unique)
    observed = np.abs(rows).sum(axis=1) > 0
    n = len(anchors)
    a_pos = inverse[:n]
    t_pos = inverse[n:]
    keep = observed[a_pos] & observed[t_pos]
    af = rows[a_pos[keep]]
    tf = rows[t_pos[keep]]
    dist = distances[keep]
    terms = {
        "absdiff": np.abs(af - tf),
        "mean": (af + tf) * 0.5,
        "product": af * tf,
    }
    correlations = {name: column_pearson(values, dist) for name, values in terms.items()}

    by_distance: dict[str, Any] = {}
    anorm = np.linalg.norm(af, axis=1)
    tnorm = np.linalg.norm(tf, axis=1)
    denom = anorm * tnorm
    denom[denom == 0] = 1.0
    cosine = 1.0 - (af * tf).sum(axis=1) / denom
    euclidean = np.linalg.norm(af - tf, axis=1)
    for value in np.unique(dist):
        sel = dist == value
        by_distance[str(int(value))] = {
            "n": int(sel.sum()),
            "mean_cosine_distance": json_float(cosine[sel].mean()),
            "mean_euclidean_distance": json_float(euclidean[sel].mean()),
        }

    dimensions = []
    for dim in range(af.shape[1]):
        term_corr = {name: json_float(correlations[name][dim]) for name in PAIR_TERMS}
        finite_items = [(name, value) for name, value in term_corr.items() if value is not None]
        best_name, best_value = max(finite_items, key=lambda item: abs(item[1]))
        dimensions.append(
            {
                "dimension": dim,
                "pearson_with_exact_distance": term_corr,
                "strongest_term": best_name,
                "strongest_abs_correlation": abs(best_value),
            }
        )
    return {
        "n_pairs_before_feature_filter": int(len(anchors)),
        "n_pairs_nonmissing": int(keep.sum()),
        "distance_min": int(dist.min()),
        "distance_max": int(dist.max()),
        "exact_distance_1000_nonmissing_pairs": int((dist == 1_000).sum()),
        "by_distance": by_distance,
        "per_dimension": dimensions,
    }


def eta_squared(values: np.ndarray, labels: np.ndarray, classes: np.ndarray) -> np.ndarray:
    overall = values.mean(axis=0)
    between = np.zeros(values.shape[1], dtype=np.float64)
    for cls in classes:
        group = values[labels == cls]
        between += len(group) * (group.mean(axis=0) - overall) ** 2
    total = np.sum((values - overall) ** 2, axis=0)
    out = np.zeros(values.shape[1], dtype=np.float64)
    np.divide(between, total, out=out, where=total > 0)
    return out


def graph_identity_diagnostics(
    samples: dict[str, np.ndarray],
    graph_names: list[str],
    seed: int,
) -> dict[str, Any]:
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score

    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    train_labels: list[np.ndarray] = []
    test_labels: list[np.ndarray] = []
    for label, name in enumerate(graph_names):
        rows = samples[name]
        order = rng.permutation(len(rows))
        split = max(1, int(0.7 * len(rows)))
        train_parts.append(rows[order[:split]])
        test_parts.append(rows[order[split:]])
        train_labels.append(np.full(split, label, dtype=np.int16))
        test_labels.append(np.full(len(rows) - split, label, dtype=np.int16))
    x_train = np.concatenate(train_parts)
    x_test = np.concatenate(test_parts)
    y_train = np.concatenate(train_labels)
    y_test = np.concatenate(test_labels)
    classes = np.arange(len(graph_names), dtype=np.int16)
    eta = eta_squared(x_train, y_train, classes)

    per_class_auc = np.empty((len(graph_names), x_train.shape[1]), dtype=np.float64)
    for label in classes:
        binary = (y_test == label).astype(np.int8)
        for dim in range(x_test.shape[1]):
            auc = roc_auc_score(binary, x_test[:, dim])
            per_class_auc[label, dim] = 0.5 + abs(auc - 0.5)

    balanced_accuracy = np.empty(x_train.shape[1], dtype=np.float64)
    variances = np.empty((len(classes), x_train.shape[1]), dtype=np.float64)
    means = np.empty_like(variances)
    priors = np.empty(len(classes), dtype=np.float64)
    for label in classes:
        group = x_train[y_train == label]
        means[label] = group.mean(axis=0)
        variances[label] = np.maximum(group.var(axis=0), 1e-8)
        priors[label] = len(group) / len(x_train)
    for dim in range(x_train.shape[1]):
        scores = np.stack(
            [
                -0.5 * np.log(variances[label, dim])
                - 0.5 * (x_test[:, dim] - means[label, dim]) ** 2 / variances[label, dim]
                + math.log(priors[label])
                for label in classes
            ],
            axis=1,
        )
        balanced_accuracy[dim] = balanced_accuracy_score(y_test, scores.argmax(axis=1))

    dimensions = []
    for dim in range(x_train.shape[1]):
        best_label = int(np.argmax(per_class_auc[:, dim]))
        dimensions.append(
            {
                "dimension": dim,
                "train_eta_squared": float(eta[dim]),
                "test_univariate_gaussian_balanced_accuracy": float(balanced_accuracy[dim]),
                "test_mean_oriented_ovr_auc": float(per_class_auc[:, dim].mean()),
                "test_max_oriented_ovr_auc": float(per_class_auc[best_label, dim]),
                "best_predicted_graph": graph_names[best_label],
                "test_oriented_ovr_auc": {
                    name: float(per_class_auc[label, dim])
                    for label, name in enumerate(graph_names)
                },
            }
        )
    return {
        "graphs": graph_names,
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "chance_balanced_accuracy": 1.0 / len(graph_names),
        "per_dimension": dimensions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="docs/graph_catalog.json")
    parser.add_argument("--data-root", default="")
    parser.add_argument("--graphs", default=",".join(DEFAULT_DATASET_KEYS))
    parser.add_argument("--graph-path", action="append", default=[], metavar="KEY=RELATIVE_PATH")
    parser.add_argument("--anchors-large", type=int, default=8)
    parser.add_argument("--anchors-small", type=int, default=24)
    parser.add_argument("--large-node-threshold", type=int, default=1_000_000)
    parser.add_argument("--candidate-targets", type=int, default=200_000)
    parser.add_argument("--targets-per-distance", type=int, default=100)
    parser.add_argument("--graph-identity-sample", type=int, default=4_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        default="scripts/experiments/analysis/path_feature_coupling/data/dimension_diagnostics.json",
    )
    args = parser.parse_args()

    catalog_root, catalog_paths = load_catalog(Path(args.catalog))
    data_root = Path(args.data_root) if args.data_root else catalog_root
    paths = dict(catalog_paths)
    paths.update(parse_overrides(args.graph_path))
    names = [item.strip() for item in args.graphs.split(",") if item.strip()]
    master = np.random.default_rng(args.seed)
    result: dict[str, Any] = {
        "meta": {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "hostname": platform.node(),
            "git_commit": git_commit(),
            "data_root": str(data_root),
            "seed": args.seed,
            "config": vars(args),
        },
        "graphs": [],
        "per_graph_distance": {},
        "graph_identity": {},
    }
    graph_samples: dict[str, np.ndarray] = {}

    for name in names:
        path = data_root / paths[name]
        if not path.exists():
            log(f"SKIP {name}: {path} missing")
            continue
        log(f"=== {name} ===")
        started = time.time()
        rng = np.random.default_rng(int(master.integers(1 << 31)))
        obj = load_graph(path)
        x = obj["x"]
        n_nodes = int(x.shape[0])
        edge_index = as_numpy(obj["edge_index"]).astype(np.int64, copy=False)
        adjacency = build_undirected_csr(edge_index, n_nodes)
        certificate = component_distance_certificate(adjacency)
        labels = certificate.pop("labels")
        certificate.pop("largest_root_distances", None)
        n_anchors = args.anchors_large if n_nodes >= args.large_node_threshold else args.anchors_small
        anchors = sample_nonmissing_anchors(
            x, labels, certificate["largest_component_label"], n_anchors, rng
        )
        pair_a, pair_t, pair_d, sample_meta = sample_exact_distance_pairs(
            adjacency,
            labels,
            certificate["largest_component_label"],
            anchors,
            rng,
            args.targets_per_distance,
            args.candidate_targets,
        )
        diagnostics = pair_feature_diagnostics(x, pair_a, pair_t, pair_d)
        graph_samples[name] = uniform_nonmissing_sample(
            x, n_nodes, args.graph_identity_sample, rng
        )
        result["graphs"].append(name)
        result["per_graph_distance"][name] = {
            "graph_path": str(path),
            "n_nodes": n_nodes,
            "n_edges": int(edge_index.shape[1]),
            "component_distance_certificate": certificate,
            "pair_sampling": sample_meta,
            "exact_finite_distance_features": diagnostics,
            "elapsed_seconds": float(time.time() - started),
        }
        log(
            f"  done in {time.time()-started:.0f}s; global diameter upper "
            f"{certificate['global_diameter_upper_bound']}; pairs "
            f"{diagnostics.get('n_pairs_nonmissing', 0):,}"
        )
        del obj, x, edge_index, adjacency, labels
        gc.collect()

    result["graph_identity"]["all_graphs"] = graph_identity_diagnostics(
        graph_samples, result["graphs"], args.seed + 10_000
    )
    same_pipeline = [name for name in SAME_PIPELINE_KEYS if name in result["graphs"]]
    result["graph_identity"]["same_pipeline_graphs"] = graph_identity_diagnostics(
        graph_samples, same_pipeline, args.seed + 20_000
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    log(f"WROTE {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
