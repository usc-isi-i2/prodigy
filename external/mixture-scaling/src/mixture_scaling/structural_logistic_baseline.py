from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx
import numpy as np
from scipy import sparse
from scipy.stats import rankdata

from .config import load_config
from .probe_strict import choose_trial, classifier
from .strict_data import induced_partition, load_raw, load_split, split_hash
from .strict_metrics import classification_metrics


def pagerank(adjacency: sparse.csr_matrix, damping: float = 0.85, iterations: int = 100) -> np.ndarray:
    n = adjacency.shape[0]
    out_degree = np.asarray(adjacency.sum(axis=1)).ravel()
    inverse = np.divide(1.0, out_degree, out=np.zeros_like(out_degree), where=out_degree > 0)
    rank = np.full(n, 1.0 / n)
    dangling = out_degree == 0
    for _ in range(iterations):
        updated = (1.0 - damping) / n + damping * (
            adjacency.T @ (rank * inverse) + rank[dangling].sum() / n
        )
        if np.abs(updated - rank).sum() < 1e-9:
            rank = updated
            break
        rank = updated
    return rank


def percentile_columns(values: np.ndarray) -> np.ndarray:
    if len(values) <= 1:
        return np.zeros_like(values, dtype=np.float64)
    return np.column_stack(
        [(rankdata(values[:, index], method="average") - 1.0) / (len(values) - 1.0) for index in range(values.shape[1])]
    )


def structural_features(edge_index, num_nodes: int) -> tuple[np.ndarray, list[str]]:
    source, target = edge_index.numpy()
    keep = source != target
    adjacency = sparse.csr_matrix(
        (np.ones(int(keep.sum()), dtype=np.float64), (source[keep], target[keep])),
        shape=(num_nodes, num_nodes),
    )
    adjacency.data[:] = 1.0
    adjacency.eliminate_zeros()
    out_degree = np.asarray(adjacency.sum(axis=1)).ravel()
    in_degree = np.asarray(adjacency.sum(axis=0)).ravel()
    reciprocal = np.asarray(adjacency.multiply(adjacency.T).sum(axis=1)).ravel()
    undirected = adjacency.maximum(adjacency.T).tocsr()
    undirected.data[:] = 1.0
    degree = np.asarray(undirected.sum(axis=1)).ravel()
    neighbor_sum = np.asarray(undirected @ degree).ravel()
    neighbor_square_sum = np.asarray(undirected @ np.square(degree)).ravel()
    neighbor_mean = np.divide(neighbor_sum, degree, out=np.zeros_like(degree), where=degree > 0)
    neighbor_variance = np.maximum(
        np.divide(neighbor_square_sum, degree, out=np.zeros_like(degree), where=degree > 0)
        - np.square(neighbor_mean),
        0.0,
    )
    neighbor_max = np.zeros(num_nodes, dtype=np.float64)
    rows = np.repeat(np.arange(num_nodes), np.diff(undirected.indptr))
    if len(rows):
        np.maximum.at(neighbor_max, rows, degree[undirected.indices])
    two_hop_walks = np.asarray(undirected @ degree).ravel()

    graph = nx.from_scipy_sparse_array(undirected, create_using=nx.Graph)
    triangles_dict = nx.triangles(graph)
    triangles = np.fromiter((triangles_dict[i] for i in range(num_nodes)), dtype=np.float64, count=num_nodes)
    clustering = np.divide(
        2.0 * triangles, degree * np.maximum(degree - 1.0, 0.0),
        out=np.zeros_like(degree), where=degree > 1,
    )
    core_dict = nx.core_number(graph)
    core = np.fromiter((core_dict[i] for i in range(num_nodes)), dtype=np.float64, count=num_nodes)
    ego_edges = degree + triangles
    ego_nodes = degree + 1.0
    ego_density = np.divide(
        2.0 * ego_edges, ego_nodes * np.maximum(ego_nodes - 1.0, 1.0),
        out=np.zeros_like(degree), where=ego_nodes > 1,
    )
    values = np.column_stack(
        [
            np.log1p(in_degree), np.log1p(out_degree), np.log1p(in_degree + out_degree),
            np.divide(in_degree, out_degree + 1.0),
            np.divide(reciprocal, out_degree, out=np.zeros_like(out_degree), where=out_degree > 0),
            np.log1p(degree), clustering, np.log1p(triangles), np.log1p(core),
            np.log1p(neighbor_mean), np.log1p(np.sqrt(neighbor_variance)), np.log1p(neighbor_max),
            np.log1p(two_hop_walks),
            np.divide(two_hop_walks, degree + 1.0),
            ego_density, pagerank(adjacency) * max(num_nodes, 1),
        ]
    )
    names = [
        "log_in_degree", "log_out_degree", "log_total_directed_degree", "in_out_ratio",
        "reciprocal_fraction", "log_undirected_degree", "clustering", "log_triangles",
        "log_core", "log_neighbor_degree_mean", "log_neighbor_degree_std", "log_neighbor_degree_max",
        "log_two_hop_walks", "two_hop_expansion", "egonet_density", "scaled_pagerank",
    ]
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    features = np.concatenate((values, percentile_columns(values)), axis=1).astype(np.float32)
    return features, names + [f"rank_{name}" for name in names]


def labeled_inputs(graph, structural: np.ndarray, include_existing: bool) -> tuple[np.ndarray, np.ndarray]:
    labels = graph.data.y.numpy()
    mask = labels >= 0
    features = structural
    if include_existing:
        features = np.concatenate((features, graph.data.x.numpy()), axis=1)
    return features[mask], labels[mask]


def select_head(train_x, train_y, validation_x, validation_y, c_values, seed: int) -> tuple[dict, list[dict]]:
    trials = []
    for c_value in map(float, c_values):
        head = classifier(c_value, seed)
        head.fit(train_x, train_y)
        trials.append({
            "c": c_value,
            **classification_metrics(
                validation_y, head.predict(validation_x), head.predict_proba(validation_x), head.classes_
            ),
        })
    return choose_trial(trials, "auc"), trials


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split-root", required=True)
    parser.add_argument("--target", default="twibot20")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    outputs = {
        False: output_dir / "structural_only.json",
        True: output_dir / "structural_plus_existing.json",
    }
    if any(path.exists() for path in outputs.values()):
        raise FileExistsError(f"refusing to overwrite structural baseline in {output_dir}")
    config = load_config(args.config)
    raw = load_raw(config["graphs"][args.target]["path"])
    split = load_split(Path(args.split_root) / f"{args.target}.pt", args.target, int(raw["x"].shape[0]))
    graphs, structure = {}, {}
    for partition in ("train", "validation"):
        graphs[partition] = induced_partition(args.target, config["graphs"][args.target]["path"], split, partition)
        structure[partition], feature_names = structural_features(
            graphs[partition].data.edge_index, int(graphs[partition].data.num_nodes)
        )
    selections = {}
    for include_existing in (False, True):
        train_x, train_y = labeled_inputs(graphs["train"], structure["train"], include_existing)
        validation_x, validation_y = labeled_inputs(
            graphs["validation"], structure["validation"], include_existing
        )
        chosen, trials = select_head(
            train_x, train_y, validation_x, validation_y, config["protocol"]["probe_c_values"], args.seed
        )
        selections[include_existing] = (chosen, trials, train_x, train_y, validation_x, validation_y)

    # Build structural test features and read test labels only after both model-selection procedures finish.
    test_graph = induced_partition(args.target, config["graphs"][args.target]["path"], split, "test")
    test_structure, _ = structural_features(test_graph.data.edge_index, int(test_graph.data.num_nodes))
    output_dir.mkdir(parents=True)
    for include_existing, output in outputs.items():
        chosen, trials, train_x, train_y, validation_x, validation_y = selections[include_existing]
        test_x, test_y = labeled_inputs(test_graph, test_structure, include_existing)
        head = classifier(float(chosen["c"]), args.seed)
        head.fit(np.concatenate((train_x, validation_x)), np.concatenate((train_y, validation_y)))
        result = {
            "status": "complete",
            "evaluation": "structural_plus_existing_logistic" if include_existing else "structural_only_logistic",
            "target": args.target,
            "target_split_hash": split_hash(split),
            "seed": args.seed,
            "selected_c": chosen["c"],
            "selection_metric": "validation_roc_auc_ovr_macro",
            "structural_feature_names": feature_names,
            "structural_feature_count": len(feature_names),
            "existing_feature_count": int(test_graph.data.x.shape[1]) if include_existing else 0,
            "validation_trials": trials,
            "train_nodes": int(len(train_y)), "validation_nodes": int(len(validation_y)),
            "test_nodes": int(len(test_y)),
            **classification_metrics(test_y, head.predict(test_x), head.predict_proba(test_x), head.classes_),
        }
        output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
