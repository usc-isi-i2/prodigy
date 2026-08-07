#!/usr/bin/env python3
"""Export the nested per-dimension JSON results as analysis-friendly CSV tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def write_node_distance(data: dict, path: Path) -> None:
    fields = [
        "graph",
        "dimension",
        "exact_distance_corr_absdiff",
        "exact_distance_corr_mean",
        "exact_distance_corr_product",
        "strongest_distance_term",
        "strongest_abs_distance_correlation",
        "edge_vs_uniform_auc_absdiff",
        "edge_vs_uniform_auc_mean",
        "edge_vs_uniform_auc_product",
        "best_edge_vs_uniform_term",
        "best_edge_vs_uniform_auc",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for graph in data["graphs"]:
            result = data["per_graph_distance"][graph]
            corr = result["uniform_connected_pair_feature_correlations"]["per_dimension"]
            auc = result["edge_vs_uniform_dimension_diagnostics"]["per_dimension"]
            for corr_row, auc_row in zip(corr, auc, strict=True):
                assert corr_row["dimension"] == auc_row["dimension"]
                writer.writerow(
                    {
                        "graph": graph,
                        "dimension": corr_row["dimension"],
                        **{
                            f"exact_distance_corr_{term}": corr_row[
                                "pearson_with_exact_distance"
                            ][term]
                            for term in ("absdiff", "mean", "product")
                        },
                        "strongest_distance_term": corr_row["strongest_term"],
                        "strongest_abs_distance_correlation": corr_row[
                            "strongest_abs_correlation"
                        ],
                        **{
                            f"edge_vs_uniform_auc_{term}": auc_row["test_oriented_auc"][term]
                            for term in ("absdiff", "mean", "product")
                        },
                        "best_edge_vs_uniform_term": auc_row["best_term"],
                        "best_edge_vs_uniform_auc": auc_row["best_test_auc"],
                    }
                )


def write_graph_identity(data: dict, path: Path) -> None:
    all_graphs = data["graphs"]
    fields = [
        "scope",
        "dimension",
        "train_eta_squared",
        "test_univariate_gaussian_balanced_accuracy",
        "chance_balanced_accuracy",
        "test_mean_oriented_ovr_auc",
        "test_max_oriented_ovr_auc",
        "best_predicted_graph",
        *[f"ovr_auc_{graph}" for graph in all_graphs],
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for scope, result in data["graph_identity"].items():
            for row in result["per_dimension"]:
                writer.writerow(
                    {
                        "scope": scope,
                        "dimension": row["dimension"],
                        "train_eta_squared": row["train_eta_squared"],
                        "test_univariate_gaussian_balanced_accuracy": row[
                            "test_univariate_gaussian_balanced_accuracy"
                        ],
                        "chance_balanced_accuracy": result["chance_balanced_accuracy"],
                        "test_mean_oriented_ovr_auc": row["test_mean_oriented_ovr_auc"],
                        "test_max_oriented_ovr_auc": row["test_max_oriented_ovr_auc"],
                        "best_predicted_graph": row["best_predicted_graph"],
                        **{
                            f"ovr_auc_{graph}": row["test_oriented_ovr_auc"].get(graph)
                            for graph in all_graphs
                        },
                    }
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="scripts/experiments/analysis/path_feature_coupling/data/dimension_diagnostics.json",
    )
    parser.add_argument(
        "--node-out",
        default="scripts/experiments/analysis/path_feature_coupling/data/node_distance_per_dimension.csv",
    )
    parser.add_argument(
        "--graph-out",
        default="scripts/experiments/analysis/path_feature_coupling/data/graph_identity_per_dimension.csv",
    )
    args = parser.parse_args()
    with Path(args.input).open(encoding="utf-8") as handle:
        data = json.load(handle)
    write_node_distance(data, Path(args.node_out))
    write_graph_identity(data, Path(args.graph_out))


if __name__ == "__main__":
    main()
