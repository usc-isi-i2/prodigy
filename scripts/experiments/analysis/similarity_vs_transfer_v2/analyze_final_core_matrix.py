#!/usr/bin/env python3
"""Rank transfer predictors against the strict three-seed final-core matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_predictors import (
    asymmetry_stat,
    graph_permutation_p,
    scalar_matrix,
    scalar_permutation_p,
    selection_stats,
    within_target_stat,
)


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "scripts/experiments/analysis/graph_divergence/data/graph_divergence_data.json"
EXTENDED = Path(__file__).resolve().parent / "data/extended_predictors.json"
CELLS = Path(__file__).resolve().parent / "data/final_core_matrix/specialist_cells_three_seed.csv"
DEFAULT_OUT = Path(__file__).resolve().parent / "data/final_core_matrix/predictors"


def matrix(frame: pd.DataFrame, graphs: list[str], seed: int | None = None) -> np.ndarray:
    if seed is not None:
        frame = frame[frame.seed == seed]
    else:
        frame = frame.groupby(["source", "target"], as_index=False).accuracy.mean()
    return (
        frame.pivot(index="source", columns="target", values="accuracy")
        .reindex(index=graphs, columns=graphs).to_numpy(float)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=Path, default=CELLS)
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--extended", type=Path, default=EXTENDED)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--permutations", type=int, default=9_999)
    parser.add_argument("--seed", type=int, default=20260808)
    args = parser.parse_args()

    base = json.loads(args.base.read_text())
    extended = json.loads(args.extended.read_text())
    graphs = base["graphs"]
    if extended["graphs"] != graphs:
        raise ValueError("extended predictor graph order does not match base data")
    cells = pd.read_csv(args.cells)
    outcome = matrix(cells, graphs)
    seed_outcomes = {seed: matrix(cells, graphs, seed) for seed in (0, 1, 2)}
    if not np.isfinite(outcome).all() or not all(np.isfinite(x).all() for x in seed_outcomes.values()):
        raise ValueError("final-core matrix is incomplete after graph-name alignment")

    rng = np.random.default_rng(args.seed)
    pairwise = {**base["pairwise"], **extended["pairwise"]}
    pair_rows, target_rows = [], []
    for name, raw in pairwise.items():
        predictor = np.asarray(raw, float)
        if predictor.shape != outcome.shape:
            continue
        is_similarity = name in {"user_jaccard", "user_source_containment", "user_target_containment"}
        stat, targets = within_target_stat(predictor, outcome)
        row = {
            "predictor": name,
            "kind": "pairwise_similarity" if is_similarity else "pairwise_distance",
            "mean_target_spearman_accuracy": stat,
            "seed_0_rho": within_target_stat(predictor, seed_outcomes[0])[0],
            "seed_1_rho": within_target_stat(predictor, seed_outcomes[1])[0],
            "seed_2_rho": within_target_stat(predictor, seed_outcomes[2])[0],
            "graph_permutation_p_two_sided": graph_permutation_p(
                predictor, outcome, stat, rng, args.permutations
            ),
            "targets_expected_direction": int(
                np.sum(np.asarray(targets) > 0 if is_similarity else np.asarray(targets) < 0)
            ),
            "targets_with_correlation": int(np.isfinite(targets).sum()),
        }
        row["priority_score"] = (
            abs(stat) * row["targets_with_correlation"] / len(graphs)
            * (1 - row["graph_permutation_p_two_sided"])
        )
        row.update(selection_stats(predictor, outcome, prefer_low=not is_similarity))
        pair_rows.append(row)
        target_rows.extend(
            {"predictor": name, "target": graph, "spearman_accuracy": value}
            for graph, value in zip(graphs, targets)
        )

    exclude = {
        "label_names", "has_labels", "indegree_ccdf", "outdegree_ccdf",
        "class_balance", "exact_edge_metrics_subsampled",
    }
    scalar_names = sorted(
        set.intersection(*[set(base["per_graph"][graph]) for graph in graphs]) - exclude
    )
    source_rows, gap_rows, direction_rows = [], [], []
    for name in scalar_names:
        try:
            values = np.asarray([base["per_graph"][graph][name] for graph in graphs], float)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(values).all() or np.unique(values).size < 3:
            continue
        for mode, sink in (("source", source_rows), ("absolute_gap", gap_rows)):
            predictor = scalar_matrix(values, mode)
            stat, targets = within_target_stat(predictor, outcome)
            row = {
                "predictor": name,
                "kind": mode,
                "mean_target_spearman_accuracy": stat,
                "seed_0_rho": within_target_stat(predictor, seed_outcomes[0])[0],
                "seed_1_rho": within_target_stat(predictor, seed_outcomes[1])[0],
                "seed_2_rho": within_target_stat(predictor, seed_outcomes[2])[0],
                "graph_permutation_p_two_sided": scalar_permutation_p(
                    values, outcome, stat, mode, rng, args.permutations
                ),
                "targets_same_sign_as_mean": int(np.sum(np.sign(targets) == np.sign(stat))),
            }
            row.update(selection_stats(predictor, outcome, prefer_low=(stat < 0)))
            sink.append(row)
        observed = asymmetry_stat(values, outcome)
        exceed = sum(
            abs(asymmetry_stat(rng.permutation(values), outcome)) >= abs(observed) - 1e-12
            for _ in range(args.permutations)
        )
        direction_rows.append({
            "predictor": name,
            "kind": "signed_source_minus_target",
            "spearman_with_accuracy_asymmetry": observed,
            "graph_permutation_p_two_sided": (exceed + 1) / (args.permutations + 1),
        })

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pair_frame = pd.DataFrame(pair_rows).sort_values("priority_score", ascending=False)
    outputs = {
        "pairwise_predictors.csv": pair_frame,
        "targetwise_correlations.csv": pd.DataFrame(target_rows),
        "source_predictors.csv": pd.DataFrame(source_rows).sort_values(
            "mean_target_spearman_accuracy", ascending=False
        ),
        "scalar_gap_predictors.csv": pd.DataFrame(gap_rows).sort_values(
            "mean_target_spearman_accuracy"
        ),
        "direction_predictors.csv": pd.DataFrame(direction_rows).sort_values(
            "spearman_with_accuracy_asymmetry", key=lambda x: x.abs(), ascending=False
        ),
    }
    for filename, frame in outputs.items():
        frame.to_csv(args.out_dir / filename, index=False)
    summary = {
        "graphs": graphs,
        "outcome": "three-seed mean fixed-test episodic NM accuracy",
        "cells_per_seed": 81,
        "self_cells_excluded": True,
        "permutations": args.permutations,
        "permutation_unit": "joint graph identity",
        "best_pairwise_by_priority": pair_frame.iloc[0].to_dict(),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
