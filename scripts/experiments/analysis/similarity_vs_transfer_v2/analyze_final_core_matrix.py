#!/usr/bin/env python3
"""Rank transfer predictors against the strict three-seed final-core matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

try:
    from .analyze_predictors import (
        asymmetry_stat,
        scalar_matrix,
        selection_stats,
        within_target_stat,
    )
except ImportError:  # Direct script execution.
    from analyze_predictors import (
        asymmetry_stat,
        scalar_matrix,
        selection_stats,
        within_target_stat,
    )


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "scripts/experiments/analysis/graph_divergence/data/graph_divergence_data.json"
EXTENDED = Path(__file__).resolve().parent / "data/extended_predictors.json"
CELLS = Path(__file__).resolve().parent / "data/final_core_matrix/specialist_cells_three_seed.csv"
DEFAULT_OUT = Path(__file__).resolve().parent / "data/final_core_matrix/predictors"


def rowwise_spearman(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Spearman correlation between every row of x and one finite vector y."""
    xr = rankdata(x, axis=1)
    yr = rankdata(y)
    xr = xr - xr.mean(axis=1, keepdims=True)
    yr = yr - yr.mean()
    denominator = np.sqrt(np.sum(xr * xr, axis=1) * np.sum(yr * yr))
    return np.divide(
        xr @ yr,
        denominator,
        out=np.full(len(xr), np.nan, dtype=float),
        where=denominator > 0,
    )


def rowwise_spearman_masked(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Rowwise Spearman with a potentially different finite mask per row."""
    valid = np.isfinite(x) & np.isfinite(y)[None, :]
    masks, inverse = np.unique(valid, axis=0, return_inverse=True)
    result = np.full(len(x), np.nan, dtype=float)
    for mask_index, mask in enumerate(masks):
        rows = np.flatnonzero(inverse == mask_index)
        if mask.sum() < 3:
            continue
        result[rows] = rowwise_spearman(x[rows][:, mask], y[mask])
    return result


def permuted_pairwise_stats(
    predictor: np.ndarray, outcome: np.ndarray, orders: np.ndarray
) -> np.ndarray:
    """Vectorized equivalent of jointly permuting predictor graph identities."""
    by_target = []
    for target in range(len(outcome)):
        keep = np.arange(len(outcome)) != target
        x = predictor[orders[:, keep], orders[:, target, None]]
        by_target.append(rowwise_spearman_masked(x, outcome[keep, target]))
    stacked = np.column_stack(by_target)
    counts = np.sum(np.isfinite(stacked), axis=1)
    return np.divide(
        np.nansum(stacked, axis=1), counts,
        out=np.full(len(stacked), np.nan, dtype=float), where=counts > 0,
    )


def permuted_scalar_stats(
    values: np.ndarray, outcome: np.ndarray, mode: str, orders: np.ndarray
) -> np.ndarray:
    permuted = values[orders]
    by_target = []
    for target in range(len(outcome)):
        keep = np.arange(len(outcome)) != target
        if mode == "source":
            x = permuted[:, keep]
        elif mode == "absolute_gap":
            x = np.abs(permuted[:, keep] - permuted[:, target, None])
        else:
            raise ValueError(mode)
        by_target.append(rowwise_spearman(x, outcome[keep, target]))
    stacked = np.column_stack(by_target)
    counts = np.sum(np.isfinite(stacked), axis=1)
    return np.divide(
        np.nansum(stacked, axis=1), counts,
        out=np.full(len(stacked), np.nan, dtype=float), where=counts > 0,
    )


def permuted_asymmetry_stats(
    values: np.ndarray, outcome: np.ndarray, orders: np.ndarray
) -> np.ndarray:
    permuted = values[orders]
    left, right, y = [], [], []
    for a in range(len(values)):
        for b in range(a + 1, len(values)):
            left.append(a)
            right.append(b)
            y.append(outcome[a, b] - outcome[b, a])
    x = permuted[:, left] - permuted[:, right]
    return rowwise_spearman(x, np.asarray(y))


def permutation_p(null: np.ndarray, observed: float) -> float:
    finite = null[np.isfinite(null)]
    if not np.isfinite(observed) or not len(finite):
        return float("nan")
    return float(
        (np.sum(np.abs(finite) >= abs(observed) - 1e-12) + 1) / (len(finite) + 1)
    )


def matrix(
    frame: pd.DataFrame, graphs: list[str], metric: str = "accuracy",
    seed: int | None = None,
) -> np.ndarray:
    if metric not in frame:
        raise ValueError(f"metric column not found: {metric}")
    if seed is not None:
        frame = frame[frame.seed == seed]
    else:
        frame = frame.groupby(["source", "target"], as_index=False)[metric].mean()
    return (
        frame.pivot(index="source", columns="target", values=metric)
        .reindex(index=graphs, columns=graphs).to_numpy(float)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=Path, default=CELLS)
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--extended", type=Path, default=EXTENDED)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--metric", default="accuracy")
    parser.add_argument("--permutations", type=int, default=9_999)
    parser.add_argument("--seed", type=int, default=20260808)
    args = parser.parse_args()

    base = json.loads(args.base.read_text())
    extended = json.loads(args.extended.read_text())
    graphs = base["graphs"]
    if extended["graphs"] != graphs:
        raise ValueError("extended predictor graph order does not match base data")
    cells = pd.read_csv(args.cells)
    outcome = matrix(cells, graphs, args.metric)
    seed_outcomes = {
        seed: matrix(cells, graphs, args.metric, seed) for seed in (0, 1, 2)
    }
    if not np.isfinite(outcome).all() or not all(np.isfinite(x).all() for x in seed_outcomes.values()):
        raise ValueError("final-core matrix is incomplete after graph-name alignment")

    rng = np.random.default_rng(args.seed)
    orders = np.stack([rng.permutation(len(graphs)) for _ in range(args.permutations)])
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
            f"mean_target_spearman_{args.metric}": stat,
            "seed_0_rho": within_target_stat(predictor, seed_outcomes[0])[0],
            "seed_1_rho": within_target_stat(predictor, seed_outcomes[1])[0],
            "seed_2_rho": within_target_stat(predictor, seed_outcomes[2])[0],
            "graph_permutation_p_two_sided": permutation_p(
                permuted_pairwise_stats(predictor, outcome, orders), stat
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
            {"predictor": name, "target": graph, f"spearman_{args.metric}": value}
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
                f"mean_target_spearman_{args.metric}": stat,
                "seed_0_rho": within_target_stat(predictor, seed_outcomes[0])[0],
                "seed_1_rho": within_target_stat(predictor, seed_outcomes[1])[0],
                "seed_2_rho": within_target_stat(predictor, seed_outcomes[2])[0],
                "graph_permutation_p_two_sided": permutation_p(
                    permuted_scalar_stats(values, outcome, mode, orders), stat
                ),
                "targets_same_sign_as_mean": int(np.sum(np.sign(targets) == np.sign(stat))),
            }
            row.update(selection_stats(predictor, outcome, prefer_low=(stat < 0)))
            sink.append(row)
        observed = asymmetry_stat(values, outcome)
        direction_rows.append({
            "predictor": name,
            "kind": "signed_source_minus_target",
            f"spearman_with_{args.metric}_asymmetry": observed,
            "graph_permutation_p_two_sided": permutation_p(
                permuted_asymmetry_stats(values, outcome, orders), observed
            ),
        })

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pair_frame = pd.DataFrame(pair_rows).sort_values("priority_score", ascending=False)
    outputs = {
        "pairwise_predictors.csv": pair_frame,
        "targetwise_correlations.csv": pd.DataFrame(target_rows),
        "source_predictors.csv": pd.DataFrame(source_rows).sort_values(
            f"mean_target_spearman_{args.metric}", ascending=False
        ),
        "scalar_gap_predictors.csv": pd.DataFrame(gap_rows).sort_values(
            f"mean_target_spearman_{args.metric}"
        ),
        "direction_predictors.csv": pd.DataFrame(direction_rows).sort_values(
            f"spearman_with_{args.metric}_asymmetry", key=lambda x: x.abs(), ascending=False
        ),
    }
    for filename, frame in outputs.items():
        frame.to_csv(args.out_dir / filename, index=False)
    summary = {
        "graphs": graphs,
        "outcome": f"three-seed mean fixed-test episodic NM {args.metric}",
        "cells_per_seed": 81,
        "self_cells_excluded": True,
        "permutations": args.permutations,
        "permutation_unit": "joint graph identity",
        "permutation_draws_shared_across_predictors": True,
        "best_pairwise_by_priority": pair_frame.iloc[0].to_dict(),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
