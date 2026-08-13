#!/usr/bin/env python3
"""Rank candidate predictors of the directed 9x9 NM transfer matrix.

The unit of resampling is a graph identity, never an individual matrix cell.
This preserves the dependence induced by every graph appearing as both source
and target.  Self-transfer cells are excluded from all headline statistics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


ROOT = next(p for p in Path(__file__).resolve().parents if (p / "AGENTS.md").is_file())
DEFAULT_TRANSFER = ROOT / "scripts/experiments/analysis/transfer/matrices/prodigy_nm/single_source/nm_single_source_matrix_facebook/data/nm_single_source_matrix_9x9_long.csv"
DEFAULT_DIVERGENCE = ROOT / "scripts/experiments/analysis/graph_characterization/statistics/graph_divergence/data/graph_divergence_data.json"
DEFAULT_OUT = Path(__file__).resolve().parent / "data"


def rho(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.unique(x[ok]).size < 2 or np.unique(y[ok]).size < 2:
        return np.nan
    return float(spearmanr(x[ok], y[ok]).statistic)


def transfer_matrix(frame: pd.DataFrame, graphs: list[str], metric: str) -> np.ndarray:
    part = frame[frame.metric == metric]
    return part.pivot(index="train", columns="test", values="value").reindex(index=graphs, columns=graphs).to_numpy(float)


def within_target_stat(predictor: np.ndarray, outcome: np.ndarray) -> tuple[float, list[float]]:
    values = []
    for target in range(len(outcome)):
        keep = np.arange(len(outcome)) != target
        values.append(rho(predictor[keep, target], outcome[keep, target]))
    return float(np.nanmean(values)), values


def graph_permutation_p(
    predictor: np.ndarray, outcome: np.ndarray, observed: float,
    rng: np.random.Generator, n_permutations: int,
) -> float:
    exceed = 0
    for _ in range(n_permutations):
        order = rng.permutation(len(outcome))
        value, _ = within_target_stat(predictor[np.ix_(order, order)], outcome)
        exceed += abs(value) >= abs(observed) - 1e-12
    return (exceed + 1) / (n_permutations + 1)


def selection_stats(score: np.ndarray, outcome: np.ndarray, prefer_low: bool) -> dict[str, float]:
    hits, regrets, ranks = 0, [], []
    for target in range(len(outcome)):
        candidates = np.flatnonzero(np.arange(len(outcome)) != target)
        candidates = candidates[np.isfinite(score[candidates, target]) & np.isfinite(outcome[candidates, target])]
        if len(candidates) < 2:
            continue
        chosen = candidates[np.argmin(score[candidates, target]) if prefer_low else np.argmax(score[candidates, target])]
        values = outcome[candidates, target]
        best = float(np.nanmax(values))
        selected = float(outcome[chosen, target])
        hits += bool(np.isclose(selected, best))
        ranks.append(float(rankdata(-values, method="average")[np.flatnonzero(candidates == chosen)[0]]))
        regrets.append(best - selected)
    return {"evaluable_targets": len(regrets), "top1_hits": hits,
            "top1_rate": hits / len(regrets) if regrets else np.nan,
            "mean_regret": float(np.mean(regrets)) if regrets else np.nan,
            "mean_selected_rank": float(np.mean(ranks)) if ranks else np.nan}


def scalar_matrix(values: np.ndarray, mode: str) -> np.ndarray:
    if mode == "source":
        return np.repeat(values[:, None], len(values), axis=1)
    if mode == "absolute_gap":
        return np.abs(values[:, None] - values[None, :])
    raise ValueError(mode)


def scalar_permutation_p(values, outcome, observed, mode, rng, n_permutations):
    exceed = 0
    for _ in range(n_permutations):
        stat, _ = within_target_stat(scalar_matrix(rng.permutation(values), mode), outcome)
        exceed += abs(stat) >= abs(observed) - 1e-12
    return (exceed + 1) / (n_permutations + 1)


def asymmetry_stat(values: np.ndarray, outcome: np.ndarray) -> float:
    xs, ys = [], []
    for a in range(len(values)):
        for b in range(a + 1, len(values)):
            xs.append(values[a] - values[b])
            ys.append(outcome[a, b] - outcome[b, a])
    return rho(np.asarray(xs), np.asarray(ys))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transfer", type=Path, default=DEFAULT_TRANSFER)
    parser.add_argument("--divergence", type=Path, default=DEFAULT_DIVERGENCE)
    parser.add_argument("--extended", type=Path, default=None,
                        help="optional JSON from compute_extended_predictors_tucker.py")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--permutations", type=int, default=9_999)
    parser.add_argument("--seed", type=int, default=20260807)
    args = parser.parse_args()

    divergence = json.loads(args.divergence.read_text())
    graphs = divergence["graphs"]
    transfer = pd.read_csv(args.transfer)
    auc = transfer_matrix(transfer, graphs, "roc_auc")
    accuracy = transfer_matrix(transfer, graphs, "accuracy")
    if not np.isfinite(auc).all():
        raise ValueError("transfer matrix is incomplete after graph-name alignment")
    rng = np.random.default_rng(args.seed)

    pairwise = dict(divergence["pairwise"])
    if args.extended:
        extended = json.loads(args.extended.read_text())
        if extended["graphs"] != graphs:
            raise ValueError("extended predictor graph order does not match graph_divergence")
        pairwise.update(extended["pairwise"])
    pair_rows, target_rows = [], []
    for name, raw in pairwise.items():
        matrix = np.asarray(raw, float)
        is_similarity = name in {"user_jaccard", "user_source_containment", "user_target_containment"}
        auc_stat, auc_targets = within_target_stat(matrix, auc)
        acc_stat, acc_targets = within_target_stat(matrix, accuracy)
        row = {"predictor": name, "kind": "pairwise_similarity" if is_similarity else "pairwise_distance", "graphs": len(graphs),
               "mean_target_spearman_auc": auc_stat, "mean_target_spearman_accuracy": acc_stat,
               "auc_graph_permutation_p_two_sided": graph_permutation_p(matrix, auc, auc_stat, rng, args.permutations),
               "targets_expected_direction": int(np.sum(np.asarray(auc_targets) > 0 if is_similarity else np.asarray(auc_targets) < 0))}
        row["targets_with_correlation"] = int(np.isfinite(auc_targets).sum())
        row["priority_score"] = abs(auc_stat) * row["targets_with_correlation"] / len(graphs) * (1 - row["auc_graph_permutation_p_two_sided"])
        row.update(selection_stats(matrix, auc, prefer_low=not is_similarity))
        pair_rows.append(row)
        target_rows.extend({"predictor": name, "target": graph, "spearman_auc": value}
                           for graph, value in zip(graphs, auc_targets))

    scalar_exclude = {"label_names", "has_labels", "indegree_ccdf", "outdegree_ccdf", "class_balance", "exact_edge_metrics_subsampled"}
    scalar_names = sorted(set.intersection(*[set(divergence["per_graph"][g]) for g in graphs]) - scalar_exclude)
    source_rows, gap_rows, direction_rows = [], [], []
    for name in scalar_names:
        try:
            values = np.asarray([divergence["per_graph"][g][name] for g in graphs], float)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(values).all() or np.unique(values).size < 3:
            continue
        for mode, sink in (("source", source_rows), ("absolute_gap", gap_rows)):
            matrix = scalar_matrix(values, mode)
            stat, targets = within_target_stat(matrix, auc)
            row = {"predictor": name, "kind": mode, "graphs": len(graphs),
                   "mean_target_spearman_auc": stat,
                   "auc_graph_permutation_p_two_sided": scalar_permutation_p(values, auc, stat, mode, rng, args.permutations),
                   "targets_same_sign_as_mean": int(np.sum(np.sign(targets) == np.sign(stat)))}
            row.update(selection_stats(matrix, auc, prefer_low=(stat < 0)))
            sink.append(row)
        observed = asymmetry_stat(values, auc)
        exceed = sum(abs(asymmetry_stat(rng.permutation(values), auc)) >= abs(observed) - 1e-12 for _ in range(args.permutations))
        direction_rows.append({"predictor": name, "kind": "signed_source_minus_target",
                               "unordered_pairs": len(graphs) * (len(graphs) - 1) // 2,
                               "spearman_with_auc_asymmetry": observed,
                               "graph_permutation_p_two_sided": (exceed + 1) / (args.permutations + 1)})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "pairwise_predictors.csv": pd.DataFrame(pair_rows).sort_values("priority_score", ascending=False),
        "targetwise_correlations.csv": pd.DataFrame(target_rows),
        "source_predictors.csv": pd.DataFrame(source_rows).sort_values("mean_target_spearman_auc", ascending=False),
        "scalar_gap_predictors.csv": pd.DataFrame(gap_rows).sort_values("mean_target_spearman_auc"),
        "direction_predictors.csv": pd.DataFrame(direction_rows).sort_values("spearman_with_auc_asymmetry", key=lambda x: x.abs(), ascending=False),
    }
    for filename, frame in outputs.items():
        frame.to_csv(args.out_dir / filename, index=False)

    summary = {
        "graphs": graphs, "n_graphs": len(graphs), "self_cells_excluded": True,
        "transfer_metrics": ["roc_auc", "accuracy"], "permutations": args.permutations,
        "permutation_unit": "joint graph identity",
        "best_pairwise_by_abs_auc_rho": max(pair_rows, key=lambda x: abs(x["mean_target_spearman_auc"])),
        "largest_graph_auc_selection": next(x for x in source_rows if x["predictor"] == "n_nodes"),
        "notes": ["Source/gap selection direction is chosen from its descriptive mean rho and is exploratory.",
                  "P-values test graph-label exchangeability; nine graphs still imply wide uncertainty and strong corpus confounding."],
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
