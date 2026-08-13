#!/usr/bin/env python3
"""Test whether AUC on one graph predicts transfer AUC on another graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parents[4]
TRANSFER = ROOT / "scripts/experiments/analysis/nm_single_source_matrix_facebook/data/nm_single_source_matrix_9x9_long.csv"
OUT = Path(__file__).resolve().parent / "data"


def correlation(x, y, kind="pearson") -> float:
    fn = pearsonr if kind == "pearson" else spearmanr
    return float(fn(np.asarray(x), np.asarray(y)).statistic)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transfer", type=Path, default=TRANSFER)
    parser.add_argument("--metric", default="roc_auc")
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args()
    frame = pd.read_csv(args.transfer)
    auc = frame[frame.metric == args.metric].pivot(
        index="train", columns="test", values="value"
    )
    if auc.shape != (9, 9) or not np.isfinite(auc.to_numpy(float)).all():
        raise ValueError(f"expected a complete 9x9 {args.metric} matrix")
    names = list(auc.index)
    values = auc.to_numpy(float)
    n = len(names)

    # Literal prediction from one other foreign target: y_hat(s,t) = AUC(s,r).
    observed, held_out = [], []
    for source in range(n):
        for reference in range(n):
            for target in range(n):
                if len({source, reference, target}) == 3:
                    observed.append(values[source, reference])
                    held_out.append(values[source, target])

    # Each source's mean on its other seven foreign targets predicts the eighth.
    row_mean_prediction, row_mean_truth = [], []
    for source in range(n):
        for target in range(n):
            if source == target:
                continue
            other = [values[source, ref] for ref in range(n) if ref not in (source, target)]
            row_mean_prediction.append(float(np.mean(other)))
            row_mean_truth.append(values[source, target])

    # Target-pair correlations: do the seven common foreign donors retain rank?
    target_pair_rows = []
    for left in range(n):
        for right in range(left + 1, n):
            common_sources = [source for source in range(n) if source not in (left, right)]
            x, y = values[common_sources, left], values[common_sources, right]
            target_pair_rows.append({
                "target_a": names[left], "target_b": names[right],
                "n_common_foreign_sources": len(common_sources),
                "pearson": correlation(x, y), "spearman": correlation(x, y, "spearman"),
                "identity_mae": float(np.mean(np.abs(x - y))),
            })

    # Self AUC is a particularly tempting but distinct predictor.
    self_auc = np.diag(values)
    mean_foreign_auc = np.asarray([
        values[source, np.arange(n) != source].mean() for source in range(n)
    ])

    # Leave-one-cell-out two-way model. This is a matrix-completion upper baseline:
    # it observes other donors on the target, so it is not zero-shot target prediction.
    cells = [(s, t, values[s, t]) for s in range(n) for t in range(n) if s != t]
    additive_prediction, additive_truth = [], []
    for held_index, (held_source, held_target, truth) in enumerate(cells):
        train = [cell for index, cell in enumerate(cells) if index != held_index]
        x_train = np.asarray([[s, t] for s, t, _ in train])
        y_train = np.asarray([y for _, _, y in train])
        encoder = OneHotEncoder(categories=[range(n), range(n)], sparse_output=False)
        encoded = encoder.fit_transform(x_train)
        model = Ridge(alpha=1e-6).fit(encoded, y_train)
        additive_prediction.append(float(model.predict(encoder.transform([[held_source, held_target]]))[0]))
        additive_truth.append(float(truth))

    def metrics(prediction, truth):
        return {
            "n": len(truth),
            "pearson": correlation(prediction, truth),
            "spearman": correlation(prediction, truth, "spearman"),
            "mae": float(mean_absolute_error(truth, prediction)),
            "rmse": float(mean_squared_error(truth, prediction) ** 0.5),
        }

    pairs = pd.DataFrame(target_pair_rows)
    summary = {
        "transfer_matrix": str(args.transfer),
        "metric": args.metric,
        "n_graphs": n,
        "foreign_auc_range": [float(np.min(row_mean_truth)), float(np.max(row_mean_truth))],
        "one_other_graph_identity_prediction": metrics(observed, held_out),
        "mean_of_other_seven_graphs_prediction": metrics(row_mean_prediction, row_mean_truth),
        "target_pair_donor_rank_stability": {
            "n_target_pairs": len(pairs),
            "mean_pearson": float(pairs.pearson.mean()),
            "median_pearson": float(pairs.pearson.median()),
            "min_pearson": float(pairs.pearson.min()),
            "mean_spearman": float(pairs.spearman.mean()),
            "median_spearman": float(pairs.spearman.median()),
            "min_spearman": float(pairs.spearman.min()),
        },
        "self_auc_vs_mean_foreign_auc": {
            "n_sources": n,
            "pearson": correlation(self_auc, mean_foreign_auc),
            "spearman": correlation(self_auc, mean_foreign_auc, "spearman"),
            "identity_mae": float(np.mean(np.abs(self_auc - mean_foreign_auc))),
        },
        "two_way_source_plus_target_leave_one_cell_out": {
            **metrics(additive_prediction, additive_truth),
            "r2": float(r2_score(additive_truth, additive_prediction)),
            "warning": "Not zero-shot: the model observes other source models on the held-out cell's target.",
        },
        "interpretation": "AUC on another foreign graph is useful for donor ranking but is not an accurate uncalibrated forecast of absolute target AUC.",
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(args.out_dir / "target_pair_auc_correlations.csv", index=False)
    (args.out_dir / "auc_predictability_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
