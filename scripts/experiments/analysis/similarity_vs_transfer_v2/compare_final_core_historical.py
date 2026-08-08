#!/usr/bin/env python3
"""Compare the final-core three-seed matrix with the historical 9x9 matrix."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[4]
OLD = ROOT / "scripts/experiments/analysis/nm_single_source_matrix_facebook/data/nm_single_source_matrix_9x9_long.csv"
NEW = Path(__file__).resolve().parent / "data/final_core_matrix/specialist_cells_three_seed.csv"
DEFAULT_OUT = Path(__file__).resolve().parent / "data/final_core_matrix/comparison"


def rho(x, y) -> float:
    return float(spearmanr(np.asarray(x, float), np.asarray(y, float)).statistic)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical", type=Path, default=OLD)
    parser.add_argument("--final-core", type=Path, default=NEW)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    old = pd.read_csv(args.historical)
    old = old[old.metric == "roc_auc"].rename(
        columns={"train": "source", "test": "target", "value": "historical_auc"}
    )[["source", "target", "historical_auc"]]
    cells = pd.read_csv(args.final_core)
    new = (
        cells.groupby(["source", "target"], as_index=False).accuracy.mean()
        .rename(columns={"accuracy": "final_core_accuracy"})
    )
    joined = old.merge(new, on=["source", "target"], validate="one_to_one")
    if len(joined) != 81:
        raise ValueError(f"expected 81 aligned cells, got {len(joined)}")
    foreign = joined[joined.source != joined.target].copy()

    target_rows = []
    for target, frame in foreign.groupby("target"):
        old_best = frame.loc[frame.historical_auc.idxmax(), "source"]
        new_best = frame.loc[frame.final_core_accuracy.idxmax(), "source"]
        target_rows.append({
            "target": target,
            "donor_rank_spearman": rho(frame.historical_auc, frame.final_core_accuracy),
            "historical_best_donor": old_best,
            "final_core_best_donor": new_best,
            "best_donor_agrees": old_best == new_best,
        })
    targets = pd.DataFrame(target_rows).sort_values("target")

    donor = foreign.groupby("source", as_index=False).agg(
        historical_mean_foreign_auc=("historical_auc", "mean"),
        final_core_mean_foreign_accuracy=("final_core_accuracy", "mean"),
    )
    seed_pair_rows = []
    for left, right in combinations(sorted(cells.seed.unique()), 2):
        for target in sorted(cells.target.unique()):
            frame = cells[(cells.target == target) & (cells.source != target)]
            a = frame[frame.seed == left].set_index("source").accuracy
            b = frame[frame.seed == right].set_index("source").accuracy.reindex(a.index)
            seed_pair_rows.append({
                "seed_left": int(left),
                "seed_right": int(right),
                "target": target,
                "donor_rank_spearman": rho(a, b),
            })
    seed_pairs = pd.DataFrame(seed_pair_rows)

    summary = {
        "aligned_cells": len(joined),
        "foreign_cells": len(foreign),
        "overall_foreign_cell_spearman": rho(
            foreign.historical_auc, foreign.final_core_accuracy
        ),
        "mean_target_donor_rank_spearman": float(targets.donor_rank_spearman.mean()),
        "targets_with_same_best_donor": int(targets.best_donor_agrees.sum()),
        "source_mean_foreign_rank_spearman": rho(
            donor.historical_mean_foreign_auc, donor.final_core_mean_foreign_accuracy
        ),
        "mean_cross_seed_target_donor_rank_spearman": float(
            seed_pairs.donor_rank_spearman.mean()
        ),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    joined.to_csv(args.out_dir / "aligned_cells.csv", index=False)
    targets.to_csv(args.out_dir / "target_donor_rank_agreement.csv", index=False)
    donor.to_csv(args.out_dir / "source_strength_agreement.csv", index=False)
    seed_pairs.to_csv(args.out_dir / "cross_seed_target_rank_stability.csv", index=False)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
