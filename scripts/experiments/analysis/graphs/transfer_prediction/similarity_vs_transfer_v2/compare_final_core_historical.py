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


ROOT = next(p for p in Path(__file__).resolve().parents if (p / "AGENTS.md").is_file())
OLD = ROOT / "scripts/experiments/analysis/transfer/matrices/prodigy_nm/single_source/nm_single_source_matrix_facebook/data/nm_single_source_matrix_9x9_long.csv"
NEW = Path(__file__).resolve().parent / "data/final_core_matrix/specialist_cells_three_seed.csv"
DEFAULT_OUT = Path(__file__).resolve().parent / "data/final_core_matrix/comparison"


def rho(x, y) -> float:
    return float(spearmanr(np.asarray(x, float), np.asarray(y, float)).statistic)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical", type=Path, default=OLD)
    parser.add_argument("--final-core", type=Path, default=NEW)
    parser.add_argument("--historical-metric", default="roc_auc")
    parser.add_argument("--final-core-metric", default="accuracy")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    old = pd.read_csv(args.historical)
    old = old[old.metric == args.historical_metric].rename(
        columns={"train": "source", "test": "target", "value": "historical_value"}
    )[["source", "target", "historical_value"]]
    cells = pd.read_csv(args.final_core)
    if args.final_core_metric not in cells:
        raise ValueError(f"metric column not found: {args.final_core_metric}")
    new = (
        cells.groupby(["source", "target"], as_index=False)[args.final_core_metric].mean()
        .rename(columns={args.final_core_metric: "final_core_value"})
    )
    joined = old.merge(new, on=["source", "target"], validate="one_to_one")
    if len(joined) != 81:
        raise ValueError(f"expected 81 aligned cells, got {len(joined)}")
    foreign = joined[joined.source != joined.target].copy()

    target_rows = []
    for target, frame in foreign.groupby("target"):
        old_best = frame.loc[frame.historical_value.idxmax(), "source"]
        new_best = frame.loc[frame.final_core_value.idxmax(), "source"]
        target_rows.append({
            "target": target,
            "donor_rank_spearman": rho(frame.historical_value, frame.final_core_value),
            "historical_best_donor": old_best,
            "final_core_best_donor": new_best,
            "best_donor_agrees": old_best == new_best,
        })
    targets = pd.DataFrame(target_rows).sort_values("target")

    donor = foreign.groupby("source", as_index=False).agg(
        historical_mean_foreign_value=("historical_value", "mean"),
        final_core_mean_foreign_value=("final_core_value", "mean"),
    )
    seed_pair_rows = []
    for left, right in combinations(sorted(cells.seed.unique()), 2):
        for target in sorted(cells.target.unique()):
            frame = cells[(cells.target == target) & (cells.source != target)]
            a = frame[frame.seed == left].set_index("source")[args.final_core_metric]
            b = frame[frame.seed == right].set_index("source")[args.final_core_metric].reindex(a.index)
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
        "historical_metric": args.historical_metric,
        "final_core_metric": args.final_core_metric,
        "overall_foreign_cell_spearman": rho(
            foreign.historical_value, foreign.final_core_value
        ),
        "mean_target_donor_rank_spearman": float(targets.donor_rank_spearman.mean()),
        "targets_with_same_best_donor": int(targets.best_donor_agrees.sum()),
        "source_mean_foreign_rank_spearman": rho(
            donor.historical_mean_foreign_value, donor.final_core_mean_foreign_value
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
