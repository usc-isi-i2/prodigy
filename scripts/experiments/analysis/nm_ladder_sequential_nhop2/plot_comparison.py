#!/usr/bin/env python3
"""Plot paired sequential-minus-interleaved AUC deltas by rung and test graph."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
DATASETS = [
    "ukr_rus_twitter", "covid19_twitter", "midterm", "covid_political",
    "election2020", "ukr_rus_suspended", "twibot20", "cp_hk_twitter",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        default=HERE / "data" / "nm_ladder_schedule_comparison_long.csv",
    )
    parser.add_argument(
        "--output", type=Path,
        default=HERE / "figures" / "sequential_minus_interleaved.png",
    )
    args = parser.parse_args()

    matrix = np.full((8, 8), np.nan)
    with args.input.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            matrix[int(row["rung"]) - 1, DATASETS.index(row["test_graph"])] = float(
                row["delta_sequential_minus_interleaved"]
            )

    limit = max(0.01, float(np.nanmax(np.abs(matrix))))
    fig, ax = plt.subplots(figsize=(10, 6))
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(8), [name.replace("_twitter", "") for name in DATASETS], rotation=40, ha="right")
    ax.set_yticks(range(8), range(1, 9))
    ax.set_xlabel("evaluation graph")
    ax.set_ylabel("ladder rung")
    ax.set_title("Blocked sequential − balanced interleaved (AUC)")
    fig.colorbar(image, ax=ax, label="Δ ROC-AUC")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
