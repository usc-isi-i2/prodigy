#!/usr/bin/env python3
"""Plot matched regression floors for Ukraine-suspended and twibot20."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np


BASELINES = ["raw_features", "raw_degree", "random_init"]
BASELINE_LABELS = {
    "raw_features": "raw bio features",
    "raw_degree": "raw directed degree",
    "random_init": "untrained encoder",
}
DATASETS = ["ukr_rus_suspended", "twibot20"]
DATASET_LABELS = {
    "ukr_rus_suspended": "ukr_susp",
    "twibot20": "twibot20",
}
TARGETS = ["followers_count", "statuses_count", "account_age_days"]
TARGET_LABELS = {
    "followers_count": "followers",
    "statuses_count": "statuses",
    "account_age_days": "account age",
}


def main() -> int:
    here = Path(__file__).resolve().parent
    with (here / "data/regression_baselines.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    values = {}
    for row in rows:
        for target in TARGETS:
            values[(row["baseline"], row["dataset"], target)] = float(row[target])
    columns = [
        (dataset, target)
        for dataset in DATASETS
        for target in TARGETS
    ]
    data = np.array([
        [
            values[(baseline, dataset, target)]
            for dataset, target in columns
        ]
        for baseline in BASELINES
    ])

    plt.rcParams.update({"font.size": 10, "figure.dpi": 140})
    fig, ax = plt.subplots(figsize=(9.2, 3.8), constrained_layout=True)
    image = ax.imshow(
        data,
        cmap="RdYlGn",
        norm=TwoSlopeNorm(vmin=-0.25, vcenter=0.0, vmax=0.45),
        aspect="auto",
    )
    ax.set_xticks(
        np.arange(len(columns)),
        [
            f"{DATASET_LABELS[dataset]}\n{TARGET_LABELS[target]}"
            for dataset, target in columns
        ],
    )
    ax.set_yticks(
        np.arange(len(BASELINES)),
        [BASELINE_LABELS[baseline] for baseline in BASELINES],
    )
    ax.set_xlabel("evaluation graph and profile target")
    ax.set_title(
        "10-shot node-regression floors\n"
        "Spearman by evaluation graph and profile target",
        fontsize=11,
    )
    for i in range(len(BASELINES)):
        for j in range(len(columns)):
            value = data[i, j]
            ax.text(
                j, i, f"{value:.3f}",
                ha="center", va="center",
                color=("white" if value < -0.15 or value > 0.35 else "black"),
            )
    ax.axvline(len(TARGETS) - 0.5, color="black", linewidth=2.0)
    ax.set_xticks(np.arange(-0.5, len(columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(BASELINES), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0, alpha=0.65)
    ax.tick_params(which="minor", bottom=False, left=False)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.05, pad=0.04)
    colorbar.set_label("Spearman")

    out_dir = here / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(out_dir / f"regression_baselines.{suffix}", dpi=180)
    plt.close(fig)
    print(f"wrote {out_dir}/regression_baselines.(png|pdf)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
