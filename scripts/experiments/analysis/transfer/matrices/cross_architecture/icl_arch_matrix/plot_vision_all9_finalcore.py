#!/usr/bin/env python3
"""Plot the VISION specialist trajectory with the all-nine final-core point."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


STEPS = (0, 20, 60, 100, 300, 900)
TARGETS = (
    ("covid_political", "COVID political"),
    ("election2020", "Election 2020"),
    ("ukr_rus_suspended", "UKR/RUS suspended"),
    ("twibot20", "TwiBot-20"),
    ("facebook_page_reference", "Facebook pages"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specialist-summary", required=True, type=Path)
    parser.add_argument("--all9-results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with args.specialist_summary.open(encoding="utf-8", newline="") as handle:
        summary = list(csv.DictReader(handle, delimiter="\t"))
    all9 = {
        row["dataset"]: float(row["roc_auc"])
        for row in (
            json.loads(line)
            for line in args.all9_results.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    if set(all9) != {target for target, _ in TARGETS}:
        raise ValueError(f"unexpected all-nine target set: {sorted(all9)}")

    curves = {}
    for target, _ in TARGETS:
        by_step = {
            int(row["checkpoint_step"]): float(row["mean_roc_auc"])
            for row in summary
            if row["architecture"] == "vision" and row["target"] == target
        }
        if set(by_step) != set(STEPS):
            raise ValueError(f"incomplete specialist trajectory for {target}: {sorted(by_step)}")
        curves[target] = [by_step[step] for step in STEPS]

    positions = np.arange(len(STEPS) + 1)
    values = [value for curve in curves.values() for value in curve] + list(all9.values())
    lo, hi = min(values), max(values)
    pad = max(0.015, (hi - lo) * 0.08)

    fig, axes = plt.subplots(1, len(TARGETS), figsize=(18, 3.8), sharey=True)
    for index, (axis, (target, title)) in enumerate(zip(axes, TARGETS)):
        axis.plot(
            positions[:-1],
            curves[target],
            color="#228833",
            linewidth=2.2,
            marker="o",
            markersize=4.5,
            label="Single-source mean",
        )
        axis.scatter(
            positions[-1],
            all9[target],
            color="#AA3377",
            marker="D",
            s=58,
            zorder=4,
            label="All-nine mixture",
        )
        axis.set_title(title)
        axis.set_xticks(positions, [*map(str, STEPS), "2500"])
        axis.set_xlabel("Training checkpoint")
        axis.set_ylim(max(0.0, lo - pad), min(1.0, hi + pad))
        axis.grid(axis="y", alpha=0.25)
        if index == 0:
            axis.set_ylabel("Classification ROC-AUC")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("VISION native pretraining: specialist trajectories and all-nine final-core model", y=0.99)
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.925))
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
