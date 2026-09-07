#!/usr/bin/env python3
"""Plot native NM and downstream classification trajectories by target graph."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ARCHITECTURES = ("prodigy", "vision", "gilt")
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
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def trajectory(
    rows: list[dict[str, str]], architecture: str, target: str, task: str
) -> list[float]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        if (
            row["architecture"] == architecture
            and row["dataset"] == target
            and row["task"] == task
        ):
            grouped[int(row["checkpoint_step"])].append(float(row["roc_auc"]))

    expected_counts = {step: 1 for step in STEPS}
    if task == "classification":
        expected_counts.update({step: 5 for step in STEPS if step > 0})
    observed_counts = {step: len(grouped[step]) for step in STEPS}
    if observed_counts != expected_counts:
        raise ValueError(
            f"unexpected cells for {architecture}/{target}/{task}: "
            f"{observed_counts} != {expected_counts}"
        )
    return [float(np.mean(grouped[step])) for step in STEPS]


def plot_architecture(rows: list[dict[str, str]], architecture: str, output_root: Path) -> None:
    positions = np.arange(len(STEPS))
    trajectories = {
        (target, task): trajectory(rows, architecture, target, task)
        for target, _ in TARGETS
        for task in ("neighbor_matching", "classification")
    }
    all_values = [value for values in trajectories.values() for value in values]
    lo, hi = min(all_values), max(all_values)
    pad = max(0.02, (hi - lo) * 0.08)

    fig, axes = plt.subplots(1, len(TARGETS), figsize=(18, 3.9), sharey=True)
    for index, (axis, (target, title)) in enumerate(zip(axes, TARGETS)):
        axis.plot(
            positions,
            trajectories[(target, "neighbor_matching")],
            linewidth=2.3,
            color="#4477AA",
            label="Native/self NM AUC",
        )
        axis.plot(
            positions,
            trajectories[(target, "classification")],
            linewidth=2.3,
            color="#EE6677",
            label="Target CLS AUC (mean over sources)",
        )
        axis.set_title(title)
        axis.set_xticks(positions, [str(step) for step in STEPS])
        axis.set_xlabel("Training checkpoint")
        axis.grid(axis="y", alpha=0.25)
        axis.set_ylim(max(0, lo - pad), min(1, hi + pad))
        if index == 0:
            axis.set_ylabel("ROC-AUC")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle(
        f"{architecture.upper()}: native NM and classification by target graph", y=0.99
    )
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.94))
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    output_root.mkdir(parents=True, exist_ok=True)
    stem = output_root / f"{architecture}_nm_cls_by_target_900_seed0"
    fig.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    rows = load_rows(args.results)
    for architecture in ARCHITECTURES:
        plot_architecture(rows, architecture, args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
