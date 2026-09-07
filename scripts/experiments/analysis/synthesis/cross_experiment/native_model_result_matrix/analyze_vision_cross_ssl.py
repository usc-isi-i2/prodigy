#!/usr/bin/env python3
"""Validate and plot VISION's native feature-similarity cross-SSL matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CHECKPOINTS = (20, 60, 100, 300, 900)
TARGETS = (
    "covid_political",
    "election2020",
    "ukr_rus_suspended",
    "twibot20",
    "facebook_page_reference",
)
DISPLAY = {
    "covid_political": "COVID",
    "election2020": "Election",
    "ukr_rus_suspended": "Ukraine",
    "twibot20": "TwiBot",
    "facebook_page_reference": "Facebook",
}


def load_cells(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(root.rglob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    frame = pd.DataFrame(rows)
    if len(frame) != 125:
        raise ValueError(f"expected 125 VISION native cross-SSL cells, got {len(frame)}")
    if set(frame.task) != {"native_feature_similarity_ssl"}:
        raise ValueError("cross-SSL input contains downstream or non-native tasks")
    if set(frame.source) != set(TARGETS) or set(frame.target) != set(TARGETS):
        raise ValueError("VISION cross-SSL source/target panel changed")
    if set(frame.checkpoint_step) != set(CHECKPOINTS) or set(frame.training_seed) != {0}:
        raise ValueError("VISION cross-SSL trajectory contract changed")
    keys = ["source", "target", "checkpoint_step", "training_seed"]
    if frame.duplicated(keys).any():
        raise ValueError("duplicate VISION cross-SSL cells")
    fingerprints = frame.groupby("target").episode_fingerprint.nunique()
    if not (fingerprints == 1).all():
        raise ValueError(f"pseudo-episode drift: {fingerprints.to_dict()}")
    return frame


def plot_matrix(frame: pd.DataFrame, output: Path) -> None:
    terminal = frame[frame.checkpoint_step == 900]
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    for axis, metric, title, cmap in (
        (axes[0], "pseudo_classification_accuracy", "pseudo-task accuracy", "viridis"),
        (axes[1], "native_ssl_loss", "native SSL loss", "magma_r"),
    ):
        matrix = terminal.pivot(index="source", columns="target", values=metric).loc[TARGETS, TARGETS]
        values = matrix.to_numpy()
        image = axis.imshow(values, cmap=cmap, aspect="auto")
        axis.set_xticks(range(len(TARGETS)), [DISPLAY[value] for value in TARGETS], rotation=40, ha="right")
        axis.set_yticks(range(len(TARGETS)), [DISPLAY[value] for value in TARGETS])
        axis.set_xlabel("target graph pseudo-tasks")
        axis.set_ylabel("native specialist source")
        axis.set_title(title)
        for row, column in np.ndindex(values.shape):
            axis.text(column, row, f"{values[row, column]:.3f}", ha="center", va="center", fontsize=8)
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle("VISION native feature-similarity cross-SSL matrix (step 900, seed 0)")
    figure.tight_layout()
    figure.savefig(output.with_suffix(".png"), dpi=180, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def plot_trajectory(frame: pd.DataFrame, output: Path) -> None:
    summary = frame.groupby("checkpoint_step", as_index=False).agg(
        accuracy=("pseudo_classification_accuracy", "mean"),
        loss=("native_ssl_loss", "mean"),
    )
    figure, left = plt.subplots(figsize=(6.5, 4))
    right = left.twinx()
    left.plot(summary.checkpoint_step, summary.accuracy, marker="o", color="#4477AA", label="accuracy")
    right.plot(summary.checkpoint_step, summary.loss, marker="s", color="#CC6677", label="loss")
    left.set_xscale("log")
    left.set_xticks(CHECKPOINTS, [str(value) for value in CHECKPOINTS])
    left.set_xlabel("native optimizer updates")
    left.set_ylabel("mean pseudo-task accuracy", color="#4477AA")
    right.set_ylabel("mean native SSL loss", color="#CC6677")
    left.grid(alpha=0.25)
    figure.suptitle("VISION native cross-SSL trajectory (25 source→target pairs)")
    figure.tight_layout()
    figure.savefig(output.with_suffix(".png"), dpi=180, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root", type=Path,
        default=ROOT / "data" / "vision_native_cross_ssl_raw",
    )
    args = parser.parse_args()
    frame = load_cells(args.input_root)
    frame.to_csv(ROOT / "data" / "vision_native_cross_ssl_cells.csv", index=False)
    plot_matrix(frame, ROOT / "figures" / "vision_native_cross_ssl_matrix")
    plot_trajectory(frame, ROOT / "figures" / "vision_native_cross_ssl_trajectory")
    print("VISION_CROSS_SSL_OK cells=125 fixed_fingerprints=5")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
