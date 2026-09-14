#!/usr/bin/env python3
"""Render the node-only FP and LP source-to-target transfer matrices."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle


SOURCES = [
    "ukr_rus_twitter", "covid19_twitter", "midterm", "covid_political",
    "election2020", "ukr_rus_suspended", "twibot20", "cp_hk_twitter",
    "facebook_page_reference",
]
LP_TARGETS = [
    "ukr_rus_twitter", "covid19_twitter", "midterm", "twibot20",
    "cp_hk_twitter", "facebook_page_reference",
]
SHORT = {
    "ukr_rus_twitter": "UKR",
    "covid19_twitter": "COVID",
    "midterm": "Midterm",
    "covid_political": "CovPol",
    "election2020": "Election",
    "ukr_rus_suspended": "Susp.",
    "twibot20": "TwiBot",
    "cp_hk_twitter": "CP/HK",
    "facebook_page_reference": "Facebook",
}


def load_matrix(path: Path, targets: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t")
    if frame[["source", "target"]].duplicated().any():
        raise ValueError(f"duplicate source-target cell in {path}")
    matrix = frame.pivot(index="source", columns="target", values="value")
    if set(matrix.index) != set(SOURCES) or set(matrix.columns) != set(targets):
        raise ValueError(f"unexpected transfer matrix shape or labels in {path}")
    return matrix.loc[SOURCES, targets]


def annotate(ax, display: np.ndarray, color_values: np.ndarray, fmt: str) -> None:
    lo, hi = float(np.nanmin(color_values)), float(np.nanmax(color_values))
    span = max(hi - lo, 1e-12)
    for row in range(display.shape[0]):
        for col in range(display.shape[1]):
            normalized = (color_values[row, col] - lo) / span
            color = "white" if normalized > 0.58 else "black"
            ax.text(col, row, format(display[row, col], fmt), ha="center", va="center",
                    fontsize=8.2, color=color)


def finish_axes(ax, targets: list[str]) -> None:
    ax.set_xticks(np.arange(len(targets)), [SHORT[value] for value in targets], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(SOURCES)), [SHORT[value] for value in SOURCES])
    ax.set_xlabel("Evaluation target")
    ax.set_ylabel("Training source")
    ax.tick_params(length=0)
    for row, source in enumerate(SOURCES):
        if source in targets:
            col = targets.index(source)
            ax.add_patch(Rectangle((col - 0.49, row - 0.49), 0.98, 0.98,
                                   fill=False, edgecolor="black", linewidth=1.8))


def plot_fp(matrix: pd.DataFrame, output: Path) -> None:
    raw = matrix.to_numpy() * 100
    diagonal = np.array([matrix.loc[target, target] for target in matrix.columns]) * 100
    penalty = raw - diagonal[None, :]
    fig, ax = plt.subplots(figsize=(9.2, 7.0))
    image = ax.imshow(penalty, cmap="Blues", aspect="auto", vmin=0)
    annotate(ax, raw, penalty, ".2f")
    finish_axes(ax, list(matrix.columns))
    ax.set_title("Node-only feature-prediction transfer", loc="left", weight="bold", pad=32)
    ax.text(0, 1.015, "Cell text: scaled cosine error ×100 (lower is better) · Color: penalty vs target-trained model",
            transform=ax.transAxes, fontsize=9, color="0.35", va="bottom")
    colorbar = fig.colorbar(image, ax=ax, shrink=0.78, pad=0.025)
    colorbar.set_label("Transfer penalty (Δ error ×100)")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_lp(matrix: pd.DataFrame, output: Path) -> None:
    auc = matrix.to_numpy()
    fig, ax = plt.subplots(figsize=(7.6, 7.0))
    image = ax.imshow(auc, cmap="viridis", aspect="auto", vmin=0.5, vmax=max(0.75, float(auc.max())))
    annotate(ax, auc, auc, ".3f")
    finish_axes(ax, list(matrix.columns))
    ax.set_title("Node-only static link-prediction transfer", loc="left", weight="bold", pad=32)
    ax.text(0, 1.015, "ROC-AUC (higher is better) · Black boxes mark source=target cells",
            transform=ax.transAxes, fontsize=9, color="0.35", va="bottom")
    colorbar = fig.colorbar(image, ax=ax, shrink=0.78, pad=0.025)
    colorbar.set_label("ROC-AUC")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp-input", type=Path, required=True)
    parser.add_argument("--lp-input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    fp = load_matrix(args.fp_input, SOURCES)
    lp = load_matrix(args.lp_input, LP_TARGETS)
    plot_fp(fp, args.output_dir / "node_mlp_fp_transfer_heatmap.png")
    plot_lp(lp, args.output_dir / "node_mlp_lp_transfer_heatmap.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
