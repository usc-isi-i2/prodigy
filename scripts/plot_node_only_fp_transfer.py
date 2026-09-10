#!/usr/bin/env python3
"""Plot node-only feature-prediction transfer against each target diagonal."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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

LABEL_OFFSETS = {
    "ukr_rus_twitter": (7, 5),
    "covid19_twitter": (7, 5),
    "midterm": (7, -12),
    "covid_political": (8, 7),
    "election2020": (-2, -18),
    "ukr_rus_suspended": (7, -9),
    "twibot20": (7, -9),
    "cp_hk_twitter": (7, -12),
    "facebook_page_reference": (10, -8),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.input, sep="\t")
    required = {"source", "target", "value"}
    if not required <= set(frame):
        raise ValueError(f"missing columns: {sorted(required - set(frame))}")
    if len(frame) != 81 or frame[["source", "target"]].duplicated().any():
        raise ValueError("expected a unique 9x9 transfer matrix")

    matrix = frame.pivot(index="source", columns="target", values="value") * 100
    targets = list(SHORT)
    if set(matrix.index) != set(targets) or set(matrix.columns) != set(targets):
        raise ValueError("graph set does not match the Social-9 contract")

    plt.rcParams.update({
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 160,
    })
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.12, top=0.86)
    colors = plt.get_cmap("tab10")(np.arange(len(targets)))
    all_values = matrix.to_numpy().ravel()
    lo, hi = float(all_values.min()), float(all_values.max())
    pad = 0.04 * (hi - lo)
    limits = (lo - pad, hi + pad)

    for color, target in zip(colors, targets):
        diagonal = float(matrix.loc[target, target])
        transfer = matrix[target].drop(index=target)
        ax.scatter(
            np.full(len(transfer), diagonal), transfer.to_numpy(),
            s=34, color=color, alpha=0.68, edgecolors="none", zorder=2,
        )
        ax.scatter(
            diagonal, diagonal, marker="D", s=62, color=color,
            edgecolors="black", linewidths=0.65, zorder=4,
        )
        ax.annotate(
            SHORT[target], (diagonal, diagonal), xytext=LABEL_OFFSETS[target],
            textcoords="offset points", color=color, fontsize=8.5,
        )

    ax.plot(limits, limits, linestyle="--", linewidth=1.1, color="0.35", zorder=1)
    ax.text(18.5, 18.9, "equal to in-domain", color="0.35", fontsize=8.5, rotation=45)
    ax.set(xlim=limits, ylim=limits)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.5, alpha=0.25)
    ax.set_xlabel("Target in-domain error (scaled cosine ×100)")
    ax.set_ylabel("Model error on target (scaled cosine ×100)")
    fig.suptitle(
        "Node-only feature-prediction transfer", x=0.13, y=0.975,
        ha="left", weight="bold", fontsize=14,
    )
    fig.text(
        0.13, 0.925,
        "Circles: cross-source models · Diamonds: target-trained model · Lower is better",
        fontsize=8.5, color="0.35", ha="left",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
