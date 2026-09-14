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
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 160,
    })
    fig, axes = plt.subplots(3, 3, figsize=(11.2, 9.0), sharex=True)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.10, top=0.86, hspace=0.42, wspace=0.34)
    max_penalty = max(
        float((matrix[target] - matrix.loc[target, target]).max()) for target in targets
    )

    for ax, target in zip(axes.flat, targets):
        penalties = matrix[target] - matrix.loc[target, target]
        ordered_sources = [target] + sorted(
            (source for source in targets if source != target), key=lambda source: penalties[source]
        )
        values = np.array([float(penalties[source]) for source in ordered_sources])
        positions = np.arange(len(ordered_sources))
        ax.axvline(0, color="0.35", linestyle="--", linewidth=1.0, zorder=1)
        ax.hlines(positions, 0, values, color="0.82", linewidth=1.0, zorder=1)
        ax.scatter(values[1:], positions[1:], s=35, color="#3b82b8", zorder=3)
        ax.scatter(
            values[0], positions[0], marker="D", s=43, facecolor="white",
            edgecolor="black", linewidth=1.0, zorder=4,
        )
        median = float(np.median(values[1:]))
        ax.axvline(median, color="black", linewidth=1.4, alpha=0.8, zorder=2)
        ax.set_title(f"Target: {SHORT[target]}", loc="left", fontsize=10, weight="bold")
        ax.set_yticks(positions, [SHORT[source] for source in ordered_sources])
        ax.invert_yaxis()
        ax.set_xlim(-0.06, max_penalty * 1.06)
        ax.grid(axis="x", linewidth=0.5, alpha=0.25)
        ax.tick_params(axis="y", length=0, labelsize=8)
        ax.tick_params(axis="x", labelsize=8)

    fig.supxlabel("Transfer penalty vs target-trained MLP (Δ scaled cosine error ×100; lower is better)", y=0.035)
    fig.suptitle(
        "Node-only feature-prediction transfer penalty", x=0.10, y=0.965,
        ha="left", weight="bold", fontsize=14,
    )
    fig.text(
        0.10, 0.92,
        "Sources are ordered best-to-worst within each target · Diamond: target-trained baseline · Vertical bar: foreign-source median",
        fontsize=8.5, color="0.35", ha="left",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
