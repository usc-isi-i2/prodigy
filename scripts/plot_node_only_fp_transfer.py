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
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 160,
    })
    fig, ax = plt.subplots(figsize=(9.4, 6.4))
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.27, top=0.84)
    colors = plt.get_cmap("tab10")(np.arange(len(targets)))

    source_offsets = np.linspace(-0.28, 0.28, len(targets))
    handles = []
    for source_index, (color, source) in enumerate(zip(colors, targets)):
        xs, penalties = [], []
        for target_index, target in enumerate(targets):
            if source == target:
                continue
            xs.append(target_index + source_offsets[source_index])
            penalties.append(float(matrix.loc[source, target] - matrix.loc[target, target]))
        handle = ax.scatter(
            xs, penalties, s=39, color=color, alpha=0.78, edgecolors="none",
            label=SHORT[source], zorder=3,
        )
        handles.append(handle)

    medians = []
    for target in targets:
        penalties = matrix[target].drop(index=target) - matrix.loc[target, target]
        medians.append(float(penalties.median()))
    for index, median in enumerate(medians):
        ax.plot([index - 0.34, index + 0.34], [median, median], color="black", linewidth=2.1, zorder=4)
    ax.scatter(
        np.arange(len(targets)), np.zeros(len(targets)), marker="D", s=36,
        facecolor="white", edgecolor="black", linewidth=1, label="Target-trained baseline", zorder=5,
    )

    ax.axhline(0, linestyle="--", linewidth=1.1, color="0.35", zorder=1)
    ax.set_xlim(-0.55, len(targets) - 0.45)
    ax.set_ylim(-0.08, None)
    ax.set_xticks(np.arange(len(targets)), [SHORT[target] for target in targets], rotation=28, ha="right")
    ax.grid(axis="y", linewidth=0.5, alpha=0.3)
    ax.set_xlabel("Target graph")
    ax.set_ylabel("Transfer penalty vs target-trained MLP\n(Δ scaled cosine error ×100; lower is better)")
    fig.suptitle(
        "Node-only feature-prediction transfer penalty", x=0.09, y=0.97,
        ha="left", weight="bold", fontsize=14,
    )
    fig.text(
        0.09, 0.91,
        "Each circle is a source-trained MLP on a target; black bars mark target medians",
        fontsize=8.5, color="0.35", ha="left",
    )
    legend = ax.legend(
        handles=ax.collections[:len(targets)] + [ax.collections[-1]],
        loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=5,
        frameon=False, fontsize=8.2, handletextpad=0.35, columnspacing=1.1,
    )
    legend.set_title("Training source", prop={"size": 8.2})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
