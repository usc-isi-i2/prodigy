#!/usr/bin/env python3
"""Render the native-model result-matrix coverage snapshot."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "coverage.csv"
OUT = ROOT / "figures"

FAMILIES = [
    ("ssl_to_cls_saturation", "SSL→CLS\nsaturation"),
    ("cross_ssl_matrix", "Cross-SSL\nmatrix"),
    ("downstream_cls_matrix", "Downstream\nCLS matrix"),
    ("mixture_diversity_to_cls", "Mixture diversity\n→ CLS"),
    ("adaptation_efficiency", "Adaptation\nefficiency"),
]
SCORE = {"missing": 0.0, "pending": 0.25, "partial": 0.55, "complete": 1.0}
COLOR = ListedColormap(["#d9d9d9", "#f4a261", "#e9c46a", "#2a9d8f"])


def main() -> None:
    with DATA.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["model"] not in {"MLP", "Logistic regression"}]

    matrix = np.array([[SCORE.get(row[key], np.nan) for key, _ in FAMILIES] for row in rows])
    quantized = np.full_like(matrix, np.nan)
    quantized[matrix == 0.0] = 0
    quantized[matrix == 0.25] = 1
    quantized[matrix == 0.55] = 2
    quantized[matrix == 1.0] = 3

    fig, ax = plt.subplots(figsize=(10.4, 4.8), constrained_layout=True)
    ax.imshow(quantized, cmap=COLOR, vmin=0, vmax=3, aspect="auto")
    ax.set_xticks(range(len(FAMILIES)), [label for _, label in FAMILIES])
    ax.set_yticks(range(len(rows)), [row["model"] for row in rows])
    ax.tick_params(axis="both", length=0, labelsize=10)
    ax.set_title("Native-pretext result-matrix coverage", fontsize=15, pad=16)
    ax.set_xlabel("Result family", labelpad=10)
    for i, row in enumerate(rows):
        for j, (key, _) in enumerate(FAMILIES):
            ax.text(j, i, row[key].replace("pending", "ready\n(no results)"), ha="center", va="center", fontsize=8.5)
    for spine in ax.spines.values():
        spine.set_visible(False)

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "coverage.png", dpi=220)
    fig.savefig(OUT / "coverage.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
