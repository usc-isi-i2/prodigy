#!/usr/bin/env python3
"""Plot native SAMGPT GraphCL training loss across the canonical 9x3 ladder."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "data" / "samgpt_9x3_carc_v100" / "graphcl_loss_summary.csv"
FIG_ROOT = HERE / "figures"

COLORS = {"A": "#2a78d6", "B": "#d85a30", "C": "#7a5aa6"}
INK = "#111111"
MUTED = "#77756f"
GRID = "#dfded8"


def load() -> dict[str, list[dict[str, float]]]:
    rows: dict[str, list[dict[str, float]]] = defaultdict(list)
    with DATA_PATH.open(newline="") as handle:
        for raw in csv.DictReader(handle):
            rows[raw["order"]].append(
                {
                    "rung": int(raw["rung"]),
                    "best_loss": float(raw["best_loss"]),
                    "final_loss": float(raw["final_loss"]),
                }
            )
    for order in rows:
        rows[order].sort(key=lambda row: row["rung"])
    if set(rows) != {"A", "B", "C"} or any(len(values) != 9 for values in rows.values()):
        raise ValueError("Expected three complete nine-rung orders")
    return rows


def main() -> None:
    rows = load()
    x = np.arange(1, 10)
    matrix = np.array(
        [[row["best_loss"] * 1_000 for row in rows[order]] for order in "ABC"]
    )

    fig, ax = plt.subplots(figsize=(9.2, 5.2), dpi=200)
    for order in "ABC":
        y = [row["best_loss"] * 1_000 for row in rows[order]]
        ax.plot(
            x,
            y,
            color=COLORS[order],
            marker="o",
            markersize=5,
            linewidth=1.7,
            label=f"Order {order}",
            zorder=3,
        )

    mean = matrix.mean(axis=0)
    ax.plot(
        x,
        mean,
        color=INK,
        marker="s",
        markersize=5.2,
        linewidth=2.4,
        label="Three-order mean",
        zorder=4,
    )

    ax.set_xticks(x)
    ax.set_xlabel("merge size (number of source graphs)", color=INK)
    ax.set_ylabel("best GraphCL training BCE (×10⁻³; lower is better)", color=INK)
    ax.set_title(
        "SAMGPT fits its native objective nearly to zero at every ladder rung",
        loc="left",
        fontsize=13,
        weight="bold",
        color=INK,
        pad=14,
    )
    ax.text(
        0,
        1.015,
        "Loss rises modestly with mixture size, but source composition remains a large factor",
        transform=ax.transAxes,
        fontsize=9.5,
        color=MUTED,
        va="bottom",
    )
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#bdbbb3")
    ax.tick_params(colors=MUTED)
    ax.legend(frameon=False, ncol=4, loc="upper left", fontsize=8.5)
    ax.text(
        0,
        -0.19,
        "200 training epochs per rung · mean of per-source BCE losses · best epoch chosen per run · no validation split",
        transform=ax.transAxes,
        fontsize=8.5,
        color=MUTED,
    )

    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        output = FIG_ROOT / f"samgpt_graphcl_training_loss.{suffix}"
        fig.savefig(output, bbox_inches="tight", dpi=220)
        print(f"wrote {output}")
    plt.close(fig)


if __name__ == "__main__":
    main()
