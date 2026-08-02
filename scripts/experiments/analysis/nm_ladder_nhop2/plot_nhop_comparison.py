#!/usr/bin/env python3
"""Plot paired 1-hop versus 2-hop ladder performance and entry jumps."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "nm_ladder_nhop_comparison_long.csv"
DATA_A = HERE / "data" / "nm_ladder_nhop_comparison_order_A_long.csv"
FIGURES = HERE / "figures"

H1 = "#8f8d87"
H2 = "#2a78d6"
INK = "#111111"
GRID = "#e1e0d9"


def load_rows(path: Path | None = None) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with (path or DATA).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "order": row["order"],
                    "rung": int(row["rung"]),
                    "graph": row["test_graph"],
                    "entry": int(row["entry_rung"]),
                    "h1": float(row["auc_h1"]),
                    "h2": float(row["auc_h2"]),
                }
            )
    return rows


def style(ax) -> None:
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def main(phase: str = "all") -> None:
    if phase not in {"A", "all"}:
        raise ValueError(f"unknown phase {phase!r}")
    data_path = DATA_A if phase == "A" else DATA
    expected = 64 if phase == "A" else 192
    rows = load_rows(data_path)
    if len(rows) != expected:
        raise ValueError(f"expected {expected} paired cells for phase {phase}, found {len(rows)}")

    by_rung = defaultdict(lambda: {"h1": [], "h2": []})
    auc = {}
    for row in rows:
        by_rung[int(row["rung"])]["h1"].append(float(row["h1"]))
        by_rung[int(row["rung"])]["h2"].append(float(row["h2"]))
        auc[(row["order"], row["rung"], row["graph"])] = (row["h1"], row["h2"])

    jumps = {"h1": [], "h2": []}
    for row in rows:
        rung = int(row["rung"])
        if rung != int(row["entry"]) or rung == 1:
            continue
        before = auc[(row["order"], rung - 1, row["graph"])]
        jumps["h1"].append(float(row["h1"]) - float(before[0]))
        jumps["h2"].append(float(row["h2"]) - float(before[1]))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), dpi=180)
    rungs = np.arange(1, 9)
    for hop, color, label in (("h1", H1, "1 hop"), ("h2", H2, "2 hops")):
        means = [np.mean(by_rung[rung][hop]) for rung in rungs]
        axes[0].plot(rungs, means, marker="o", linewidth=2.4, color=color, label=label)
    axes[0].set_xticks(rungs)
    axes[0].set_xlabel("rung (number of source graphs)")
    scope = "test graphs" if phase == "A" else "orders and test graphs"
    axes[0].set_ylabel(f"mean NM AUC across {scope}")
    axes[0].set_title("Overall ladder performance")
    axes[0].legend(frameon=False)
    style(axes[0])

    positions = [0, 1]
    boxes = axes[1].boxplot(
        [jumps["h1"], jumps["h2"]], positions=positions, widths=0.5,
        patch_artist=True, showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": INK, "markeredgecolor": "white"},
        medianprops={"color": INK, "linewidth": 1.5},
    )
    for patch, color in zip(boxes["boxes"], (H1, H2)):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
    rng = np.random.default_rng(0)
    for position, hop, color in zip(positions, ("h1", "h2"), (H1, H2)):
        jitter = rng.normal(0, 0.045, len(jumps[hop]))
        axes[1].scatter(position + jitter, jumps[hop], s=17, color=color, alpha=0.8, zorder=3)
    axes[1].axhline(0, color=INK, linewidth=1, linestyle="--")
    axes[1].set_xticks(positions, ["1 hop", "2 hops"])
    axes[1].set_ylabel("entry jump (AUC after entry − before entry)")
    axes[1].set_title(f"{len(jumps['h1'])} paired entry events")
    style(axes[1])

    prefix = "Order A: " if phase == "A" else ""
    fig.suptitle(
        f"{prefix}does a 2-hop neighborhood change the graph-ladder result?",
        fontweight="bold",
    )
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    stem = "nm_ladder_nhop_comparison_order_A" if phase == "A" else "nm_ladder_nhop_comparison"
    for extension in ("pdf", "png"):
        output = FIGURES / f"{stem}.{extension}"
        fig.savefig(output, bbox_inches="tight", dpi=220)
        print(f"wrote {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["A", "all"], default="all")
    main(parser.parse_args().phase)
