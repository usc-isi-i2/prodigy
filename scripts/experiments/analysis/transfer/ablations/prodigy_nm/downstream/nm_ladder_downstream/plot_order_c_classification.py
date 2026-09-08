#!/usr/bin/env python3
"""Plot downstream classification AUC along the original NM ladder Order C."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
DATA = HERE / "data/nm_ladder_downstream_long.csv"
PNG = HERE / "figures/order_c_classification_auc.png"
PDF = HERE / "figures/order_c_classification_auc.pdf"

RUNG_LABELS = (
    "Election ’20",
    "+Cov-Pol",
    "+CP-HK",
    "+UKR-susp.",
    "+Midterm",
    "+TwiBot-20",
    "+UKR–RUS",
    "+COVID",
)
TARGETS = (
    ("election2020", "Election ’20"),
    ("covid_political", "COVID political"),
    ("ukr_rus_suspended", "UKR–RUS suspended"),
    ("twibot20", "TwiBot-20"),
)


def read_order_c_auc() -> dict[str, dict[int, tuple[float, int]]]:
    values: dict[str, dict[int, tuple[float, int]]] = {}
    with DATA.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["order"] != "C" or row["task"] != "pl" or row["metric"] != "roc_auc":
                continue
            values.setdefault(row["dataset"], {})[int(row["rung"])] = (
                float(row["value"]),
                int(row["entry_rung"]),
            )
    return values


def main() -> None:
    values = read_order_c_auc()
    blue = "#2474B5"
    entry_color = "#D55E00"
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.0), sharex=True, sharey=True)

    for ax, (target, title) in zip(axes.flat, TARGETS):
        series = values[target]
        rungs = sorted(series)
        aucs = [series[rung][0] for rung in rungs]
        entry = series[rungs[0]][1]

        ax.plot(rungs, aucs, color=blue, linewidth=2.4, marker="o", markersize=5.2)
        ax.axvline(entry, color=entry_color, linewidth=1.4, linestyle="--", alpha=0.9)
        ax.scatter(
            [entry], [series[entry][0]], s=68, color=entry_color,
            edgecolor="white", linewidth=0.9, zorder=4,
        )

        if entry == 1:
            annotation = "target starts in mix"
        else:
            delta = series[entry][0] - series[entry - 1][0]
            annotation = f"entry Δ {delta:+.3f}"
        annotation_y = 0.88 if max(aucs) < 0.75 else 0.08
        ax.text(
            0.03, annotation_y, annotation, transform=ax.transAxes, color=entry_color,
            fontsize=10, fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.5},
        )

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylim(0.48, 1.005)
        ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax.grid(axis="y", color="#dddddd", linewidth=0.75)
        ax.spines[["top", "right"]].set_visible(False)

    for ax in axes[:, 0]:
        ax.set_ylabel("Classification ROC-AUC")
    for ax in axes[-1, :]:
        ax.set_xticks(range(1, 9), RUNG_LABELS, rotation=35, ha="right")
        ax.set_xlabel("Order C mixture rung")

    fig.suptitle(
        "Order C downstream classification across mixture growth",
        fontsize=14, fontweight="bold", y=0.99,
    )
    fig.text(
        0.5, 0.925,
        "Election ’20 → COVID political → CP-HK → UKR–RUS suspended → "
        "Midterm → TwiBot-20 → UKR–RUS → COVID",
        ha="center", color="#555555", fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91), h_pad=1.8, w_pad=1.2)
    PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG, dpi=220, bbox_inches="tight")
    fig.savefig(PDF, bbox_inches="tight")
    print(PNG)


if __name__ == "__main__":
    main()
