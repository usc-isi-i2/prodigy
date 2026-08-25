#!/usr/bin/env python3
"""Plot fixed-exposure NM and classification AUC before target entry."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


HERE = Path(__file__).resolve().parent
NM_PATH = (
    HERE.parent.parent
    / "source_exposure/nm_ladder_fixed_exposure_nhop2/data/logical_results.csv"
)
CLS_PATH = HERE / "data/downstream_long.csv"
PNG = HERE / "figures/fixed_exposure_nm_vs_cls_auc_entry_aligned.png"
PDF = HERE / "figures/fixed_exposure_nm_vs_cls_auc_entry_aligned.pdf"

TARGETS = (
    "covid_political",
    "election2020",
    "twibot20",
    "ukr_rus_suspended",
)
TITLES = {
    "covid_political": "COVID political",
    "election2020": "Election 2020",
    "twibot20": "TwiBot-20",
    "ukr_rus_suspended": "UKR–RUS suspended",
}
ORDERS = ("A", "C")
ORDER_COLORS = {"A": "#0072B2", "C": "#009E73"}


def read_nm() -> dict[tuple[str, str], list[tuple[int, float]]]:
    rows: dict[tuple[str, str], list[tuple[int, float]]] = {}
    with NM_PATH.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["order"] not in ORDERS or row["dataset"] not in TARGETS:
                continue
            point = (int(row["rel_to_entry"]), float(row["test_roc_auc"]))
            rows.setdefault((row["order"], row["dataset"]), []).append(point)
    return rows


def read_classification() -> dict[tuple[str, str], list[tuple[int, float]]]:
    rows: dict[tuple[str, str], list[tuple[int, float]]] = {}
    with CLS_PATH.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (
                row["variant"] != "fixed10k"
                or row["task"] != "classification"
                or row["metric"] != "roc_auc"
                or row["primary"] != "1"
                or row["order"] not in ORDERS
                or row["dataset"] not in TARGETS
            ):
                continue
            point = (int(row["rel_to_entry"]), float(row["value"]))
            rows.setdefault((row["order"], row["dataset"]), []).append(point)
    return rows


def validate(rows: dict[tuple[str, str], list[tuple[int, float]]], name: str) -> None:
    expected = {(order, target) for order in ORDERS for target in TARGETS}
    if set(rows) != expected:
        raise ValueError(f"{name}: missing or extra order-target series")
    for key, points in rows.items():
        xs = [x for x, _ in points]
        if len(points) != 8 or len(set(xs)) != 8 or 0 not in xs:
            raise ValueError(f"{name}: incomplete ladder for {key}: {sorted(xs)}")


def main() -> None:
    nm = read_nm()
    classification = read_classification()
    validate(nm, "NM")
    validate(classification, "classification")

    fig, axes = plt.subplots(2, 4, figsize=(13.4, 6.4), sharex=True, sharey=True)
    for column, target in enumerate(TARGETS):
        axes[0, column].set_title(TITLES[target], fontsize=10.5, fontweight="bold")
        for row_index, table in enumerate((nm, classification)):
            ax = axes[row_index, column]
            for order in ORDERS:
                points = sorted(point for point in table[(order, target)] if point[0] <= 0)
                xs, ys = zip(*points)
                ax.plot(
                    xs, ys, color=ORDER_COLORS[order], lw=2.0,
                    ls="-" if row_index == 0 else "--",
                    marker="o" if row_index == 0 else "s", ms=3.2,
                )
            ax.axvline(0, color="#777777", lw=1.15, ls=":", zorder=0)
            ax.set_xlim(-7.35, .25)
            ax.set_xticks((-7, -6, -5, -4, -3, -2, -1, 0))
            ax.set_ylim(.4, 1.01)
            ax.grid(axis="y", color="#d9d9d9", linewidth=.7)
            ax.spines[["top", "right"]].set_visible(False)
        axes[1, column].set_xlabel("rungs relative to target entry")

    axes[0, 0].set_ylabel("NM AUC")
    axes[1, 0].set_ylabel("Classification AUC")
    handles = [
        Line2D([0], [0], color=ORDER_COLORS[order], lw=2.3, label=f"Order {order}")
        for order in ORDERS
    ]
    handles.append(
        Line2D([0], [0], color="#777777", lw=1.2, ls=":", label="target enters mix")
    )
    fig.legend(
        handles=handles, loc="lower center", ncol=3,
        frameon=False, bbox_to_anchor=(.5, -.005),
    )
    fig.suptitle(
        "Fixed exposure (10k updates/source): entry-aligned NM vs classification AUC",
        y=.99, fontsize=13, fontweight="bold",
    )
    fig.text(.995, .012, "1 training seed", ha="right", va="bottom", fontsize=8, color="#666666")
    fig.tight_layout(rect=(0, .08, 1, .94), w_pad=1.0, h_pad=2.0)
    PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PDF, bbox_inches="tight")
    fig.savefig(PNG, dpi=220, bbox_inches="tight")
    print(PDF)


if __name__ == "__main__":
    main()
