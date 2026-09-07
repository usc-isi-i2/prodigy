#!/usr/bin/env python3
"""Order-robustness figures: the three single-order ladder plots, aggregated over orders.

The published NM ladder (nm_ladder_fillin) added the 8 source graphs in ONE order. This
experiment reran it under three orders (A published, B donor-strength descending, C its
reverse). These three figures are the order-aggregated counterparts of the three original
ladder figures, so the "adding a graph" story can be told without depending on a sequence:

  fig 1  entry-aligned trajectory   <- plot_nm_ladder.py (trajectory / diagonal cascade)
         Each test graph's AUC re-indexed by rungs-relative-to-its-own-entry, every
         (graph, order) overlaid, with the mean and min/max band. The absolute rung a
         graph enters differs by order; aligned on entry, the jump is one shape.

  fig 2  in-/out-of-distribution gap <- plot_nm_ladder_means.py (ID vs all-8 mean)
         Per rung (= merge size), the in-distribution mean vs the all-8 mean, averaged
         over the three orders with a min/max band. Rung k = k sources in every order, so
         this axis IS comparable across orders. The gap closes regardless of order.

  fig 3  per-source role decomposition <- plot_nm_ladder_deltas.py (newcomer/incumbent/held-out)
         Every source-addition event, pooled over the three orders (7 rungs x 3 = 21
         newcomer events), split by role: newcomer (its own OOD->ID jump = benefit),
         incumbents (already in training = cost), held-out (not yet added = interference).
         Distribution, not a single order's 7 bars.

Reads data/nm_ladder_order_robustness_long.csv (entry-aligned long form from
assemble_order_table.py). Writes figures/ here. Local homebrew python:
  /opt/homebrew/bin/python3.11 plot_order_robustness.py
"""
import csv
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "nm_ladder_order_robustness_long.csv")
FIGS = os.path.join(HERE, "figures")
SINGLE_SOURCE_DATA = os.path.normpath(os.path.join(
    HERE,
    "../../../../matrices/prodigy_nm/single_source/nm_single_source_matrix/data/"
    "nm_single_source_matrix.csv",
))

# palette (matches the nm_ladder figures)
BLUE = "#2a78d6"    # in-distribution / newcomer benefit
CORAL = "#d85a30"   # incumbents (cost)
GRAY = "#8f8d87"    # held-out / out-of-distribution
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
ORDER_C = {"A": "#2a78d6", "B": "#e0a51f", "C": "#8f3fbf"}
TARGET_C = {
    "ukr_rus_twitter": "#174f73",
    "covid19_twitter": "#2a78d6",
    "midterm": "#e68613",
    "covid_political": "#2a9d65",
    "election2020": "#d34e4e",
    "ukr_rus_suspended": "#8b63c7",
    "twibot20": "#c79a17",
    "cp_hk_twitter": "#66717e",
}
TARGET_LABEL = {
    "ukr_rus_twitter": "Ukr-Rus",
    "covid19_twitter": "COVID-19",
    "midterm": "Midterm",
    "covid_political": "COVID-pol.",
    "election2020": "Election '20",
    "ukr_rus_suspended": "Ukr-Rus susp.",
    "twibot20": "TwiBot-20",
    "cp_hk_twitter": "CP-HK",
}
SOURCE_LABEL = {
    "ukr_rus": "ukr",
    "covid": "covid",
    "midterm": "midterm",
    "covid_political": "cov-pol",
    "election2020": "elec '20",
    "ukr_rus_suspended": "ukr-susp",
    "twibot20": "twibot",
    "cp_hk": "cp-hk",
}
SOURCE_TO_GRAPH = {
    "ukr_rus": "ukr_rus_twitter",
    "covid": "covid19_twitter",
    "midterm": "midterm",
    "covid_political": "covid_political",
    "election2020": "election2020",
    "ukr_rus_suspended": "ukr_rus_suspended",
    "twibot20": "twibot20",
    "cp_hk": "cp_hk_twitter",
}

RUNGS = list(range(1, 9))

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "svg.fonttype": "none",
})


def load():
    rows = []
    with open(DATA, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["auc"] == "":
                continue
            rows.append(dict(
                order=r["order"], rung=int(r["rung"]), n_sources=int(r["n_sources"]),
                graph=r["test_graph"], auc=float(r["auc"]),
                entry_rung=int(r["entry_rung"]), rel=int(r["rel_to_entry"]),
                in_training=r["in_training"] == "1", added=r["added"],
                sources=r["sources"].split(),
            ))
    return rows


def chrome(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#c3c2b7")
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9)


def save(fig, stem):
    for ext in ("pdf", "png"):
        out = os.path.join(FIGS, f"{stem}.{ext}")
        fig.savefig(out, bbox_inches="tight", dpi=200)
        print("wrote", out)


# ------------------------------------------------------- held-out-only trajectories
def fig_heldout_only_by_order(rows):
    """Per-target NM AUC while the target is absent from each training prefix."""
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.5), dpi=200, sharey=True)

    for ax, order in zip(axes, ("A", "B", "C")):
        order_rows = [r for r in rows if r["order"] == order]
        added_by_rung = {
            r["rung"]: r["added"] for r in order_rows if r["graph"] == "ukr_rus_twitter"
        }

        for graph in TARGET_C:
            trajectory = sorted(
                (r for r in order_rows if r["graph"] == graph),
                key=lambda r: r["rung"],
            )
            heldout = sorted(
                (r for r in order_rows if r["graph"] == graph and r["rel"] < 0),
                key=lambda r: r["rung"],
            )
            # Full trajectory remains visible as context after target entry.
            ax.plot(
                [r["rung"] for r in trajectory],
                [r["auc"] for r in trajectory],
                color=TARGET_C[graph],
                lw=2.1,
                alpha=0.20,
                solid_capstyle="round",
                zorder=2,
            )
            if len(heldout) < 2:
                continue
            # Opaque portion contains only evaluations made while the target is absent.
            ax.plot(
                [r["rung"] for r in heldout],
                [r["auc"] for r in heldout],
                color=TARGET_C[graph],
                lw=2.1,
                solid_capstyle="round",
                zorder=3,
            )

        ax.set_xlim(0.7, 8.3)
        ax.set_ylim(0.53, 1.0)
        ax.set_xticks(range(1, 9))
        ax.set_xticklabels(
            [SOURCE_LABEL[added_by_rung[rung]] for rung in range(1, 9)],
            rotation=35,
            ha="right",
            fontsize=8.4,
        )
        ax.set_xlabel("source added to pre-training mixture", fontsize=9.3, color=INK)
        ax.set_title(f"Order {order}", fontsize=12, color=INK, fontweight="bold")
        chrome(ax)

    axes[0].set_ylabel("NM AUC  (3-shot, 30-way)", fontsize=10.5, color=INK)
    handles = [
        Line2D(
            [0], [0], color=TARGET_C[graph], lw=2.1,
            label=TARGET_LABEL[graph],
        )
        for graph in TARGET_C
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=8,
        frameon=False,
        fontsize=8.7,
        handlelength=2.0,
        columnspacing=1.15,
    )
    fig.suptitle(
        "Held-out NM scaling is source-order dependent",
        fontsize=14,
        color=INK,
        fontweight="bold",
        y=1.01,
    )
    fig.text(
        0.5,
        0.962,
        "matched step 40k  ·  within-balanced sampling  ·  opaque = held out; transparent = after target entry",
        ha="center",
        va="top",
        fontsize=9.2,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.93))
    save(fig, "order_heldout_only_trajectories")
    plt.close(fig)


# -------------------------------------- Order C vs best constituent specialist
def load_single_source_matrix():
    matrix = {}
    with open(SINGLE_SOURCE_DATA, newline="") as fh:
        for row in csv.DictReader(fh):
            source = row["train_graph"]
            for target, value in row.items():
                if target != "train_graph" and value != "":
                    matrix[(source, target)] = float(value)
    return matrix


def fig_order_c_best_constituent_residual(rows):
    """Order-C held-out mixture AUC minus the strongest included single source."""
    specialists = load_single_source_matrix()
    order_rows = [r for r in rows if r["order"] == "C"]
    added_by_rung = {
        r["rung"]: r["added"] for r in order_rows if r["graph"] == "ukr_rus_twitter"
    }
    residual_rows = []
    for row in order_rows:
        # Only true held-out targets, and omit rung 1 where the max-rule comparison
        # is tautologically the same single-source model.
        if row["rel"] >= 0 or row["rung"] == 1:
            continue
        constituent_graphs = [SOURCE_TO_GRAPH[source] for source in row["sources"]]
        constituent_scores = [
            specialists[(source, row["graph"])] for source in constituent_graphs
        ]
        best_index = int(np.argmax(constituent_scores))
        best_auc = constituent_scores[best_index]
        residual_rows.append({
            "order": "C",
            "rung": row["rung"],
            "target": row["graph"],
            "sources": " ".join(row["sources"]),
            "mixture_auc": row["auc"],
            "best_constituent": constituent_graphs[best_index],
            "best_constituent_auc": best_auc,
            "mixture_minus_best": row["auc"] - best_auc,
        })

    data_out = os.path.join(HERE, "data", "order_C_heldout_best_constituent_residual.csv")
    with open(data_out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(residual_rows[0]))
        writer.writeheader()
        writer.writerows(residual_rows)
    print("wrote", data_out)

    fig, ax = plt.subplots(figsize=(9.2, 5.7), dpi=200)
    for graph in TARGET_C:
        target_rows = sorted(
            (r for r in residual_rows if r["target"] == graph),
            key=lambda r: r["rung"],
        )
        if not target_rows:
            continue
        ax.plot(
            [r["rung"] for r in target_rows],
            [r["mixture_minus_best"] for r in target_rows],
            color=TARGET_C[graph],
            lw=2.2,
            marker="o",
            ms=5.7,
            markerfacecolor="white",
            markeredgecolor=TARGET_C[graph],
            markeredgewidth=1.5,
            label=TARGET_LABEL[graph],
            zorder=4,
        )

    ax.axhline(0, color=INK, lw=1.1, ls=(0, (4, 3)), zorder=2)
    ax.text(8.28, 0.002, "best constituent", ha="right", va="bottom",
            fontsize=8.8, color=MUTED)
    ax.set_xlim(0.7, 8.3)
    ax.set_ylim(-0.015, 0.10)
    ax.set_xticks(range(1, 9))
    ax.set_xticklabels(
        [SOURCE_LABEL[added_by_rung[rung]] for rung in range(1, 9)],
        rotation=30,
        ha="right",
        fontsize=8.8,
    )
    ax.set_xlabel("source added to Order C pre-training mixture", fontsize=10, color=INK)
    ax.set_ylabel("mixture AUC − best constituent AUC", fontsize=10.5, color=INK)
    chrome(ax)
    ax.set_title(
        "Order C mixtures often outperform their best constituent on held-out graphs",
        fontsize=12.3,
        color=INK,
        fontweight="bold",
        loc="left",
        pad=26,
    )
    ax.text(
        0.0,
        1.02,
        "NM 3-shot / 30-way  ·  matched step 40k  ·  held-out targets only  ·  rung 1 shown empty",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.8,
        color=MUTED,
    )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=6,
        frameon=False,
        fontsize=8.7,
        handlelength=2.0,
        columnspacing=1.15,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, "order_C_heldout_best_constituent_residual")
    plt.close(fig)


def fig_best_constituent_residual_by_order(rows):
    """Held-out mixture-minus-best-constituent residuals for Orders A, B, and C."""
    specialists = load_single_source_matrix()
    residual_rows = []
    for row in rows:
        # Keep only unseen targets. Rung 1 remains on the axis but has no point because
        # a one-source model equals its sole constituent by construction.
        if row["rel"] >= 0 or row["rung"] == 1:
            continue
        constituent_graphs = [SOURCE_TO_GRAPH[source] for source in row["sources"]]
        constituent_scores = [
            specialists[(source, row["graph"])] for source in constituent_graphs
        ]
        best_index = int(np.argmax(constituent_scores))
        best_auc = constituent_scores[best_index]
        residual_rows.append({
            "order": row["order"],
            "rung": row["rung"],
            "target": row["graph"],
            "sources": " ".join(row["sources"]),
            "mixture_auc": row["auc"],
            "best_constituent": constituent_graphs[best_index],
            "best_constituent_auc": best_auc,
            "mixture_minus_best": row["auc"] - best_auc,
        })

    data_out = os.path.join(
        HERE, "data", "all_orders_heldout_best_constituent_residual.csv"
    )
    with open(data_out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(residual_rows[0]))
        writer.writeheader()
        writer.writerows(residual_rows)
    print("wrote", data_out)

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.7), dpi=200, sharey=True)
    for ax, order in zip(axes, ("A", "B", "C")):
        order_source_rows = [r for r in rows if r["order"] == order]
        added_by_rung = {
            r["rung"]: r["added"]
            for r in order_source_rows
            if r["graph"] == "ukr_rus_twitter"
        }
        order_residuals = [r for r in residual_rows if r["order"] == order]
        for graph in TARGET_C:
            target_rows = sorted(
                (r for r in order_residuals if r["target"] == graph),
                key=lambda r: r["rung"],
            )
            if not target_rows:
                continue
            ax.plot(
                [r["rung"] for r in target_rows],
                [r["mixture_minus_best"] for r in target_rows],
                color=TARGET_C[graph],
                lw=2.1,
                marker="o",
                ms=5.2,
                markerfacecolor="white",
                markeredgecolor=TARGET_C[graph],
                markeredgewidth=1.4,
                zorder=4,
            )

        ax.axhline(0, color=INK, lw=1.0, ls=(0, (4, 3)), zorder=2)
        ax.set_xlim(0.7, 8.3)
        ax.set_ylim(-0.035, 0.10)
        ax.set_xticks(range(1, 9))
        ax.set_xticklabels(
            [SOURCE_LABEL[added_by_rung[rung]] for rung in range(1, 9)],
            rotation=35,
            ha="right",
            fontsize=8.4,
        )
        ax.set_xlabel("source added to pre-training mixture", fontsize=9.3, color=INK)
        ax.set_title(f"Order {order}", fontsize=12, color=INK, fontweight="bold")
        chrome(ax)

    axes[0].set_ylabel("mixture AUC − best constituent AUC", fontsize=10.5, color=INK)
    axes[-1].text(8.25, 0.002, "best constituent", ha="right", va="bottom",
                  fontsize=8.4, color=MUTED)
    handles = [
        Line2D(
            [0], [0], color=TARGET_C[graph], lw=2.1, marker="o",
            markerfacecolor="white", markeredgecolor=TARGET_C[graph], ms=6,
            label=TARGET_LABEL[graph],
        )
        for graph in TARGET_C
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=8,
        frameon=False,
        fontsize=8.7,
        handlelength=2.0,
        columnspacing=1.15,
    )
    fig.suptitle(
        "Held-out mixture advantage over the best constituent depends on source order",
        fontsize=14,
        color=INK,
        fontweight="bold",
        y=1.01,
    )
    fig.text(
        0.5,
        0.962,
        "NM 3-shot / 30-way  ·  matched step 40k  ·  specialists excluded  ·  rung 1 shown empty",
        ha="center",
        va="top",
        fontsize=9.2,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.93))
    save(fig, "all_orders_heldout_best_constituent_residual")
    plt.close(fig)


def fig_mean_mixture_vs_best_constituent_gain_by_order(rows):
    """Compare mean held-out gains from rung 1 for mixtures and their best donors."""
    specialists = load_single_source_matrix()
    summary_rows = []

    for order in ("A", "B", "C"):
        order_rows = [r for r in rows if r["order"] == order]
        mixture_at_rung1 = {
            r["graph"]: r["auc"] for r in order_rows if r["rung"] == 1
        }
        first_source = next(r["sources"][0] for r in order_rows if r["rung"] == 1)
        first_source_graph = SOURCE_TO_GRAPH[first_source]

        for rung in range(1, 8):
            heldout = [
                r for r in order_rows if r["rung"] == rung and r["rel"] < 0
            ]
            if rung == 1:
                mixture_gain = 0.0
                constituent_gain = 0.0
            else:
                mixture_deltas = []
                constituent_deltas = []
                for row in heldout:
                    target = row["graph"]
                    constituent_graphs = [
                        SOURCE_TO_GRAPH[source] for source in row["sources"]
                    ]
                    best_auc = max(
                        specialists[(source, target)] for source in constituent_graphs
                    )
                    mixture_deltas.append(row["auc"] - mixture_at_rung1[target])
                    constituent_deltas.append(
                        best_auc - specialists[(first_source_graph, target)]
                    )
                mixture_gain = float(np.mean(mixture_deltas))
                constituent_gain = float(np.mean(constituent_deltas))

            added = next(r["added"] for r in order_rows if r["rung"] == rung)
            summary_rows.append({
                "order": order,
                "rung": rung,
                "added": added,
                "n_heldout": len(heldout),
                "mean_mixture_gain_from_rung1": mixture_gain,
                "mean_best_constituent_gain_from_rung1": constituent_gain,
                "mixture_minus_best_gain": mixture_gain - constituent_gain,
            })

    data_out = os.path.join(
        HERE, "data", "all_orders_heldout_mixture_vs_best_constituent_gain.csv"
    )
    with open(data_out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    print("wrote", data_out)

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.7), dpi=200, sharey=True)
    for ax, order in zip(axes, ("A", "B", "C")):
        order_summary = [r for r in summary_rows if r["order"] == order]
        order_source_rows = [r for r in rows if r["order"] == order]
        x = [r["rung"] for r in order_summary]
        ax.plot(
            x,
            [r["mean_mixture_gain_from_rung1"] for r in order_summary],
            color=BLUE,
            lw=2.6,
            marker="o",
            ms=6,
            markerfacecolor=BLUE,
            markeredgecolor="white",
            markeredgewidth=1.0,
            zorder=5,
        )
        ax.plot(
            x,
            [r["mean_best_constituent_gain_from_rung1"] for r in order_summary],
            color=GRAY,
            lw=2.3,
            marker="o",
            ms=5.7,
            markerfacecolor="white",
            markeredgecolor=GRAY,
            markeredgewidth=1.4,
            zorder=4,
        )
        ax.axhline(0, color=INK, lw=0.9, ls=(0, (4, 3)), zorder=2)
        ax.set_xlim(0.7, 8.3)
        ax.set_ylim(-0.025, 0.34)
        ax.set_xticks(range(1, 9))
        added_by_rung = {r["rung"]: r["added"] for r in order_summary}
        # Rung 8 has no held-out targets but remains visible to complete the order.
        order_rows = [r for r in rows if r["order"] == order]
        added_by_rung[8] = next(r["added"] for r in order_rows if r["rung"] == 8)
        ax.set_xticklabels(
            [SOURCE_LABEL[added_by_rung[rung]] for rung in range(1, 9)],
            rotation=35,
            ha="right",
            fontsize=8.4,
        )
        ax.set_xlabel("source added to pre-training mixture", fontsize=9.3, color=INK)
        ax.set_title(f"Order {order}", fontsize=12, color=INK, fontweight="bold")
        chrome(ax)

    axes[0].set_ylabel(
        "mean held-out NM AUC gain from rung 1", fontsize=10.5, color=INK
    )
    fig.legend(
        handles=[
            Line2D([0], [0], color=BLUE, lw=2.6, marker="o",
                   markerfacecolor=BLUE, markeredgecolor="white", ms=7,
                   label="mixture model"),
            Line2D([0], [0], color=GRAY, lw=2.3, marker="o",
                   markerfacecolor="white", markeredgecolor=GRAY, ms=7,
                   label="best constituent per target"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=2,
        frameon=False,
        fontsize=9.2,
        handlelength=2.4,
        columnspacing=2.2,
    )
    fig.suptitle(
        "Order C creates mixture gains beyond simply adding a stronger donor",
        fontsize=14,
        color=INK,
        fontweight="bold",
        y=1.01,
    )
    fig.text(
        0.5,
        0.962,
        "per-target change from rung 1, then averaged over targets still held out  ·  specialists excluded  ·  rung 8 has no held-out targets",
        ha="center",
        va="top",
        fontsize=9.2,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.09, 1, 0.93))
    save(fig, "all_orders_heldout_mixture_vs_best_constituent_gain")
    plt.close(fig)


def fig_mean_mixture_vs_best_constituent_absolute_by_order(rows):
    """Compare absolute mean AUC for held-out mixtures and their best donors."""
    specialists = load_single_source_matrix()
    summary_rows = []

    for order in ("A", "B", "C"):
        order_rows = [r for r in rows if r["order"] == order]
        for rung in range(1, 8):
            heldout = [
                r for r in order_rows if r["rung"] == rung and r["rel"] < 0
            ]
            mixture_values = []
            constituent_values = []
            for row in heldout:
                target = row["graph"]
                constituent_graphs = [
                    SOURCE_TO_GRAPH[source] for source in row["sources"]
                ]
                mixture_values.append(row["auc"])
                constituent_values.append(max(
                    specialists[(source, target)] for source in constituent_graphs
                ))
            added = next(r["added"] for r in order_rows if r["rung"] == rung)
            mixture_mean = float(np.mean(mixture_values))
            constituent_mean = float(np.mean(constituent_values))
            summary_rows.append({
                "order": order,
                "rung": rung,
                "added": added,
                "n_heldout": len(heldout),
                "mean_mixture_auc": mixture_mean,
                "mean_best_constituent_auc": constituent_mean,
                "mixture_minus_best_auc": mixture_mean - constituent_mean,
            })

    data_out = os.path.join(
        HERE, "data", "all_orders_heldout_mixture_vs_best_constituent_absolute.csv"
    )
    with open(data_out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    print("wrote", data_out)

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.7), dpi=200, sharey=True)
    for ax, order in zip(axes, ("A", "B", "C")):
        order_summary = [r for r in summary_rows if r["order"] == order]
        order_source_rows = [r for r in rows if r["order"] == order]
        x = [r["rung"] for r in order_summary]
        ax.plot(
            x,
            [r["mean_mixture_auc"] for r in order_summary],
            color=BLUE,
            lw=2.6,
            marker="o",
            ms=6,
            markerfacecolor=BLUE,
            markeredgecolor="white",
            markeredgewidth=1.0,
            zorder=5,
        )
        ax.plot(
            x,
            [r["mean_best_constituent_auc"] for r in order_summary],
            color=GRAY,
            lw=2.3,
            marker="o",
            ms=5.7,
            markerfacecolor="white",
            markeredgecolor=GRAY,
            markeredgewidth=1.4,
            zorder=4,
        )
        ax.set_xlim(0.7, 8.3)
        ax.set_ylim(0.54, 1.0)
        ax.set_xticks(range(1, 9))
        added_by_rung = {r["rung"]: r["added"] for r in order_summary}
        added_by_rung[8] = next(
            r["added"] for r in order_source_rows if r["rung"] == 8
        )
        ax.set_xticklabels(
            [SOURCE_LABEL[added_by_rung[rung]] for rung in range(1, 9)],
            rotation=35,
            ha="right",
            fontsize=8.4,
        )
        ax.set_xlabel("source added to pre-training mixture", fontsize=9.3, color=INK)
        ax.set_title(f"Order {order}", fontsize=12, color=INK, fontweight="bold")
        chrome(ax)

    axes[0].set_ylabel("mean held-out NM AUC", fontsize=10.5, color=INK)
    fig.legend(
        handles=[
            Line2D([0], [0], color=BLUE, lw=2.6, marker="o",
                   markerfacecolor=BLUE, markeredgecolor="white", ms=7,
                   label="mixture model"),
            Line2D([0], [0], color=GRAY, lw=2.3, marker="o",
                   markerfacecolor="white", markeredgecolor=GRAY, ms=7,
                   label="best constituent per target"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=2,
        frameon=False,
        fontsize=9.2,
        handlelength=2.4,
        columnspacing=2.2,
    )
    fig.suptitle(
        "Absolute held-out NM AUC: mixture versus best constituent",
        fontsize=14,
        color=INK,
        fontweight="bold",
        y=1.01,
    )
    fig.text(
        0.5,
        0.962,
        "means over targets still held out at each rung  ·  best constituent chosen separately per target  ·  specialists excluded  ·  rung 8 empty",
        ha="center",
        va="top",
        fontsize=9.2,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.09, 1, 0.93))
    save(fig, "all_orders_heldout_mixture_vs_best_constituent_absolute")
    plt.close(fig)


def fig_mean_mixture_minus_best_constituent_by_order(rows):
    """Mean mixture advantage over the best constituent on non-identity targets."""
    specialists = load_single_source_matrix()
    summary_rows = []

    for order in ("A", "B", "C"):
        order_rows = [r for r in rows if r["order"] == order]
        for rung in range(1, 8):
            heldout = [
                r for r in order_rows if r["rung"] == rung and r["rel"] < 0
            ]
            gaps = []
            for row in heldout:
                target = row["graph"]
                constituent_graphs = [
                    SOURCE_TO_GRAPH[source] for source in row["sources"]
                ]
                best_auc = max(
                    specialists[(source, target)] for source in constituent_graphs
                )
                gaps.append(row["auc"] - best_auc)

            # A one-source mixture is its own sole constituent. Anchor rung 1 at
            # zero rather than exposing small evaluation/artifact discrepancies.
            mean_gap = 0.0 if rung == 1 else float(np.mean(gaps))
            added = next(r["added"] for r in order_rows if r["rung"] == rung)
            summary_rows.append({
                "order": order,
                "rung": rung,
                "added": added,
                "n_non_identity_targets": len(heldout),
                "mean_mixture_minus_best_constituent_auc": mean_gap,
            })

    data_out = os.path.join(
        HERE, "data", "all_orders_nonidentity_mean_mixture_minus_best_constituent.csv"
    )
    with open(data_out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    print("wrote", data_out)

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.7), dpi=200, sharey=True)
    for ax, order in zip(axes, ("A", "B", "C")):
        order_summary = [r for r in summary_rows if r["order"] == order]
        order_source_rows = [r for r in rows if r["order"] == order]
        x = [r["rung"] for r in order_summary]
        values = [
            r["mean_mixture_minus_best_constituent_auc"] for r in order_summary
        ]
        colors = [
            GRAY if abs(value) < 1e-12 else BLUE if value > 0 else CORAL
            for value in values
        ]
        ax.bar(
            x,
            values,
            width=0.68,
            color=colors,
            edgecolor="white",
            linewidth=0.8,
            zorder=4,
        )
        # Make the deliberately zero-height rung-1 bar visible.
        ax.scatter([1], [0], s=28, color=GRAY, edgecolor="white", linewidth=0.8,
                   zorder=6)
        for rung, value in zip(x, values):
            value_label = (
                "0" if rung == 1
                else f"{value:+.4f}" if 0 < abs(value) < 0.001
                else f"{value:+.3f}"
            )
            ax.text(
                rung,
                value + (0.0022 if value >= 0 else -0.0022),
                value_label,
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=7.8,
                color=MUTED,
            )
        ax.axhline(0, color=INK, lw=1.0, ls=(0, (4, 3)), zorder=2)
        ax.set_xlim(0.7, 8.3)
        ax.set_ylim(-0.025, 0.08)
        ax.set_xticks(range(1, 9))
        added_by_rung = {r["rung"]: r["added"] for r in order_summary}
        # Rung 8 has no non-identity targets, but its final source label completes
        # the displayed order.
        added_by_rung[8] = next(
            r["added"] for r in order_source_rows if r["rung"] == 8
        )
        ax.set_xticklabels(
            [SOURCE_LABEL[added_by_rung[rung]] for rung in range(1, 9)],
            rotation=35,
            ha="right",
            fontsize=8.4,
        )
        ax.set_xlabel("source added to pre-training mixture", fontsize=9.3, color=INK)
        ax.set_title(f"Order {order}", fontsize=12, color=INK, fontweight="bold")
        chrome(ax)

    axes[0].set_ylabel(
        "mean (mixture AUC − best constituent AUC)", fontsize=10.5, color=INK
    )
    fig.suptitle(
        "Mean mixture advantage over the best non-specialist constituent",
        fontsize=14,
        color=INK,
        fontweight="bold",
        y=1.01,
    )
    fig.text(
        0.5,
        0.962,
        "best constituent chosen separately per target  ·  mean over targets still held out  ·  rung 1 = 0  ·  rung 8 empty",
        ha="center",
        va="top",
        fontsize=9.2,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    save(fig, "all_orders_nonidentity_mean_mixture_minus_best_constituent_bars")
    plt.close(fig)


def fig_final_heldout_mixture_minus_best_constituent_bars(rows):
    """Mixture-minus-best-constituent bars for each order's last-added target."""
    specialists = load_single_source_matrix()
    result_rows = []

    for order in ("A", "B", "C"):
        order_rows = [r for r in rows if r["order"] == order]
        added_by_rung = {
            rung: next(r["added"] for r in order_rows if r["rung"] == rung)
            for rung in range(1, 9)
        }
        final_target = SOURCE_TO_GRAPH[added_by_rung[8]]
        for rung in range(1, 8):
            row = next(
                r for r in order_rows
                if r["rung"] == rung and r["graph"] == final_target
            )
            constituent_graphs = [
                SOURCE_TO_GRAPH[source] for source in row["sources"]
            ]
            constituent_aucs = [
                specialists[(source, final_target)] for source in constituent_graphs
            ]
            best_index = int(np.argmax(constituent_aucs))
            best_source = constituent_graphs[best_index]
            best_auc = constituent_aucs[best_index]
            # A one-source mixture and its only constituent are the same model class;
            # preserve the established rung-1 zero anchor.
            gap = 0.0 if rung == 1 else row["auc"] - best_auc
            result_rows.append({
                "order": order,
                "rung": rung,
                "added": added_by_rung[rung],
                "final_heldout_target": final_target,
                "mixture_auc": row["auc"],
                "best_constituent": best_source,
                "best_constituent_auc": best_auc,
                "mixture_minus_best_constituent_auc": gap,
            })

    data_out = os.path.join(
        HERE, "data", "all_orders_final_heldout_mixture_minus_best_constituent.csv"
    )
    with open(data_out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(result_rows[0]))
        writer.writeheader()
        writer.writerows(result_rows)
    print("wrote", data_out)

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.7), dpi=200, sharey=True)
    for ax, order in zip(axes, ("A", "B", "C")):
        order_results = [r for r in result_rows if r["order"] == order]
        order_source_rows = [r for r in rows if r["order"] == order]
        target = order_results[0]["final_heldout_target"]
        x = [r["rung"] for r in order_results]
        values = [r["mixture_minus_best_constituent_auc"] for r in order_results]
        colors = [
            GRAY if abs(value) < 1e-12 else BLUE if value > 0 else CORAL
            for value in values
        ]
        ax.bar(
            x,
            values,
            width=0.68,
            color=colors,
            edgecolor="white",
            linewidth=0.8,
            zorder=4,
        )
        ax.scatter([1], [0], s=28, color=GRAY, edgecolor="white", linewidth=0.8,
                   zorder=6)
        for rung, value in zip(x, values):
            value_label = (
                "0" if rung == 1
                else f"{value:+.4f}" if 0 < abs(value) < 0.001
                else f"{value:+.3f}"
            )
            ax.text(
                rung,
                value + (0.0022 if value >= 0 else -0.0022),
                value_label,
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=7.8,
                color=MUTED,
            )
        added_by_rung = {
            rung: next(
                r["added"] for r in order_source_rows if r["rung"] == rung
            )
            for rung in range(1, 9)
        }
        ax.axhline(0, color=INK, lw=1.0, ls=(0, (4, 3)), zorder=2)
        ax.set_xlim(0.7, 8.3)
        ax.set_ylim(-0.02, 0.09)
        ax.set_xticks(range(1, 9))
        ax.set_xticklabels(
            [SOURCE_LABEL[added_by_rung[rung]] for rung in range(1, 9)],
            rotation=35,
            ha="right",
            fontsize=8.4,
        )
        ax.set_xlabel("source added to pre-training mixture", fontsize=9.3, color=INK)
        ax.set_title(
            f"Order {order}  ·  {TARGET_LABEL[target]}",
            fontsize=12,
            color=INK,
            fontweight="bold",
        )
        chrome(ax)

    axes[0].set_ylabel(
        "mixture AUC − best constituent AUC", fontsize=10.5, color=INK
    )
    fig.suptitle(
        "Mixture advantage on each order's final held-out graph",
        fontsize=14,
        color=INK,
        fontweight="bold",
        y=1.01,
    )
    fig.text(
        0.5,
        0.962,
        "best non-specialist constituent selected at each rung  ·  rung 1 = 0  ·  target specialist enters at empty rung 8",
        ha="center",
        va="top",
        fontsize=9.2,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    save(fig, "all_orders_final_heldout_mixture_minus_best_constituent_bars")
    plt.close(fig)


def fig_final_heldout_mixture_vs_best_constituent_lines(rows):
    """Two-line view on each order's final held-out target."""
    specialists = load_single_source_matrix()
    result_rows = []

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.7), dpi=200)
    for ax, order in zip(axes, ("A", "B", "C")):
        order_rows = [r for r in rows if r["order"] == order]
        added_by_rung = {
            rung: next(r["added"] for r in order_rows if r["rung"] == rung)
            for rung in range(1, 9)
        }
        final_target = SOURCE_TO_GRAPH[added_by_rung[8]]
        mixture_values = []
        best_values = []

        for rung in range(1, 8):
            row = next(
                r for r in order_rows
                if r["rung"] == rung and r["graph"] == final_target
            )
            constituent_graphs = [
                SOURCE_TO_GRAPH[source] for source in row["sources"]
            ]
            best_source = max(
                constituent_graphs,
                key=lambda source: specialists[(source, final_target)],
            )
            raw_best_auc = specialists[(best_source, final_target)]
            # At rung 1 the mixture has exactly one constituent, so the two curves
            # coincide by definition. This also avoids displaying run-to-run noise
            # between the ladder checkpoint and the standalone matrix artifact.
            best_auc = row["auc"] if rung == 1 else raw_best_auc
            mixture_values.append(row["auc"])
            best_values.append(best_auc)
            result_rows.append({
                "order": order,
                "rung": rung,
                "added": added_by_rung[rung],
                "final_heldout_target": final_target,
                "mixture_auc": row["auc"],
                "best_constituent": best_source,
                "best_constituent_auc": best_auc,
                "raw_single_source_auc": raw_best_auc,
            })

        x = list(range(1, 8))
        ax.plot(
            x,
            best_values,
            color=GRAY,
            lw=2.5,
            marker="o",
            ms=5.8,
            markerfacecolor="white",
            markeredgecolor=GRAY,
            markeredgewidth=1.3,
            zorder=4,
        )
        ax.plot(
            x,
            mixture_values,
            color=BLUE,
            lw=2.8,
            marker="o",
            ms=6.2,
            markerfacecolor=BLUE,
            markeredgecolor="white",
            markeredgewidth=1.0,
            zorder=6,
        )
        values = mixture_values + best_values
        pad = max(0.012, 0.10 * (max(values) - min(values)))
        ax.set_ylim(max(0.0, min(values) - pad), min(1.0, max(values) + pad))
        ax.set_xlim(0.7, 8.3)
        ax.set_xticks(range(1, 9))
        ax.set_xticklabels(
            [SOURCE_LABEL[added_by_rung[rung]] for rung in range(1, 9)],
            rotation=35,
            ha="right",
            fontsize=8.4,
        )
        ax.set_xlabel("source added to pre-training mixture", fontsize=9.3, color=INK)
        ax.set_title(
            f"Order {order}  ·  {TARGET_LABEL[final_target]}",
            fontsize=12,
            color=INK,
            fontweight="bold",
        )
        chrome(ax)

    axes[0].set_ylabel("NM AUC on final held-out graph", fontsize=10.5, color=INK)
    fig.legend(
        handles=[
            Line2D(
                [0], [0], color=GRAY, lw=2.5, marker="o",
                markerfacecolor="white", markeredgecolor=GRAY, ms=6.2,
                label="best non-target constituent in mixture",
            ),
            Line2D(
                [0], [0], color=BLUE, lw=2.8, marker="o",
                markerfacecolor=BLUE, markeredgecolor="white", ms=6.5,
                label="mixture",
            ),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=2,
        frameon=False,
        fontsize=9.2,
        handlelength=2.5,
        columnspacing=2.3,
    )
    fig.suptitle(
        "Final held-out graph: mixture versus its best constituent",
        fontsize=14,
        color=INK,
        fontweight="bold",
        y=1.01,
    )
    fig.text(
        0.5,
        0.962,
        "best single-source constituent recomputed inside each mixture  ·  rung 1 curves coincide  ·  target specialist enters at empty rung 8",
        ha="center",
        va="top",
        fontsize=9.2,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.09, 1, 0.93))
    save(fig, "all_orders_final_heldout_mixture_vs_best_constituent_lines")
    plt.close(fig)

    data_out = os.path.join(
        HERE, "data", "all_orders_final_heldout_mixture_vs_best_constituent_lines.csv"
    )
    with open(data_out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(result_rows[0]))
        writer.writeheader()
        writer.writerows(result_rows)
    print("wrote", data_out)


def fig_heldout_mixture_vs_best_constituent_target_grid(rows):
    """Eight targets by three orders: mixture vs best in-mixture constituent."""
    specialists = load_single_source_matrix()
    target_ceiling = {}
    for target in TARGET_C:
        candidates = [source for source in TARGET_C if source != target]
        target_ceiling[target] = max(
            specialists[(source, target)] for source in candidates
        )
    targets = sorted(TARGET_C, key=target_ceiling.get, reverse=True)
    orders = ("A", "B", "C")
    grid_rows = []

    fig, axes = plt.subplots(
        len(targets), len(orders), figsize=(15.2, 24.0), dpi=200, sharey="row"
    )
    for target_index, target in enumerate(targets):
        row_values = []
        for order_index, order in enumerate(orders):
            ax = axes[target_index, order_index]
            order_rows = [r for r in rows if r["order"] == order]
            target_rows = sorted(
                (
                    r for r in order_rows
                    if r["graph"] == target and r["rel"] < 0
                ),
                key=lambda r: r["rung"],
            )
            entry_rung = next(
                r["entry_rung"] for r in order_rows if r["graph"] == target
            )
            mixture_values = []
            best_values = []
            for row in target_rows:
                constituent_graphs = [
                    SOURCE_TO_GRAPH[source] for source in row["sources"]
                ]
                best_source = max(
                    constituent_graphs,
                    key=lambda source: specialists[(source, target)],
                )
                raw_best_auc = specialists[(best_source, target)]
                best_auc = row["auc"] if row["rung"] == 1 else raw_best_auc
                mixture_values.append(row["auc"])
                best_values.append(best_auc)
                grid_rows.append({
                    "target": target,
                    "order": order,
                    "rung": row["rung"],
                    "target_entry_rung": entry_rung,
                    "added": row["added"],
                    "sources": " ".join(row["sources"]),
                    "mixture_auc": row["auc"],
                    "best_constituent": best_source,
                    "best_constituent_auc": best_auc,
                    "raw_single_source_auc": raw_best_auc,
                })

            if target_rows:
                x = [r["rung"] for r in target_rows]
                ax.plot(
                    x,
                    best_values,
                    color=GRAY,
                    lw=2.0,
                    marker="o",
                    ms=4.7,
                    markerfacecolor="white",
                    markeredgecolor=GRAY,
                    markeredgewidth=1.1,
                    zorder=4,
                )
                ax.plot(
                    x,
                    mixture_values,
                    color=BLUE,
                    lw=2.3,
                    marker="o",
                    ms=5.0,
                    markerfacecolor=BLUE,
                    markeredgecolor="white",
                    markeredgewidth=0.8,
                    zorder=6,
                )
                row_values.extend(mixture_values)
                row_values.extend(best_values)
            else:
                ax.text(
                    0.5,
                    0.5,
                    "enters at rung 1\n(no held-out point)",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=9.0,
                    color=MUTED,
                )

            ax.text(
                0.98,
                0.92,
                f"enters r{entry_rung}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8.0,
                color=MUTED,
            )
            ax.set_xlim(0.7, 8.3)
            ax.set_xticks(range(1, 9))
            if target_index == len(targets) - 1:
                added_by_rung = {
                    rung: next(
                        r["added"]
                        for r in order_rows
                        if r["rung"] == rung
                    )
                    for rung in range(1, 9)
                }
                ax.set_xticklabels(
                    [SOURCE_LABEL[added_by_rung[rung]] for rung in range(1, 9)],
                    rotation=35,
                    ha="right",
                    fontsize=8.0,
                )
                ax.set_xlabel("source added", fontsize=9.0, color=INK)
            else:
                ax.set_xticklabels([])
            if target_index == 0:
                ax.set_title(
                    f"Order {order}", fontsize=12, color=INK, fontweight="bold"
                )
            chrome(ax)

        pad = max(0.012, 0.10 * (max(row_values) - min(row_values)))
        axes[target_index, 0].set_ylim(
            max(0.0, min(row_values) - pad), min(1.0, max(row_values) + pad)
        )
        axes[target_index, 0].set_ylabel(
            f"{TARGET_LABEL[target]}\nNM AUC",
            fontsize=9.3,
            color=INK,
            fontweight="bold",
        )

    fig.legend(
        handles=[
            Line2D(
                [0], [0], color=GRAY, lw=2.2, marker="o",
                markerfacecolor="white", markeredgecolor=GRAY, ms=5.5,
                label="best non-target constituent in mixture",
            ),
            Line2D(
                [0], [0], color=BLUE, lw=2.4, marker="o",
                markerfacecolor=BLUE, markeredgecolor="white", ms=5.8,
                label="mixture",
            ),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.006),
        ncol=2,
        frameon=False,
        fontsize=9.5,
        handlelength=2.5,
        columnspacing=2.4,
    )
    fig.suptitle(
        "Held-out NM trajectories by target and source order",
        fontsize=15,
        color=INK,
        fontweight="bold",
        y=0.997,
    )
    fig.text(
        0.5,
        0.985,
        "blue = mixture  ·  gray = best single-source constituent currently inside the mixture  ·  each trajectory stops before target entry",
        ha="center",
        va="top",
        fontsize=9.4,
        color=MUTED,
    )
    fig.tight_layout(rect=(0.015, 0.025, 1, 0.975), h_pad=1.15, w_pad=1.0)
    save(fig, "all_targets_heldout_mixture_vs_best_constituent_grid")
    plt.close(fig)

    data_out = os.path.join(
        HERE, "data", "all_targets_heldout_mixture_vs_best_constituent_grid.csv"
    )
    with open(data_out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(grid_rows[0]))
        writer.writeheader()
        writer.writerows(grid_rows)
    print("wrote", data_out)


def fig_heldout_mixture_vs_best_constituent_entry_aligned_grid(rows):
    """Grid aligned by target entry rung, with longest held-out trajectories first."""
    specialists = load_single_source_matrix()
    orders = ("A", "B", "C")
    entry_rungs = list(range(8, 0, -1))
    grid_rows = []

    fig, axes = plt.subplots(
        len(entry_rungs), len(orders), figsize=(15.2, 24.0), dpi=200,
        sharey="row",
    )
    for row_index, entry_rung in enumerate(entry_rungs):
        row_values = []
        for order_index, order in enumerate(orders):
            ax = axes[row_index, order_index]
            order_rows = [r for r in rows if r["order"] == order]
            target = next(
                r["graph"]
                for r in order_rows
                if r["entry_rung"] == entry_rung
            )
            target_rows = sorted(
                (
                    r for r in order_rows
                    if r["graph"] == target and r["rel"] < 0
                ),
                key=lambda r: r["rung"],
            )
            mixture_values = []
            best_values = []
            for row in target_rows:
                constituent_graphs = [
                    SOURCE_TO_GRAPH[source] for source in row["sources"]
                ]
                best_source = max(
                    constituent_graphs,
                    key=lambda source: specialists[(source, target)],
                )
                raw_best_auc = specialists[(best_source, target)]
                best_auc = row["auc"] if row["rung"] == 1 else raw_best_auc
                mixture_values.append(row["auc"])
                best_values.append(best_auc)
                grid_rows.append({
                    "entry_rung": entry_rung,
                    "target": target,
                    "order": order,
                    "rung": row["rung"],
                    "added": row["added"],
                    "sources": " ".join(row["sources"]),
                    "mixture_auc": row["auc"],
                    "best_constituent": best_source,
                    "best_constituent_auc": best_auc,
                    "raw_single_source_auc": raw_best_auc,
                })

            if target_rows:
                x = [r["rung"] for r in target_rows]
                ax.plot(
                    x,
                    best_values,
                    color=GRAY,
                    lw=2.0,
                    marker="o",
                    ms=4.7,
                    markerfacecolor="white",
                    markeredgecolor=GRAY,
                    markeredgewidth=1.1,
                    zorder=4,
                )
                ax.plot(
                    x,
                    mixture_values,
                    color=BLUE,
                    lw=2.3,
                    marker="o",
                    ms=5.0,
                    markerfacecolor=BLUE,
                    markeredgecolor="white",
                    markeredgewidth=0.8,
                    zorder=6,
                )
                row_values.extend(mixture_values)
                row_values.extend(best_values)
            else:
                ax.text(
                    0.5,
                    0.5,
                    "no held-out point",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=9.0,
                    color=MUTED,
                )

            ax.text(
                0.02,
                0.92,
                TARGET_LABEL[target],
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9.0,
                color=INK,
                fontweight="bold",
            )
            ax.set_xlim(0.7, 8.3)
            ax.set_xticks(range(1, 9))
            if row_index == len(entry_rungs) - 1:
                added_by_rung = {
                    rung: next(
                        r["added"] for r in order_rows if r["rung"] == rung
                    )
                    for rung in range(1, 9)
                }
                ax.set_xticklabels(
                    [SOURCE_LABEL[added_by_rung[rung]] for rung in range(1, 9)],
                    rotation=35,
                    ha="right",
                    fontsize=8.0,
                )
                ax.set_xlabel("source added", fontsize=9.0, color=INK)
            else:
                ax.set_xticklabels([])
            if row_index == 0:
                ax.set_title(
                    f"Order {order}", fontsize=12, color=INK, fontweight="bold"
                )
            chrome(ax)

        if row_values:
            pad = max(0.012, 0.10 * (max(row_values) - min(row_values)))
            axes[row_index, 0].set_ylim(
                max(0.0, min(row_values) - pad), min(1.0, max(row_values) + pad)
            )
        else:
            axes[row_index, 0].set_ylim(0.5, 1.0)
        axes[row_index, 0].set_ylabel(
            f"enters r{entry_rung}\nNM AUC",
            fontsize=9.3,
            color=INK,
            fontweight="bold",
        )

    fig.legend(
        handles=[
            Line2D(
                [0], [0], color=GRAY, lw=2.2, marker="o",
                markerfacecolor="white", markeredgecolor=GRAY, ms=5.5,
                label="best non-target constituent in mixture",
            ),
            Line2D(
                [0], [0], color=BLUE, lw=2.4, marker="o",
                markerfacecolor=BLUE, markeredgecolor="white", ms=5.8,
                label="mixture",
            ),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.006),
        ncol=2,
        frameon=False,
        fontsize=9.5,
        handlelength=2.5,
        columnspacing=2.4,
    )
    fig.suptitle(
        "Held-out NM trajectories aligned by target entry rung",
        fontsize=15,
        color=INK,
        fontweight="bold",
        y=0.997,
    )
    fig.text(
        0.5,
        0.985,
        "rows run from final held-out target (r8) to first-added target (r1)  ·  blue = mixture  ·  gray = best constituent inside mixture",
        ha="center",
        va="top",
        fontsize=9.4,
        color=MUTED,
    )
    fig.tight_layout(rect=(0.015, 0.025, 1, 0.975), h_pad=1.15, w_pad=1.0)
    save(fig, "all_targets_heldout_mixture_vs_best_constituent_grid")
    plt.close(fig)

    data_out = os.path.join(
        HERE, "data", "all_targets_heldout_mixture_vs_best_constituent_grid.csv"
    )
    with open(data_out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(grid_rows[0]))
        writer.writeheader()
        writer.writerows(grid_rows)
    print("wrote", data_out)


def fig_heldout_mixture_advantage_entry_aligned_grid(rows):
    """Per-target mixture-minus-best-constituent bars aligned by entry rung."""
    specialists = load_single_source_matrix()
    orders = ("A", "B", "C")
    entry_rungs = list(range(8, 0, -1))
    result_rows = []

    fig, axes = plt.subplots(
        len(entry_rungs), len(orders), figsize=(15.2, 24.0), dpi=200,
        sharex=True, sharey=True,
    )
    for row_index, entry_rung in enumerate(entry_rungs):
        for order_index, order in enumerate(orders):
            ax = axes[row_index, order_index]
            order_rows = [r for r in rows if r["order"] == order]
            target = next(
                r["graph"]
                for r in order_rows
                if r["entry_rung"] == entry_rung
            )
            target_rows = sorted(
                (
                    r for r in order_rows
                    if r["graph"] == target and r["rel"] < 0
                ),
                key=lambda r: r["rung"],
            )

            x = []
            gaps = []
            for row in target_rows:
                constituent_graphs = [
                    SOURCE_TO_GRAPH[source] for source in row["sources"]
                ]
                best_source = max(
                    constituent_graphs,
                    key=lambda source: specialists[(source, target)],
                )
                best_auc = specialists[(best_source, target)]
                gap = 0.0 if row["rung"] == 1 else row["auc"] - best_auc
                x.append(row["rung"])
                gaps.append(gap)
                result_rows.append({
                    "entry_rung": entry_rung,
                    "target": target,
                    "order": order,
                    "rung": row["rung"],
                    "added": row["added"],
                    "sources": " ".join(row["sources"]),
                    "mixture_auc": row["auc"],
                    "best_constituent": best_source,
                    "best_constituent_auc": best_auc,
                    "mixture_minus_best_constituent_auc": gap,
                })

            if x:
                colors = [
                    GRAY if abs(gap) < 1e-12 else BLUE if gap > 0 else CORAL
                    for gap in gaps
                ]
                ax.bar(
                    x,
                    gaps,
                    width=0.66,
                    color=colors,
                    edgecolor="white",
                    linewidth=0.7,
                    zorder=4,
                )
                if x[0] == 1:
                    ax.scatter(
                        [1], [0], s=22, color=GRAY, edgecolor="white",
                        linewidth=0.7, zorder=6,
                    )
            else:
                ax.text(
                    0.5,
                    0.5,
                    "no held-out point",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=9.0,
                    color=MUTED,
                )

            ax.axhline(0, color=INK, lw=0.9, ls=(0, (4, 3)), zorder=2)
            ax.text(
                0.02,
                0.91,
                TARGET_LABEL[target],
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9.0,
                color=INK,
                fontweight="bold",
            )
            ax.set_xlim(0.7, 8.3)
            ax.set_ylim(-0.04, 0.11)
            ax.set_xticks(range(1, 9))
            if row_index == len(entry_rungs) - 1:
                added_by_rung = {
                    rung: next(
                        r["added"] for r in order_rows if r["rung"] == rung
                    )
                    for rung in range(1, 9)
                }
                ax.set_xticklabels(
                    [SOURCE_LABEL[added_by_rung[rung]] for rung in range(1, 9)],
                    rotation=35,
                    ha="right",
                    fontsize=8.0,
                )
                ax.set_xlabel("source added", fontsize=9.0, color=INK)
            if row_index == 0:
                ax.set_title(
                    f"Order {order}", fontsize=12, color=INK, fontweight="bold"
                )
            chrome(ax)

        axes[row_index, 0].set_ylabel(
            f"enters r{entry_rung}\nΔ AUC",
            fontsize=9.3,
            color=INK,
            fontweight="bold",
        )

    fig.legend(
        handles=[
            Patch(facecolor=BLUE, edgecolor="white", label="mixture beats constituent"),
            Patch(facecolor=CORAL, edgecolor="white", label="mixture trails constituent"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.006),
        ncol=2,
        frameon=False,
        fontsize=9.5,
        handlelength=1.5,
        columnspacing=2.4,
    )
    fig.suptitle(
        "Mixture advantage over the best constituent, by held-out target",
        fontsize=15,
        color=INK,
        fontweight="bold",
        y=0.997,
    )
    fig.text(
        0.5,
        0.985,
        "Δ AUC = mixture − best single-source constituent inside the mixture  ·  common y-scale  ·  rows aligned by target entry rung",
        ha="center",
        va="top",
        fontsize=9.4,
        color=MUTED,
    )
    fig.tight_layout(rect=(0.015, 0.025, 1, 0.975), h_pad=1.15, w_pad=1.0)
    save(fig, "all_targets_heldout_mixture_advantage_grid")
    plt.close(fig)

    data_out = os.path.join(
        HERE, "data", "all_targets_heldout_mixture_advantage_grid.csv"
    )
    with open(data_out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(result_rows[0]))
        writer.writeheader()
        writer.writerows(result_rows)
    print("wrote", data_out)


def fig_all_heldout_targets_mixture_minus_best_constituent_bars(rows):
    """Grouped residual bars for every target that is still held out at each rung."""
    specialists = load_single_source_matrix()
    result_rows = []

    for order in ("A", "B", "C"):
        order_rows = [r for r in rows if r["order"] == order]
        for rung in range(1, 8):
            heldout_rows = [
                r for r in order_rows if r["rung"] == rung and r["rel"] < 0
            ]
            for row in heldout_rows:
                constituent_graphs = [
                    SOURCE_TO_GRAPH[source] for source in row["sources"]
                ]
                constituent_aucs = [
                    specialists[(source, row["graph"])]
                    for source in constituent_graphs
                ]
                best_index = int(np.argmax(constituent_aucs))
                best_source = constituent_graphs[best_index]
                best_auc = constituent_aucs[best_index]
                gap = 0.0 if rung == 1 else row["auc"] - best_auc
                result_rows.append({
                    "order": order,
                    "rung": rung,
                    "added": row["added"],
                    "target": row["graph"],
                    "target_entry_rung": row["entry_rung"],
                    "mixture_auc": row["auc"],
                    "best_constituent": best_source,
                    "best_constituent_auc": best_auc,
                    "mixture_minus_best_constituent_auc": gap,
                })

    data_out = os.path.join(
        HERE, "data", "all_orders_all_heldout_mixture_minus_best_constituent.csv"
    )
    with open(data_out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(result_rows[0]))
        writer.writeheader()
        writer.writerows(result_rows)
    print("wrote", data_out)

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.8), dpi=200, sharey=True)
    for ax, order in zip(axes, ("A", "B", "C")):
        order_results = [r for r in result_rows if r["order"] == order]
        order_source_rows = [r for r in rows if r["order"] == order]
        for rung in range(1, 8):
            rung_rows = sorted(
                (r for r in order_results if r["rung"] == rung),
                key=lambda r: list(TARGET_C).index(r["target"]),
            )
            n_targets = len(rung_rows)
            width = 0.68 / n_targets
            offsets = (np.arange(n_targets) - (n_targets - 1) / 2) * width
            for offset, result in zip(offsets, rung_rows):
                value = result["mixture_minus_best_constituent_auc"]
                ax.bar(
                    rung + offset,
                    value,
                    width=width * 0.92,
                    color=TARGET_C[result["target"]],
                    edgecolor="white",
                    linewidth=0.65,
                    zorder=4,
                )
                if rung == 1:
                    ax.scatter(
                        [rung + offset], [0], s=18,
                        color=TARGET_C[result["target"]], edgecolor="white",
                        linewidth=0.6, zorder=6,
                    )

        added_by_rung = {
            rung: next(
                r["added"] for r in order_source_rows if r["rung"] == rung
            )
            for rung in range(1, 9)
        }
        ax.axhline(0, color=INK, lw=1.0, ls=(0, (4, 3)), zorder=2)
        ax.set_xlim(0.55, 8.35)
        ax.set_ylim(-0.035, 0.105)
        ax.set_xticks(range(1, 9))
        ax.set_xticklabels(
            [SOURCE_LABEL[added_by_rung[rung]] for rung in range(1, 9)],
            rotation=35,
            ha="right",
            fontsize=8.4,
        )
        ax.set_xlabel("source added to pre-training mixture", fontsize=9.3, color=INK)
        ax.set_title(f"Order {order}", fontsize=12, color=INK, fontweight="bold")
        chrome(ax)

    axes[0].set_ylabel(
        "mixture AUC − best constituent AUC", fontsize=10.5, color=INK
    )
    handles = [
        Patch(facecolor=TARGET_C[target], edgecolor="white", label=TARGET_LABEL[target])
        for target in TARGET_C
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.005),
        ncol=8,
        frameon=False,
        fontsize=8.7,
        handlelength=1.3,
        columnspacing=1.15,
    )
    fig.suptitle(
        "Mixture advantage on every graph while it remains held out",
        fontsize=14,
        color=INK,
        fontweight="bold",
        y=1.01,
    )
    fig.text(
        0.5,
        0.962,
        "one bar per held-out target  ·  best non-specialist constituent selected per target and rung  ·  rung 1 = 0  ·  rung 8 empty",
        ha="center",
        va="top",
        fontsize=9.2,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.085, 1, 0.93))
    save(fig, "all_orders_all_heldout_mixture_minus_best_constituent_bars")
    plt.close(fig)


def fig_heldout_auc_profiles_vs_best_non_specialist(rows):
    """Held-out AUC bars with each mixture's best constituent as the reference."""
    specialists = load_single_source_matrix()
    targets = list(TARGET_C)
    oracle = {}
    for target in targets:
        candidates = [source for source in targets if source != target]
        best_source = max(candidates, key=lambda source: specialists[(source, target)])
        oracle[target] = {
            "source": best_source,
            "auc": specialists[(best_source, target)],
        }
    targets = sorted(targets, key=lambda target: oracle[target]["auc"], reverse=True)

    profile_rows = []
    cmap = plt.get_cmap("Blues")
    rung_colors = {
        rung: cmap(0.30 + 0.10 * (rung - 1)) for rung in range(1, 8)
    }

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.8), dpi=200, sharey=True)
    for ax, order in zip(axes, ("A", "B", "C")):
        order_rows = [r for r in rows if r["order"] == order]
        for rung in range(1, 8):
            heldout = {
                r["graph"]: r
                for r in order_rows
                if r["rung"] == rung and r["rel"] < 0
            }
            x = [index for index, target in enumerate(targets) if target in heldout]
            y = [heldout[target]["auc"] for target in targets if target in heldout]
            offset = (rung - 4) * 0.09
            bar_x = [value + offset for value in x]
            ax.bar(
                bar_x,
                [value - 0.5 for value in y],
                bottom=0.5,
                width=0.085,
                color=rung_colors[rung],
                edgecolor="white",
                linewidth=0.65,
                alpha=0.96,
                zorder=3 + rung,
            )
            for bar_position, target in zip(
                bar_x, (target for target in targets if target in heldout)
            ):
                row = heldout[target]
                constituent_graphs = [
                    SOURCE_TO_GRAPH[source] for source in row["sources"]
                ]
                best_source = max(
                    constituent_graphs,
                    key=lambda source: specialists[(source, target)],
                )
                best_auc = specialists[(best_source, target)]
                ax.plot(
                    [bar_position - 0.037, bar_position + 0.037],
                    [best_auc, best_auc],
                    color=GRAY,
                    lw=2.0,
                    solid_capstyle="round",
                    zorder=15,
                )
                profile_rows.append({
                    "order": order,
                    "rung": rung,
                    "n_sources": rung,
                    "sources": " ".join(row["sources"]),
                    "target": target,
                    "mixture_auc": row["auc"],
                    "best_constituent_source": best_source,
                    "best_constituent_auc": best_auc,
                    "mixture_minus_best_constituent": (
                        row["auc"] - best_auc
                    ),
                })

        ax.set_xlim(-0.3, len(targets) - 0.7)
        ax.set_ylim(0.5, 1.0)
        ax.set_xticks(range(len(targets)))
        ax.set_xticklabels(
            [TARGET_LABEL[target] for target in targets],
            rotation=35,
            ha="right",
            fontsize=8.4,
        )
        ax.set_xlabel("target graph", fontsize=9.5, color=INK)
        ax.set_title(f"Order {order}", fontsize=12, color=INK, fontweight="bold")
        chrome(ax)

    axes[0].set_ylabel("held-out NM AUC", fontsize=10.5, color=INK)
    handles = [
        Line2D(
            [0], [0], color=GRAY, lw=2.6,
            label="best constituent in mixture",
        )
    ] + [
        Patch(
            facecolor=rung_colors[rung], edgecolor="white",
            label=f"{rung}-source mixture",
        )
        for rung in range(1, 8)
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.005),
        ncol=8,
        frameon=False,
        fontsize=8.4,
        handlelength=2.1,
        columnspacing=1.05,
    )
    fig.suptitle(
        "Held-out NM performance across targets and mixture sizes",
        fontsize=14,
        color=INK,
        fontweight="bold",
        y=1.01,
    )
    fig.text(
        0.5,
        0.962,
        "gray tick = best single-source constituent inside that mixture  ·  bars show held-out targets only  ·  darker = larger mixture",
        ha="center",
        va="top",
        fontsize=9.2,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.09, 1, 0.93))
    save(fig, "all_orders_heldout_auc_by_target_mixture_size")
    plt.close(fig)

    data_out = os.path.join(
        HERE, "data", "all_orders_heldout_auc_by_target_mixture_size.csv"
    )
    with open(data_out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(profile_rows[0]))
        writer.writeheader()
        writer.writerows(profile_rows)
    print("wrote", data_out)


def fig_final_heldout_mixture_and_constituents_by_order(rows):
    """Track each order's last-held-out target against every included constituent."""
    specialists = load_single_source_matrix()
    data_rows = []

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.9), dpi=200)
    for ax, order in zip(axes, ("A", "B", "C")):
        order_rows = [r for r in rows if r["order"] == order]
        added_by_rung = {
            r["rung"]: r["added"]
            for r in order_rows
            if r["graph"] == "ukr_rus_twitter"
        }
        final_source = added_by_rung[8]
        final_target = SOURCE_TO_GRAPH[final_source]
        mixture_rows = sorted(
            (
                r for r in order_rows
                if r["graph"] == final_target and r["rung"] < 8
            ),
            key=lambda r: r["rung"],
        )

        mixture_x = [r["rung"] for r in mixture_rows]
        mixture_y = [r["auc"] for r in mixture_rows]
        ax.plot(
            mixture_x,
            mixture_y,
            color=INK,
            lw=3.0,
            marker="o",
            ms=6.2,
            markerfacecolor=INK,
            markeredgecolor="white",
            markeredgewidth=1.0,
            label="mixture",
            zorder=8,
        )
        for row in mixture_rows:
            data_rows.append({
                "order": order,
                "final_heldout_target": final_target,
                "series": "mixture",
                "source": "mixture",
                "entry_rung": "",
                "rung": row["rung"],
                "auc": row["auc"],
            })

        constituent_values = []
        for entry_rung in range(1, 8):
            source_key = added_by_rung[entry_rung]
            source_graph = SOURCE_TO_GRAPH[source_key]
            auc = specialists[(source_graph, final_target)]
            constituent_values.append(auc)
            # A source becomes a constituent at its entry rung, so draw only the
            # portion of its single-source baseline that is available to that prefix.
            ax.plot(
                [entry_rung, 7],
                [auc, auc],
                color=TARGET_C[source_graph],
                lw=1.8,
                ls=(0, (3, 2)),
                alpha=0.95,
                solid_capstyle="round",
                zorder=3,
            )
            ax.scatter(
                [entry_rung],
                [auc],
                s=34,
                color=TARGET_C[source_graph],
                edgecolor="white",
                linewidth=0.8,
                zorder=6,
            )
            for rung in range(entry_rung, 8):
                data_rows.append({
                    "order": order,
                    "final_heldout_target": final_target,
                    "series": "constituent",
                    "source": source_graph,
                    "entry_rung": entry_rung,
                    "rung": rung,
                    "auc": auc,
                })

        values = mixture_y + constituent_values
        pad = max(0.015, 0.08 * (max(values) - min(values)))
        ax.set_ylim(max(0.0, min(values) - pad), min(1.0, max(values) + pad))
        ax.set_xlim(0.7, 8.3)
        ax.set_xticks(range(1, 9))
        ax.set_xticklabels(
            [SOURCE_LABEL[added_by_rung[rung]] for rung in range(1, 9)],
            rotation=35,
            ha="right",
            fontsize=8.4,
        )
        ax.set_xlabel("source added to pre-training mixture", fontsize=9.3, color=INK)
        ax.set_title(
            f"Order {order}  ·  {TARGET_LABEL[final_target]}",
            fontsize=12,
            color=INK,
            fontweight="bold",
        )
        chrome(ax)

    axes[0].set_ylabel("NM AUC on final held-out graph", fontsize=10.5, color=INK)
    handles = [
        Line2D(
            [0], [0], color=INK, lw=3.0, marker="o", markerfacecolor=INK,
            markeredgecolor="white", ms=6.5, label="mixture",
        )
    ] + [
        Line2D(
            [0], [0], color=TARGET_C[graph], lw=1.8, ls=(0, (3, 2)),
            marker="o", markerfacecolor=TARGET_C[graph], markeredgecolor="white",
            ms=5.5, label=TARGET_LABEL[graph],
        )
        for graph in TARGET_C
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.005),
        ncol=9,
        frameon=False,
        fontsize=8.3,
        handlelength=2.2,
        columnspacing=1.0,
    )
    fig.suptitle(
        "Final held-out graph: mixture trajectory versus every constituent",
        fontsize=14,
        color=INK,
        fontweight="bold",
        y=1.01,
    )
    fig.text(
        0.5,
        0.962,
        "constituent baseline begins when its source enters the mixture  ·  final target specialist is excluded  ·  rung 8 empty",
        ha="center",
        va="top",
        fontsize=9.2,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.085, 1, 0.93))
    save(fig, "all_orders_final_heldout_mixture_and_constituents")
    plt.close(fig)

    data_out = os.path.join(
        HERE, "data", "all_orders_final_heldout_mixture_and_constituents.csv"
    )
    with open(data_out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(data_rows[0]))
        writer.writeheader()
        writer.writerows(data_rows)
    print("wrote", data_out)


def fig_final_heldout_swap_first_two_by_order(rows):
    """Counterfactual final-target curves with each order's first two sources swapped."""
    specialists = load_single_source_matrix()
    data_rows = []

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.9), dpi=200)
    for ax, order in zip(axes, ("A", "B", "C")):
        order_rows = [r for r in rows if r["order"] == order]
        original_sources = [
            next(r["added"] for r in order_rows if r["rung"] == rung)
            for rung in range(1, 9)
        ]
        swapped_sources = [
            original_sources[1], original_sources[0], *original_sources[2:]
        ]
        final_target = SOURCE_TO_GRAPH[swapped_sources[-1]]

        observed_mixture = {
            r["rung"]: r["auc"]
            for r in order_rows
            if r["graph"] == final_target and 2 <= r["rung"] < 8
        }
        first_source_graph = SOURCE_TO_GRAPH[swapped_sources[0]]
        mixture_y = [specialists[(first_source_graph, final_target)]] + [
            observed_mixture[rung] for rung in range(2, 8)
        ]
        ax.plot(
            range(1, 8),
            mixture_y,
            color=INK,
            lw=3.0,
            marker="o",
            ms=6.2,
            markerfacecolor=INK,
            markeredgecolor="white",
            markeredgewidth=1.0,
            label="mixture",
            zorder=8,
        )
        for rung, auc in enumerate(mixture_y, start=1):
            data_rows.append({
                "order": order,
                "final_heldout_target": final_target,
                "series": "mixture",
                "source": first_source_graph if rung == 1 else "mixture",
                "entry_rung": "" if rung > 1 else 1,
                "rung": rung,
                "auc": auc,
                "value_origin": "single_source_reconstruction" if rung == 1 else "observed_same_source_set",
            })

        constituent_values = []
        for entry_rung, source_key in enumerate(swapped_sources[:7], start=1):
            source_graph = SOURCE_TO_GRAPH[source_key]
            auc = specialists[(source_graph, final_target)]
            constituent_values.append(auc)
            ax.plot(
                [entry_rung, 7],
                [auc, auc],
                color=TARGET_C[source_graph],
                lw=1.8,
                ls=(0, (3, 2)),
                alpha=0.95,
                solid_capstyle="round",
                zorder=3,
            )
            ax.scatter(
                [entry_rung],
                [auc],
                s=34,
                color=TARGET_C[source_graph],
                edgecolor="white",
                linewidth=0.8,
                zorder=6,
            )
            for rung in range(entry_rung, 8):
                data_rows.append({
                    "order": order,
                    "final_heldout_target": final_target,
                    "series": "constituent",
                    "source": source_graph,
                    "entry_rung": entry_rung,
                    "rung": rung,
                    "auc": auc,
                    "value_origin": "single_source_matrix",
                })

        values = mixture_y + constituent_values
        pad = max(0.015, 0.08 * (max(values) - min(values)))
        ax.set_ylim(max(0.0, min(values) - pad), min(1.0, max(values) + pad))
        ax.set_xlim(0.7, 8.3)
        ax.set_xticks(range(1, 9))
        ax.set_xticklabels(
            [SOURCE_LABEL[source] for source in swapped_sources],
            rotation=35,
            ha="right",
            fontsize=8.4,
        )
        ax.set_xlabel("source added to pre-training mixture", fontsize=9.3, color=INK)
        ax.set_title(
            f"Order {order}′  ·  {TARGET_LABEL[final_target]}",
            fontsize=12,
            color=INK,
            fontweight="bold",
        )
        chrome(ax)

    axes[0].set_ylabel("NM AUC on final held-out graph", fontsize=10.5, color=INK)
    handles = [
        Line2D(
            [0], [0], color=INK, lw=3.0, marker="o", markerfacecolor=INK,
            markeredgecolor="white", ms=6.5, label="mixture",
        )
    ] + [
        Line2D(
            [0], [0], color=TARGET_C[graph], lw=1.8, ls=(0, (3, 2)),
            marker="o", markerfacecolor=TARGET_C[graph], markeredgecolor="white",
            ms=5.5, label=TARGET_LABEL[graph],
        )
        for graph in TARGET_C
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.005),
        ncol=9,
        frameon=False,
        fontsize=8.3,
        handlelength=2.2,
        columnspacing=1.0,
    )
    fig.suptitle(
        "Final held-out graph after swapping each order's first two sources",
        fontsize=14,
        color=INK,
        fontweight="bold",
        y=1.01,
    )
    fig.text(
        0.5,
        0.962,
        "rung 1 reconstructed from the new first source  ·  rungs 2–7 reuse observed mixtures with identical source sets  ·  rung 8 empty",
        ha="center",
        va="top",
        fontsize=9.2,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.085, 1, 0.93))
    save(fig, "all_orders_final_heldout_swap_first_two")
    plt.close(fig)

    data_out = os.path.join(
        HERE, "data", "all_orders_final_heldout_swap_first_two.csv"
    )
    with open(data_out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(data_rows[0]))
        writer.writeheader()
        writer.writerows(data_rows)
    print("wrote", data_out)


def fig_heldout_swap_first_two_mixture_vs_best_by_order(rows):
    """Per-target held-out mixture and max-constituent curves after swapping rung 1."""
    specialists = load_single_source_matrix()
    comparison_rows = []

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.9), dpi=200, sharey=True)
    for ax, order in zip(axes, ("A", "B", "C")):
        order_rows = [r for r in rows if r["order"] == order]
        original_sources = [
            next(r["added"] for r in order_rows if r["rung"] == rung)
            for rung in range(1, 9)
        ]
        swapped_sources = [
            original_sources[1], original_sources[0], *original_sources[2:]
        ]
        entry_by_target = {
            SOURCE_TO_GRAPH[source]: rung
            for rung, source in enumerate(swapped_sources, start=1)
        }

        for target in TARGET_C:
            entry_rung = entry_by_target[target]
            heldout_rungs = list(range(1, entry_rung))
            if not heldout_rungs:
                continue

            mixture_values = []
            best_values = []
            for rung in heldout_rungs:
                if rung == 1:
                    mixture_auc = specialists[
                        (SOURCE_TO_GRAPH[swapped_sources[0]], target)
                    ]
                    value_origin = "single_source_reconstruction"
                else:
                    mixture_auc = next(
                        r["auc"]
                        for r in order_rows
                        if r["graph"] == target and r["rung"] == rung
                    )
                    value_origin = "observed_same_source_set"

                constituent_graphs = [
                    SOURCE_TO_GRAPH[source] for source in swapped_sources[:rung]
                ]
                constituent_aucs = [
                    specialists[(source, target)] for source in constituent_graphs
                ]
                best_index = int(np.argmax(constituent_aucs))
                best_source = constituent_graphs[best_index]
                best_auc = constituent_aucs[best_index]
                mixture_values.append(mixture_auc)
                best_values.append(best_auc)
                comparison_rows.append({
                    "order": order,
                    "rung": rung,
                    "target": target,
                    "target_entry_rung": entry_rung,
                    "sources": " ".join(swapped_sources[:rung]),
                    "mixture_auc": mixture_auc,
                    "mixture_value_origin": value_origin,
                    "best_constituent": best_source,
                    "best_constituent_auc": best_auc,
                    "mixture_minus_best": mixture_auc - best_auc,
                })

            ax.plot(
                heldout_rungs,
                mixture_values,
                color=TARGET_C[target],
                lw=2.2,
                marker="o",
                ms=5.2,
                markerfacecolor=TARGET_C[target],
                markeredgecolor="white",
                markeredgewidth=0.9,
                zorder=6,
            )
            ax.plot(
                heldout_rungs,
                best_values,
                color=TARGET_C[target],
                lw=1.7,
                ls=(0, (3, 2)),
                marker="o",
                ms=4.7,
                markerfacecolor="white",
                markeredgecolor=TARGET_C[target],
                markeredgewidth=1.1,
                alpha=0.95,
                zorder=4,
            )

        ax.set_xlim(0.7, 8.3)
        ax.set_ylim(0.53, 1.0)
        ax.set_xticks(range(1, 9))
        ax.set_xticklabels(
            [SOURCE_LABEL[source] for source in swapped_sources],
            rotation=35,
            ha="right",
            fontsize=8.4,
        )
        ax.set_xlabel("source added to pre-training mixture", fontsize=9.3, color=INK)
        ax.set_title(f"Order {order}′", fontsize=12, color=INK, fontweight="bold")
        chrome(ax)

    axes[0].set_ylabel("held-out NM AUC", fontsize=10.5, color=INK)
    handles = [
        Line2D(
            [0], [0], color=INK, lw=2.2, marker="o", markerfacecolor=INK,
            markeredgecolor="white", ms=5.5, label="mixture",
        ),
        Line2D(
            [0], [0], color=INK, lw=1.7, ls=(0, (3, 2)), marker="o",
            markerfacecolor="white", markeredgecolor=INK, ms=5.2,
            label="best constituent so far",
        ),
    ] + [
        Line2D([0], [0], color=TARGET_C[target], lw=2.2,
               label=TARGET_LABEL[target])
        for target in TARGET_C
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.005),
        ncol=10,
        frameon=False,
        fontsize=8.1,
        handlelength=2.1,
        columnspacing=0.9,
    )
    fig.suptitle(
        "Held-out mixture performance versus the best constituent available so far",
        fontsize=14,
        color=INK,
        fontweight="bold",
        y=1.01,
    )
    fig.text(
        0.5,
        0.962,
        "first two sources swapped  ·  max constituent selected separately for every target and rung  ·  each target stops at entry",
        ha="center",
        va="top",
        fontsize=9.2,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.085, 1, 0.93))
    save(fig, "all_orders_heldout_swap_first_two_mixture_vs_best_per_target")
    plt.close(fig)

    data_out = os.path.join(
        HERE, "data", "all_orders_heldout_swap_first_two_mixture_vs_best_per_target.csv"
    )
    with open(data_out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(comparison_rows[0]))
        writer.writeheader()
        writer.writerows(comparison_rows)
    print("wrote", data_out)


def fig_heldout_swap_first_two_mean_mixture_vs_best_by_order(rows):
    """Mean held-out mixture and per-target max-constituent AUC after rung-1 swap."""
    specialists = load_single_source_matrix()
    summary_rows = []

    for order in ("A", "B", "C"):
        order_rows = [r for r in rows if r["order"] == order]
        original_sources = [
            next(r["added"] for r in order_rows if r["rung"] == rung)
            for rung in range(1, 9)
        ]
        swapped_sources = [
            original_sources[1], original_sources[0], *original_sources[2:]
        ]

        for rung in range(1, 8):
            source_graphs = [
                SOURCE_TO_GRAPH[source] for source in swapped_sources[:rung]
            ]
            heldout_targets = [
                graph for graph in TARGET_C if graph not in source_graphs
            ]
            mixture_values = []
            best_values = []
            for target in heldout_targets:
                if rung == 1:
                    mixture_auc = specialists[(source_graphs[0], target)]
                else:
                    mixture_auc = next(
                        r["auc"]
                        for r in order_rows
                        if r["graph"] == target and r["rung"] == rung
                    )
                best_auc = max(
                    specialists[(source, target)] for source in source_graphs
                )
                mixture_values.append(mixture_auc)
                best_values.append(best_auc)

            summary_rows.append({
                "order": order,
                "rung": rung,
                "added": swapped_sources[rung - 1],
                "sources": " ".join(swapped_sources[:rung]),
                "n_heldout": len(heldout_targets),
                "mean_mixture_auc": float(np.mean(mixture_values)),
                "mean_best_constituent_auc": float(np.mean(best_values)),
                "mean_mixture_minus_best": float(
                    np.mean(np.asarray(mixture_values) - np.asarray(best_values))
                ),
            })

    data_out = os.path.join(
        HERE, "data", "all_orders_heldout_swap_first_two_mean_mixture_vs_best.csv"
    )
    with open(data_out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    print("wrote", data_out)

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.7), dpi=200, sharey=True)
    for ax, order in zip(axes, ("A", "B", "C")):
        order_summary = [r for r in summary_rows if r["order"] == order]
        original_sources = [
            next(
                r["added"]
                for r in rows
                if r["order"] == order and r["rung"] == rung
            )
            for rung in range(1, 9)
        ]
        swapped_sources = [
            original_sources[1], original_sources[0], *original_sources[2:]
        ]
        x = [r["rung"] for r in order_summary]
        ax.plot(
            x,
            [r["mean_mixture_auc"] for r in order_summary],
            color=BLUE,
            lw=2.7,
            marker="o",
            ms=6.2,
            markerfacecolor=BLUE,
            markeredgecolor="white",
            markeredgewidth=1.0,
            zorder=6,
        )
        ax.plot(
            x,
            [r["mean_best_constituent_auc"] for r in order_summary],
            color=GRAY,
            lw=2.3,
            marker="o",
            ms=5.8,
            markerfacecolor="white",
            markeredgecolor=GRAY,
            markeredgewidth=1.3,
            zorder=5,
        )
        ax.set_xlim(0.7, 8.3)
        ax.set_ylim(0.63, 1.0)
        ax.set_xticks(range(1, 9))
        ax.set_xticklabels(
            [SOURCE_LABEL[source] for source in swapped_sources],
            rotation=35,
            ha="right",
            fontsize=8.4,
        )
        ax.set_xlabel("source added to pre-training mixture", fontsize=9.3, color=INK)
        ax.set_title(f"Order {order}′", fontsize=12, color=INK, fontweight="bold")
        chrome(ax)

    axes[0].set_ylabel("mean held-out NM AUC", fontsize=10.5, color=INK)
    fig.legend(
        handles=[
            Line2D(
                [0], [0], color=BLUE, lw=2.7, marker="o",
                markerfacecolor=BLUE, markeredgecolor="white", ms=6.5,
                label="mixture",
            ),
            Line2D(
                [0], [0], color=GRAY, lw=2.3, marker="o",
                markerfacecolor="white", markeredgecolor=GRAY, ms=6.2,
                label="best constituent per target",
            ),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=2,
        frameon=False,
        fontsize=9.2,
        handlelength=2.4,
        columnspacing=2.2,
    )
    fig.suptitle(
        "Mean held-out performance after swapping each order's first two sources",
        fontsize=14,
        color=INK,
        fontweight="bold",
        y=1.01,
    )
    fig.text(
        0.5,
        0.962,
        "best constituent selected separately per target, then averaged  ·  held-out set shrinks from 7 to 1  ·  rung 8 empty",
        ha="center",
        va="top",
        fontsize=9.2,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.09, 1, 0.93))
    save(fig, "all_orders_heldout_swap_first_two_mean_mixture_vs_best")
    plt.close(fig)


# --------------------------------------------------------------------- figure 1
def fig_entry_aligned(rows):
    """Every (graph, order) AUC curve aligned on its own entry rung, + mean and band."""
    by_pair = defaultdict(dict)                       # (graph, order) -> {rel: auc}
    for r in rows:
        by_pair[(r["graph"], r["order"])][r["rel"]] = r["auc"]

    rels = list(range(-4, 8))
    at_rel = defaultdict(list)
    for series in by_pair.values():
        for rel, auc in series.items():
            at_rel[rel].append(auc)
    # keep offsets covered by a healthy number of (graph, order) pairs
    keep = [rel for rel in rels if len(at_rel.get(rel, [])) >= 6]
    mean = [float(np.mean(at_rel[rel])) for rel in keep]
    lo = [float(np.min(at_rel[rel])) for rel in keep]
    hi = [float(np.max(at_rel[rel])) for rel in keep]

    fig, ax = plt.subplots(figsize=(9.2, 5.6), dpi=200)

    for (graph, order), series in by_pair.items():
        xs = sorted(series)
        ax.plot(xs, [series[x] for x in xs], color=GRAY, lw=0.7, alpha=0.28, zorder=2)

    ax.fill_between(keep, lo, hi, color=BLUE, alpha=0.10, zorder=3, linewidth=0)
    ax.plot(keep, mean, color=INK, lw=2.8, zorder=6, marker="o", ms=6,
            markerfacecolor=INK, markeredgecolor="white", markeredgewidth=1.2)

    ax.axvline(0, color=CORAL, lw=1.4, ls=(0, (3, 2)), zorder=4)
    ax.annotate("graph enters\nthe merge", xy=(0.12, 0.775), fontsize=9,
                color=CORAL, ha="left", va="center", fontweight="bold")

    # the mean jump at entry
    m = dict(zip(keep, mean))
    if -1 in m and 0 in m:
        ax.annotate(f"mean jump +{m[0]-m[-1]:.3f}",
                    xy=(0, m[0]), xytext=(1.2, m[0] - 0.055),
                    fontsize=9.5, color=INK, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=INK, lw=1.1))

    ax.set_xlim(min(keep) - 0.4, max(keep) + 0.4)
    ax.set_xticks(keep)
    ax.set_xlabel("rungs relative to the graph's own entry into the merge  "
                  "(0 = entry; <0 held out, >0 in training)", fontsize=10.3, color=INK)
    ax.set_ylabel("NM AUC  (3-shot, 30-way)", fontsize=10.5, color=INK)
    chrome(ax)
    ax.set_title("Entering the merge causes the jump — in every order",
                 fontsize=12.5, color=INK, fontweight="bold", loc="left", pad=26)
    ax.text(0.0, 1.02, "each thin line = one (graph, order); bold = mean over all "
            "24 (graph, order) pairs, band = min/max  ·  21/21 entry jumps positive, "
            "sign-test p = 5e-7", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=8.6, color=MUTED)
    ax.legend(handles=[
        Line2D([0], [0], color=GRAY, lw=1.2, alpha=0.5, label="individual (graph, order)"),
        Line2D([0], [0], color=INK, lw=2.8, marker="o", markerfacecolor=INK,
               markeredgecolor="white", ms=7, label="mean (24 pairs)"),
    ], loc="lower right", frameon=False, fontsize=9, handlelength=2.4)

    fig.tight_layout()
    save(fig, "order_entry_aligned_trajectory")
    plt.close(fig)


# --------------------------------------------------------------------- figure 2
def fig_id_ood_gap(rows):
    """In-distribution mean vs all-8 mean per rung, averaged over the three orders."""
    # per (order, rung): all-8 mean and in-dist mean (graphs already in the merge)
    by_or = defaultdict(lambda: defaultdict(list))    # (order, rung) -> {'all','in'}
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["order"], r["rung"])].append(r)
    all_mean = defaultdict(list)   # rung -> [order means]
    in_mean = defaultdict(list)
    for (order, rung), rs in grouped.items():
        all_mean[rung].append(float(np.mean([x["auc"] for x in rs])))
        inc = [x["auc"] for x in rs if x["in_training"]]
        in_mean[rung].append(float(np.mean(inc)))

    x = [rg - 1 for rg in RUNGS]
    all_mu = [float(np.mean(all_mean[rg])) for rg in RUNGS]
    all_lo = [float(np.min(all_mean[rg])) for rg in RUNGS]
    all_hi = [float(np.max(all_mean[rg])) for rg in RUNGS]
    in_mu = [float(np.mean(in_mean[rg])) for rg in RUNGS]
    in_lo = [float(np.min(in_mean[rg])) for rg in RUNGS]
    in_hi = [float(np.max(in_mean[rg])) for rg in RUNGS]

    fig, ax = plt.subplots(figsize=(8.8, 5.3), dpi=200)
    ax.fill_between(x, in_mu, all_mu, color=BLUE, alpha=0.10, zorder=1, linewidth=0)
    ax.fill_between(x, in_lo, in_hi, color=BLUE, alpha=0.16, zorder=2, linewidth=0)
    ax.fill_between(x, all_lo, all_hi, color=INK, alpha=0.10, zorder=2, linewidth=0)
    ax.plot(x, in_mu, color=BLUE, lw=2.6, zorder=5, marker="o", ms=6.5,
            markerfacecolor=BLUE, markeredgecolor="white", markeredgewidth=1.2)
    ax.plot(x, all_mu, color=INK, lw=2.6, zorder=6, marker="s", ms=6,
            markerfacecolor=INK, markeredgecolor="white", markeredgewidth=1.2)

    ax.annotate("in training\n(in-dist. mean)", xy=(x[-1] + 0.12, in_mu[-1] + 0.004),
                ha="left", va="center", fontsize=9.5, color=BLUE, fontweight="bold")
    ax.annotate("all 8 graphs\n(incl. held-out)", xy=(x[-1] + 0.12, all_mu[-1] - 0.012),
                ha="left", va="center", fontsize=9.5, color=INK, fontweight="bold")
    gi = 1
    ax.annotate("", xy=(gi, in_mu[gi]), xytext=(gi, all_mu[gi]),
                arrowprops=dict(arrowstyle="<->", color=MUTED, lw=1.1))
    ax.annotate("out-of-distribution\npenalty", xy=(gi + 0.12, (in_mu[gi] + all_mu[gi]) / 2),
                ha="left", va="center", fontsize=8.8, color=MUTED)

    ax.set_xlim(-0.5, 8.7)
    ax.set_xticks(x)
    ax.set_xticklabels([str(rg) for rg in RUNGS], fontsize=9.5)
    ax.set_xlabel("merge size  (number of source graphs in SSL pre-training)",
                  fontsize=10.5, color=INK)
    ax.set_ylabel("mean NM AUC  (3-shot, 30-way)", fontsize=10.5, color=INK)
    chrome(ax)
    ax.set_title("Adding sources closes the in-/out-of-distribution gap — order-robustly",
                 fontsize=12.3, color=INK, fontweight="bold", loc="left", pad=26)
    ax.text(0.0, 1.02, "lines = mean over orders A/B/C, bands = min/max over orders  ·  "
            "in-dist. mean = graphs already in the merge at that size",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=8.6, color=MUTED)
    ax.legend(handles=[
        Line2D([0], [0], color=BLUE, lw=2.6, marker="o", markerfacecolor=BLUE,
               markeredgecolor="white", ms=7, label="in training (in-dist. mean)"),
        Line2D([0], [0], color=INK, lw=2.6, marker="s", markerfacecolor=INK,
               markeredgecolor="white", ms=7, label="all 8 graphs (incl. held-out)"),
    ], loc="lower right", frameon=False, fontsize=9, handlelength=2.4)

    fig.tight_layout()
    save(fig, "order_id_ood_gap")
    plt.close(fig)


# --------------------------------------------------------------------- figure 3
def fig_role_deltas(rows):
    """Per-source impact split by role, colored by order."""
    # index AUC by (order, rung, graph) and by (order, graph)->entry
    auc = {(r["order"], r["rung"], r["graph"]): r["auc"] for r in rows}
    entry = {}
    graphs_by_order = defaultdict(set)
    for r in rows:
        entry[(r["order"], r["graph"])] = r["entry_rung"]
        graphs_by_order[r["order"]].add(r["graph"])

    # Collect data by (order, role)
    data = defaultdict(lambda: defaultdict(list))  # order -> role -> list of deltas
    for order in ("A", "B", "C"):
        graphs = graphs_by_order[order]
        for rr in range(2, 9):                                  # rung rr vs rr-1
            deltas = {}
            for g in graphs:
                a0 = auc.get((order, rr - 1, g))
                a1 = auc.get((order, rr, g))
                if a0 is not None and a1 is not None:
                    deltas[g] = a1 - a0
            newcomer = next(g for g in graphs if entry[(order, g)] == rr)
            incs = [g for g in graphs if entry[(order, g)] < rr]
            hels = [g for g in graphs if entry[(order, g)] > rr]
            if newcomer in deltas:
                data[order]["newc"].append(deltas[newcomer])
            data[order]["inc"] += [deltas[g] for g in incs if g in deltas]
            data[order]["hel"] += [deltas[g] for g in hels if g in deltas]

    roles = [("newcomer\n(just added)", "newc"),
             ("incumbents\n(already in)", "inc"),
             ("held-out\n(not yet added)", "hel")]

    fig, ax = plt.subplots(figsize=(8.4, 5.6), dpi=200)
    rng = np.random.default_rng(0)

    # Plot each role, with each order as a separate color with x-offset
    orders = ["A", "B", "C"]
    order_colors = [ORDER_C[o] for o in orders]
    n_orders = len(orders)

    handles = []
    for role_idx, (lab, role_key) in enumerate(roles):
        for order_idx, (order, color) in enumerate(zip(orders, order_colors)):
            vals = np.array(data[order][role_key])
            if len(vals) == 0:
                continue

            # Spread orders across the x-position to avoid overlap
            x_offset = (order_idx - n_orders / 2 + 0.5) * 0.12
            jitter = rng.uniform(-0.06, 0.06, size=len(vals))
            ax.scatter(np.full(len(vals), role_idx) + x_offset + jitter, vals, s=26,
                      color=color, alpha=0.60, edgecolor="white", linewidth=0.4,
                      zorder=4, label=f"order {order}" if role_idx == 0 else "")

            # Mean line for this (role, order)
            mu = float(vals.mean())
            line_x = [role_idx + x_offset - 0.13, role_idx + x_offset + 0.13]
            ax.plot(line_x, [mu, mu], color=color, lw=1.8, zorder=5, alpha=0.8)

        # Pooled summary annotation
        all_vals = np.array(data["A"][role_key] + data["B"][role_key] + data["C"][role_key])
        if len(all_vals) > 0:
            mu_pool = float(all_vals.mean())
            ax.annotate(f"pool\n{mu_pool:+.3f}", xy=(role_idx + 0.35, mu_pool),
                       fontsize=7.8, color=MUTED, va="center", style="italic")

    ax.axhline(0, color="#c3c2b7", lw=1.0, zorder=2)
    ax.set_xticks(range(len(roles)))
    ax.set_xticklabels([r[0] for r in roles], fontsize=9.8)
    ax.set_ylabel("Δ NM AUC at each source-addition step\n(this rung − rung below)",
                  fontsize=10.3, color=INK)
    ax.set_xlim(-0.6, len(roles) - 0.4)
    chrome(ax)
    ax.set_title("Per-source impact by order: large in-domain benefit, ~zero cost to other graphs",
                 fontsize=12.0, color=INK, fontweight="bold", loc="left", pad=26)
    ax.text(0.0, 1.02, "each dot = one graph at one addition step  ·  newcomer = its own "
            "OOD-to-ID jump  ·  colored by order; lines = mean per (role, order)  ·  italic "
            "label = pooled mean over all orders",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=8.6, color=MUTED)

    # Custom legend
    legend_handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=ORDER_C[o],
                            markersize=7, markeredgecolor="white", markeredgewidth=0.8,
                            label=f"order {o}") for o in orders]
    ax.legend(handles=legend_handles, loc="upper left", frameon=False, fontsize=9,
             handlelength=1.5)

    fig.tight_layout()
    save(fig, "order_role_deltas")
    plt.close(fig)

    # console summary by role
    print("\nrole            n     mean      min      max")
    for lab, role_key in roles:
        all_vals = np.array(data["A"][role_key] + data["B"][role_key] + data["C"][role_key])
        if len(all_vals) > 0:
            print(f"  {lab.split(chr(10))[0]:<14}{len(all_vals):<4}{all_vals.mean():+.4f}  "
                  f"{all_vals.min():+.4f}  {all_vals.max():+.4f}")

    # Summary by order
    print("\norder  newcomer      incumbent      held-out")
    for order in orders:
        n_vals = np.array(data[order]["newc"])
        i_vals = np.array(data[order]["inc"])
        h_vals = np.array(data[order]["hel"])
        print(f"  {order}     {n_vals.mean() if len(n_vals) > 0 else 0:+.4f}        "
              f"{i_vals.mean() if len(i_vals) > 0 else 0:+.4f}       "
              f"{h_vals.mean() if len(h_vals) > 0 else 0:+.4f}")


# --------------------------------------------------------------- figure 3b (bars)
def fig_role_deltas_bars(rows):
    """Per-source impact as BARS by merge size, aggregated over orders (the bar form of
    the original plot_nm_ladder_deltas.py two-panel figure).

    x = merge size after the addition (2..8). At each step and in each order, the added
    graph is different, so a bar is the MEAN over orders A/B/C of that role's delta, and
    the error bar is the min/max over the three orders. Top panel = newcomer benefit at
    full scale; bottom = incumbent cost + held-out interference, zoomed (note the
    y-scales differ, as in the single-order figure).
    """
    auc = {(r["order"], r["rung"], r["graph"]): r["auc"] for r in rows}
    entry, graphs_by_order = {}, defaultdict(set)
    for r in rows:
        entry[(r["order"], r["graph"])] = r["entry_rung"]
        graphs_by_order[r["order"]].add(r["graph"])

    steps = list(range(2, 9))
    # per step: lists (over the 3 orders) of newcomer Δ, mean-incumbent Δ, mean-held-out Δ
    newc, incu, held = defaultdict(list), defaultdict(list), defaultdict(list)
    for order in ("A", "B", "C"):
        graphs = graphs_by_order[order]
        for rr in steps:
            d = {}
            for g in graphs:
                a0, a1 = auc.get((order, rr - 1, g)), auc.get((order, rr, g))
                if a0 is not None and a1 is not None:
                    d[g] = a1 - a0
            newcomer = next(g for g in graphs if entry[(order, g)] == rr)
            incs = [d[g] for g in graphs if entry[(order, g)] < rr and g in d]
            hels = [d[g] for g in graphs if entry[(order, g)] > rr and g in d]
            if newcomer in d:
                newc[rr].append(d[newcomer])
            if incs:
                incu[rr].append(float(np.mean(incs)))
            if hels:
                held[rr].append(float(np.mean(hels)))

    def agg(dct):
        mu = [float(np.mean(dct[s])) if dct[s] else np.nan for s in steps]
        lo = [mu[i] - float(np.min(dct[s])) if dct[s] else 0 for i, s in enumerate(steps)]
        hi = [float(np.max(dct[s])) - mu[i] if dct[s] else 0 for i, s in enumerate(steps)]
        return np.array(mu), np.array([lo, hi])

    n_mu, n_err = agg(newc)
    i_mu, i_err = agg(incu)
    h_mu, h_err = agg(held)

    x = np.arange(len(steps))
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9.4, 6.8), dpi=200, sharex=True, constrained_layout=True,
        gridspec_kw={"height_ratios": [2.3, 1.0]})

    # top: newcomer benefit, full scale
    ax1.bar(x, n_mu, width=0.62, color=BLUE, edgecolor="white", linewidth=0.8, zorder=3)
    ax1.errorbar(x, n_mu, yerr=n_err, fmt="none", ecolor=INK, elinewidth=1.1,
                 capsize=3, zorder=5)
    for xi, v in zip(x, n_mu):
        ax1.annotate(f"+{v:.3f}", xy=(xi, v), xytext=(0, -13), textcoords="offset points",
                     ha="center", va="top", fontsize=8.6, color="white", fontweight="bold")
    ax1.set_ylim(0, max(n_mu + n_err[1]) * 1.12)
    ax1.set_ylabel("newcomer Δ\n(its OOD-to-ID jump)", fontsize=10, color=INK)
    ax1.axhline(0, color="#c3c2b7", lw=0.9, zorder=2)
    chrome(ax1)

    # bottom: incumbent (cost) + held-out (interference), zoomed, grouped
    w = 0.36
    ax2.bar(x - w / 2, i_mu, width=w, color=CORAL, edgecolor="white", linewidth=0.8,
            zorder=3, label="incumbents (already in) — cost")
    ax2.bar(x + w / 2, h_mu, width=w, color=GRAY, edgecolor="white", linewidth=0.8,
            zorder=3, label="held-out (not yet added) — interference")
    ax2.errorbar(x - w / 2, i_mu, yerr=i_err, fmt="none", ecolor="#8a3b1c",
                 elinewidth=1.0, capsize=2.5, zorder=5)
    ax2.errorbar(x + w / 2, h_mu, yerr=h_err, fmt="none", ecolor="#5f5e5a",
                 elinewidth=1.0, capsize=2.5, zorder=5)
    ax2.axhline(0, color="#c3c2b7", lw=0.9, zorder=2)
    ax2.set_ylabel("mean Δ of\nother graphs", fontsize=10, color=INK)
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(s) for s in steps], fontsize=9.5)
    ax2.set_xlabel("merge size after the addition  (newcomer identity differs by order)",
                   fontsize=10.5, color=INK)
    chrome(ax2)
    ax2.legend(loc="upper right", frameon=False, fontsize=8.6, handlelength=1.2)

    ax1.set_title("Per-source impact by merge size: large newcomer benefit throughout, "
                  "~zero incumbent cost",
                  fontsize=11.6, color=INK, fontweight="bold", loc="left", pad=24)
    ax1.text(0.0, 1.03, "bars = mean over orders A/B/C, whiskers = min/max over orders  ·  "
             "NM 3-shot/30-way · matched-40k  ·  panels differ in y-scale",
             transform=ax1.transAxes, ha="left", va="bottom", fontsize=8.5, color=MUTED)

    save(fig, "order_role_deltas_bars")
    plt.close(fig)

    print("\nsize  newcomer(mean)  incumbent  held-out   [mean over 3 orders]")
    for i, s in enumerate(steps):
        print(f"  {s}     {n_mu[i]:+.4f}       {i_mu[i]:+.4f}   {h_mu[i]:+.4f}")


# ------------------------------------------------------------ figure 4 (held-out)
def fig_heldout_headroom(rows):
    """Why some held-out graphs gain a lot: headroom, not donor-matching.

    Each held-out event (a graph not yet in the merge, at one addition step, in one
    order) as a point: x = the graph's AUC just before the step, y = the change at the
    step. The gains are almost all in order C (weak-first), and they scale with how far
    below its normal level the graph currently sits -- a graph already covered well barely
    moves. That is recovery toward the achievable level as coverage grows, not a targeted
    cross-transfer from the specific source being added (which shows ~0 correlation).
    """
    auc = {(r["order"], r["rung"], r["graph"]): r["auc"] for r in rows}
    entry = {(r["order"], r["graph"]): r["entry_rung"] for r in rows}
    graphs_by_order = defaultdict(set)
    for r in rows:
        graphs_by_order[r["order"]].add(r["graph"])

    pts = defaultdict(list)   # order -> [(before, delta)]
    for order in ("A", "B", "C"):
        for rr in range(2, 9):
            for g in graphs_by_order[order]:
                if entry[(order, g)] > rr:
                    before = auc[(order, rr - 1, g)]
                    pts[order].append((before, auc[(order, rr, g)] - before))

    allx = [b for order in pts for b, _ in pts[order]]
    ally = [d for order in pts for _, d in pts[order]]
    mx, my = np.mean(allx), np.mean(ally)
    r = (np.mean([(b - mx) * (d - my) for b, d in zip(allx, ally)])
         / (np.std(allx) * np.std(ally)))
    # least-squares trend line
    b_slope = np.polyfit(allx, ally, 1)

    fig, ax = plt.subplots(figsize=(8.6, 5.6), dpi=200)
    ax.axhline(0, color="#c3c2b7", lw=1.0, zorder=2)
    xs = np.array([min(allx) - 0.01, max(allx) + 0.01])
    ax.plot(xs, np.polyval(b_slope, xs), color=MUTED, lw=1.4, ls=(0, (5, 3)), zorder=3)

    for order in ("A", "B", "C"):
        bx = [b for b, _ in pts[order]]
        dy = [d for _, d in pts[order]]
        ax.scatter(bx, dy, s=46, color=ORDER_C[order], alpha=0.75, edgecolor="white",
                   linewidth=0.6, zorder=5, label=f"order {order}")

    ax.text(0.035, 0.52, f"r = {r:+.2f}", transform=ax.transAxes, fontsize=13,
            color=INK, fontweight="bold", ha="left")
    ax.text(0.035, 0.475, "lower current AUC,\nlarger recovery", transform=ax.transAxes,
            fontsize=9, color=MUTED, ha="left", va="top")

    ax.set_xlabel("held-out graph's NM AUC just before the source is added",
                  fontsize=10.5, color=INK)
    ax.set_ylabel("Δ NM AUC at that step\n(while still held out)", fontsize=10.5, color=INK)
    chrome(ax)
    ax.set_title("Held-out gains are headroom, not targeted transfer",
                 fontsize=12.5, color=INK, fontweight="bold", loc="left", pad=26)
    ax.text(0.0, 1.02, "each dot = one held-out graph at one addition step  ·  lower current "
            "AUC, larger recovery  ·  A/B ≈ 0, gains concentrated in weak-first order C",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=8.6, color=MUTED)
    ax.legend(loc="upper right", frameon=False, fontsize=9.5, handlelength=1.0,
              scatterpoints=1)

    fig.tight_layout()
    save(fig, "order_heldout_headroom")
    plt.close(fig)

    print("\nheld-out mean Δ by order:  " +
          "  ".join(f"{o}: {np.mean([d for _, d in pts[o]]):+.4f}" for o in "ABC"))
    print(f"corr(Δ, current AUC) = {r:+.3f}")


def main():
    os.makedirs(FIGS, exist_ok=True)
    rows = load()
    print(f"[data] {len(rows)} cells from {os.path.relpath(DATA, HERE)}")
    fig_heldout_only_by_order(rows)
    fig_order_c_best_constituent_residual(rows)
    fig_best_constituent_residual_by_order(rows)
    fig_mean_mixture_vs_best_constituent_gain_by_order(rows)
    fig_mean_mixture_minus_best_constituent_by_order(rows)
    fig_final_heldout_mixture_minus_best_constituent_bars(rows)
    fig_final_heldout_mixture_vs_best_constituent_lines(rows)
    fig_heldout_mixture_vs_best_constituent_entry_aligned_grid(rows)
    fig_heldout_mixture_advantage_entry_aligned_grid(rows)
    fig_all_heldout_targets_mixture_minus_best_constituent_bars(rows)
    fig_heldout_auc_profiles_vs_best_non_specialist(rows)
    fig_final_heldout_mixture_and_constituents_by_order(rows)
    fig_heldout_swap_first_two_mean_mixture_vs_best_by_order(rows)
    fig_entry_aligned(rows)
    fig_id_ood_gap(rows)
    fig_role_deltas(rows)
    fig_role_deltas_bars(rows)
    fig_heldout_headroom(rows)


if __name__ == "__main__":
    main()
