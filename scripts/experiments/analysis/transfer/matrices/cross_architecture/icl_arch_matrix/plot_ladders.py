#!/usr/bin/env python3
"""Plot architecture-specific and joint nine-rung downstream source ladders."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402


HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "raw_aggregate" / "summary" / "classification_matrix.csv"
FIGURES = HERE / "figures"
ARCHITECTURES = ("prodigy", "vision", "gilt")
TARGETS = ("covid_political", "election2020", "ukr_rus_suspended", "twibot20")
SAMGPT_DATA = (
    HERE.parents[6]
    / "scripts/experiments/analysis/transfer/ladders/prodigy_nm/robustness"
    / "nm_ladder_order_robustness/data/samgpt_9x3_carc_v100/roc_auc.csv"
)
SAMGPT_TARGETS = (
    "ukr_rus_twitter", "covid19_twitter", "midterm", "covid_political",
    "election2020", "ukr_rus_suspended", "twibot20", "cp_hk_twitter",
    "facebook_page_reference",
)

BLUE = "#2a78d6"
GRAY = "#8f8d87"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"

SHORT = {
    "ukr_rus": "Ukr-Rus",
    "covid": "COVID-19",
    "midterm": "Midterm",
    "covid_political": "COVID-pol.",
    "election2020": "Election '20",
    "ukr_rus_suspended": "Ukr-Rus susp.",
    "twibot20": "TwiBot-20",
    "cp_hk": "CP-HK",
    "facebook_page_reference": "Facebook",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "pdf.fonttype": 42,
})


def load() -> list[dict]:
    with DATA.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["n_sources"] = int(row["n_sources"])
        row["source_list"] = row["sources"].split(",")
        for field in (*TARGETS, "mean_roc_auc"):
            row[field] = float(row[field])
    return rows


def load_samgpt() -> list[dict]:
    with SAMGPT_DATA.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["rung"] = int(row["rung"])
        row["target_mean"] = float(row["target_mean"])
        for target in SAMGPT_TARGETS:
            row[target] = float(row[target])
    expected = {(order, rung) for order in "ABC" for rung in range(1, 10)}
    observed = {(row["order"], row["rung"]) for row in rows}
    if observed != expected:
        raise ValueError(f"SAMGPT: expected 27 order/rung cells, got {len(observed)}")
    return rows


def ladder(rows: list[dict], architecture: str, order: str) -> list[dict]:
    selected = [row for row in rows if row["architecture"] == architecture]
    by_id = {row["model_id"]: row for row in selected}
    middle = [by_id[f"ord{order}_r{rung}"] for rung in range(2, 9)]
    first_source = middle[0]["source_list"][0]
    result = [by_id[f"ss_{first_source}"], *middle, by_id["all9"]]
    expected = list(range(1, 10))
    observed = [row["n_sources"] for row in result]
    if observed != expected:
        raise ValueError(f"{architecture} order {order}: expected rungs {expected}, got {observed}")
    return result


def chrome(ax: plt.Axes) -> None:
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#c3c2b7")
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8.5)


def plot_architecture(rows: list[dict], architecture: str) -> None:
    x = np.arange(1, 10)
    ladders = {order: ladder(rows, architecture, order) for order in "ABC"}
    all_values = [row[target] for order_rows in ladders.values() for row in order_rows for target in TARGETS]
    lower = max(0.0, min(all_values) - 0.035)
    upper = min(1.0, max(all_values) + 0.025)

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.1), dpi=200, sharey=True)
    for ax, order in zip(axes, "ABC"):
        order_rows = ladders[order]
        sequence = order_rows[-1]["source_list"]
        # The all9 row has canonical order A, so infer each order from its rung prefixes.
        sequence = [order_rows[0]["source_list"][0]] + [row["source_list"][-1] for row in order_rows[1:8]]
        final_source = next(source for source in order_rows[-1]["source_list"] if source not in sequence)
        sequence.append(final_source)

        for target in TARGETS:
            values = [row[target] for row in order_rows]
            entry = sequence.index(target) + 1 if target in sequence else None
            for left in range(1, 9):
                in_mixture = entry is not None and left + 1 >= entry
                ax.plot(
                    [left, left + 1], values[left - 1:left + 1],
                    color=BLUE if in_mixture else GRAY,
                    lw=1.55 if in_mixture else 1.1,
                    ls="-" if in_mixture else (0, (3, 2)),
                    alpha=0.78 if in_mixture else 0.52,
                    zorder=2,
                )
            if entry is not None:
                ax.scatter(entry, values[entry - 1], s=36, facecolor=BLUE,
                           edgecolor="white", linewidth=0.9, zorder=4)

        means = [row["mean_roc_auc"] for row in order_rows]
        ax.plot(x, means, color=INK, lw=2.6, marker="s", ms=5.8,
                markerfacecolor=INK, markeredgecolor="white", markeredgewidth=0.9,
                zorder=6)
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT[source] for source in sequence], rotation=55,
                           ha="right", fontsize=7.6)
        ax.set_xlabel("source added at this rung", fontsize=9.2, color=INK)
        ax.set_title(f"order {order}", loc="left", fontsize=11, color=INK,
                     fontweight="bold")
        chrome(ax)

    display = architecture.upper()
    axes[0].set_ylabel(f"{display} ROC-AUC", fontsize=10.5, color=INK)
    axes[0].set_ylim(lower, upper)
    fig.suptitle(
        f"{display} nine-source ladder at 100 updates",
        x=0.07, ha="left", y=1.045, fontsize=13.2, color=INK, fontweight="bold",
    )
    fig.text(
        0.07, 0.99,
        "thin lines = four evaluation targets; blue begins when that target enters "
        "the mixture; black = four-target mean · seed 0",
        ha="left", va="top", fontsize=9, color=MUTED,
    )
    fig.legend(handles=[
        Line2D([0], [0], color=GRAY, lw=1.2, ls=(0, (3, 2)), label="target held out"),
        Line2D([0], [0], color=BLUE, lw=1.55, label="target in mixture"),
        Line2D([0], [0], color=INK, lw=2.6, marker="s", markerfacecolor=INK,
               markeredgecolor="white", label="four-target mean"),
    ], loc="lower center", bbox_to_anchor=(0.5, -0.17), ncol=3, frameon=False,
       fontsize=8.7, handlelength=2.2)
    fig.tight_layout(w_pad=1.4)
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        output = FIGURES / f"{architecture}_ladder_orders.{suffix}"
        fig.savefig(output, bbox_inches="tight", dpi=220)
        print("wrote", output)
    plt.close(fig)


def plot_joint(rows: list[dict], samgpt_rows: list[dict]) -> None:
    """Place the three matched architectures and separate SAMGPT ladder in one PNG."""
    x = np.arange(1, 10)
    displays = {"prodigy": "PRODIGY", "vision": "VISION", "gilt": "GILT", "samgpt": "SAMGPT"}
    fig, axes = plt.subplots(4, 3, figsize=(15.8, 15.2), dpi=200, sharex=True, sharey=True)

    for row_index, architecture in enumerate(displays):
        for col_index, order in enumerate("ABC"):
            ax = axes[row_index, col_index]
            if architecture == "samgpt":
                order_rows = sorted(
                    (row for row in samgpt_rows if row["order"] == order),
                    key=lambda row: row["rung"],
                )
                targets = SAMGPT_TARGETS
                sequence = [row["added"] for row in order_rows]
                means = [row["target_mean"] for row in order_rows]
            else:
                order_rows = ladder(rows, architecture, order)
                targets = TARGETS
                sequence = [order_rows[0]["source_list"][0]] + [
                    row["source_list"][-1] for row in order_rows[1:8]
                ]
                sequence.append(next(
                    source for source in order_rows[-1]["source_list"] if source not in sequence
                ))
                means = [row["mean_roc_auc"] for row in order_rows]

            for target in targets:
                values = [row[target] for row in order_rows]
                canonical_target = {
                    "ukr_rus_twitter": "ukr_rus",
                    "covid19_twitter": "covid",
                    "cp_hk_twitter": "cp_hk",
                }.get(target, target)
                canonical_sequence = [{
                    "ukr_rus_twitter": "ukr_rus",
                    "covid19_twitter": "covid",
                    "cp_hk_twitter": "cp_hk",
                }.get(source, source) for source in sequence]
                entry = canonical_sequence.index(canonical_target) + 1
                for left in range(1, 9):
                    in_mixture = left + 1 >= entry
                    ax.plot(
                        [left, left + 1], values[left - 1:left + 1],
                        color=BLUE if in_mixture else GRAY,
                        lw=1.05 if in_mixture else 0.75,
                        ls="-" if in_mixture else (0, (3, 2)),
                        alpha=0.62 if in_mixture else 0.35,
                        zorder=2,
                    )
                ax.scatter(entry, values[entry - 1], s=20, facecolor=BLUE,
                           edgecolor="white", linewidth=0.65, zorder=4)

            ax.plot(x, means, color=INK, lw=2.35, marker="s", ms=4.8,
                    markerfacecolor=INK, markeredgecolor="white", markeredgewidth=0.7,
                    zorder=6)
            chrome(ax)
            ax.set_xlim(0.7, 9.3)
            ax.set_ylim(0.4, 1.005)
            ax.set_xticks(x)
            if row_index == 3:
                ax.set_xlabel("mixture rung / number of sources", fontsize=9.2, color=INK)
            if row_index == 0:
                ax.set_title(f"order {order}", fontsize=11.2, color=INK,
                             fontweight="bold", pad=8)
            if col_index == 0:
                protocol = "matched · 4 targets · 100 updates" if architecture != "samgpt" else "separate protocol · 9 targets"
                ax.set_ylabel(f"{displays[architecture]} ROC-AUC\n{protocol}", fontsize=9.5,
                              color=INK, labelpad=9)

    fig.suptitle(
        "Nine-source downstream transfer ladders",
        x=0.075, ha="left", y=0.995, fontsize=14.5, color=INK, fontweight="bold",
    )
    fig.text(
        0.075, 0.972,
        "PRODIGY, VISION, and GILT share a controlled seed-0 evaluation; SAMGPT is shown "
        "as separate corroborating evidence and is not directly comparable",
        ha="left", va="top", fontsize=9.4, color=MUTED,
    )
    fig.legend(handles=[
        Line2D([0], [0], color=GRAY, lw=1.0, ls=(0, (3, 2)), label="target held out"),
        Line2D([0], [0], color=BLUE, lw=1.2, label="target in mixture"),
        Line2D([0], [0], color=INK, lw=2.35, marker="s", markerfacecolor=INK,
               markeredgecolor="white", label="target mean"),
    ], loc="lower center", bbox_to_anchor=(0.5, 0.012), ncol=3, frameon=False,
       fontsize=9, handlelength=2.2)
    fig.tight_layout(rect=(0.025, 0.055, 0.995, 0.95), h_pad=1.6, w_pad=1.2)
    output = FIGURES / "joint_downstream_ladder_orders.png"
    fig.savefig(output, bbox_inches="tight", dpi=220)
    print("wrote", output)
    plt.close(fig)


def main() -> None:
    rows = load()
    for architecture in ARCHITECTURES:
        plot_architecture(rows, architecture)
    plot_joint(rows, load_samgpt())


if __name__ == "__main__":
    main()
