#!/usr/bin/env python3
"""Visualize the canonical 3-order x 9-rung SAMGPT source ladder.

This is the SAMGPT analogue of the PRODIGY ladder figures in this folder. It reads
the mixed-hardware provisional snapshot and writes three figures:

* ``samgpt_ladder_orders``: the original ladder view, faceted by source order;
* ``samgpt_entry_aligned_trajectory``: targets aligned on their own entry rung;
* ``samgpt_id_ood_gap``: in-mixture versus all-target mean by merge size;
* ``samgpt_role_deltas``: marginal changes for newcomer/incumbent/held-out targets.

Run locally with ``/opt/homebrew/bin/python3.11 plot_samgpt_ladder.py``.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


HERE = Path(__file__).resolve().parent
DATA_ROOT = HERE / "data" / "samgpt_9x3_carc_v100"
FIG_ROOT = HERE / "figures"

BLUE = "#2a78d6"
CORAL = "#d85a30"
GRAY = "#8f8d87"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
H100 = "#e0a51f"

SHORT = {
    "ukr_rus_twitter": "Ukr-Rus",
    "covid19_twitter": "COVID-19",
    "midterm": "Midterm",
    "covid_political": "COVID-pol.",
    "election2020": "Election '20",
    "ukr_rus_suspended": "Ukr-Rus susp.",
    "twibot20": "TwiBot-20",
    "cp_hk_twitter": "CP-HK",
    "facebook_page_reference": "Facebook",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})


def load() -> tuple[list[dict], dict[str, list[str]], list[str]]:
    manifest = json.loads((DATA_ROOT / "manifest.json").read_text())
    orders = manifest["orders"]
    targets = manifest["targets"]
    rows = []
    with (DATA_ROOT / "roc_auc.csv").open(newline="") as handle:
        for raw in csv.DictReader(handle):
            rows.append({
                "order": raw["order"],
                "rung": int(raw["rung"]),
                "added": raw["added"],
                "hardware": raw.get("execution_source", "CARC-V100"),
                "target_mean": float(raw["target_mean"]),
                "targets": {target: float(raw[target]) for target in targets},
            })
    expected = {(order, rung) for order in orders for rung in range(1, 10)}
    observed = {(row["order"], row["rung"]) for row in rows}
    if observed != expected:
        raise ValueError(f"Expected 27 order/rung rows, found {len(observed)}")
    return rows, orders, targets


def source_note(rows: list[dict]) -> str:
    sources = sorted({row["hardware"] for row in rows})
    if sources == ["CARC-V100"]:
        return "canonical all-CARC V100"
    return "provisional mixed hardware: " + " + ".join(sources)


def chrome(ax: plt.Axes) -> None:
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#c3c2b7")
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8.5)


def save(fig: plt.Figure, stem: str) -> None:
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        path = FIG_ROOT / f"{stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight", dpi=220)
        print("wrote", path)


def index(rows: list[dict]) -> dict[tuple[str, int], dict]:
    return {(row["order"], row["rung"]): row for row in rows}


def entry_map(orders: dict[str, list[str]]) -> dict[tuple[str, str], int]:
    return {
        (order, target): position
        for order, sequence in orders.items()
        for position, target in enumerate(sequence, start=1)
    }


def entry_deltas(
    table: dict[tuple[str, int], dict],
    orders: dict[str, list[str]],
) -> list[float]:
    deltas = []
    for order, sequence in orders.items():
        for rung, target in enumerate(sequence, start=1):
            if rung > 1:
                deltas.append(
                    table[order, rung]["targets"][target]
                    - table[order, rung - 1]["targets"][target]
                )
    return deltas


def plot_order_ladders(
    rows: list[dict], orders: dict[str, list[str]], targets: list[str]
) -> None:
    table = index(rows)
    deltas = entry_deltas(table, orders)
    n_positive = sum(delta > 0 for delta in deltas)
    x = np.arange(1, 10)

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.1), dpi=200, sharey=True)
    for ax, order in zip(axes, ("A", "B", "C")):
        sequence = orders[order]
        hardware = [table[order, rung]["hardware"] for rung in x]

        # Mark the provisional H100-filled rungs without covering the data.
        for rung, source in zip(x, hardware):
            if source == "Tucker-H100":
                ax.axvspan(rung - 0.48, rung + 0.48, color=H100, alpha=0.07,
                           zorder=0, linewidth=0)

        for target in targets:
            values = [table[order, rung]["targets"][target] for rung in x]
            entered = sequence.index(target) + 1
            for left in range(1, 9):
                in_mixture = left + 1 >= entered
                ax.plot(
                    [left, left + 1], values[left - 1:left + 1],
                    color=BLUE if in_mixture else GRAY,
                    lw=1.45 if in_mixture else 1.05,
                    ls="-" if in_mixture else (0, (3, 2)),
                    alpha=0.72 if in_mixture else 0.48,
                    zorder=2,
                )
            ax.scatter(
                [entered], [values[entered - 1]], s=34,
                facecolor=BLUE, edgecolor="white", linewidth=0.9, zorder=4,
            )

        means = [table[order, rung]["target_mean"] for rung in x]
        ax.plot(x, means, color=INK, lw=2.5, zorder=6)
        for rung, value, source in zip(x, means, hardware):
            marker = "^" if source == "Tucker-H100" else "s"
            face = H100 if source == "Tucker-H100" else INK
            ax.scatter([rung], [value], marker=marker, s=45, facecolor=face,
                       edgecolor="white", linewidth=0.9, zorder=7)

        ax.set_xticks(x)
        ax.set_xticklabels([SHORT[name] for name in sequence], rotation=55,
                           ha="right", fontsize=7.6)
        ax.set_xlabel("source added at this rung", fontsize=9.2, color=INK)
        ax.set_title(f"order {order}", loc="left", fontsize=11, color=INK,
                     fontweight="bold")
        chrome(ax)

    axes[0].set_ylabel("SAMGPT ROC-AUC", fontsize=10.5, color=INK)
    axes[0].set_ylim(0.56, 0.92)
    axes[0].set_yticks(np.arange(0.60, 0.91, 0.05))
    fig.suptitle(
        "SAMGPT source ladder: target performance stays largely flat as the mixture grows",
        x=0.07, ha="left", y=1.045, fontsize=13.2, color=INK, fontweight="bold",
    )
    fig.text(
        0.07, 0.99,
        f"thin lines = targets; blue begins when that target enters the mixture; "
        f"black = nine-target mean  ·  entry increases: {n_positive}/24, "
        f"paired mean {np.mean(deltas):+.4f}",
        ha="left", va="top", fontsize=9, color=MUTED,
    )
    legend_handles = [
        Line2D([0], [0], color=GRAY, lw=1.2, ls=(0, (3, 2)),
               label="target held out"),
        Line2D([0], [0], color=BLUE, lw=1.5, label="target in mixture"),
        Line2D([0], [0], color=INK, lw=2.5, marker="s", markerfacecolor=INK,
               markeredgecolor="white", label="mean · CARC V100"),
    ]
    if any(row["hardware"] == "Tucker-H100" for row in rows):
        legend_handles.extend([
            Line2D([0], [0], color=INK, lw=2.5, marker="^", markerfacecolor=H100,
                   markeredgecolor="white", label="mean · Tucker H100 fill"),
            Patch(facecolor=H100, alpha=0.12, edgecolor="none", label="H100-filled rung"),
        ])
    fig.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, -0.17),
       ncol=len(legend_handles), frameon=False,
       fontsize=8.5, handlelength=2.0)
    fig.tight_layout(w_pad=1.4)
    save(fig, "samgpt_ladder_orders")
    plt.close(fig)


def plot_entry_aligned(
    rows: list[dict], orders: dict[str, list[str]], targets: list[str]
) -> None:
    table = index(rows)
    entries = entry_map(orders)
    by_pair: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
    for order in orders:
        for rung in range(1, 10):
            for target in targets:
                relative = rung - entries[order, target]
                by_pair[target, order][relative] = table[order, rung]["targets"][target]

    at_relative: dict[int, list[float]] = defaultdict(list)
    for series in by_pair.values():
        for relative, value in series.items():
            at_relative[relative].append(value)
    keep = [relative for relative in range(-8, 9)
            if len(at_relative[relative]) >= 6]
    mean = np.array([np.mean(at_relative[relative]) for relative in keep])
    low = np.array([np.min(at_relative[relative]) for relative in keep])
    high = np.array([np.max(at_relative[relative]) for relative in keep])
    deltas = entry_deltas(table, orders)
    n_positive = sum(delta > 0 for delta in deltas)

    fig, ax = plt.subplots(figsize=(9.4, 5.5), dpi=200)
    for series in by_pair.values():
        xs = sorted(series)
        ax.plot(xs, [series[x] for x in xs], color=GRAY, lw=0.7,
                alpha=0.25, zorder=2)
    ax.fill_between(keep, low, high, color=BLUE, alpha=0.09, linewidth=0,
                    zorder=1)
    ax.plot(keep, mean, color=INK, lw=2.8, marker="o", ms=6,
            markerfacecolor=INK, markeredgecolor="white", markeredgewidth=1.0,
            zorder=5)
    ax.axvline(0, color=CORAL, lw=1.3, ls=(0, (3, 2)), zorder=3)
    ax.annotate("target enters\nthe mixture", xy=(0.13, 0.612), fontsize=9,
                color=CORAL, ha="left", va="center")
    zero_y = mean[keep.index(0)]
    ax.annotate(
        f"paired mean entry change {np.mean(deltas):+.4f}",
        xy=(0, zero_y), xytext=(1.25, zero_y - 0.065),
        fontsize=9.3, color=INK,
        arrowprops=dict(arrowstyle="->", color=INK, lw=1.0),
    )
    ax.set_xlim(min(keep) - 0.4, max(keep) + 0.4)
    ax.set_ylim(0.55, 0.92)
    ax.set_xticks(keep)
    ax.set_xlabel("rungs relative to the target's own entry  "
                  "(0 = entry; <0 held out; >0 in mixture)",
                  fontsize=10.2, color=INK)
    ax.set_ylabel("SAMGPT ROC-AUC", fontsize=10.5, color=INK)
    chrome(ax)
    ax.set_title("SAMGPT does not show PRODIGY's systematic entry jump",
                 fontsize=12.6, color=INK, fontweight="bold", loc="left", pad=25)
    ax.text(
        0, 1.02,
        f"each thin line = one (target, order); bold = mean; band = min/max  ·  "
        f"{n_positive}/24 measurable entry changes positive  ·  {source_note(rows)}",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=8.6, color=MUTED,
    )
    ax.legend(handles=[
        Line2D([0], [0], color=GRAY, lw=1.2, alpha=0.5,
               label="individual (target, order)"),
        Line2D([0], [0], color=INK, lw=2.8, marker="o", markerfacecolor=INK,
               markeredgecolor="white", ms=7, label="mean at each relative rung"),
    ], loc="lower right", frameon=False, fontsize=9, handlelength=2.3)
    fig.tight_layout()
    save(fig, "samgpt_entry_aligned_trajectory")
    plt.close(fig)


def plot_id_ood_gap(
    rows: list[dict], orders: dict[str, list[str]], targets: list[str]
) -> None:
    table = index(rows)
    rungs = np.arange(1, 10)
    all_by_rung: dict[int, list[float]] = defaultdict(list)
    in_by_rung: dict[int, list[float]] = defaultdict(list)
    for order, sequence in orders.items():
        for rung in rungs:
            row = table[order, int(rung)]
            all_by_rung[int(rung)].append(row["target_mean"])
            included = sequence[:int(rung)]
            in_by_rung[int(rung)].append(
                float(np.mean([row["targets"][target] for target in included]))
            )

    def summarize(values: dict[int, list[float]]) -> tuple[np.ndarray, ...]:
        mean = np.array([np.mean(values[int(rung)]) for rung in rungs])
        low = np.array([np.min(values[int(rung)]) for rung in rungs])
        high = np.array([np.max(values[int(rung)]) for rung in rungs])
        return mean, low, high

    all_mean, all_low, all_high = summarize(all_by_rung)
    in_mean, in_low, in_high = summarize(in_by_rung)

    fig, ax = plt.subplots(figsize=(8.8, 5.3), dpi=200)
    ax.fill_between(rungs, in_low, in_high, color=BLUE, alpha=0.14,
                    linewidth=0, zorder=1)
    ax.fill_between(rungs, all_low, all_high, color=INK, alpha=0.09,
                    linewidth=0, zorder=1)
    ax.plot(rungs, in_mean, color=BLUE, lw=2.6, marker="o", ms=6.5,
            markerfacecolor=BLUE, markeredgecolor="white", markeredgewidth=1.0,
            zorder=4)
    ax.plot(rungs, all_mean, color=INK, lw=2.6, marker="s", ms=6,
            markerfacecolor=INK, markeredgecolor="white", markeredgewidth=1.0,
            zorder=5)

    ax.annotate(
        f"in-mixture minus all targets\n{in_mean[0] - all_mean[0]:+.3f}",
        xy=(1, (in_mean[0] + all_mean[0]) / 2), xytext=(1.55, 0.662),
        fontsize=8.8, color=MUTED,
        arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.9),
    )
    ax.annotate("coincide by definition\n(all nine sources included)",
                xy=(9, all_mean[-1]), xytext=(7.15, 0.679), fontsize=8.8,
                color=MUTED, ha="left",
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.9))
    ax.set_xlim(0.65, 9.35)
    ax.set_ylim(0.64, 0.75)
    ax.set_xticks(rungs)
    ax.set_xlabel("merge size (number of source graphs in SAMGPT pre-training)",
                  fontsize=10.3, color=INK)
    ax.set_ylabel("mean SAMGPT ROC-AUC", fontsize=10.5, color=INK)
    chrome(ax)
    ax.set_title("SAMGPT shows no stable in-/out-of-mixture advantage",
                 fontsize=12.6, color=INK, fontweight="bold", loc="left", pad=25)
    ax.text(
        0, 1.02,
        "lines = mean over orders A/B/C; bands = min/max over orders  ·  "
        f"in-mixture mean uses targets already included at that rung  ·  {source_note(rows)}",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=8.6, color=MUTED,
    )
    ax.legend(handles=[
        Line2D([0], [0], color=BLUE, lw=2.6, marker="o", markerfacecolor=BLUE,
               markeredgecolor="white", ms=7, label="targets already in mixture"),
        Line2D([0], [0], color=INK, lw=2.6, marker="s", markerfacecolor=INK,
               markeredgecolor="white", ms=7, label="all nine targets"),
    ], loc="upper right", frameon=False, fontsize=9, handlelength=2.3)
    fig.tight_layout()
    save(fig, "samgpt_id_ood_gap")
    plt.close(fig)


def plot_role_deltas(
    rows: list[dict], orders: dict[str, list[str]], targets: list[str]
) -> None:
    table = index(rows)
    entries = entry_map(orders)
    role_values: dict[str, list[float]] = defaultdict(list)
    role_hardware: dict[str, list[str]] = defaultdict(list)
    for order in orders:
        for rung in range(2, 10):
            current = table[order, rung]
            previous = table[order, rung - 1]
            transition = (
                "hardware change"
                if current["hardware"] != previous["hardware"]
                else current["hardware"]
            )
            for target in targets:
                entry = entries[order, target]
                role = "newcomer" if entry == rung else (
                    "incumbent" if entry < rung else "held-out"
                )
                role_values[role].append(
                    current["targets"][target] - previous["targets"][target]
                )
                role_hardware[role].append(transition)

    roles = ["newcomer", "incumbent", "held-out"]
    labels = ["newcomer\n(just added)", "incumbents\n(already in)",
              "held-out\n(not yet added)"]
    colors = [BLUE, CORAL, GRAY]
    rng = np.random.default_rng(0)
    has_hardware_boundary = any(
        source == "hardware change"
        for values in role_hardware.values()
        for source in values
    )

    fig, ax = plt.subplots(figsize=(8.8, 5.5), dpi=200)
    for position, (role, color) in enumerate(zip(roles, colors)):
        values = np.array(role_values[role])
        jitter = rng.uniform(-0.13, 0.13, size=len(values))
        normal = np.array([source != "hardware change"
                           for source in role_hardware[role]])
        ax.scatter(np.full(normal.sum(), position) + jitter[normal], values[normal],
                   s=28, color=color, alpha=0.60, edgecolor="white",
                   linewidth=0.4, zorder=4)
        if (~normal).any():
            ax.scatter(np.full((~normal).sum(), position) + jitter[~normal],
                       values[~normal], s=35, facecolor="white", edgecolor=color,
                       linewidth=1.2, zorder=5)
        mean = float(values.mean())
        ax.plot([position - 0.27, position + 0.27], [mean, mean], color=INK,
                lw=2.5, zorder=6)
        ax.annotate(f"mean {mean:+.4f}\nn={len(values)}",
                    xy=(position + 0.31, mean), ha="left", va="center",
                    fontsize=8.8, color=color)

    ax.axhline(0, color="#b8b7af", lw=1.1, zorder=1)
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_xlim(-0.58, 2.62)
    ax.set_ylabel("change in ROC-AUC at a source-addition step\n"
                  "(this rung − previous rung)", fontsize=10.2, color=INK)
    chrome(ax)
    ax.set_title("SAMGPT source additions have near-zero marginal effects",
                 fontsize=12.6, color=INK, fontweight="bold", loc="left", pad=25)
    subtitle = "each point = one target at one step, pooled over orders A/B/C"
    if has_hardware_boundary:
        subtitle += "  ·  hollow = hardware boundary"
    subtitle += f"  ·  {source_note(rows)}"
    ax.text(
        0, 1.02, subtitle,
        transform=ax.transAxes, ha="left", va="bottom", fontsize=8.6, color=MUTED,
    )
    if has_hardware_boundary:
        ax.legend(handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor=GRAY,
                   markeredgecolor="white", ms=7, label="same hardware across step"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
                   markeredgecolor=GRAY, ms=7, label="hardware boundary"),
        ], loc="lower right", frameon=False, fontsize=8.6)
    fig.tight_layout()
    save(fig, "samgpt_role_deltas")
    plt.close(fig)

    print("\nrole         n      mean      median       min       max")
    for role in roles:
        values = np.array(role_values[role])
        print(f"{role:<11} {len(values):>3}  {values.mean():+.4f}  "
              f"{np.median(values):+.4f}  {values.min():+.4f}  {values.max():+.4f}")


def main() -> None:
    rows, orders, targets = load()
    plot_order_ladders(rows, orders, targets)
    plot_entry_aligned(rows, orders, targets)
    plot_id_ood_gap(rows, orders, targets)
    plot_role_deltas(rows, orders, targets)


if __name__ == "__main__":
    main()
