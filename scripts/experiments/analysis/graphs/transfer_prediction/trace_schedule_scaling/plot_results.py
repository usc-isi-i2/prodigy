#!/usr/bin/env python3
"""Plot the controlled schedule result, mechanism, and TRACE fusion gain."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
TARGET_ORDER = (
    "covid_political",
    "election2020",
    "facebook_page_reference",
    "twibot20",
)
TARGET_LABELS = {
    "covid_political": "COVID\npolitical",
    "election2020": "Election\n2020",
    "facebook_page_reference": "Facebook\npages",
    "twibot20": "TwiBot-20",
}
TARGET_COLORS = {
    "covid_political": "#0072B2",
    "election2020": "#E69F00",
    "facebook_page_reference": "#009E73",
    "twibot20": "#CC79A7",
}


def panel_schedule(ax, contrasts):
    frame = contrasts[
        contrasts.stream.eq("fresh")
        & ~contrasts.target.eq("ukr_rus_suspended")
    ]
    styles = {
        "blocked": ("#D55E00", "o", "Blocked"),
        "replay100": ("#0072B2", "s", "Replay-100"),
    }
    for schedule_index, (schedule, (color, marker, label)) in enumerate(styles.items()):
        part = frame[frame.schedule.eq(schedule)]
        means = []
        for rung in (2, 3, 4):
            values = part[part.rung.eq(rung)].delta_accuracy.to_numpy() * 100
            jitter = np.linspace(-0.09, 0.09, len(values)) + (schedule_index - .5) * .035
            ax.scatter(
                rung + jitter,
                values,
                s=17,
                color=color,
                alpha=.34,
                linewidth=0,
            )
            means.append(values.mean())
        ax.plot(
            (2, 3, 4), means, color=color, marker=marker, markersize=5,
            linewidth=2, label=label,
        )
    ax.axhline(0, color="#777777", linewidth=.8, linestyle="--")
    ax.set_xticks((2, 3, 4))
    ax.set_xlabel("Number of source graphs")
    ax.set_ylabel("Accuracy minus interleaving (points)")
    ax.set_title("a  No universal pair-to-many reversal", loc="left", weight="bold")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower right")


def panel_health(ax, contrasts, correlations):
    frame = contrasts[
        contrasts.stream.eq("fresh")
        & ~contrasts.target.eq("ukr_rus_suspended")
    ]
    for target in TARGET_ORDER:
        part = frame[frame.target.eq(target)]
        ax.scatter(
            part.delta_u1_agreement * 100,
            part.delta_accuracy * 100,
            s=24,
            color=TARGET_COLORS[target],
            alpha=.72,
            edgecolor="white",
            linewidth=.3,
            label=TARGET_LABELS[target].replace("\n", " "),
        )
    x = frame.delta_u1_agreement.to_numpy() * 100
    y = frame.delta_accuracy.to_numpy() * 100
    coefficient = np.polyfit(x, y, 1)
    grid = np.linspace(x.min(), x.max(), 100)
    ax.plot(grid, np.polyval(coefficient, grid), color="#222222", linewidth=1.3)
    ax.axhline(0, color="#888888", linewidth=.7, linestyle="--")
    ax.axvline(0, color="#888888", linewidth=.7, linestyle="--")
    row = correlations[
        correlations.stream.eq("fresh")
        & correlations.scope.eq("above_chance_targets")
        & correlations.metric.eq("accuracy")
    ].iloc[0]
    ax.text(
        .03, .96,
        rf"Spearman $\rho$={row.spearman:.3f} "
        rf"[{row.crossed_seed_target_bootstrap_low:.3f}, "
        rf"{row.crossed_seed_target_bootstrap_high:.3f}]",
        transform=ax.transAxes, va="top", fontsize=8,
    )
    ax.set_xlabel("Change in U1 agreement (points)")
    ax.set_ylabel("Change in accuracy (points)")
    ax.set_title("b  Preservation tracks schedule effects", loc="left", weight="bold")
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="lower right")


def panel_fusion(ax, target_summary, gain_summary):
    part = target_summary[
        target_summary.method.eq("trace_health_fusion")
        & target_summary.target.isin(TARGET_ORDER)
    ].set_index("target")
    macro = gain_summary[
        gain_summary.scope.eq("supported_targets")
        & gain_summary.method.eq("trace_health_fusion")
    ].set_index("metric")
    accuracy = [part.loc[target].mean_delta_accuracy * 100 for target in TARGET_ORDER]
    auc = [part.loc[target].mean_delta_roc_auc * 100 for target in TARGET_ORDER]
    accuracy.append(macro.loc["accuracy"].mean_delta * 100)
    auc.append(macro.loc["roc_auc"].mean_delta * 100)
    positions = np.arange(5)
    width = .34
    ax.bar(positions - width / 2, accuracy, width, color="#0072B2", label="Accuracy")
    ax.bar(positions + width / 2, auc, width, color="#E69F00", label="ROC-AUC")
    for index, metric in enumerate(("accuracy", "roc_auc")):
        row = macro.loc[metric]
        value = row.mean_delta * 100
        ax.errorbar(
            4 + (-width / 2 if metric == "accuracy" else width / 2),
            value,
            yerr=np.array([[
                value - row.crossed_seed_target_bootstrap_low * 100
            ], [
                row.crossed_seed_target_bootstrap_high * 100 - value
            ]]),
            fmt="none", ecolor="#222222", capsize=3, linewidth=1,
        )
    ax.axhline(0, color="#777777", linewidth=.8, linestyle="--")
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [TARGET_LABELS[target] for target in TARGET_ORDER] + ["Macro"],
        fontsize=8,
    )
    ax.set_ylabel("TRACE fusion gain (points)")
    ax.set_title("c  Support-only fusion improves accuracy", loc="left", weight="bold")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper right")


def main() -> int:
    contrasts = pd.read_csv(DATA / "paired_contrasts.csv")
    correlations = pd.read_csv(DATA / "health_correlations.csv")
    target_summary = pd.read_csv(DATA / "selector_contrast_summary_by_target.csv")
    gain_summary = pd.read_csv(DATA / "selector_gain_summary.csv")
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlepad": 8,
        }
    )
    figure, axes = plt.subplots(1, 3, figsize=(11.4, 3.25))
    panel_schedule(axes[0], contrasts)
    panel_health(axes[1], contrasts, correlations)
    panel_fusion(axes[2], target_summary, gain_summary)
    figure.tight_layout(w_pad=2.0)
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        figure.savefig(
            FIGURES / f"trace_schedule_mechanism.{suffix}",
            dpi=300, bbox_inches="tight",
        )
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
