#!/usr/bin/env python3
"""Plot TRACE health fusion against fixed and equal-ensemble controls."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
TARGETS = (
    "covid_political",
    "election2020",
    "facebook_page_reference",
    "twibot20",
)
LABELS = {
    "covid_political": "COVID\npolitical",
    "election2020": "Election\n2020",
    "facebook_page_reference": "Facebook\npages",
    "twibot20": "TwiBot-20",
}


def main() -> int:
    results = pd.read_csv(DATA / "health_fusion_results.csv")
    results = results[
        results.stream.eq("fresh") & results.target_supported_at_0_55
    ]
    hard = pd.read_csv(DATA / "support_only_covered_seed_macro.csv")
    by_target = pd.read_csv(DATA / "health_fusion_by_target.csv")
    gains = pd.read_csv(DATA / "health_fusion_gain_summary.csv")

    absolute = results.groupby("method")[["accuracy", "roc_auc"]].mean()
    names = ("Fixed\nexpert", "Equal\nensemble", "Hard\nTRACE", "TRACE\nfusion")
    accuracy = (
        absolute.loc["discovery_auc_fixed", "accuracy"],
        absolute.loc["equal_probability", "accuracy"],
        hard.mean_accuracy.mean(),
        absolute.loc["trace_health_fusion", "accuracy"],
    )
    auc = (
        absolute.loc["discovery_auc_fixed", "roc_auc"],
        absolute.loc["equal_probability", "roc_auc"],
        hard.mean_auc.mean(),
        absolute.loc["trace_health_fusion", "roc_auc"],
    )
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlepad": 8,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(8.4, 3.15))
    x = np.arange(4)
    axes[0].plot(x, np.array(accuracy) * 100, "o-", color="#0072B2", label="Accuracy")
    axes[0].plot(x, np.array(auc) * 100, "s-", color="#E69F00", label="ROC-AUC")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names)
    axes[0].set_ylabel("Fresh supported-target macro (%)")
    axes[0].set_ylim(78, 87)
    axes[0].set_title("a  Health-aware fusion beats selection", loc="left", weight="bold")
    axes[0].legend(frameon=False, fontsize=8, ncol=2, loc="lower right")

    comparison = by_target[
        by_target.stream.eq("fresh")
        & by_target.method.eq("trace_health_fusion")
        & by_target.reference.eq("equal_probability")
        & by_target.target.isin(TARGETS)
    ].set_index("target")
    macro = gains[
        gains.stream.eq("fresh")
        & gains.scope.eq("supported_targets")
        & gains.method.eq("trace_health_fusion")
        & gains.reference.eq("equal_probability")
    ].set_index("metric")
    accuracy_gain = [comparison.loc[target].mean_delta_accuracy * 100 for target in TARGETS]
    auc_gain = [comparison.loc[target].mean_delta_roc_auc * 100 for target in TARGETS]
    accuracy_gain.append(macro.loc["accuracy"].mean_delta * 100)
    auc_gain.append(macro.loc["roc_auc"].mean_delta * 100)
    x = np.arange(5)
    width = .34
    axes[1].bar(x - width / 2, accuracy_gain, width, color="#0072B2", label="Accuracy")
    axes[1].bar(x + width / 2, auc_gain, width, color="#E69F00", label="ROC-AUC")
    for offset, metric in ((-width / 2, "accuracy"), (width / 2, "roc_auc")):
        row = macro.loc[metric]
        value = row.mean_delta * 100
        axes[1].errorbar(
            4 + offset,
            value,
            yerr=np.array(
                [
                    [value - row.crossed_seed_target_bootstrap_low * 100],
                    [row.crossed_seed_target_bootstrap_high * 100 - value],
                ]
            ),
            fmt="none", ecolor="#222222", capsize=3, linewidth=1,
        )
    axes[1].axhline(0, color="#777777", linewidth=.8, linestyle="--")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([LABELS[target] for target in TARGETS] + ["Macro"])
    axes[1].set_ylabel("Gain over equal ensemble (points)")
    axes[1].set_title("b  Gain is not ordinary ensembling", loc="left", weight="bold")
    axes[1].legend(frameon=False, fontsize=8, ncol=2, loc="upper left")

    figure.tight_layout(w_pad=2.2)
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        figure.savefig(
            FIGURES / f"trace_health_fusion.{suffix}",
            dpi=300, bbox_inches="tight",
        )
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
