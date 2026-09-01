#!/usr/bin/env python3
"""Plot transfer-only means for specialist and held-out-mixture models."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).parent
DATA = ROOT / "data" / "matched_results.csv"
FIGURES = ROOT / "figures"
ARMS = ["MT", "NM", "NM_MT"]
ARM_LABELS = {"MT": "MT", "NM": "NM", "NM_MT": "NM+MT"}
SOURCES = [
    "heldout_mixture",
    "covid_political",
    "election2020",
    "facebook_page_reference",
    "twibot20",
    "ukr_rus_suspended",
]
LABELS = {
    "heldout_mixture": "Held-out mix",
    "covid_political": "COVID",
    "election2020": "Election",
    "facebook_page_reference": "Facebook",
    "twibot20": "TwiBot",
    "ukr_rus_suspended": "UKR-RUS",
}


def transfer_means(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for arm in ARMS:
        for source in SOURCES:
            subset = results[(results.arm == arm) & (results.source == source)]
            if source != "heldout_mixture":
                subset = subset[subset.target != source]
            rows.append(
                {
                    "arm": arm,
                    "source": source,
                    "n_targets": len(subset),
                    "accuracy": subset.accuracy.mean(),
                    "roc_auc": subset.roc_auc.mean(),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    results = pd.read_csv(DATA)
    means = transfer_means(results)
    means.to_csv(ROOT / "data" / "transfer_only_model_means.csv", index=False)

    colors = ["#D55E00", "#0072B2", "#56B4E9", "#009E73", "#CC79A7", "#E69F00"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.2), sharey="row", constrained_layout=True)
    for col, arm in enumerate(ARMS):
        frame = means[means.arm == arm].set_index("source").loc[SOURCES]
        for row, (metric, title) in enumerate((("accuracy", "Accuracy"), ("roc_auc", "ROC-AUC"))):
            ax = axes[row, col]
            values = frame[metric].to_numpy() * 100
            bars = ax.bar(np.arange(len(SOURCES)), values, color=colors, width=0.74)
            ax.set_title(f"{ARM_LABELS[arm]} — {title}", fontweight="semibold")
            ax.set_xticks(np.arange(len(SOURCES)), [LABELS[s] for s in SOURCES], rotation=35, ha="right")
            ax.set_ylim(30 if metric == "accuracy" else 40, 80)
            ax.grid(axis="y", alpha=0.22, linewidth=0.7)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
            if col == 0:
                ax.set_ylabel("Mean transfer performance (%)")
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, value + 0.8, f"{value:.1f}",
                        ha="center", va="bottom", fontsize=8)
    fig.suptitle(
        "Transfer-only mean by pretrained model\n"
        "Specialists exclude their own graph; held-out mix averages five leave-one-graph-out models",
        fontsize=14,
    )
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "transfer_only_model_means.png", dpi=220)
    fig.savefig(FIGURES / "transfer_only_model_means.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
