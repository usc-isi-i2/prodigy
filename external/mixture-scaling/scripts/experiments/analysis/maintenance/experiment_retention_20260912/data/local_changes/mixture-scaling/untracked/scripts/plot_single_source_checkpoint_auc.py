#!/usr/bin/env python3
"""Plot matched single-source GraphSAGE probe AUC over SSL checkpoints."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT = RESULTS / "figures" / "single-source-label-cls-auc-over-steps.png"

DISPLAY_NAMES = {
    "cora": "Cora",
    "covid_political": "COVID political",
    "election2020": "Election 2020",
    "facebook_page_reference": "Facebook pages",
    "pubmed": "PubMed",
    "twibot20": "TwiBot-20",
    "ukr_rus_suspended": "UKR/RUS suspended",
}

COLORS = {
    "cora": "#0072B2",
    "covid_political": "#D55E00",
    "election2020": "#009E73",
    "facebook_page_reference": "#CC79A7",
    "pubmed": "#E69F00",
    "twibot20": "#56B4E9",
    "ukr_rus_suspended": "#6F4E7C",
}


def main() -> None:
    frames = [
        pd.read_csv(RESULTS / "primary_s0" / "primary_results.csv"),
        pd.read_csv(RESULTS / "primary_s1_s2" / "primary_results.csv"),
    ]
    data = pd.concat(frames, ignore_index=True)
    data = data[data["sources"].eq(data["target"])].copy()

    counts = data.groupby(["target", "checkpoint_step"])["seed"].nunique()
    if not counts.eq(3).all():
        raise RuntimeError("Expected three seeds for every target/checkpoint pair")

    summary = (
        data.groupby(["target", "checkpoint_step"], as_index=False)
        .agg(auc=("roc_auc_ovr_macro", "mean"))
        .sort_values("checkpoint_step")
    )

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(13.5, 7.6))
    for target in DISPLAY_NAMES:
        group = summary[summary["target"].eq(target)]
        ax.plot(
            group["checkpoint_step"],
            group["auc"],
            color=COLORS[target],
            linewidth=2.7,
            label=DISPLAY_NAMES[target],
        )

    ax.set_xscale("log")
    ax.set_xticks([100, 300, 900, 2500], labels=["100", "300", "900", "2,500"])
    ax.set_xlim(90, 2850)
    ax.set_ylim(0.43, 1.005)
    ax.set_xlabel("GraphSAGE SSL training step", fontsize=14)
    ax.set_ylabel("Label classification ROC-AUC", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(axis="x", which="minor", visible=False)
    ax.grid(axis="both", which="major", color="#D7D7D7", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "Single-source GraphSAGE: downstream label classification over SSL training",
        x=0.075,
        y=0.97,
        ha="left",
        fontsize=20,
        fontweight="semibold",
    )
    ax.set_title(
        "Matched source = target; frozen linear probe; mean across 3 seeds",
        loc="left",
        fontsize=12.5,
        color="#444444",
        pad=14,
    )
    ax.legend(
        title="Training / target graph",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=4,
        frameon=False,
        fontsize=11.5,
        title_fontsize=11.5,
    )
    fig.text(
        0.075,
        0.012,
        "Older 10-labels-per-class protocol; downstream AUC was measured at fixed checkpoints and not used to select SSL checkpoints.",
        fontsize=10.5,
        color="#555555",
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.84, bottom=0.25)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
