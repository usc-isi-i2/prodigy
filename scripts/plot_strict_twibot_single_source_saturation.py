#!/usr/bin/env python3
"""Plot strict TwiBot probe AUC across single-source SSL checkpoints."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results" / "twibot_saturation" / "saturation.csv"
OUTPUT = (
    ROOT
    / "results"
    / "figures"
    / "strict-twibot-single-source-label-cls-auc-over-steps.png"
)

ARMS = {
    "cora": ("Cora-only SSL", "#0072B2"),
    "election": ("Election 2020-only SSL", "#D55E00"),
}
COMMON_STEPS = [100, 300, 600, 900, 1500, 2500]


def main() -> None:
    data = pd.read_csv(INPUT)
    data = data[data["arm"].isin(ARMS) & data["checkpoint_step"].isin(COMMON_STEPS)]

    counts = data.groupby(["arm", "checkpoint_step"])["seed"].nunique()
    if len(counts) != len(ARMS) * len(COMMON_STEPS) or not counts.eq(3).all():
        raise RuntimeError("Expected three seeds at every shared checkpoint")

    summary = (
        data.groupby(["arm", "checkpoint_step"], as_index=False)
        .agg(auc=("auc", "mean"))
        .sort_values("checkpoint_step")
    )

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    for arm, (label, color) in ARMS.items():
        group = summary[summary["arm"].eq(arm)]
        ax.plot(
            group["checkpoint_step"],
            group["auc"],
            color=color,
            linewidth=3.2,
            label=label,
        )

    ax.set_xscale("log")
    ax.set_xticks(
        COMMON_STEPS,
        labels=["100", "300", "600", "900", "1,500", "2,500"],
    )
    ax.set_xlim(90, 2850)
    observed = summary["auc"]
    padding = max(0.006, (observed.max() - observed.min()) * 0.35)
    ax.set_ylim(observed.min() - padding, observed.max() + padding)
    ax.set_xlabel("GraphSAGE SSL training step", fontsize=14)
    ax.set_ylabel("TwiBot-20 test ROC-AUC", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(axis="x", which="minor", visible=False)
    ax.grid(axis="both", which="major", color="#D7D7D7", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "Strict single-source GraphSAGE saturation on TwiBot-20",
        x=0.09,
        y=0.965,
        ha="left",
        fontsize=20,
        fontweight="semibold",
    )
    ax.set_title(
        "70/15/15 node split; frozen linear probe; mean across 3 seeds",
        loc="left",
        fontsize=12.5,
        color="#444444",
        pad=14,
    )
    ax.legend(loc="best", frameon=False, fontsize=12.5)
    fig.text(
        0.09,
        0.02,
        "Each encoder was SSL-pretrained only on the named non-target graph; every checkpoint uses the same TwiBot train/validation/test split.",
        fontsize=10.5,
        color="#555555",
    )
    fig.subplots_adjust(left=0.09, right=0.985, top=0.84, bottom=0.17)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
