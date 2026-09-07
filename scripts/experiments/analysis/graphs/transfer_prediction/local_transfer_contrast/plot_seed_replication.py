"""Paper-facing summary of checkpoint-seed replication and TRACE routing."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).parent
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
TARGETS = [
    "covid_political", "election2020", "facebook_page_reference", "twibot20",
    "ukr_rus_suspended",
]
LABELS = {
    "covid_political": "COVID-pol.", "election2020": "Election",
    "facebook_page_reference": "Facebook", "twibot20": "TwiBot-20",
    "ukr_rus_suspended": "Suspended",
}
COLORS = dict(zip(TARGETS, ["#376795", "#d4743c", "#4c956c", "#8a5ca7", "#777777"]))


def main():
    FIGURES.mkdir(exist_ok=True)
    health = pd.read_csv(DATA / "model_health_all_seeds.csv")
    replication = pd.read_csv(DATA / "multi_health_replication.csv")
    scope = pd.read_csv(DATA / "support_only_scope.csv")
    fresh = health[health.stream.eq("fresh")].copy()
    grouping = fresh.groupby(["target", "seed"])
    fresh["health_centered"] = fresh.u1_agreement_rate - grouping.u1_agreement_rate.transform("mean")
    fresh["accuracy_centered"] = fresh.accuracy - grouping.accuracy.transform("mean")

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.45), gridspec_kw={"width_ratios": [1.25, 1, 1]})
    markers = ["o", "s", "^"]
    for target in TARGETS:
        for seed in range(3):
            part = fresh[fresh.target.eq(target) & fresh.seed.eq(seed)]
            axes[0].scatter(100 * part.health_centered, 100 * part.accuracy_centered,
                            color=COLORS[target], marker=markers[seed], s=23, alpha=.78,
                            label=LABELS[target] if seed == 0 else None)
    coefficient = np.polyfit(fresh.health_centered, fresh.accuracy_centered, 1)
    grid = np.linspace(fresh.health_centered.min(), fresh.health_centered.max(), 100)
    axes[0].plot(100 * grid, 100 * np.polyval(coefficient, grid), color="#222222", linewidth=1.4)
    axes[0].axhline(0, color="#bbbbbb", linewidth=.8)
    axes[0].axvline(0, color="#bbbbbb", linewidth=.8)
    axes[0].set_xlabel("U1 agreement relative to target/seed mean (points)")
    axes[0].set_ylabel("Accuracy relative to target/seed mean (points)")
    axes[0].set_title("A. Health predicts transfer", loc="left", weight="bold")
    axes[0].text(.97, .05, r"$n=135,\ \rho=.710$", transform=axes[0].transAxes,
                 ha="right", bbox={"facecolor": "white", "edgecolor": "none", "alpha": .8})
    axes[0].legend(frameon=False, ncol=2, fontsize=7.5, loc="upper left")

    routed = replication[replication.method.eq("u1_agreement")].set_index("target").loc[TARGETS]
    y = np.arange(len(routed))
    gain = 100 * routed.mean_accuracy_gain.to_numpy()
    low = 100 * routed.min_accuracy_gain.to_numpy()
    high = 100 * routed.max_accuracy_gain.to_numpy()
    axes[1].errorbar(gain, y, xerr=np.vstack((gain - low, high - gain)), fmt="none",
                     ecolor="#555555", elinewidth=1.1, capsize=3)
    axes[1].scatter(gain, y, color=[COLORS[target] for target in routed.index], s=38, zorder=3)
    axes[1].axvline(0, color="#888888", linewidth=.8)
    axes[1].set_yticks(y, [LABELS[target] for target in routed.index])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Accuracy gain over fixed expert (points)")
    axes[1].set_title("B. Health gating replicates", loc="left", weight="bold")
    axes[1].text(.98, .96, "mean and range over 3 seeds", transform=axes[1].transAxes,
                 ha="right", va="top", fontsize=7.5)

    summary = scope.groupby("target").agg(
        competence=("target_support_competence", "mean"),
        gain=("accuracy_gain_over_fixed_best", "mean"),
        supported=("target_supported_at_0_55", "all"),
    ).loc[TARGETS]
    for target, row in summary.iterrows():
        axes[2].scatter(row.competence, 100 * row.gain, color=COLORS[target], s=48)
        axes[2].annotate(LABELS[target], (row.competence, 100 * row.gain),
                         xytext=(4, 4), textcoords="offset points", fontsize=7.5)
    axes[2].axvline(.55, color="#b53b3b", linestyle="--", linewidth=1.1)
    axes[2].axhline(0, color="#aaaaaa", linewidth=.8)
    axes[2].set_xlabel("Maximum support-only competence")
    axes[2].set_ylabel("TRACE accuracy gain (points)")
    axes[2].set_title("C. Competence enables abstention", loc="left", weight="bold")
    axes[2].text(.55, axes[2].get_ylim()[1] * .92, " abstain", color="#9a2f2f", fontsize=7.5)
    fig.tight_layout(w_pad=1.4)
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURES / f"trace_seed_replication.{suffix}", dpi=220, bbox_inches="tight")


if __name__ == "__main__":
    main()
