"""Plot layer-agreement predictiveness across the full source-target grid."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).parent
DATA = ROOT / "data"
FIGURES = ROOT / "figures"


def main():
    FIGURES.mkdir(exist_ok=True)
    health = pd.read_csv(DATA / "model_health.csv")
    correlations = pd.read_csv(DATA / "health_correlations.csv")
    fresh = health[health.stream.eq("fresh")].copy()
    fresh["health_centered"] = fresh.u1_agreement_rate - fresh.groupby("target").u1_agreement_rate.transform("mean")
    fresh["accuracy_centered"] = fresh.accuracy - fresh.groupby("target").accuracy.transform("mean")
    target_labels = {
        "covid_political": "COVID-pol.", "election2020": "Election",
        "facebook_page_reference": "Facebook", "twibot20": "TwiBot-20",
        "ukr_rus_suspended": "Suspended",
    }
    colors = dict(zip(target_labels, ["#376795", "#d4743c", "#4c956c", "#8a5ca7", "#777777"]))
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4), gridspec_kw={"width_ratios": [1.15, 1]})
    for target, label in target_labels.items():
        part = fresh[fresh.target.eq(target)]
        axes[0].scatter(100 * part.health_centered, 100 * part.accuracy_centered,
                        label=label, color=colors[target], s=30, alpha=.88)
    coefficient = np.polyfit(fresh.health_centered, fresh.accuracy_centered, 1)
    grid = np.linspace(fresh.health_centered.min(), fresh.health_centered.max(), 100)
    axes[0].plot(100 * grid, 100 * np.polyval(coefficient, grid), color="#222222", linewidth=1.5)
    axes[0].axhline(0, color="#bbbbbb", linewidth=.8)
    axes[0].axvline(0, color="#bbbbbb", linewidth=.8)
    axes[0].set_xlabel("U1 agreement relative to target mean (points)")
    axes[0].set_ylabel("Accuracy relative to target mean (points)")
    axes[0].set_title("A. Health predicts source transfer", loc="left", weight="bold")
    axes[0].legend(frameon=False, ncol=2, fontsize=8)
    axes[0].text(.97, .05, r"target-centered $\rho=.740$", transform=axes[0].transAxes,
                 va="bottom", ha="right", bbox={"facecolor": "white", "edgecolor": "none", "alpha": .8})

    corr = correlations[(correlations.stream == "fresh") &
                        (correlations.metric == "accuracy") &
                        (correlations.scope != "target_centered_pool")].set_index("scope").loc[list(target_labels)]
    y = np.arange(len(corr))
    axes[1].barh(y, corr.spearman, color=[colors[target] for target in corr.index])
    axes[1].set_yticks(y, [target_labels[target] for target in corr.index])
    axes[1].invert_yaxis()
    axes[1].axvline(0, color="#888888", linewidth=.8)
    axes[1].set_xlim(-.45, 1.05)
    axes[1].set_xlabel("Within-target Spearman rho")
    axes[1].set_title("B. Signal vanishes at chance", loc="left", weight="bold")
    for index, value in enumerate(corr.spearman):
        axes[1].text(value + .03, index, f"{value:.2f}", va="center", ha="left")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURES / f"health_predictiveness.{suffix}", dpi=220, bbox_inches="tight")


if __name__ == "__main__":
    main()
