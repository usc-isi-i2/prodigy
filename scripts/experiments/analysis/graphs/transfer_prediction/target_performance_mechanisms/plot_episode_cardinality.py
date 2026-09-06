"""Show every foreign-donor count counterfactual, not only the primary contrast."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.experiments.setup.target_performance_mechanisms.episode_cardinality_contract import VARIANTS


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--figures", type=Path, required=True)
    args = parser.parse_args()
    cells = pd.read_csv(args.data / "episode_cardinality_cells.csv")
    cells = cells[cells.foreign & cells.variant.ne("baseline")]
    if len(cells) != 480:
        raise ValueError("complete 480-cell foreign intervention panel required")
    targets = {"covid_political": "COVID Political", "election2020": "Election",
        "facebook_page_reference": "Facebook pages", "twibot20": "TwiBot", "ukr_rus_suspended": "Suspension"}
    names = ["Joint", "Way", "Shot", "Inverse", "Virtual", "Bias"]
    colors = ["#237da1", "#818b93", "#818b93", "#d58036", "#818b93", "#818b93"]
    fig, axes = plt.subplots(2, 5, figsize=(16.8, 6.4), sharey="col")
    for col, (target, title) in enumerate(targets.items()):
        for row, stream in enumerate(("original", "fresh")):
            ax = axes[row, col]
            group = cells[cells.target.eq(target) & cells.stream.eq(stream)]
            for pos, (variant, color) in enumerate(zip(VARIANTS[1:], colors)):
                values = group[group.variant.eq(variant)].sort_values("source").delta_roc_auc.to_numpy() * 100
                if len(values) != 8:
                    raise ValueError("eight foreign donors per condition required")
                ax.scatter(pos + np.linspace(-.19, .19, 8), values, s=17, color=color, alpha=.7, linewidths=0)
                ax.plot([pos - .24, pos + .24], [values.mean()] * 2, color="black", lw=2)
            ax.axhline(0, color="#75818a", lw=.8, ls="--", zorder=0)
            ax.set_xticks(range(6), names, rotation=45, ha="right", fontsize=8)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", color="#dfe3e6", lw=.5)
            ax.set_axisbelow(True)
            if row == 0:
                ax.set_title(title, fontsize=11)
            if col == 0:
                ax.set_ylabel(f"{stream.capitalize()} episodes\nAUC change (points)")
    fig.suptitle("Does restoring training-time attention multiplicity improve transfer?", x=.055, ha="left", fontsize=15, weight="bold")
    fig.text(.055, .92, "Dots: eight foreign donors per panel; black bars: mean. Blue: primary joint attention condition; orange: reciprocal control.", fontsize=10)
    fig.text(.055, .025, "Way/Shot/Joint/Inverse change attention only. Virtual also scales affine bias; Bias changes only that bias. Fixed examples and weights; one training seed.\nThese reweighted messages do not create 30 distinct classes. All six prespecified interventions and both test streams are shown.", fontsize=9)
    fig.tight_layout(rect=(.02, .09, 1, .9), h_pad=1.6)
    args.figures.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(args.figures / f"episode_cardinality_foreign.{extension}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
