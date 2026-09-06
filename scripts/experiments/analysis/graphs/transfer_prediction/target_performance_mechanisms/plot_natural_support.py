"""Two compact views of the prespecified political support experiment."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    data = args.root / "data"
    validation = json.loads((data / "natural_support_validation.json").read_text())
    if validation["metric_cells"] != 660 or validation["support_positions_reproduced"] != 204800:
        raise ValueError("complete independently validated results required")
    sensitivity = pd.read_csv(data / "natural_support_political_sensitivity.csv")
    effects = pd.read_csv(data / "natural_support_selection_effects.csv")
    effects = effects[effects.target.eq("covid_political") & effects.condition.eq("support_cv_selected")]
    if len(sensitivity) != 6 or len(effects) != 12:
        raise ValueError("all three seeds and both streams required")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.8))
    colors = ["#0072B2", "#D55E00", "#009E73"]
    markers = {"original": "o", "fresh": "^"}
    for row in sensitivity.itertuples():
        axes[0].scatter(row.ukr_rus, row.cp_hk, color=colors[row.seed], marker=markers[row.stream], s=62,
                        facecolors=colors[row.seed] if row.stream == "original" else "none", linewidths=1.6, zorder=3)
    axes[0].plot([0, .065], [0, .065], color=".65", ls="--", lw=1)
    axes[0].text(.037, .030, "Equal sensitivity", rotation=42, color=".45", fontsize=9)
    axes[0].set(xlim=(0, .065), ylim=(0, .065), xlabel="Ukraine: mean query-probability variance",
                ylabel="Hong Kong: mean query-probability variance", title="Hong Kong is more sensitive to support choice")
    axes[0].set_aspect("equal", adjustable="box")
    axes[1].axhline(0, color=".5", lw=1)
    for row in effects.itertuples():
        x = (0 if row.source == "cp_hk" else 1) + (row.seed-1)*.15 + (-.025 if row.stream == "original" else .025)
        axes[1].scatter(x, row.delta_nll, color=colors[row.seed], marker=markers[row.stream], s=62,
                        facecolors=colors[row.seed] if row.stream == "original" else "none", linewidths=1.6, zorder=3)
    axes[1].set(xticks=[0, 1], xticklabels=["Hong Kong", "Ukraine"], xlim=(-.4, 1.4), ylim=(-.59, .09),
                ylabel="Selected − mean random-support query loss",
                title="Support-only selection helps Hong Kong\nbut usually worsens Ukraine's loss")
    axes[1].text(.98, .16, "Below zero = lower loss", ha="right", transform=axes[1].transAxes, color=".4", fontsize=9)
    for ax in axes:
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
    handles = [Line2D([], [], color=c, marker="o", linestyle="none", label=f"Seed {i}") for i, c in enumerate(colors)]
    handles += [Line2D([], [], color=".3", marker="o", linestyle="none", label="Original episodes"),
                Line2D([], [], color=".3", marker="^", markerfacecolor="none", linestyle="none", label="Fresh episodes")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .11), ncol=5, frameon=False, fontsize=9)
    fig.text(.5, .052, "COVID Political · 3 initialization seeds × 2 episode streams · 8 valid support sets per fixed query episode",
             ha="center", fontsize=9)
    fig.text(.5, .017, "Selection uses a larger labeled pool; this is not a standard 10-shot benchmark comparison.", ha="center", fontsize=9, color=".4")
    fig.subplots_adjust(left=.075, right=.97, top=.87, bottom=.28, wspace=.39)
    output = args.root / "figures" / "natural_support_political.png"
    output.parent.mkdir(exist_ok=True)
    fig.savefig(output, dpi=200, facecolor="white")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
