"""All-target dose and checkpoint curves; seed/stream ranges are not CIs."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TARGETS = ["covid_political", "election2020", "facebook_page_reference", "twibot20", "ukr_rus_suspended"]
NAMES = {"covid_political": "Political (CP)", "election2020": "Political (E20)",
         "facebook_page_reference": "Pages (FB)", "twibot20": "Bots (T20)", "ukr_rus_suspended": "Suspension"}
SOURCES = {"cp_hk": ("Hong Kong", "#b3587a"), "ukr_rus": ("Ukraine", "#128878")}


def curve_plot(frame, xcol, ticks, metric, output, xlabel, title):
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.7))
    for ax, target in zip(axes, TARGETS):
        ax.axhline(0, color="0.5", lw=.7, zorder=1)
        for source, (label, color) in SOURCES.items():
            part = frame[(frame.target == target) & (frame.source == source)]
            summary = part.groupby(xcol)[metric].agg(["mean", "min", "max"]).reindex(ticks)*100
            if summary.isna().any().any():
                raise ValueError("incomplete curve")
            x = np.arange(len(ticks))
            ax.fill_between(x, summary["min"], summary["max"], color=color, alpha=.13, linewidth=0)
            ax.plot(x, summary["mean"], "o-", color=color, label=label, ms=4, lw=1.5)
        ax.set_title(NAMES[target], fontsize=11)
        ax.set_xticks(np.arange(len(ticks)), [str(t) for t in ticks], fontsize=9)
        ax.tick_params(axis="y", labelsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.18)
    axes[0].set_ylabel("AUC change from intact (points)", fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .94), ncol=2, frameon=False, fontsize=10)
    fig.suptitle(title, fontsize=13, y=1.0)
    fig.supxlabel(xlabel, fontsize=10, y=.13)
    fig.text(.5, .02, "Lines: mean across three initialization seeds and two streams. Bands: their observed range, not confidence intervals. Panel y-scales differ.", ha="center", fontsize=9)
    fig.subplots_adjust(left=.05, right=.99, top=.76, bottom=.3, wspace=.36)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    dose = pd.read_csv(args.input / "dose_cells.csv")
    cells = pd.read_csv(args.input / "cells.csv")
    if len(dose) != 300 or len(cells) != 1140:
        raise ValueError("complete analyzed panel required")
    args.output.mkdir(parents=True, exist_ok=True)
    curve_plot(dose, "suppression_percent", [0, 25, 50, 75, 100], "delta_roc_auc", args.output / "support_dose_curves.png",
               "Requested support-edge suppression (%)", "Support suppression dose at update 2,500 (three draws averaged within each seed/stream)")
    curve_plot(cells[cells.suppression_percent == 100], "step", [0, 100, 300, 900, 2500], "delta_roc_auc", args.output / "support_suppression_saved_steps.png",
               "Saved training updates (equally spaced checkpoints)", "Support-removal effects through training (same cached target inputs)")


if __name__ == "__main__":
    main()
