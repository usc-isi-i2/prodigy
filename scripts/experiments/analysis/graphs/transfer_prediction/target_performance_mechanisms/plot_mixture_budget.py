"""All foreign cases across fixed saved-update budgets; no selected best step."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from .analyze_mixture_budget import STEPS, budget_step
from .plot_mixture_complementarity import TARGETS


def panel_data(cells):
    keys = ["stream", "target", "model_id", "specialist_step"]
    if len(cells) != 1800 or cells.duplicated(keys).any():
        raise ValueError("all 1800 budget cells required")
    shown = cells[~cells.target_seen].copy()
    if len(shown) != 1160 or not np.isfinite(shown.ensemble_minus_mixture_auc).all():
        raise ValueError("all 1160 foreign cases required")
    for (_, _, count), group in shown.groupby(["stream", "target", "source_count"]):
        if len(group) != {2: 112, 8: 4}.get(count) or set(group.specialist_step) != set(STEPS):
            raise ValueError("foreign source/step panel incomplete")
    return shown


def main():
    root = Path(__file__).parent
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--data", type=Path, default=root / "data")
    parser.add_argument("--output", type=Path, default=root / "figures")
    args = parser.parse_args()
    receipt = json.loads((args.data / "mixture_budget_validation.json").read_text())
    if receipt.get("comparisons") != 1800 or receipt.get("terminal_reference_reproduced") is not True:
        raise ValueError("complete verified budget analysis required")
    shown = panel_data(pd.read_csv(args.data / "mixture_budget_comparisons.csv"))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    colors = {"original": "#236d9f", "fresh": "#cc632f"}
    fig, axes = plt.subplots(2, 5, figsize=(16, 8.8))
    fig.subplots_adjust(left=.07, right=.985, bottom=.25, top=.82, wspace=.32, hspace=.4)
    for col, (target, title) in enumerate(TARGETS):
        for row, count in enumerate((2, 8)):
            ax = axes[row, col]
            group = shown[shown.target.eq(target) & shown.source_count.eq(count)]
            x = count * np.array(STEPS)
            ax.axvspan(x[0], 2500, color="#eef1f3", zorder=0)
            ax.axhline(0, color="#6c7379", ls="--", lw=.8)
            ax.axvline(2500, color="#6c7379", ls=":", lw=1)
            for stream, color in colors.items():
                selected = group[group.stream.eq(stream)]
                for _, case in selected.groupby("model_id"):
                    series = case.set_index("specialist_step").loc[list(STEPS)]
                    ax.plot(x, 100 * series.ensemble_minus_mixture_auc, color=color, alpha=.15 if count == 2 else .9, lw=.6 if count == 2 else 1.5)
                means = selected.groupby("specialist_step").ensemble_minus_mixture_auc.mean().loc[list(STEPS)]
                ax.plot(x, 100 * means, color=color, lw=2, marker="o", ms=3)
                selected_step = budget_step(count)
                ax.scatter([count * selected_step], [100 * means.loc[selected_step]], color=color, s=48, marker="D", zorder=5)
            ax.set_xscale("log")
            ax.set_xticks(x, [f"{v:,}" for v in x], rotation=25, ha="right")
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_title(title, loc="left", fontsize=10)
            if col == 0:
                ax.set_ylabel(("Foreign pairs" if count == 2 else "Foreign LOO") + "\nEnsemble minus mixture AUC (pp)")
            if row == 1:
                ax.set_xlabel("Total specialist training updates")
    fig.suptitle("Specialist averaging across saved training-update budgets", fontsize=17, y=.97)
    fig.text(.07, .91, "Joint models stay fixed at 2,500 updates. Above zero favors the specialist ensemble; shaded region uses no more total updates.", fontsize=10.5)
    handles = [Line2D([], [], color=c, lw=2, label=s.capitalize() + " episodes") for s, c in colors.items()]
    handles += [Line2D([], [], color="#666", marker="D", ls="", label="Fixed under-budget checkpoint rule")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.062, .89), ncol=3, frameon=False)
    fig.text(.07, .12, "Pairs: every one of 28 foreign-source cases per target/stream is shown faintly; bold line is their mean. LOO: one foreign case per target/stream.\n"
             "Diamonds select the largest saved step with K × step ≤ 2,500: pairs at 900 each (1,800 total), LOO at 300 each (2,400 total).\n"
             "All four saved checkpoints and both episode streams are retained; no target-based checkpoint or ensemble-weight fitting.", fontsize=9, linespacing=1.6)
    fig.text(.07, .035, "One historical training seed; equal-probability averaging. Saved-update/episode budgets are not matched FLOPs or wall time.\n"
             "Ensembles retain 2x/8x model capacity and inference forwards. Shared sources are not independent replications; no causal interference claim. Axis ranges vary.", fontsize=9, linespacing=1.5)
    args.output.mkdir(exist_ok=True, parents=True)
    for suffix in ("png", "pdf"):
        fig.savefig(args.output / f"mixture_training_budget_foreign.{suffix}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
