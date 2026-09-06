"""Display every foreign-source pair and LOO case; no confidence-interval fiction."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

TARGETS = [("covid_political", "COVID Political"), ("election2020", "Election"),
           ("facebook_page_reference", "Facebook pages"), ("twibot20", "TwiBot"),
           ("ukr_rus_suspended", "Ukraine suspension")]


def panel_data(cells):
    if len(cells) != 450 or cells.duplicated(["stream", "target", "model_id"]).any():
        raise ValueError("complete 450-cell analysis required")
    shown = cells[~cells.target_seen].copy()
    if len(shown) != 290:
        raise ValueError("140 foreign pairs and five foreign LOO cases per stream required")
    for (stream, target, count), group in shown.groupby(["stream", "target", "source_count"]):
        if stream not in {"original", "fresh"} or target not in dict(TARGETS) or len(group) != {2: 28, 8: 1}.get(count):
            raise ValueError("foreign source panel incomplete")
    shown["ensemble_gain_pp"] = 100 * (shown.probability_ensemble_roc_auc - shown.constituent_mean_roc_auc)
    shown["mixture_gain_pp"] = 100 * (shown.mixture_roc_auc - shown.constituent_mean_roc_auc)
    shown["fixes_pp"] = 100 * shown.mixture_fixes_all_wrong_fraction
    shown["harms_pp"] = 100 * shown.mixture_harms_all_correct_fraction
    if not np.isfinite(shown[["ensemble_gain_pp", "mixture_gain_pp", "fixes_pp", "harms_pp"]]).all().all():
        raise ValueError("nonfinite plotted diagnostic")
    return shown


def main():
    root = Path(__file__).parent
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--data-dir", type=Path, default=root / "data")
    parser.add_argument("--output-dir", type=Path, default=root / "figures")
    args = parser.parse_args()
    receipt = json.loads((args.data_dir / "mixture_complementarity_validation.json").read_text())
    if receipt["mixture_comparisons"] != 450 or receipt["model_cells"] != 540 or receipt["strata"] != 1350:
        raise ValueError("complete verified mixture analysis required")
    shown = panel_data(pd.read_csv(args.data_dir / "mixture_complementarity_comparisons.csv"))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    colors = {"original": "#236d9f", "fresh": "#cc632f"}
    fig, axes = plt.subplots(2, 5, figsize=(16, 8.2))
    fig.subplots_adjust(left=.06, right=.99, bottom=.21, top=.83, wspace=.32, hspace=.48)
    for col, (target, title) in enumerate(TARGETS):
        group = shown[shown.target == target]
        for row, (x, y) in enumerate((("ensemble_gain_pp", "mixture_gain_pp"), ("fixes_pp", "harms_pp"))):
            ax = axes[row, col]
            values = group[[x, y]].to_numpy()
            lo, hi = min(float(values.min()), 0), max(float(values.max()), 0)
            pad = max((hi - lo) * .1, .1)
            lo, hi = lo - pad, hi + pad
            ax.plot([lo, hi], [lo, hi], color="#888f94", lw=.8, ls="--", zorder=0)
            if row == 0:
                ax.axhline(0, color="#d0d5d9", lw=.7, zorder=0)
                ax.axvline(0, color="#d0d5d9", lw=.7, zorder=0)
            for stream, color in colors.items():
                for count, marker, size in ((2, "o", 25), (8, "D", 70)):
                    points = group[(group.stream == stream) & (group.source_count == count)]
                    ax.scatter(points[x], points[y], c=color, marker=marker, s=size,
                               alpha=.68 if count == 2 else 1, edgecolors="white", linewidths=.4,
                               zorder=2 if count == 2 else 3)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect("equal", adjustable="box")
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_title(title, loc="left", fontsize=10)
            ax.set_xlabel("Ensemble AUC gain (pp)" if row == 0 else "Shared errors fixed (% queries)")
            if col == 0:
                ax.set_ylabel("Trained-mixture AUC gain (pp)" if row == 0 else "Unanimous-correct harmed (% queries)")
    fig.suptitle("What does joint training add beyond averaging specialists?", fontsize=17, y=.975)
    fig.text(.06, .918, "All foreign-source mixtures; fixed equal-probability ensembles; identical test inputs within each stream.", fontsize=11)
    handles = [Line2D([], [], color=c, marker="o", ls="", label=f"{s.capitalize()} episodes") for s, c in colors.items()]
    handles += [Line2D([], [], color="#555d62", marker=m, ls="", label=label) for m, label in
                (("o", "28 pairs per target/stream"), ("D", "1 foreign LOO per target/stream"))]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.052, .895), ncol=4, frameon=False)
    fig.text(.06, .115,
             "Top: both gains are relative to mean constituent AUC; above the diagonal favors the trained mixture over the ensemble.\n"
             "Bottom: below the diagonal means more shared errors are fixed than unanimous-correct decisions are harmed.\n"
             "The bottom panel is an accuracy decomposition, not an AUC decomposition; mixed-correctness queries are reported in the tables.",
             fontsize=9, linespacing=1.6)
    fig.text(.06, .035,
             "Historical sampler; one training seed. Ensembles cost 2x/8x total training and inference. Cases share sources and are not independent trials.\n"
             "No fitted ensemble weights, target-selected checkpoints, or causal interference claim. Axis ranges vary by target.", fontsize=9, linespacing=1.5)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(args.output_dir / f"mixture_complementarity_foreign.{suffix}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
