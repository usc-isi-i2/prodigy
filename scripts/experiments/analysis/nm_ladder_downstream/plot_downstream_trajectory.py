#!/usr/bin/env python3
"""Downstream ladder trajectories — per-graph performance as the merge grows.

The direct analogue of ``analysis/nm_ladder/plot_nm_ladder.py``, but scored on the
DOWNSTREAM tasks instead of on neighbor matching (the objective the encoders were
pretrained with). One panel per task; one trajectory per eval graph; colour encodes
distribution state relative to the current training merge -- blue = the graph is in
the merge, grey dashed = held out. The marker is the rung at which that graph enters.

Order A only (the published topical order), so the x axis reads as "one source added
per rung". The three orders together are the event-study figure
(``plot_downstream_event_study.py``).

Read the two effects separately, because they point opposite ways:
  * the local step AT a marker is the entry effect (small, mostly positive);
  * the overall drift across the panel is budget dilution (mostly downward).
A figure that showed only one of them would misrepresent the result.

Regression is averaged over its three profile targets. Static LP additionally carries
a dotted line per graph at the best heuristic floor (common-neighbours / Adamic-Adar /
preferential-attachment / Jaccard / raw-feature cosine on the identical pair set) --
an encoder below its own floor has not learned anything a counting rule does not know.

Writes figures/nm_ladder_downstream_trajectory.pdf/.png.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
FIGS = os.path.join(HERE, "figures")

# palette (shared with the nm_ladder figures)
BLUE = "#2a78d6"    # in the training merge
GRAY = "#8f8d87"    # held out
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"

LABEL = {
    "ukr_rus_twitter": "Ukr-Rus", "covid19_twitter": "COVID-19", "midterm": "Midterm",
    "covid_political": "COVID-pol.", "election2020": "Election '20",
    "ukr_rus_suspended": "Ukr-Rus susp.", "twibot20": "TwiBot-20",
    "cp_hk_twitter": "CP-HK",
}
# order A: one source added per rung
XTICKS = ["ukr", "+covid", "+midterm", "+cov-pol", "+elec '20",
          "+ukr-susp", "+twibot", "+cp-hk (all 8)"]


def declutter(entries, min_gap):
    """Nudge (y, text) label positions apart so end-of-line labels stay readable.

    Greedy single pass from the bottom: each label is pushed up to at least
    ``min_gap`` above the previous one. Only the label moves, never the data.
    """
    out = []
    for y, text in sorted(entries):
        if out and y - out[-1][0] < min_gap:
            y = out[-1][0] + min_gap
        out.append((y, text))
    return out
PANELS = [
    ("slp", "Static link prediction", "AUC (degree-matched)"),
    ("pl", "Node classification", "ROC-AUC (10-shot)"),
    ("reg", "Node regression", "Spearman (10-shot, mean of 3 targets)"),
]


def main():
    long_csv = os.path.join(DATA, "nm_ladder_downstream_long.csv")
    df = pd.read_csv(long_csv)
    df = df[(df["primary"] == 1) & (df["order"] == "A")]

    floors = {}
    fpath = os.path.join(DATA, "nm_ladder_downstream_slp_floors.csv")
    if os.path.isfile(fpath):
        fl = pd.read_csv(fpath).set_index("dataset")
        floors = fl.max(axis=1, numeric_only=True).to_dict()

    os.makedirs(FIGS, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.0))

    for ax, (task, title, ylab) in zip(axes, PANELS):
        sub = df[df["task"] == task]
        # regression fans out over targets; average them for the trajectory
        cells = (sub.groupby(["dataset", "rung"], as_index=False)
                    .agg(value=("value", "mean"), entry=("entry_rung", "first")))
        datasets = sorted(cells["dataset"].unique(),
                          key=lambda d: cells.loc[cells.dataset == d, "entry"].iloc[0])
        labels = []

        for ds in datasets:
            g = cells[cells.dataset == ds].sort_values("rung")
            entry = int(g["entry"].iloc[0])
            x, y = g["rung"].tolist(), g["value"].tolist()

            pre = [(a, b) for a, b in zip(x, y) if a < entry]
            post = [(a, b) for a, b in zip(x, y) if a >= entry]
            if pre:
                # bridge the two segments so the entry step is visible as a step
                bridge = pre + post[:1]
                ax.plot([a for a, _ in bridge], [b for _, b in bridge],
                        color=GRAY, lw=1.4, ls="--", zorder=2)
            if post:
                ax.plot([a for a, _ in post], [b for _, b in post],
                        color=BLUE, lw=1.9, zorder=3)
                ax.plot(post[0][0], post[0][1], "o", ms=6.5, color=BLUE,
                        mec="white", mew=1.2, zorder=4)
            labels.append((y[-1], LABEL.get(ds, ds)))
            if task == "slp" and ds in floors:
                ax.plot([1, 8], [floors[ds]] * 2, color=GRAY, lw=0.8, ls=":",
                        alpha=0.75, zorder=1)

        lo, hi = ax.get_ylim()
        for y, text in declutter(labels, (hi - lo) * 0.055):
            ax.annotate(text, xy=(8, y), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=8, color=MUTED, annotation_clip=False)

        ax.set_title(title, fontsize=11, color=INK, pad=8)
        ax.set_ylabel(ylab, fontsize=9, color=MUTED)
        ax.set_xticks(range(1, 9))
        ax.set_xticklabels(XTICKS, fontsize=8, color=MUTED, rotation=35, ha="right")
        ax.set_xlim(0.7, 8.35)
        ax.grid(True, color=GRID, lw=0.7, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8)

    handles = [
        Line2D([], [], color=BLUE, lw=1.9, label="graph is in the training merge"),
        Line2D([], [], color=GRAY, lw=1.4, ls="--", label="held out (zero-shot transfer)"),
        Line2D([], [], color=BLUE, marker="o", ls="", ms=6.5, mec="white",
               label="rung at which the graph enters"),
        Line2D([], [], color=GRAY, lw=0.8, ls=":", label="best heuristic floor (static LP)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=8.5, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Downstream performance as sources are added to pre-training "
                 "(order A, matched 40k)", fontsize=12.5, color=INK, y=1.0)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))

    for ext in ("pdf", "png"):
        out = os.path.join(FIGS, f"nm_ladder_downstream_trajectory.{ext}")
        fig.savefig(out, bbox_inches="tight", dpi=200)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
