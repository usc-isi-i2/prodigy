#!/usr/bin/env python3
"""Downstream ladder — the whole result as one figure: task/target × graph order.

Five rows (static LP, classification, and one per regression target) by three columns
(the graph orders). Every panel is the same 8-rung ladder; what changes down the rows is
what is being measured, and across the columns is the order sources were added in.

ENCODING. Colour is the eval graph, held constant everywhere it appears -- so a line can
be followed down a column and across a row. Merge state moves to line STYLE: dashed while
the graph is held out (zero-shot transfer), solid once its source is in the training
merge, with a marker at the entry rung. The dotted line in a graph's own colour is its
floor: best heuristic scorer for static LP, raw-feature ridge probe for regression.
Classification has no floor line because no such baseline was ever computed for it.

y is shared ACROSS a row and never down a column: the three orders are three routes to
the same rung-8 model, so panels in a row must be visually comparable, while AUC and
Spearman obviously must not share a scale.

PALETTE. Five hues from the reference categorical palette -- blue / yellow / magenta /
green / violet -- chosen by validating every 5-subset of the 8 documented hues under the
all-pairs pairlist (the right list here: within a panel any two lines can cross, so all
pairs are seen together). This set is the only passing one whose worst-case CVD
separation, ΔE 13.0, clears the 6-8 warn band outright; worst normal-vision pair is 16.3
against a floor of 15. Yellow and magenta fall below 3:1 contrast on a light surface, so
the relief rule applies and every line is direct-labelled.

EIGHT GRAPHS, FIVE HUES -- deliberate, and the one compromise in this figure. No
assignment of 8 distinct hues from the documented palette can satisfy both the 5-graph
static-LP panel and the 4-graph classification panel under all-pairs: orange clears the
floor only alongside the four unconflicted hues, which forces red and magenta together in
the other panel, and that pair fails. Since the classification graphs (covid-political,
election2020, ukr-rus-suspended) share no panel with the static-LP/regression graphs,
three hues do double duty across those two disjoint families. TwiBot-20 is the one graph
appearing in every row, so it keeps a hue of its own throughout.

Writes figures/nm_ladder_downstream_grid.pdf/.png.
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

INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"

# Reference categorical palette, slots validated as a set (see module docstring).
BLUE, YELLOW, MAGENTA, GREEN, VIOLET = "#2a78d6", "#eda100", "#e87ba4", "#008300", "#4a3aa7"
COLOR = {
    # static-LP / regression family
    "ukr_rus_twitter": BLUE, "covid19_twitter": YELLOW, "midterm": MAGENTA,
    "twibot20": GREEN, "cp_hk_twitter": VIOLET,
    # classification-only family -- never shares a panel with the three above
    "covid_political": BLUE, "election2020": YELLOW, "ukr_rus_suspended": MAGENTA,
}
LABEL = {
    "ukr_rus_twitter": "Ukr-Rus", "covid19_twitter": "COVID-19", "midterm": "Midterm",
    "covid_political": "COVID-pol.", "election2020": "Election '20",
    "ukr_rus_suspended": "Ukr-Rus susp.", "twibot20": "TwiBot-20",
    "cp_hk_twitter": "CP-HK",
}
SHORT = {"ukr_rus": "ukr", "covid": "covid", "midterm": "midterm",
         "covid_political": "cov-pol", "election2020": "elec '20",
         "ukr_rus_suspended": "ukr-susp", "twibot20": "twibot", "cp_hk": "cp-hk"}

ROWS = [
    ("slp", "", "Static link prediction", "AUC (degree-matched)"),
    ("pl", "", "Node classification", "ROC-AUC (10-shot)"),
    ("reg", "followers_count", "Regression · followers", "Spearman (ridge probe)"),
    ("reg", "statuses_count", "Regression · statuses", "Spearman (ridge probe)"),
    ("reg", "account_age_days", "Regression · account age", "Spearman (ridge probe)"),
]
ORDERS = ["A", "B", "C"]


def declutter(entries, min_gap):
    """Nudge (y, text) label positions apart so end-of-line labels stay readable."""
    out = []
    for y, text in sorted(entries):
        if out and y - out[-1][0] < min_gap:
            y = out[-1][0] + min_gap
        out.append((y, text))
    return out


def xticks_for(df, order):
    seq = df[df["order"] == order].drop_duplicates("rung").sort_values("rung")
    return [(SHORT.get(str(r.added), str(r.added)) if r.rung == 1
             else f"+{SHORT.get(str(r.added), str(r.added))}") for r in seq.itertuples()]


def load_floors():
    slp, reg = {}, {}
    p = os.path.join(DATA, "nm_ladder_downstream_slp_floors.csv")
    if os.path.isfile(p):
        slp = pd.read_csv(p).set_index("dataset").max(axis=1, numeric_only=True).to_dict()
    p = os.path.join(DATA, "nm_ladder_downstream_reg_floors.csv")
    if os.path.isfile(p):
        reg = {(r.dataset, r.target): r.features_only_spearman
               for r in pd.read_csv(p).itertuples()}
    return slp, reg


def main():
    df = pd.read_csv(os.path.join(DATA, "nm_ladder_downstream_long.csv"))
    df = df[df["primary"] == 1]
    df["target"] = df["target"].fillna("")
    slp_floor, reg_floor = load_floors()
    os.makedirs(FIGS, exist_ok=True)

    fig, axes = plt.subplots(len(ROWS), len(ORDERS), figsize=(15.5, 20.5),
                             sharey="row")

    for i, (task, target, row_title, ylab) in enumerate(ROWS):
        ends = {}
        for j, order in enumerate(ORDERS):
            ax = axes[i][j]
            cells = df[(df.task == task) & (df.target == target) & (df.order == order)]
            deltas = {}
            for ds, g in cells.groupby("dataset"):
                g = g.sort_values("rung")
                c = COLOR.get(ds, MUTED)
                entry = int(g["entry_rung"].iloc[0])
                x, y = g["rung"].tolist(), g["value"].tolist()
                pre = [(a, b) for a, b in zip(x, y) if a < entry]
                post = [(a, b) for a, b in zip(x, y) if a >= entry]
                if pre:
                    bridge = pre + post[:1]     # so the entry step reads as a step
                    ax.plot([a for a, _ in bridge], [b for _, b in bridge], color=c,
                            lw=1.6, ls=(0, (4, 2)), alpha=0.85, zorder=2)
                if post:
                    ax.plot([a for a, _ in post], [b for _, b in post], color=c, lw=2.0,
                            zorder=3)
                    ax.plot(post[0][0], post[0][1], "o", ms=6.5, color=c, mec="white",
                            mew=1.3, zorder=5)
                if pre and post:
                    deltas[ds] = post[0][1] - pre[-1][1]
                ends.setdefault(j, []).append((y[-1], LABEL.get(ds, ds)))

                f = (slp_floor.get(ds) if task == "slp"
                     else reg_floor.get((ds, target)) if task == "reg" else None)
                if f is not None:
                    ax.plot([min(x), max(x)], [f] * 2, color=c, lw=0.9, ls=":",
                            alpha=0.55, zorder=1)

            xt = xticks_for(df, order)
            ax.set_xticks(range(1, len(xt) + 1))
            ax.set_xticklabels(xt, fontsize=7.5, color=MUTED, rotation=40, ha="right")
            ax.set_xlim(0.7, len(xt) + 0.5)
            ax.grid(True, color=GRID, lw=0.7, zorder=0)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_color(GRID)
            ax.tick_params(colors=MUTED, labelsize=8)
            if j == 0:
                ax.set_ylabel(ylab, fontsize=9, color=MUTED)
            head = f"order {order}"
            if deltas:
                pos = sum(1 for v in deltas.values() if v > 0)
                head += (f"   ·   Δ>0 in {pos}/{len(deltas)},  "
                         f"mean {sum(deltas.values()) / len(deltas):+.3f}")
            ax.set_title(head, fontsize=9.5, color=INK, pad=6)

        # End labels need the row's final shared limits, so they go on after the row.
        for j, order in enumerate(ORDERS):
            ax = axes[i][j]
            lo, hi = ax.get_ylim()
            n = len(xticks_for(df, order))
            for y, text in declutter(ends.get(j, []), (hi - lo) * 0.075):
                ax.annotate(text, xy=(n, y), xytext=(4, 0), textcoords="offset points",
                            va="center", fontsize=7, color=MUTED, annotation_clip=False)
        axes[i][0].annotate(row_title, xy=(-0.30, 0.5), xycoords="axes fraction",
                            rotation=90, ha="center", va="center", fontsize=11.5,
                            color=INK)

    style_handles = [
        Line2D([], [], color=MUTED, lw=2.0, label="in the training merge"),
        Line2D([], [], color=MUTED, lw=1.6, ls=(0, (4, 2)), label="held out (zero-shot)"),
        Line2D([], [], color=MUTED, marker="o", ls="", ms=6.5, mec="white",
               label="entry rung"),
        Line2D([], [], color=MUTED, lw=0.9, ls=":", label="floor (heuristic / raw-feature)"),
    ]
    graph_handles = [Line2D([], [], color=COLOR[d], lw=2.6, label=LABEL[d])
                     for d in ("ukr_rus_twitter", "covid19_twitter", "midterm",
                               "twibot20", "cp_hk_twitter")]
    cls_handles = [Line2D([], [], color=COLOR[d], lw=2.6, label=LABEL[d])
                   for d in ("covid_political", "election2020", "ukr_rus_suspended")]

    leg1 = fig.legend(handles=style_handles, loc="lower center", ncol=4, frameon=False,
                      fontsize=9, bbox_to_anchor=(0.5, 0.012))
    leg2 = fig.legend(handles=graph_handles, loc="lower center", ncol=5, frameon=False,
                      fontsize=9, bbox_to_anchor=(0.5, -0.004),
                      title="eval graph — static LP & regression rows")
    leg3 = fig.legend(handles=cls_handles, loc="lower center", ncol=3, frameon=False,
                      fontsize=9, bbox_to_anchor=(0.5, -0.026),
                      title="eval graph — classification row")
    for lg in (leg2, leg3):
        lg.get_title().set_fontsize(8)
        lg.get_title().set_color(MUTED)
    for lg in (leg1, leg2, leg3):
        fig.add_artist(lg)

    fig.suptitle("The NM interpolation ladder, scored downstream: "
                 "task × graph order", fontsize=15, color=INK, y=1.006)
    fig.text(0.5, 0.9855, "colour = eval graph · dashed = held out, solid = in the "
                         "training merge · y shared across each row",
             ha="center", fontsize=9.5, color=MUTED)
    fig.tight_layout(rect=(0.015, 0.035, 1, 0.978))

    for ext in ("pdf", "png"):
        out = os.path.join(FIGS, f"nm_ladder_downstream_grid.{ext}")
        fig.savefig(out, bbox_inches="tight", dpi=200)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
