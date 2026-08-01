#!/usr/bin/env python3
"""Downstream ladder — one figure per slice, instead of three summary panels.

The three headline figures each collapse something: the trajectory shows order A only,
the event study pools all orders into one cloud of grey lines, and the means figure
averages over eval graphs. This emits the un-collapsed views, one file each.

Default granularity is **(order × task × target)** -- 3 orders × (3 regression targets +
classification + static LP) = 15 figures, each carrying one line per eval graph. Pass
``--per-dataset`` to split those apart as well, one line per figure (63 figures).

Visual language is the trajectory figure's, so the slices read against it directly:
grey dashed while the graph is held out, blue once its source is in the training merge,
a marker at the entry rung, and a dotted line at the relevant floor -- best heuristic
floor for static LP, raw-feature ridge probe for regression. Classification has no floor
line because no such baseline was computed for it; that absence is real, not an omission
here.

The x axis is built per order from the ``added`` column, so order B and C figures are
labelled with the sources THEY add at each rung rather than order A's.

Each figure annotates its own entry Δ (value at the entry rung minus the rung before).
Series whose graph enters at rung 1 have no "before" and are drawn without a Δ -- they
are single-source specialists, the dilution confound the README warns about.

    python3 plot_downstream_breakdown.py                  # 15 figures
    python3 plot_downstream_breakdown.py --per-dataset    # 63 figures
    python3 plot_downstream_breakdown.py --orders A       # just one order

Writes into figures/breakdown/.
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
FIGS = os.path.join(HERE, "figures", "breakdown")

BLUE = "#2a78d6"
GRAY = "#8f8d87"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"

LABEL = {
    "ukr_rus_twitter": "Ukr-Rus", "covid19_twitter": "COVID-19", "midterm": "Midterm",
    "covid_political": "COVID-pol.", "election2020": "Election '20",
    "ukr_rus_suspended": "Ukr-Rus susp.", "twibot20": "TwiBot-20",
    "cp_hk_twitter": "CP-HK",
}
TASK_TITLE = {"slp": "Static link prediction", "pl": "Node classification",
              "reg": "Node regression"}
TASK_YLAB = {"slp": "AUC (degree-matched)", "pl": "ROC-AUC (10-shot)",
             "reg": "Spearman (10-shot ridge probe)"}
TARGET_LABEL = {"followers_count": "followers", "statuses_count": "statuses",
                "account_age_days": "account age"}
# Short rung names, matching plot_downstream_trajectory.py's hardcoded XTICKS. Not
# cosmetic: spelled out, "+ukr_rus_suspended" rotated 35 degrees descends far enough to
# collide with the per-panel entry-delta line below the axes.
SHORT = {"ukr_rus": "ukr", "covid": "covid", "midterm": "midterm",
         "covid_political": "cov-pol", "election2020": "elec '20",
         "ukr_rus_suspended": "ukr-susp", "twibot20": "twibot", "cp_hk": "cp-hk"}


def declutter(entries, min_gap):
    """Nudge (y, text) label positions apart so end-of-line labels stay readable."""
    out = []
    for y, text in sorted(entries):
        if out and y - out[-1][0] < min_gap:
            y = out[-1][0] + min_gap
        out.append((y, text))
    return out


def xticks_for(df, order):
    """Rung labels for one order, read off the data rather than hardcoded.

    ``added`` is the source that joined at that rung, so rung 1 is the seed graph and
    the rest read as "+name". Order A gives the familiar ukr / +covid / +midterm / ...
    """
    seq = (df[df["order"] == order].drop_duplicates("rung").sort_values("rung"))
    out = []
    for r in seq.itertuples():
        key = str(r.added)
        name = SHORT.get(key, key.replace("_twitter", "").replace("_", "-"))
        out.append(name if r.rung == 1 else f"+{name}")
    return out


def load_floors():
    """(task-specific) floor lookups: slp[dataset] and reg[(dataset, target)]."""
    slp, reg = {}, {}
    p = os.path.join(DATA, "nm_ladder_downstream_slp_floors.csv")
    if os.path.isfile(p):
        slp = pd.read_csv(p).set_index("dataset").max(axis=1, numeric_only=True).to_dict()
    p = os.path.join(DATA, "nm_ladder_downstream_reg_floors.csv")
    if os.path.isfile(p):
        reg = {(r.dataset, r.target): r.features_only_spearman
               for r in pd.read_csv(p).itertuples()}
    return slp, reg


def draw(ax, cells, datasets, floor_of):
    """One axes: a trajectory per dataset, plus its floor. Returns the entry deltas."""
    labels, deltas = [], {}
    for ds in datasets:
        g = cells[cells.dataset == ds].sort_values("rung")
        if g.empty:
            continue
        entry = int(g["entry_rung"].iloc[0])
        x, y = g["rung"].tolist(), g["value"].tolist()

        pre = [(a, b) for a, b in zip(x, y) if a < entry]
        post = [(a, b) for a, b in zip(x, y) if a >= entry]
        if pre:
            bridge = pre + post[:1]      # bridge so the entry step is visible as a step
            ax.plot([a for a, _ in bridge], [b for _, b in bridge],
                    color=GRAY, lw=1.5, ls="--", zorder=2)
        if post:
            ax.plot([a for a, _ in post], [b for _, b in post], color=BLUE, lw=2.0,
                    zorder=3)
            ax.plot(post[0][0], post[0][1], "o", ms=7.0, color=BLUE, mec="white",
                    mew=1.2, zorder=4)
        if pre and post:
            deltas[ds] = post[0][1] - pre[-1][1]
        labels.append((y[-1], LABEL.get(ds, ds)))

        f = floor_of(ds)
        if f is not None:
            ax.plot([min(x), max(x)], [f] * 2, color=GRAY, lw=0.9, ls=":", alpha=0.8,
                    zorder=1)
    return labels, deltas


def style(ax, xt, ylab):
    ax.set_ylabel(ylab, fontsize=9.5, color=MUTED)
    ax.set_xticks(range(1, len(xt) + 1))
    ax.set_xticklabels(xt, fontsize=8.5, color=MUTED, rotation=35, ha="right")
    ax.set_xlim(0.7, len(xt) + 0.45)
    ax.grid(True, color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8.5)


def combine(df, orders, tasks, slp_floor, reg_floor, handles, out_dir):
    """One figure per (task, target); the orders are panels on a SHARED y axis.

    Shared y is the whole point: the orders are three routes to the same rung-8 model,
    so the comparison worth making is across panels, and independent y limits would
    make a +0.01 entry in one order look like a +0.09 entry in another.
    """
    written = 0
    for task in tasks:
        sub_t = df[df["task"] == task]
        if sub_t.empty:
            continue
        for target in sorted(sub_t["target"].fillna("").unique()):
            cells_t = sub_t[sub_t["target"].fillna("") == target]

            def floor_of(ds, _t=target, _task=task):
                if _task == "slp":
                    return slp_floor.get(ds)
                if _task == "reg":
                    return reg_floor.get((ds, _t))
                return None

            fig, axes = plt.subplots(1, len(orders), figsize=(5.9 * len(orders), 5.5),
                                     sharey=True)
            axes = [axes] if len(orders) == 1 else list(axes)
            drawn = False
            for ax, order in zip(axes, orders):
                cells = cells_t[cells_t["order"] == order]
                if cells.empty:
                    ax.set_visible(False)
                    continue
                datasets = sorted(cells["dataset"].unique())
                labels, deltas = draw(ax, cells, datasets, floor_of)
                if not labels:
                    ax.set_visible(False)
                    continue
                drawn = True
                xt = xticks_for(df, order)
                style(ax, xt, TASK_YLAB[task] if ax is axes[0] else "")
                pos = sum(1 for v in deltas.values() if v > 0)
                head = f"order {order}"
                if deltas:
                    head += (f"   —   entry Δ > 0 in {pos}/{len(deltas)},  "
                             f"mean {sum(deltas.values()) / len(deltas):+.3f}")
                ax.set_title(head, fontsize=10.5, color=INK, pad=8)
                if deltas:
                    # Clear of the rotated tick labels: order C's "+ukr-rus-suspended"
                    # is long enough to reach well below the axis.
                    ax.annotate("  ".join(f"{LABEL.get(d, d)} {v:+.3f}"
                                          for d, v in sorted(deltas.items())),
                                xy=(0.5, -0.30), xycoords="axes fraction", ha="center",
                                fontsize=7.5, color=MUTED)
            if not drawn:
                plt.close(fig)
                continue

            # End labels need the final shared y limits, so they are placed after every
            # panel is drawn -- doing it inside the loop would use panel A's limits.
            for ax, order in zip(axes, orders):
                if not ax.get_visible():
                    continue
                cells = cells_t[cells_t["order"] == order]
                lo, hi = ax.get_ylim()
                ends = [(float(g.sort_values("rung")["value"].iloc[-1]),
                         LABEL.get(ds, ds))
                        for ds, g in cells.groupby("dataset")]
                n_rungs = len(xticks_for(df, order))
                for y, text in declutter(ends, (hi - lo) * 0.062):
                    ax.annotate(text, xy=(n_rungs, y), xytext=(5, 0),
                                textcoords="offset points", va="center", fontsize=7.5,
                                color=MUTED, annotation_clip=False)

            bits = [TASK_TITLE[task]]
            if target:
                bits.append(TARGET_LABEL.get(target, target))
            fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
                       fontsize=8.5, bbox_to_anchor=(0.5, -0.055))
            fig.suptitle(" · ".join(bits) + "   (the same 8-rung ladder under 3 graph "
                                            "orders, shared y axis)",
                         fontsize=13.0, color=INK, y=1.0)
            fig.tight_layout(rect=(0, 0.04, 1, 0.96))

            stem = f"nm_ladder_downstream_orders_{task}"
            if target:
                stem += f"_{target}"
            for ext in ("pdf", "png"):
                fig.savefig(os.path.join(out_dir, f"{stem}.{ext}"), bbox_inches="tight",
                            dpi=200)
            plt.close(fig)
            written += 1
            print(f"wrote {stem}")
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-dataset", action="store_true",
                    help="one eval graph per figure (63 files) instead of one line each")
    ap.add_argument("--combine-orders", action="store_true",
                    help="one figure per (task, target) with the 3 orders as panels "
                         "on a shared y axis (5 files)")
    ap.add_argument("--orders", default="A,B,C")
    ap.add_argument("--tasks", default="slp,pl,reg")
    ap.add_argument("--out-dir", default=FIGS)
    args = ap.parse_args()

    df = pd.read_csv(os.path.join(DATA, "nm_ladder_downstream_long.csv"))
    df = df[df["primary"] == 1]
    slp_floor, reg_floor = load_floors()
    os.makedirs(args.out_dir, exist_ok=True)

    handles = [
        Line2D([], [], color=BLUE, lw=2.0, label="graph is in the training merge"),
        Line2D([], [], color=GRAY, lw=1.5, ls="--", label="held out (zero-shot transfer)"),
        Line2D([], [], color=BLUE, marker="o", ls="", ms=7.0, mec="white",
               label="entry rung"),
        Line2D([], [], color=GRAY, lw=0.9, ls=":", label="floor (heuristic / raw-feature)"),
    ]

    orders = [o.strip() for o in args.orders.split(",") if o.strip()]
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    if args.combine_orders:
        written = combine(df, orders, tasks, slp_floor, reg_floor, handles, args.out_dir)
        print(f"\n{written} figures ({written * 2} files) in {args.out_dir}")
        return

    written = 0
    for order in orders:
        xt = xticks_for(df, order)
        for task in [t.strip() for t in args.tasks.split(",") if t.strip()]:
            sub = df[(df["order"] == order) & (df["task"] == task)]
            if sub.empty:
                continue
            for target in sorted(sub["target"].fillna("").unique()):
                cells = sub[sub["target"].fillna("") == target]

                def floor_of(ds, _t=target, _task=task):
                    if _task == "slp":
                        return slp_floor.get(ds)
                    if _task == "reg":
                        return reg_floor.get((ds, _t))
                    return None

                groups = ([[d] for d in sorted(cells["dataset"].unique())]
                          if args.per_dataset else [sorted(cells["dataset"].unique())])
                for datasets in groups:
                    c = cells[cells["dataset"].isin(datasets)]
                    fig, ax = plt.subplots(figsize=(6.4, 4.4))
                    labels, deltas = draw(ax, c, datasets, floor_of)
                    if not labels:
                        plt.close(fig)
                        continue

                    lo, hi = ax.get_ylim()
                    for y, text in declutter(labels, (hi - lo) * 0.06):
                        ax.annotate(text, xy=(len(xt), y), xytext=(6, 0),
                                    textcoords="offset points", va="center",
                                    fontsize=8.5, color=MUTED, annotation_clip=False)

                    bits = [TASK_TITLE[task], f"order {order}"]
                    if target:
                        bits.insert(1, TARGET_LABEL.get(target, target))
                    if args.per_dataset:
                        bits.insert(1, LABEL.get(datasets[0], datasets[0]))
                    ax.set_title(" · ".join(bits), fontsize=11.5, color=INK, pad=10)

                    if deltas:
                        txt = "  ".join(f"{LABEL.get(d, d)} {v:+.3f}"
                                        for d, v in sorted(deltas.items()))
                        ax.annotate(f"entry Δ:  {txt}", xy=(0.5, -0.30),
                                    xycoords="axes fraction", ha="center", fontsize=8,
                                    color=MUTED)
                    style(ax, xt, TASK_YLAB[task])
                    ax.legend(handles=handles, loc="best", frameon=False, fontsize=7.5)

                    stem = f"nm_ladder_downstream_{order}_{task}"
                    if target:
                        stem += f"_{target}"
                    if args.per_dataset:
                        stem += f"_{datasets[0]}"
                    for ext in ("pdf", "png"):
                        fig.savefig(os.path.join(args.out_dir, f"{stem}.{ext}"),
                                    bbox_inches="tight", dpi=200)
                    plt.close(fig)
                    written += 1
                    print(f"wrote {stem}")

    print(f"\n{written} figures ({written * 2} files) in {args.out_dir}")


if __name__ == "__main__":
    main()
