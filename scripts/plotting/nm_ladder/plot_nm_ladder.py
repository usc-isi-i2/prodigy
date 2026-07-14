#!/usr/bin/env python3
"""NM interpolation-ladder figure (complete 8-rung ladder).

Per-graph NM AUC (3-shot, 30-way, matched step 40k) as the SSL pre-training graph
grows ONE SOURCE AT A TIME: ukr -> +covid -> +midterm -> +covid_political ->
+election2020 -> +ukr_rus_suspended -> +twibot20 -> +cp_hk (= all 8). Each of the 8
single-source eval graphs is one trajectory; color encodes distribution state
relative to the current training merge -- blue = in-training (in-distribution),
gray dashed = held-out (out-of-distribution). A line turns blue at the rung its
graph enters the merge (the interpolation event); the vertical step there is the
OOD->ID delta. With one graph entering per rung, the entry markers trace a diagonal
cascade across the ladder.

Data: nm_ladder_full.csv (within_balanced rows) from the nm_ladder_fillin
experiment; searched next to this script and in the experiment folder, with an
embedded fallback (the CSV is gitignored). Writes nm_ladder_trajectory.pdf/.png here.
"""
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))

# palette (dataviz reference instance)
BLUE = "#2a78d6"   # in-distribution / in training
GRAY = "#8f8d87"   # out-of-distribution / held out
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"

# canonical column order (matches nm_ladder_full.csv).
CANON = [
    "ukr_rus_twitter", "covid19_twitter", "midterm", "covid_political",
    "election2020", "ukr_rus_suspended", "twibot20", "cp_hk_twitter",
]

# eval graph -> (display label, rung at which it ENTERS the training merge).
# One graph enters per rung now (the fill-in), so entries are 1..8.
GRAPHS = [
    ("ukr_rus_twitter",    "Ukr-Rus",       1),
    ("covid19_twitter",    "COVID-19",      2),
    ("midterm",            "Midterm",       3),
    ("covid_political",    "COVID-pol.",    4),
    ("election2020",       "Election '20",  5),
    ("ukr_rus_suspended",  "Ukr-Rus susp.", 6),
    ("twibot20",           "TwiBot-20",     7),
    ("cp_hk_twitter",      "CP-HK",         8),
]
RUNGS = [1, 2, 3, 4, 5, 6, 7, 8]
XTICKS = ["ukr", "+covid", "+midterm", "+cov-pol", "+elec '20",
          "+ukr-susp", "+twibot", "+cp-hk\n(all 8)"]

# Verified 8-rung within-balanced AUCs (2026-07-13; == nm_ladder_full.csv). Used
# only if no CSV is found. CSV wins when present.
EMBED = {
    1: [0.9480, 0.9730, 0.8740, 0.8490, 0.8280, 0.7710, 0.9210, 0.7240],
    2: [0.9450, 0.9800, 0.8850, 0.8430, 0.8280, 0.7750, 0.9250, 0.7260],
    3: [0.9410, 0.9780, 0.9150, 0.8300, 0.8150, 0.7770, 0.9270, 0.7200],
    4: [0.9344, 0.9753, 0.9093, 0.9113, 0.8297, 0.7768, 0.9234, 0.7235],
    5: [0.9346, 0.9754, 0.9086, 0.9102, 0.9259, 0.7693, 0.9254, 0.7261],
    6: [0.9325, 0.9744, 0.9073, 0.9106, 0.9241, 0.9340, 0.9242, 0.7239],
    7: [0.9321, 0.9748, 0.9033, 0.9076, 0.9198, 0.9256, 0.9377, 0.7267],
    8: [0.9340, 0.9750, 0.9080, 0.9060, 0.9200, 0.9310, 0.9370, 0.8670],
}
CSV_CANDIDATES = [
    os.path.join(HERE, "nm_ladder_full.csv"),
    os.path.join(HERE, "..", "..", "experiments", "nm_ladder_fillin", "nm_ladder_full.csv"),
]


def load():
    table, src = None, "embedded fallback"
    for path in CSV_CANDIDATES:
        if os.path.exists(path):
            table = {}
            with open(path, newline="") as fh:
                for r in csv.DictReader(fh):
                    if r.get("sampling", "within_balanced") != "within_balanced":
                        continue
                    try:
                        table[int(r["rung"])] = {k: float(r[k]) for k in CANON}
                    except (KeyError, ValueError):
                        continue
            if all(rg in table for rg in RUNGS):
                src = os.path.relpath(path, HERE)
                break
            table = None  # incomplete -> keep looking / fall back
    if table is None:
        table = {rg: dict(zip(CANON, vals)) for rg, vals in EMBED.items()}
    print(f"[data] {src}")
    return {key: [table[rg][key] for rg in RUNGS] for key, _l, _e in GRAPHS}


def declutter(items, gap, lo, hi):
    """items: list of dicts with 'y_true'; set 'y_lbl' spread >= gap, clamped [lo,hi]."""
    items = sorted(items, key=lambda d: d["y_true"])
    for it in items:
        it["y_lbl"] = it["y_true"]
    for i in range(1, len(items)):  # push up
        if items[i]["y_lbl"] - items[i - 1]["y_lbl"] < gap:
            items[i]["y_lbl"] = items[i - 1]["y_lbl"] + gap
    if items[-1]["y_lbl"] > hi:      # overflowed top -> pull the stack down
        items[-1]["y_lbl"] = hi
        for i in range(len(items) - 2, -1, -1):
            if items[i + 1]["y_lbl"] - items[i]["y_lbl"] < gap:
                items[i]["y_lbl"] = items[i + 1]["y_lbl"] - gap
    for it in items:
        it["y_lbl"] = min(max(it["y_lbl"], lo), hi)
    return items


def main():
    series = load()
    x = list(range(len(RUNGS)))  # 0..7

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "svg.fonttype": "none",
    })
    fig, ax = plt.subplots(figsize=(10.6, 5.9), dpi=200)

    # per-graph trajectories
    label_items = []  # right-edge direct labels (identity + entry delta)
    for key, label, entry in GRAPHS:
        y = series[key]
        e = entry - 1  # entry index
        for i in range(len(x) - 1):
            in_dist_right = (i + 1) >= e
            ax.plot(
                [x[i], x[i + 1]], [y[i], y[i + 1]],
                color=BLUE if in_dist_right else GRAY,
                lw=1.9 if in_dist_right else 1.5,
                ls="-" if in_dist_right else (0, (4, 2)),
                zorder=3, solid_capstyle="round",
            )
        for i in range(len(x)):
            in_dist = i >= e
            if i == e and entry > 1:  # entry marker (just added): emphasized ring
                ax.scatter([x[i]], [y[i]], s=95, facecolor=BLUE, edgecolor="white",
                           linewidth=1.6, zorder=6)
            else:
                ax.scatter([x[i]], [y[i]], s=24,
                           facecolor=BLUE if in_dist else "white",
                           edgecolor=BLUE if in_dist else GRAY,
                           linewidth=1.4, zorder=5)
        d = y[e] - y[e - 1] if entry > 1 else 0.0
        text = f"{label}   +{d:.2f}" if d >= 0.03 else label
        label_items.append({"y_true": y[-1], "text": text, "big": d >= 0.03})

    # aggregate: mean over the fixed 8-graph set (no composition confound)
    mean = [sum(series[k][i] for k, _, _ in GRAPHS) / len(GRAPHS) for i in range(len(x))]
    ax.plot(x, mean, color=INK, lw=2.8, zorder=8, marker="s", ms=6,
            markerfacecolor=INK, markeredgecolor="white", markeredgewidth=1.2)
    ax.annotate("mean (all 8)", xy=(x[0] - 0.06, mean[0]), ha="right", va="center",
                fontsize=9, color=INK, fontweight="bold")

    # decluttered right-edge direct labels (identity via label, not color) + leaders
    xr = x[-1]
    for it in declutter(label_items, gap=0.019, lo=0.712, hi=0.992):
        ax.plot([xr + 0.06, xr + 0.28], [it["y_true"], it["y_lbl"]],
                color=MUTED, lw=0.6, zorder=2)
        ax.annotate(it["text"], xy=(xr + 0.34, it["y_lbl"]), ha="left", va="center",
                    fontsize=9.3, color=BLUE, fontweight="bold")

    # axes / chrome
    ax.set_xlim(-0.62, 8.75)
    ax.set_ylim(0.70, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(XTICKS, fontsize=9.2)
    ax.set_xlabel("SSL pre-training graph  (one source added per rung, merge grows to the right)",
                  fontsize=10.5, color=INK)
    ax.set_ylabel("NM AUC  (3-shot, 30-way)", fontsize=10.5, color=INK)
    ax.set_yticks([0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00])
    ax.tick_params(colors=MUTED, labelsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#c3c2b7")
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)

    ax.set_title("Interpolation ladder: each graph jumps to in-distribution when it "
                 "enters SSL pre-training",
                 fontsize=12.5, color=INK, fontweight="bold", loc="left", pad=26)
    ax.text(0.0, 1.02, "NM  3-shot / 30-way  ·  matched step 40k  ·  within-balanced "
            "sampling  ·  +d = out-of-dist. gain at entry rung",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=9, color=MUTED)

    legend_handles = [
        Line2D([0], [0], color=BLUE, lw=1.9, marker="o", markerfacecolor=BLUE,
               markeredgecolor="white", ms=7, label="in training (in-dist.)"),
        Line2D([0], [0], color=GRAY, lw=1.5, ls=(0, (4, 2)), marker="o",
               markerfacecolor="white", markeredgecolor=GRAY, ms=7,
               label="held out (out-of-dist.)"),
        Line2D([0], [0], color=INK, lw=2.8, marker="s", markerfacecolor=INK,
               markeredgecolor="white", ms=7, label="mean (all 8 graphs)"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=False,
              fontsize=9, handlelength=2.4, borderaxespad=0.6)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = os.path.join(HERE, f"nm_ladder_trajectory.{ext}")
        fig.savefig(out, bbox_inches="tight")
        print("wrote", out)

    # console summary of entry deltas
    print("\nOOD->ID entry delta (AUC at entry rung minus previous rung):")
    for key, label, entry in GRAPHS:
        if entry > 1:
            y = series[key]
            print(f"  {label:14s} enters rung {entry}: {y[entry-2]:.3f} -> "
                  f"{y[entry-1]:.3f}  (+{y[entry-1]-y[entry-2]:.3f})")
    print(f"\nmean(all 8) by rung: " + ", ".join(f"{m:.3f}" for m in mean))


if __name__ == "__main__":
    main()
