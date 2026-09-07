#!/usr/bin/env python3
"""NM interpolation-ladder figure (complete 8-rung ladder).

Per-graph held-out NM AUC (3-shot, 30-way, matched step 40k) as the SSL
pre-training graph grows ONE SOURCE AT A TIME: ukr -> +covid -> +midterm ->
+covid_political -> +election2020 -> +ukr_rus_suspended -> +twibot20 -> +cp_hk
(= all 8). Each target has its own color and its trajectory stops immediately
before that target enters the pre-training mixture. The UKR target is omitted
because it is already present at rung 1 and therefore has no held-out observation.

Data: nm_ladder_full.csv (within_balanced rows) from the same folder
experiment; searched next to this script and in the experiment folder, with an
embedded fallback (the CSV is gitignored). Writes nm_ladder_trajectory.pdf/.png here.
"""
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

# colorblind-friendly target palette
COLORS = {
    "covid19_twitter": "#2a78d6",
    "midterm": "#e68613",
    "covid_political": "#2a9d65",
    "election2020": "#d34e4e",
    "ukr_rus_suspended": "#8b63c7",
    "twibot20": "#c79a17",
    "cp_hk_twitter": "#66717e",
}
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
    os.path.join(HERE, "data", "nm_ladder_full.csv"),
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


def main():
    series = load()
    x = list(range(len(RUNGS)))  # 0..7

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "svg.fonttype": "none",
    })
    fig, ax = plt.subplots(figsize=(10.6, 5.9), dpi=200)

    # Plot only the observations for which the evaluation target is still held out.
    # UKR enters at rung 1, so it has no held-out point and is intentionally absent.
    for key, label, entry in GRAPHS:
        if entry == 1:
            continue
        y = series[key]
        heldout_x = x[: entry - 1]
        heldout_y = y[: entry - 1]
        color = COLORS[key]
        ax.plot(
            heldout_x,
            heldout_y,
            color=color,
            lw=2.2,
            marker="o",
            ms=5.5,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=1.5,
            zorder=3,
            solid_capstyle="round",
        )
        ax.annotate(
            label,
            xy=(heldout_x[-1], heldout_y[-1]),
            xytext=(7, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=8.8,
            color=color,
            fontweight="bold",
            zorder=5,
        )

    # axes / chrome
    ax.set_xlim(-0.35, 6.75)
    ax.set_ylim(0.70, 1.0)
    ax.set_xticks(x[:-1])
    ax.set_xticklabels(XTICKS[:-1], fontsize=9.2)
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

    ax.set_title("Held-out NM transfer does not improve as unrelated sources are added",
                 fontsize=12.5, color=INK, fontweight="bold", loc="left", pad=26)
    ax.text(0.0, 1.02, "NM  3-shot / 30-way  ·  matched step 40k  ·  within-balanced "
            "sampling  ·  each line ends before its target enters pre-training",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=9, color=MUTED)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = os.path.join(HERE, "figures", f"nm_ladder_trajectory.{ext}")
        fig.savefig(out, bbox_inches="tight")
        print("wrote", out)

    # Console summary of the final observed held-out value for each target.
    print("\nFinal held-out NM AUC before target entry:")
    for key, label, entry in GRAPHS:
        if entry > 1:
            y = series[key]
            print(f"  {label:14s} before rung {entry}: {y[entry-2]:.3f}")


if __name__ == "__main__":
    main()
