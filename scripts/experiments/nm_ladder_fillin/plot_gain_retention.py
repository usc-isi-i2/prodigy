#!/usr/bin/env python3
"""Adding a source graph: newcomers jump up, incumbents hold.

One-panel figure(s) for the NM interpolation ladder, making two points at a glance:
  (1) each newly-added graph's AUC jumps from its out-of-distribution (pre-add) level
      up to its in-distribution (post-add) level  -> we get BETTER on new graphs;
  (2) the mean AUC over graphs ALREADY in training stays high, sagging only slightly
      -> we DON'T get much worse on existing ones.

x = training-merge size (# source graphs). The blue line is the incumbent mean; each
coral arrow is the graph added at that rung, drawn from before (hollow, OOD — plotted
at the PREVIOUS rung, where it was last measured out-of-distribution) up-and-right to
after (filled, in-training). Three band variants show the dispersion of the in-training
graphs behind the mean line:
  band=flat   -> horizontal span of the mean line   (nm_ladder_gain_retention.pdf)
  band=std    -> mean +/- 1 std across in-training   (..._std.pdf)
  band=minmax -> min..max across in-training          (..._minmax.pdf)

Reads nm_ladder_full.csv if present (this folder), else uses the embedded matched-40k
values. Stdlib + matplotlib.
"""
from __future__ import annotations
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

COLS = ["ukr_rus_twitter", "covid19_twitter", "midterm", "covid_political",
        "election2020", "ukr_rus_suspended", "twibot20", "cp_hk_twitter"]
SHORT = ["ukr", "covid", "midterm", "cov_pol", "elec20", "ukr_susp", "twibot20", "cp_hk"]
# rung r (1-indexed) trains on the first r graphs, in COLS order (matched-40k, NM 3-shot/30-way).
FALLBACK = {
    1: [.9480, .9730, .8740, .8490, .8280, .7710, .9210, .7240],
    2: [.9450, .9800, .8850, .8430, .8280, .7750, .9250, .7260],
    3: [.9410, .9780, .9150, .8300, .8150, .7770, .9270, .7200],
    4: [.9344, .9753, .9093, .9113, .8297, .7768, .9234, .7235],
    5: [.9346, .9754, .9086, .9102, .9259, .7693, .9254, .7261],
    6: [.9325, .9744, .9073, .9106, .9241, .9340, .9242, .7239],
    7: [.9321, .9748, .9033, .9076, .9198, .9256, .9377, .7267],
    8: [.9340, .9750, .9080, .9060, .9200, .9310, .9370, .8670],
}
BLUE, CORAL, DCORAL = "#185FA5", "#D85A30", "#993C1D"


def load() -> dict[int, list[float]]:
    p = Path(__file__).resolve().parent / "nm_ladder_full.csv"
    if p.is_file():
        rows: dict[int, list[float]] = {}
        with p.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    rows[int(row["rung"])] = [float(row[c]) for c in COLS]
                except (KeyError, ValueError):
                    pass
        if len(rows) == 8:
            print(f"loaded {p}")
            return rows
    print("using embedded fallback")
    return dict(FALLBACK)


def _stats(xs: list[float]) -> tuple[float, float, float, float]:
    m = sum(xs) / len(xs)
    sd = (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5  # population std
    return m, sd, min(xs), max(xs)


def make_figure(M: dict[int, list[float]], band: str, out: Path) -> None:
    rungs = list(range(1, 9))
    stat = {r: _stats(M[r][:r]) for r in rungs}          # over the r in-training graphs
    inc = {r: stat[r][0] for r in rungs}
    # newcomer at rung r: graph added is col (r-1). before = rung r-1 model on it (still
    # OOD) plotted at x=r-1 (where it was last measured); after = rung r model, at x=r.
    newcomer = {r: (M[r - 1][r - 1], M[r][r - 1]) for r in range(2, 9)}

    fig, ax = plt.subplots(figsize=(8.4, 5.0))

    # (2) dispersion of the in-training graphs behind the mean line
    if band == "flat":
        ax.axhspan(min(inc.values()), max(inc.values()), color=BLUE, alpha=0.07, zorder=0)
        band_label = "in-training mean range"
    elif band == "std":
        lo = [inc[r] - stat[r][1] for r in rungs]
        hi = [inc[r] + stat[r][1] for r in rungs]
        ax.fill_between(rungs, lo, hi, color=BLUE, alpha=0.14, zorder=0, lw=0)
        band_label = "in-training graphs: mean ± 1 std"
    elif band == "minmax":
        lo = [stat[r][2] for r in rungs]
        hi = [stat[r][3] for r in rungs]
        ax.fill_between(rungs, lo, hi, color=BLUE, alpha=0.12, zorder=0, lw=0)
        band_label = "in-training graphs: min–max"
    else:
        raise ValueError(band)

    # (1) newcomer jumps: before at x=r-1 (OOD), after at x=r (in-training) -> right & up
    for r, (b, a) in newcomer.items():
        ax.annotate("", xy=(r, a), xytext=(r - 1, b),
                    arrowprops=dict(arrowstyle="-|>", color=CORAL, lw=2.2,
                                    shrinkA=0, shrinkB=0), zorder=4)
        ax.plot(r - 1, b, "o", mfc="white", mec=CORAL, mew=1.6, ms=7, zorder=5)
        ax.plot(r, a, "o", color=CORAL, ms=7, zorder=5)
        ax.annotate(f"{SHORT[r - 1]}\n+{a - b:.2f}", xy=(r, a), xytext=(r + 0.13, a),
                    fontsize=8.5, color=DCORAL, va="center", ha="left")

    # (2) incumbent mean line
    ax.plot(rungs, [inc[r] for r in rungs], "-o", color=BLUE, lw=2.4, ms=6, zorder=3)

    ax.set_xlabel("training-merge size (# source graphs)")
    ax.set_ylabel("NM AUC  (3-shot, 30-way)")
    ax.set_xticks(rungs)
    ax.set_xticklabels(["1\nukr"] + [f"{r}\n+{SHORT[r - 1]}" for r in range(2, 9)])
    ax.set_xlim(0.4, 9.15)
    ax.set_ylim(0.70, 1.0)
    ax.set_title("Adding a source graph: newcomers jump up, incumbents hold")
    ax.grid(axis="y", ls=":", alpha=0.45)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    handles = [
        Line2D([], [], color=BLUE, lw=2.4, marker="o", ms=6,
               label="in-training graphs (mean AUC)"),
        Patch(facecolor=BLUE, alpha=0.14, label=band_label),
        Line2D([], [], color=CORAL, lw=2.2, marker="o", ms=7,
               label="newly added graph: after (in-training)"),
        Line2D([], [], color=CORAL, lw=0, marker="o", mfc="white", mec=CORAL, mew=1.6,
               ms=7, label="…before it was added (out-of-dist)"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8.5, framealpha=0.92)

    fig.tight_layout()
    fig.savefig(f"{out}.pdf")
    fig.savefig(f"{out}.png", dpi=150)
    plt.close(fig)
    print(f"wrote {out}.pdf / .png")


def main() -> None:
    M = load()
    here = Path(__file__).resolve().parent
    make_figure(M, "flat", here / "nm_ladder_gain_retention")
    make_figure(M, "std", here / "nm_ladder_gain_retention_std")
    make_figure(M, "minmax", here / "nm_ladder_gain_retention_minmax")


if __name__ == "__main__":
    main()
