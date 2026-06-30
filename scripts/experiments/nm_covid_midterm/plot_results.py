#!/usr/bin/env python3
"""Plot the cov/mid results from matrix.csv (written by build_matrix.py).

Produces grouped bar charts (per test domain: single in-domain vs the three merged
regimes) for a given metric, at both @match (matched compute) and @full (per-domain
exposure). Saves PNGs next to this script.

    python build_matrix.py ... --out-csv matrix.csv      # first produce the CSV
    python plot_results.py --csv matrix.csv --metric accuracy

Needs matplotlib (available in the prodigy env).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# regime -> (display label, colour). The single in-domain bar is filled per test
# domain from the diagonal (midterm->midterm, covid->covid).
SERIES = [
    ("single", "single (in-domain)", "#898781"),
    ("merged-naive", "merged-naive", "#2a78d6"),
    ("merged-within", "merged-within", "#1baf7a"),
    ("merged-within-bal", "merged-within-balanced", "#eb6834"),
]
TESTS = [("midterm", "test: midterm (small)"), ("covid", "test: covid (large)"),
         ("ukr (held-out)", "test: ukr (held-out)")]


def load(csv_path: Path) -> dict:
    cells = {}
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            cells[(row["metric"], row["train"], row["test"])] = float(row["value"])
    return cells


def value(cells, metric, regime, suffix, test):
    if regime == "single":
        return cells.get((metric, test, test))  # diagonal = in-domain single source
    return cells.get((metric, f"{regime} {suffix}", test))


def plot_panel(ax, cells, metric, suffix, title):
    import numpy as np
    x = np.arange(len(TESTS))
    n = len(SERIES)
    width = 0.8 / n
    for i, (regime, label, colour) in enumerate(SERIES):
        raw = [value(cells, metric, regime, suffix, t) for t, _ in TESTS]
        vals = [v if v is not None else 0.0 for v in raw]
        bars = ax.bar(x + (i - (n - 1) / 2) * width, vals, width, label=label, color=colour)
        for b, v in zip(bars, raw):
            if v is not None:  # skip held-out cells with no in-domain single
                ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in TESTS], fontsize=9)
    ax.set_ylim(0, max(0.75, max((value(cells, metric, r, suffix, t) or 0)
                                 for r, _, _ in SERIES for t, _ in TESTS) + 0.1))
    ax.set_ylabel(metric, fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.8)
    ax.set_axisbelow(True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="matrix.csv")
    ap.add_argument("--metric", default="accuracy", choices=["accuracy", "f1", "roc_auc"])
    ap.add_argument("--out", default=None, help="Output PNG (default: results_<metric>.png next to this script).")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = (here / csv_path) if not csv_path.exists() else csv_path
    cells = load(csv_path)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    plot_panel(axes[0], cells, args.metric, "@match", f"{args.metric} @match (matched compute)")
    plot_panel(axes[1], cells, args.metric, "@full", f"{args.metric} @full (per-domain exposure)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9, frameon=False)
    fig.suptitle(f"cov/mid NM transfer — {args.metric} (3-shot, 30-way)", fontsize=12)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))

    out = Path(args.out) if args.out else here / f"results_{args.metric}.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
