#!/usr/bin/env python3
"""Plot the ukr/cov cross-source-shortcut results from compare_shortcut --out-csv.

Per test domain: single (in-domain) vs merged-proportional vs merged-within, at
@match (matched compute) and @full (per-domain exposure). Saves results_<metric>.png.

    python compare_shortcut.py ... --out-csv shortcut.csv
    python plot_shortcut.py --csv shortcut.csv --metric accuracy

Needs matplotlib (prodigy env).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (csv-regime base, display label, colour). single in-domain is filled per test domain.
SERIES = [
    ("single", "single (in-domain)", "#898781"),
    ("merged proportional", "merged-naive", "#2a78d6"),
    ("merged within-source", "merged-within", "#1baf7a"),
]
TESTS = [("test:ukr", "test: ukr"), ("test:covid", "test: covid")]
SINGLE_BY_TEST = {"test:ukr": "single ukr", "test:covid": "single covid"}


def load(csv_path: Path) -> dict:
    cells = {}
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            cells[(row["metric"], row["regime"], row["test"])] = float(row["value"])
    return cells


def value(cells, metric, base, suffix, test):
    if base == "single":
        return cells.get((metric, SINGLE_BY_TEST[test], test))  # single only has one ckpt
    return cells.get((metric, f"{base} {suffix}", test))


def plot_panel(ax, cells, metric, suffix, title):
    import numpy as np
    x = np.arange(len(TESTS)); n = len(SERIES); width = 0.8 / n
    for i, (base, label, colour) in enumerate(SERIES):
        vals = [value(cells, metric, base, suffix, t) or 0.0 for t, _ in TESTS]
        bars = ax.bar(x + (i - (n - 1) / 2) * width, vals, width, label=label, color=colour)
        for b, v in zip(bars, vals):
            if v:
                ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels([lbl for _, lbl in TESTS], fontsize=9)
    ax.set_ylim(0, 0.75 if metric != "roc_auc" else 1.0)
    ax.set_ylabel(metric, fontsize=10); ax.set_title(title, fontsize=11)
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.8); ax.set_axisbelow(True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="shortcut.csv")
    ap.add_argument("--metric", default="accuracy", choices=["accuracy", "f1", "roc_auc"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    csv_path = Path(args.csv)
    if not csv_path.is_absolute() and not csv_path.exists():
        csv_path = here / csv_path
    cells = load(csv_path)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    plot_panel(axes[0], cells, args.metric, "@match", f"{args.metric} @match (matched compute)")
    plot_panel(axes[1], cells, args.metric, "@full", f"{args.metric} @full (per-domain exposure)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.suptitle(f"ukr/cov NM transfer — {args.metric} (3-shot, 30-way)", fontsize=12)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    out = Path(args.out) if args.out else here / f"results_{args.metric}.png"
    fig.savefig(out, dpi=150); print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
