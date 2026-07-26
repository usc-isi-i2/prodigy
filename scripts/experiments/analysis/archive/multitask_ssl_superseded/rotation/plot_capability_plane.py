#!/usr/bin/env python3
"""Capability-plane figure for multitask_ssl_rotation (the headline figure).

One encoder, all SSL tasks. Plots each arm as a point on the plane whose axes are
the two transfer capabilities that define a *general* encoder:

    x = feature capability     = node-classification ROC-AUC (mean over datasets)
    y = topological capability = static-link-prediction ROC-AUC (mean over datasets)

Chance = 0.50 on both axes carves four quadrants; only the top-right is "good at
both". The single finding this figure carries: three single-objective controls are
pinned in/under the topological-chance floor, and MIX (rotation) is the only arm that
climbs into the generalist quadrant. Regression (the secondary feature axis, where FP
is the specialist) is folded in as marker size so FP's off-plane strength is visible.

Reads the same raw CSVs as aggregate_results.py; recomputes every number (no
hardcoded results). Writes figures/0_capability_plane.{pdf,png}.

Usage:
    python scripts/experiments/analysis/multitask_ssl_rotation/plot_capability_plane.py \
        --plotting-root scripts/experiments/analysis/multitask_ssl_rotation
"""
from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

ARMS = ["NM", "CL", "FP", "MIX"]
CONTROLS = ["NM", "CL", "FP"]
# House palette — extracted from the existing per-task figures so all four render
# identically across the experiment's figure set.
COLOR = {"NM": "#3987e5", "CL": "#1baf7a", "FP": "#eda100", "MIX": "#6a5acd"}
FULLNAME = {
    "NM": "NM · neighbor matching",
    "CL": "CL · contrastive",
    "FP": "FP · feature recon",
    "MIX": "MIX · rotation",
}


def _read(path: Path, metric: str, split: str = "test") -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        raise SystemExit(f"missing raw data: {path}")
    with path.open() as fh:
        for r in csv.DictReader(fh):
            if r.get("split") != split:
                continue
            v = r.get(metric, "")
            if v in ("", None):
                continue
            try:
                r[metric] = float(v)
            except ValueError:
                continue
            rows.append(r)
    return rows


def _by_arm(rows: list[dict], metric: str) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {a: [] for a in ARMS}
    for r in rows:
        if r.get("model") in out:
            out[r["model"]].append(r[metric])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plotting-root", default="scripts/experiments/analysis/multitask_ssl_rotation")
    ap.add_argument("--outdir", default=None, help="default: <this dir>/figures")
    args = ap.parse_args()
    root = Path(args.plotting_root)
    outdir = Path(args.outdir) if args.outdir else Path(__file__).resolve().parent / "figures"
    outdir.mkdir(parents=True, exist_ok=True)

    cls = _by_arm(_read(root / "node_classification/data/node_classification.csv", "roc_auc"), "roc_auc")
    slp = _by_arm(_read(root / "static_link_prediction/data/static_link_prediction.csv", "roc_auc"), "roc_auc")
    reg = _by_arm(_read(root / "node_regression/data/node_regression.csv", "spearman"), "spearman")

    x = {a: statistics.fmean(cls[a]) for a in ARMS}          # feature axis
    y = {a: statistics.fmean(slp[a]) for a in ARMS}          # topological axis
    ylo = {a: min(slp[a]) for a in ARMS}                     # sLP min/max across datasets
    yhi = {a: max(slp[a]) for a in ARMS}                     # (1-seed spread proxy)
    rho = {a: statistics.fmean(reg[a]) for a in ARMS}        # regression -> marker size

    # marker area scales with regression rho (floored so a negative arm is still visible)
    rmin, rmax = min(rho.values()), max(rho.values())
    def area(a: float) -> float:
        return 340.0 + (a - rmin) / (rmax - rmin) * 1350.0

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 12,
        "axes.edgecolor": "#9aa0a6", "axes.linewidth": 1.0,
        "figure.dpi": 120,
    })
    fig, ax = plt.subplots(figsize=(8.4, 6.6))

    XL, XR = 0.40, 1.0
    YB, YT = 0.24, 0.90
    ax.set_xlim(XL, XR)
    ax.set_ylim(YB, YT)

    # generalist quadrant (x>0.5, y>0.5) — faint tint + label
    ax.add_patch(Rectangle((0.5, 0.5), XR - 0.5, YT - 0.5, facecolor="#6a5acd",
                           alpha=0.055, edgecolor="none", zorder=0))
    ax.text(0.985, 0.878, "GENERALIST\ngood at both", ha="right", va="top",
            fontsize=10.5, color="#5a4bb0", fontweight="bold", linespacing=1.15, zorder=1)
    ax.text(0.985, 0.262, "feature specialist\n(topology ≈ chance)", ha="right", va="bottom",
            fontsize=9.5, color="#9aa0a6", style="italic", linespacing=1.15, zorder=1)

    # chance lines
    for v, ho in ((0.5, "x"), (0.5, "y")):
        (ax.axvline if ho == "x" else ax.axhline)(v, color="#b9bec4", lw=1.2, ls=(0, (5, 4)), zorder=1)
    ax.text(0.505, YB + 0.006, "feature chance 0.5", color="#9aa0a6", fontsize=8.5, rotation=90, va="bottom")
    ax.text(XR - 0.01, 0.508, "topological chance 0.5", color="#9aa0a6", fontsize=8.5, ha="right", va="bottom")

    # the emergent-topology gap: MIX vs best control on the topological axis
    best_ctrl = max(CONTROLS, key=lambda c: y[c])
    gap = y["MIX"] - y[best_ctrl]
    ax_x = 0.905
    ax.add_patch(FancyArrowPatch((ax_x, y[best_ctrl]), (ax_x, y["MIX"]),
                                 arrowstyle="<|-|>", mutation_scale=15,
                                 color="#5a4bb0", lw=1.8, zorder=3))
    ax.plot([x[best_ctrl], ax_x], [y[best_ctrl], y[best_ctrl]], color="#5a4bb0", lw=0.8, ls=":", zorder=2)
    ax.plot([x["MIX"], ax_x], [y["MIX"], y["MIX"]], color="#5a4bb0", lw=0.8, ls=":", zorder=2)
    ax.text(ax_x - 0.012, (y["MIX"] + y[best_ctrl]) / 2, f"+{gap:.2f}\nemergent\ntopology",
            ha="right", va="center", fontsize=10, color="#5a4bb0", fontweight="bold", linespacing=1.15)

    # per-arm: sLP min–max whisker (1-seed spread across datasets) + point
    for a in ARMS:
        ax.plot([x[a], x[a]], [ylo[a], yhi[a]], color=COLOR[a], lw=1.6, alpha=0.5,
                solid_capstyle="round", zorder=3)
    for a in ARMS:
        ax.scatter([x[a]], [y[a]], s=area(rho[a]), c=COLOR[a], edgecolors="white",
                   linewidths=2.0, zorder=5)

    # direct labels (identity in dark ink; colored marker carries the hue).
    # name_y = baseline of the bold arm name; the value line sits 0.032 below it.
    lab = {
        "NM":  dict(dx=+0.028, name_y=y["NM"] + 0.005, ha="left"),
        "CL":  dict(dx=+0.030, name_y=y["CL"] + 0.010, ha="left"),
        "FP":  dict(dx=+0.030, name_y=y["FP"] + 0.010, ha="left"),
        "MIX": dict(dx=-0.028, name_y=y["MIX"] + 0.086, ha="right"),
    }
    for a in ARMS:
        L = lab[a]
        tx = x[a] + L["dx"]
        ax.text(tx, L["name_y"], a, ha=L["ha"], va="baseline",
                fontsize=13, fontweight="bold", color="#202124")
        sub = f"cls {x[a]:.2f} · LP {y[a]:.2f} · ρ {rho[a]:+.2f}"
        ax.text(tx, L["name_y"] - 0.032, sub, ha=L["ha"], va="baseline",
                fontsize=9, color="#5f6368")

    ax.set_xlabel("feature capability  —  node-classification ROC-AUC  (10-shot, held-out)", fontsize=11.5)
    ax.set_ylabel("topological capability  —  static-link-prediction ROC-AUC  (0-shot)", fontsize=11.5)
    ax.set_title("Only rotation learns both feature and topological structure",
                 fontsize=15, fontweight="bold", pad=30)
    ax.text(0.5, 1.045, "frozen-encoder transfer · one shared encoder per arm · 1 seed, 30k matched checkpoint",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=10, color="#5f6368")

    ax.grid(True, color="#eceef1", lw=0.9, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    # legend: objective full names + the size-encoding note, as a horizontal row
    # BELOW the plot so it never overlaps a data point.
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", ls="", markerfacecolor=COLOR[a], markeredgecolor="white",
                      markeredgewidth=1.2, markersize=11, label=FULLNAME[a]) for a in ARMS]
    leg = ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.11),
                    ncol=4, fontsize=9.5, frameon=False, handletextpad=0.4, columnspacing=1.6,
                    title="marker size ∝ regression ρ  (secondary feature axis — FP is its specialist)")
    leg.get_title().set_fontsize(9)
    leg.get_title().set_color("#5f6368")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = outdir / f"0_capability_plane.{ext}"
        fig.savefig(p, bbox_inches="tight", facecolor="white")
        print("wrote", p)
    # echo the plotted values for the record
    print("\nplotted (mean over datasets, test split):")
    print(f"{'arm':<4}{'cls(x)':>8}{'sLP(y)':>8}{'sLP min':>9}{'sLP max':>9}{'reg ρ':>8}")
    for a in ARMS:
        print(f"{a:<4}{x[a]:>8.3f}{y[a]:>8.3f}{ylo[a]:>9.3f}{yhi[a]:>9.3f}{rho[a]:>8.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
