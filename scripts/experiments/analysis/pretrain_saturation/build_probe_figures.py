#!/usr/bin/env python3
"""Figures for the RE-SCORED regression channel (fitted frozen-encoder ridge probe).

The other two figure scripts read the shared per-task CSVs, whose 144 ``sat_*`` regression
rows come from the episodic path with the never-fitted random ``regression_head``. Those
rows are void (FINDINGS §4/§4b), so anything they draw in a regression panel is void with
them. This script reads ``data/reg_probe/`` instead — the only valid regression numbers in
this experiment.

Emits:
    figures/probe_regression_curves.png   8 small multiples (dataset x target), 3 arms,
                                          raw-feature floor drawn per panel
    figures/probe_regression_heatmap.png  step x cell, one panel per arm

Colour job is SEQUENTIAL, not diverging: 142 of 144 probe values are positive and the
quantity is "how much rank signal", so zero is a floor rather than a pole.

    /opt/homebrew/bin/python3.11 build_probe_figures.py
"""
from __future__ import annotations

import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize

HERE = Path(__file__).resolve().parent
ARM_ORDER = ["all8", "ukr", "covid"]
ARM_LABEL = {"all8": "all8 (8-source merge)", "ukr": "ukr (single source)",
             "covid": "covid (single source)"}
ARM_COLOR = {"all8": "#2a78d6", "ukr": "#eb6834", "covid": "#1baf7a"}
INK, INK_MUTED, GRID = "#1a1a19", "#6b6a63", "#e4e3dd"
BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
        "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("seq_blue", BLUE)


def load():
    files = sorted(glob.glob(str(HERE / "data" / "reg_probe" / "*__reg_probe.csv")))
    if not files:
        raise SystemExit("no data/reg_probe/*.csv — run regression_probe_sweep.py first")
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    floors = (d[d.model == "__features_only__"]
              .set_index(["dataset", "target"])["spearman"].to_dict())
    enc = d[d.model != "__features_only__"].copy()
    k = enc.model.str.extract(r"^sat_(?P<arm>[a-z0-9]+)_s(?P<step>\d+)$")
    enc["arm"] = k["arm"]
    enc["step"] = k["step"].astype(int)
    enc["cell"] = enc.dataset + " · " + enc.target.str.replace("_count", "", regex=False)
    return enc, floors


def luminance(rgba):
    r, g, b = rgba[:3]
    f = lambda c: c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)


def curves(enc, floors):
    cells = sorted(enc.cell.unique())
    fig, axes = plt.subplots(2, 4, figsize=(16, 6.8), constrained_layout=True)
    for ax, cell in zip(axes.ravel(), cells):
        sub = enc[enc.cell == cell]
        ds, tgt = sub.dataset.iloc[0], sub.target.iloc[0]
        floor = floors.get((ds, tgt))
        if floor is not None:
            ax.axhline(floor, color=INK_MUTED, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
            ax.annotate(f"raw-feature floor {floor:.3f}", (1.0, floor), xycoords=("axes fraction", "data"),
                        xytext=(-2, 4), textcoords="offset points", ha="right", va="bottom",
                        fontsize=7.5, color=INK_MUTED)
        for arm in ARM_ORDER:
            s = sub[sub.arm == arm].groupby("step")["spearman"].mean().sort_index()
            ax.plot(s.index, s.values, color=ARM_COLOR[arm], linewidth=2, marker="o",
                    markersize=5, markeredgecolor="white", markeredgewidth=1.1,
                    label=ARM_LABEL[arm], zorder=3, solid_capstyle="round")
        ax.set_xscale("log")
        ax.set_title(cell, fontsize=9.5, color=INK, loc="left", pad=6)
        ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_MUTED, labelsize=8)
        ax.set_xlabel("pretraining steps (log)", fontsize=8, color=INK_MUTED)
        ax.set_ylabel("Spearman", fontsize=8, color=INK_MUTED)
    # Figure-level legend, not panel-level: every panel carries a floor annotation, so an
    # in-panel legend collides with one of them whichever panel it is placed in.
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8.5, labelcolor=INK,
               loc="upper left", bbox_to_anchor=(0.004, 0.975), ncol=3)
    fig.suptitle("Regression, re-scored with a fitted probe — the effect of pretraining flips by target",
                 fontsize=13, color=INK, x=0.004, ha="left", y=1.06)
    fig.text(0.004, -0.02,
             "Frozen encoder, ridge fitted on the support set, held-out queries; 500 shared "
             "episodes per cell. Dashed line = ridge on raw features. account_age_days RISES "
             "with steps (12/12 series positive, +179%, saturating ~10k); followers_count is "
             "flat-to-declining (-13%), its best encoder the least-trained one.",
             fontsize=8, color=INK_MUTED, ha="left")
    out = HERE / "figures" / "probe_regression_curves.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")


def heatmap(enc):
    cells = sorted(enc.cell.unique())
    norm = Normalize(vmin=0.0, vmax=0.42)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4), constrained_layout=True)
    for ax, arm in zip(axes, ARM_ORDER):
        mat = (enc[enc.arm == arm]
               .pivot_table(index="cell", columns="step", values="spearman").reindex(cells))
        im = ax.imshow(mat.values, cmap=SEQ, norm=norm, aspect="auto")
        ax.set_xticks(range(len(mat.columns)))
        ax.set_xticklabels([f"{c:,}" for c in mat.columns], fontsize=8.5, color=INK_MUTED)
        ax.set_yticks(range(len(mat.index)))
        ax.set_yticklabels(mat.index, fontsize=8.5, color=INK_MUTED)
        ax.set_title(ARM_LABEL[arm], fontsize=10, color=INK, loc="left", pad=8)
        ax.set_xlabel("pretraining steps", fontsize=8.5, color=INK_MUTED)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)
        ax.set_xticks([x - 0.5 for x in range(1, len(mat.columns))], minor=True)
        ax.set_yticks([y - 0.5 for y in range(1, len(mat.index))], minor=True)
        ax.grid(which="minor", color="white", linewidth=2)
        ax.tick_params(which="minor", length=0)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat.values[i, j]
                if pd.isna(v):
                    continue
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.5,
                        color="white" if luminance(SEQ(norm(v))) < 0.45 else INK)
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01)
    cb.set_label("Spearman (probe)", fontsize=8.5, color=INK_MUTED)
    cb.ax.tick_params(labelsize=8, colors=INK_MUTED)
    cb.outline.set_visible(False)
    fig.suptitle("Regression probe by cell — account_age rises with steps, followers does not",
                 fontsize=12.5, color=INK, x=0.004, ha="left")
    out = HERE / "figures" / "probe_regression_heatmap.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")


def main() -> int:
    enc, floors = load()
    print(f"{len(enc)} probe rows, {enc.arm.nunique()} arms, {enc.cell.nunique()} cells")
    curves(enc, floors)
    heatmap(enc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
