#!/usr/bin/env python3
"""Per-graph heatmaps of the saturation sweep — step x test graph, one panel per arm.

The line figure plots the MEAN over test graphs, which is exactly what hides the main
result: classification transfer works on two of the four labelled graphs, sits at chance
on a third, and goes backwards on the fourth. A heatmap shows every cell, so the reader
sees the spread instead of an average that no single graph exhibits.

Two colour jobs, deliberately different (see the data, not the convention):

  classification -> SEQUENTIAL blue, anchored at 0.5.
      ROC-AUC runs 0.487-0.987 here, so the values sit essentially all on one side of
      chance; the quantity of interest is "how far above chance". Anchoring vmin at 0.5
      lets a chance cell recede to near-white, which is the correct reading of "nothing
      is happening here". A diverging map would spend half its range on an empty arm.

  regression -> DIVERGING blue<->red, centred on 0, grey midpoint.
      Spearman runs -0.38..+0.48 and the SIGN is meaningful (anti-correlated is a
      different statement from uncorrelated), so polarity is the job.

Every cell is labelled, so colour never carries the value alone.

    /opt/homebrew/bin/python3.11 build_heatmaps.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm

HERE = Path(__file__).resolve().parent
ARM_ORDER = ["all8", "ukr", "covid"]
ARM_TITLE = {"all8": "all8  (8-source merge)", "ukr": "ukr  (single source)",
             "covid": "covid  (single source)"}
INK, INK_MUTED, GRID = "#1a1a19", "#6b6a63", "#e4e3dd"

# Documented blue ramp, steps 100..700 (references/palette.md).
BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
        "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("seq_blue", BLUE)
# Diverging: blue <-> red poles with the documented neutral grey midpoint. Red arm is
# mirrored from the categorical red slot; no red ramp is documented, so it is built by
# interpolating slot-8 red through the same midpoint rather than inventing new steps.
DIV = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#8f2b2a", "#e34948", "#f0efec", "#2a78d6", "#0d366b"])


def luminance(rgba) -> float:
    r, g, b = rgba[:3]
    f = lambda c: c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)


def panel(ax, mat: pd.DataFrame, cmap, norm, fmt: str, title: str):
    im = ax.imshow(mat.values, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(len(mat.columns)))
    ax.set_xticklabels([f"{c:,}" for c in mat.columns], fontsize=8.5, color=INK_MUTED)
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels(mat.index, fontsize=8.5, color=INK_MUTED)
    ax.set_title(title, fontsize=10, color=INK, loc="left", pad=8)
    ax.set_xlabel("pretraining steps", fontsize=8.5, color=INK_MUTED)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    # 2px surface gap between cells -- the marks spec's spacer, done with a white grid.
    ax.set_xticks([x - 0.5 for x in range(1, len(mat.columns))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(mat.index))], minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.values[i, j]
            if pd.isna(v):
                continue
            ax.text(j, i, format(v, fmt), ha="center", va="center", fontsize=8.5,
                    color="white" if luminance(cmap(norm(v))) < 0.45 else INK)
    return im


def main() -> int:
    long = pd.read_csv(HERE / "data" / "pretrain_saturation_long.csv")

    # ---------------- classification ----------------
    c = long[long.task == "classification"]
    # One row order shared by all three panels, or the panels cannot be compared.
    # Sorted by all8's best value so the two graphs that carry the effect sit together.
    order = (c[c.arm == "all8"].groupby("dataset")["value"].max()
             .sort_values(ascending=False).index.tolist())
    norm_c = Normalize(vmin=0.5, vmax=1.0)   # 0.5 = chance
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 3.5), constrained_layout=True)
    for ax, arm in zip(axes, ARM_ORDER):
        mat = (c[c.arm == arm].pivot_table(index="dataset", columns="step", values="value")
               .reindex(order))
        im = panel(ax, mat, SEQ, norm_c, ".3f", ARM_TITLE[arm])
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01)
    cb.set_label("ROC-AUC   (0.5 = chance, floor of the scale)", fontsize=8.5, color=INK_MUTED)
    cb.ax.tick_params(labelsize=8, colors=INK_MUTED)
    cb.outline.set_visible(False)
    fig.suptitle("Node classification by test graph — the mean hides that only two graphs transfer",
                 fontsize=12.5, color=INK, x=0.005, ha="left")
    out = HERE / "figures" / "heatmap_classification.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")

    # ---------------- regression ----------------
    r = long[long.task == "regression"].copy()
    r["cell"] = r["dataset"] + "  ·  " + r["target"].str.replace("_count", "", regex=False)
    rorder = sorted(r["cell"].unique())
    norm_r = TwoSlopeNorm(vmin=-0.5, vcenter=0.0, vmax=0.5)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), constrained_layout=True)
    for ax, arm in zip(axes, ARM_ORDER):
        mat = (r[r.arm == arm].pivot_table(index="cell", columns="step", values="value")
               .reindex(rorder))
        im = panel(ax, mat, DIV, norm_r, "+.2f", ARM_TITLE[arm])
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01)
    cb.set_label("Spearman   (0 = no signal)", fontsize=8.5, color=INK_MUTED)
    cb.ax.tick_params(labelsize=8, colors=INK_MUTED)
    cb.outline.set_visible(False)
    fig.suptitle("Node regression — VOID (random unfitted head). Kept as evidence of the noise; "
                 "valid numbers in probe_regression_heatmap.png",
                 fontsize=12.5, color=INK, x=0.005, ha="left")
    out = HERE / "figures" / "heatmap_regression.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
