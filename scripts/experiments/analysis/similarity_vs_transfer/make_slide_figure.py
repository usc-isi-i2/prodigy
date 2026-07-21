#!/usr/bin/env python
"""Slide-ready summary figure for the similarity-vs-transfer pilot.

Two panels in one figure:
  A) within-target Spearman rho heatmap (predictor x target, + across-target mean)
  B) scatter of the standout predictor (proxy-A-distance) vs NM transfer accuracy.

Reproduces the numbers in FINDINGS.md / the notebook. Run from this directory:

    ~/.pyenv/versions/myenv/bin/python make_slide_figure.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(exist_ok=True)

xfer = pd.read_csv(HERE / "transfer_matrix.csv")
sim = json.load(open(HERE.parent / "graph_divergence" / "graph_divergence_data.json"))

FULL = {"covid": "covid19_twitter", "ukr": "ukr_rus_twitter", "midterm": "midterm",
        "cp_hk": "cp_hk_twitter", "twibot20": "twibot20"}
PW, GRAPHS, PG = sim["pairwise"], sim["graphs"], sim["per_graph"]
METRICS = ["indegree_ks", "outdegree_ks", "feat_centroid_cosdist",
           "feat_frechet", "feat_mmd2", "proxy_a_distance"]

# pretty labels, ordered topology -> feature -> coupling
LABELS = {
    "indegree_ks": "In-degree KS  (topology)",
    "outdegree_ks": "Out-degree KS  (topology)",
    "feat_centroid_cosdist": "Feat. centroid cos  (feature)",
    "feat_frechet": "Feat. Fréchet  (feature)",
    "feat_mmd2": "Feat. MMD²  (feature)",
    "proxy_a_distance": "Proxy-A-dist  (feature sep.)",
    "homophily_gap": "Homophily gap  (coupling, signed)",
}
ROW_ORDER = list(LABELS.keys())


def pairwise_val(metric, s, t):
    i, j = GRAPHS.index(FULL[s]), GRAPHS.index(FULL[t])
    return PW[metric][i][j]


for m in METRICS:
    xfer[m] = [pairwise_val(m, s, t) for s, t in zip(xfer.source, xfer.target)]
xfer["homophily_gap"] = [
    PG[FULL[s]]["feature_homophily"] - PG[FULL[t]]["feature_homophily"]
    for s, t in zip(xfer.source, xfer.target)
]

TARGETS = sorted(xfer.target.unique())
DV = "accuracy"


def within_target_rhos(df, drop_self=False):
    rows = []
    for t in TARGETS:
        sub = df[df.target == t]
        if drop_self:
            sub = sub[~sub.is_self]
        if len(sub) < 3:
            continue
        row = {"target": t}
        for m in ROW_ORDER:
            row[m], _ = spearmanr(sub[m], sub[DV])
        rows.append(row)
    return pd.DataFrame(rows).set_index("target")


rho_full = within_target_rhos(xfer, drop_self=False)      # index=target, cols=metric
rho_noself = within_target_rhos(xfer, drop_self=True)
mean_full = rho_full[ROW_ORDER].mean()                     # across-target mean (with self)
mean_noself = rho_noself[ROW_ORDER].mean()

# matrix for the heatmap: rows = predictors, cols = targets + a spacer + mean
M = rho_full[ROW_ORDER].T                                  # predictor x target
col_labels = list(M.columns) + ["", "mean"]
grid = np.column_stack([M.values, np.full(len(ROW_ORDER), np.nan), mean_full.values])

# ---- figure -------------------------------------------------------------
plt.rcParams.update({"font.size": 11})
fig, (axA, axB) = plt.subplots(
    1, 2, figsize=(15.5, 6.6), gridspec_kw={"width_ratios": [1.35, 1.0]}
)

# Panel A: rho heatmap
im = axA.imshow(grid, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
axA.set_xticks(range(len(col_labels)))
axA.set_xticklabels(col_labels, fontsize=11)
axA.set_yticks(range(len(ROW_ORDER)))
axA.set_yticklabels([LABELS[m] for m in ROW_ORDER], fontsize=11)
axA.set_xlabel("target graph", fontsize=11)
for i in range(grid.shape[0]):
    for j in range(grid.shape[1]):
        v = grid[i, j]
        if np.isnan(v):
            continue
        axA.text(j, i, f"{v:+.1f}", ha="center", va="center",
                 fontsize=10.5, fontweight="bold" if j == grid.shape[1] - 1 else "normal",
                 color="white" if abs(v) > 0.55 else "black")
# separate the mean column visually
axA.axvline(len(TARGETS) - 0.5, color="0.35", lw=1)
axA.axvline(len(TARGETS) + 0.5, color="0.35", lw=1)
axA.set_title("Within-target Spearman ρ: source divergence vs. NM transfer\n"
              "(blue = more-divergent source transfers worse — the expected sign)",
              fontsize=12.5, pad=10)
cbar = fig.colorbar(im, ax=axA, fraction=0.046, pad=0.02)
cbar.set_label("Spearman ρ", fontsize=10)

# Panel B: scatter of the standout predictor
colors = dict(zip(TARGETS, plt.cm.tab10.colors))
for t in TARGETS:
    sub = xfer[xfer.target == t]
    s0, s1 = sub[~sub.is_self], sub[sub.is_self]
    axB.scatter(s0["proxy_a_distance"], s0[DV], color=colors[t], s=70, label=t, zorder=3)
    axB.scatter(s1["proxy_a_distance"], s1[DV], color=colors[t], marker="*", s=320,
                edgecolor="black", linewidth=0.7, zorder=4)
axB.set_xlabel("Proxy-A-distance  (feature-cloud separability: 0 = same, 2 = disjoint)",
               fontsize=10.5)
axB.set_xlim(-0.08, 1.5)
axB.set_ylabel("NM transfer accuracy  (30-way, 3-shot, test)", fontsize=11)
axB.set_title("Standout predictor: feature separability vs. transfer\n"
              f"mean within-target ρ = {mean_full['proxy_a_distance']:+.2f}  "
              f"({mean_noself['proxy_a_distance']:+.2f} excl. self)",
              fontsize=12.5, pad=10)
axB.spines[["top", "right"]].set_visible(False)
axB.legend(title="target", fontsize=9.5, frameon=False, loc="upper right")
axB.text(0.02, 0.03, "★ = in-domain (self) source", transform=axB.transAxes,
         fontsize=9, color="0.35")

fig.suptitle("Graph similarity predicts NM transfer — feature axes beat raw topology",
             fontsize=15, fontweight="bold", y=1.02)
fig.tight_layout()
out = FIG_DIR / "similarity_transfer_slide.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
