#!/usr/bin/env python3
"""A 'graph of the graphs': feature similarity x transfer asymmetry, one meta-graph.

  position   classical MDS of the (symmetrized) proxy-A-distance matrix
             -> feature-similar graphs sit close; outliers on the rim
  node color feature-centrality (mean distance to others): dark green hub -> coral outlier
  node size  in-domain AUC ceiling (diagonal of the transfer matrix)
  arrows     one per graph pair, drawn donor -> beneficiary (the LOWER-regret /
             better-transfer direction); width = transfer asymmetry
             |reg(A->B) - reg(B->A)|. Near-symmetric pairs are hairlines.

Reproducible: loads nm_single_source_matrix.csv + graph_divergence_data.json.
PDF+PNG to ./figures/.  NM 3-shot/30-way, matched-40k, 1 seed.
"""
import os, csv, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
FIGDIR = os.path.join(HERE, "figures"); os.makedirs(FIGDIR, exist_ok=True)
TRANSFER = os.path.join(HERE, "nm_single_source_matrix.csv")
DIVJSON = os.path.join(ROOT, "scripts", "plotting", "graph_divergence", "graph_divergence_data.json")

INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
CORAL = "#D85A30"
CENTMAP = LinearSegmentedColormap.from_list(
    "cent", ["#24500d", "#5b9a2f", "#b5e086", "#f0b48a", CORAL])
NAME = {"ukr_rus_twitter": "Ukraine", "covid19_twitter": "COVID", "midterm": "Midterm",
        "covid_political": "COVID-pol", "election2020": "Election-20",
        "ukr_rus_suspended": "Ukr-susp", "twibot20": "TwiBot-20", "cp_hk_twitter": "CP-HK"}

# ---------- load ----------
rows = list(csv.reader(open(TRANSFER)))
cols = rows[0][1:]; AUC = {}; diag = {}
for r in rows[1:]:
    s = r[0]
    for j, t in enumerate(cols):
        AUC[(s, t)] = float(r[j + 1])
    diag[s] = AUC[(s, s)]
reg = {(a, b): diag[b] - AUC[(a, b)] for a in cols for b in cols}

dv = json.load(open(DIVJSON))
G = dv["graphs"]; gi = {g: i for i, g in enumerate(G)}
D = np.array(dv["pairwise"]["proxy_a_distance"]); D = (D + D.T) / 2.0
cent = {g: D[gi[g]][[gi[h] for h in G if h != g]].mean() for g in G}

# ---------- classical (Torgerson) MDS -> 2D ----------
n = len(G)
J = np.eye(n) - np.ones((n, n)) / n
B = -0.5 * J @ (D ** 2) @ J
w, V = np.linalg.eigh(B)
o = np.argsort(w)[::-1]
pos = V[:, o[:2]] * np.sqrt(np.clip(w[o[:2]], 0, None))
pos = pos / np.abs(pos).max()  # normalize to ~[-1,1]
# orient for legibility: COVID/ukr hub to the left, outliers to the right/top
if pos[gi["covid19_twitter"], 0] > 0: pos[:, 0] *= -1
if pos[gi["election2020"], 1] < 0: pos[:, 1] *= -1
# overlap removal: the 4 feature-hubs land near-coincident (they ARE that similar);
# push any two nodes apart to a minimum spacing so markers/labels are legible.
MIN_SEP = 0.62
for _ in range(400):
    for i in range(n):
        for j in range(i + 1, n):
            d = pos[i] - pos[j]; dist = np.hypot(*d)
            if dist < MIN_SEP:
                shove = (MIN_SEP - dist) / 2 * d / (dist + 1e-9)
                pos[i] += shove; pos[j] -= shove
P = {g: pos[gi[g]] for g in G}
centroid = pos.mean(0)

# ---------- figure ----------
fig, ax = plt.subplots(figsize=(10.4, 8.4))
cvals = np.array([cent[g] for g in G])
cnorm = (cvals - cvals.min()) / (cvals.max() - cvals.min())

# ---- arrows: donor -> beneficiary, width = asymmetry (only directional pairs) ----
THRESH = 0.02  # AUC (2 pts): below this a pair is ~symmetric -> omit its hairline
pairs = [(a, b) for a in G for b in G if gi[a] < gi[b]]
asym = {p: abs(reg[(p[0], p[1])] - reg[(p[1], p[0])]) for p in pairs}
shown = [p for p in pairs if asym[p] >= THRESH]
amax = max(asym.values())
for a, b in shown:
    src, dst = (a, b) if reg[(a, b)] <= reg[(b, a)] else (b, a)  # lower regret = donor->beneficiary
    aa = asym[(a, b)]
    ar = FancyArrowPatch(P[src], P[dst], connectionstyle="arc3,rad=0.11",
                         arrowstyle="-|>", mutation_scale=6 + 15 * (aa / amax),
                         lw=0.5 + (aa * 100) * 0.30, color="#3a3a3a",
                         alpha=0.20 + 0.62 * (aa / amax),
                         shrinkA=15, shrinkB=18, zorder=2)
    ax.add_patch(ar)

# ---- nodes ----
sizes = [360 + (diag[g] - 0.90) * 4200 for g in G]
sc = ax.scatter([P[g][0] for g in G], [P[g][1] for g in G], s=sizes,
                c=cnorm, cmap=CENTMAP, edgecolor="white", linewidth=1.8, zorder=5)
# labels fanned radially outward from the layout centroid so they clear the markers
for g in G:
    v = P[g] - centroid
    v = v / (np.hypot(*v) + 1e-9)
    dx, dy = v * 30
    ax.annotate(NAME[g], P[g], xytext=(dx, dy), textcoords="offset points",
                ha="center", va="center", fontsize=9.6, color=INK, fontweight="medium",
                zorder=6)

# ---- colorbar for centrality ----
cb = fig.colorbar(sc, ax=ax, fraction=0.036, pad=0.02)
cb.set_label("feature-centrality  (mean proxy-A-distance)", fontsize=9)
cb.set_ticks([0, 1]); cb.set_ticklabels(["hub", "outlier"]); cb.ax.tick_params(labelsize=8.5)

# ---- legend below the plot ----
handles = [
    Line2D([0], [0], color="#3a3a3a", lw=2.4, alpha=0.9, marker=">", markersize=8,
           markerfacecolor="#3a3a3a", label="thicker = more one-way (bigger regret gap)"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor="#888", markeredgecolor="white",
           markersize=13, label="node size = in-domain AUC ceiling"),
]
leg = ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02),
                ncol=2, fontsize=9, frameon=True, framealpha=0.95, edgecolor=MUTED,
                title="arrow = donor → beneficiary  ·  near-symmetric pairs (gap < 2 pts) omitted")
leg.get_title().set_fontsize(9); leg.get_title().set_color(INK)

ax.set_title("A graph of the graphs — feature similarity (layout)  ×  transfer asymmetry (arrows)\n"
             "single-source NM, 8 graphs · close = feature-similar · arrows flow from good donors outward",
             fontsize=12.5, color=INK, pad=12)
ax.set_xticks([]); ax.set_yticks([])
for sp in ax.spines.values():
    sp.set_visible(False)
ax.set_aspect("equal", adjustable="datalim")
ax.margins(0.16)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(FIGDIR, f"nmss_graph_of_graphs.{ext}"), dpi=200, bbox_inches="tight")
print("wrote figures/nmss_graph_of_graphs.{pdf,png}")
# report the 5 most one-way pairs for the caption
top = sorted(pairs, key=lambda p: -asym[p])[:5]
for a, b in top:
    src, dst = (a, b) if reg[(a, b)] <= reg[(b, a)] else (b, a)
    print(f"  {NAME[src]:>11} -> {NAME[dst]:<11} asym={asym[(a,b)]*100:4.1f}pts  "
          f"({reg[(src,dst)]*100:.1f} vs {reg[(dst,src)]*100:.1f})")
