#!/usr/bin/env python3
"""Is the single-source NM transfer asymmetry a graph-similarity artifact?

Three-beat argument, one row of panels (paper-ready PDF+PNG in ./figures/):
  A  CONFOUND   donor strength (outflow) is ~perfectly explained by feature-centrality
                -> the "best donor" ranking is largely a restatement of centrality.
  B  DECOUPLING source-strength (outflow) tracks centrality; target-reachability
                (inflow) does NOT. A *symmetric* similarity cannot produce that split,
                so the donor/receiver asymmetry is real, not clustering.
  C  DIRECTION  a symmetric distance sets how big a pair's asymmetry is, but the *sign*
                is carried by the signed feature-homophily gap.

Reproducible: loads the transfer matrix (nm_single_source_matrix.csv) and the graph
divergence data (graph_divergence_data.json). NM 3-shot/30-way, matched-40k, 1 seed.
"""
import os, csv, json, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
FIGDIR = os.path.join(HERE, "figures"); os.makedirs(FIGDIR, exist_ok=True)
TRANSFER = os.path.join(HERE, "nm_single_source_matrix.csv")
DIVJSON = os.path.join(ROOT, "scripts", "plotting", "graph_divergence", "graph_divergence_data.json")

INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
CORAL, CORAL_TXT = "#D85A30", "#8a3517"
CENTMAP = LinearSegmentedColormap.from_list(
    "cent", ["#24500d", "#5b9a2f", "#b5e086", "#f0b48a", CORAL])  # central (dark green) -> peripheral (coral)

NAME = {"ukr_rus_twitter": "Ukraine", "covid19_twitter": "COVID", "midterm": "Midterm",
        "covid_political": "COVID-pol", "election2020": "Election-20",
        "ukr_rus_suspended": "Ukr-susp", "twibot20": "TwiBot-20", "cp_hk_twitter": "CP-HK"}

# ---------- load ----------
rows = list(csv.reader(open(TRANSFER)))
cols = rows[0][1:]
AUC = {}; diag = {}
for r in rows[1:]:
    s = r[0]
    for j, t in enumerate(cols):
        AUC[(s, t)] = float(r[j + 1])
    diag[s] = AUC[(s, s)]
reg = {(a, b): diag[b] - AUC[(a, b)] for a in cols for b in cols}

dv = json.load(open(DIVJSON))
G = dv["graphs"]; PW = dv["pairwise"]
P = np.array(PW["proxy_a_distance"])
P = (P + P.T) / 2.0  # symmetrize the ~7% directional residual
gi = {g: i for i, g in enumerate(G)}
cent = {g: P[gi[g]][[gi[h] for h in G if h != g]].mean() for g in G}  # mean feat-distance (low = central)
fh = {g: dv["per_graph"][g]["feature_homophily"] for g in G}

order = cols  # 8 graphs, canonical CSV order
outflow = {g: np.mean([AUC[(g, h)] for h in order if h != g]) for g in order}  # source strength
inflow = {g: np.mean([AUC[(h, g)] for h in order if h != g]) for g in order}   # target reachability


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    def rank(v):
        o = v.argsort(); r = np.empty(len(v)); r[o] = np.arange(len(v))
        # average ties
        _, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
        sums = np.zeros(len(cnt)); np.add.at(sums, inv, r)
        return (sums / cnt)[inv] + 1
    rx, ry = rank(x), rank(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color(MUTED); ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(labelsize=8.5, color=MUTED)
    ax.grid(True, color=GRID, lw=0.7, zorder=0)


fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(15.2, 5.3))

# ================= A: CONFOUND =================
xc = [cent[g] for g in order]; yo = [outflow[g] for g in order]
m, b = np.polyfit(xc, yo, 1)
xs = np.linspace(min(xc) - .02, max(xc) + .02, 50)
axA.plot(xs, m * xs + b, "-", color=MUTED, lw=1.4, zorder=2)
axA.scatter(xc, yo, s=95, c=[cent[g] for g in order], cmap=CENTMAP,
            edgecolor="white", linewidth=1.1, zorder=4)
axA.set_xlim(min(xc) - 0.03, max(xc) + 0.11)
off = {"ukr_rus_twitter": (8, -10), "covid19_twitter": (8, 9), "twibot20": (7, 4),
       "midterm": (7, -3), "ukr_rus_suspended": (-7, -11), "cp_hk_twitter": (7, 2),
       "covid_political": (-7, 7), "election2020": (7, -2)}
haA = {"ukr_rus_suspended": "right", "covid_political": "right"}
for g in order:
    axA.annotate(NAME[g], (cent[g], outflow[g]), xytext=off[g], textcoords="offset points",
                 fontsize=8.5, color=INK, ha=haA.get(g, "left"), va="center")
axA.set_xlabel("feature-centrality  =  mean proxy-A-distance to other graphs\n(left = central hub   ·   right = outlier)", fontsize=9.5)
axA.set_ylabel("donor strength  (mean off-diagonal AUC as source)", fontsize=9.5)
axA.set_title("A · The donor ranking IS centrality", fontsize=11.5, color=INK, pad=8)
axA.text(0.04, 0.06, f"Spearman ρ = {spearman(xc, yo):+.2f}", transform=axA.transAxes,
         fontsize=11, color=INK, fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=MUTED, lw=0.8))
style(axA)

# ================= B: DECOUPLING (hero) =================
xo = [outflow[g] for g in order]; yi = [inflow[g] for g in order]
lo = min(min(xo), min(yi)) - .02; hi = max(max(xo), max(yi)) + .055
axB.plot([lo, hi], [lo, hi], "--", color="#9a988e", lw=1.3, zorder=1)
axB.annotate("symmetric-transfer line  (outflow = inflow)", (0.70, 0.70), rotation=45,
             rotation_mode="anchor", ha="center", va="bottom", fontsize=8.3, color="#8a887e")
sc = axB.scatter(xo, yi, s=150, c=[cent[g] for g in order], cmap=CENTMAP,
                 edgecolor="white", linewidth=1.3, zorder=4)
offB = {"ukr_rus_twitter": (8, -4), "covid19_twitter": (8, 5), "twibot20": (8, 3),
        "midterm": (8, -3), "ukr_rus_suspended": (8, -3), "cp_hk_twitter": (8, 2),
        "covid_political": (-8, 6), "election2020": (-8, -10)}
haB = {"covid_political": "right", "election2020": "right"}
for g in order:
    axB.annotate(NAME[g], (outflow[g], inflow[g]), xytext=offB[g], textcoords="offset points",
                 fontsize=8.6, color=INK, ha=haB.get(g, "left"), va="center")
axB.text(0.63, 0.905, "net donors", transform=axB.transAxes, fontsize=9, color=CORAL_TXT,
         style="italic", ha="center")
axB.text(0.30, 0.055, "net receivers", transform=axB.transAxes, fontsize=9, color="#2f6d1f",
         style="italic", ha="center")
axB.set_xlim(lo, hi); axB.set_ylim(lo, hi); axB.set_aspect("equal", adjustable="box")
axB.set_xlabel("outflow  =  source strength  (as donor)", fontsize=9.5)
axB.set_ylabel("inflow  =  target reachability  (as receiver)", fontsize=9.5)
axB.set_title("B · Source strength ≠ target reachability", fontsize=11.5, color=INK, pad=8)
txt = (f"outflow ↔ centrality:  ρ = {spearman([cent[g] for g in order], xo):+.2f}\n"
       f"inflow  ↔ centrality:  ρ = {spearman([cent[g] for g in order], yi):+.2f}\n"
       f"outflow ↔ inflow:       ρ = {spearman(xo, yi):+.2f}")
axB.text(0.985, 0.30, txt, transform=axB.transAxes, fontsize=8.8, color=INK, family="monospace",
         va="top", ha="right", bbox=dict(boxstyle="round,pad=0.4", fc="#faf9f5", ec=MUTED, lw=0.8))
cb = fig.colorbar(sc, ax=axB, fraction=0.046, pad=0.02)
cb.set_label("feature-centrality (distance)", fontsize=8.3); cb.ax.tick_params(labelsize=7.5)
cb.ax.text(0.5, 1.02, "outlier", transform=cb.ax.transAxes, ha="center", fontsize=7, color=MUTED)
cb.ax.text(0.5, -0.04, "hub", transform=cb.ax.transAxes, ha="center", va="top", fontsize=7, color=MUTED)
style(axB); axB.grid(True, color=GRID, lw=0.7, zorder=0)

# ================= C: DIRECTION =================
pairs = [(a, b) for a in order for b in order if gi[a] < gi[b]]  # 28 unordered, canonical orientation
xh = [fh[a] - fh[b] for a, b in pairs]                          # signed homophily gap
ya = [(reg[(a, b)] - reg[(b, a)]) / 2 * 100 for a, b in pairs]  # antisymmetric regret (pts)
m2, b2 = np.polyfit(xh, ya, 1)
xs2 = np.linspace(min(xh) - .002, max(xh) + .002, 50)
axC.axhline(0, color=GRID, lw=1.0); axC.axvline(0, color=GRID, lw=1.0)
axC.plot(xs2, m2 * xs2 + b2, "-", color=CORAL, lw=1.6, zorder=2)
axC.scatter(xh, ya, s=55, color=INK, alpha=0.78, edgecolor="white", linewidth=0.6, zorder=4)
axC.set_xlabel("signed feature-homophily gap   fh(A) − fh(B)", fontsize=9.5)
axC.set_ylabel("antisymmetric regret  (A→B minus B→A)/2   [AUC×100]", fontsize=9.5)
axC.set_title("C · Direction rides on homophily, not distance", fontsize=11.5, color=INK, pad=8)
axC.text(0.04, 0.93, f"Spearman ρ = {spearman(xh, ya):+.2f}   (n=28 pairs)",
         transform=axC.transAxes, fontsize=10.5, color=CORAL_TXT, fontweight="bold", va="top")
axC.text(0.96, 0.06, "|asymmetry| ↔ distance:  ρ = %+.2f" %
         spearman([abs(v) for v in ya], [P[gi[a]][gi[b]] for a, b in pairs]),
         transform=axC.transAxes, fontsize=8.5, color=MUTED, ha="right")
style(axC)

fig.suptitle("Domain-transfer asymmetry is not a graph-similarity artifact   ·   single-source NM, 8 graphs (1 seed)",
             fontsize=13.5, color=INK, y=1.0, x=0.5)
fig.tight_layout(rect=[0, 0, 1, 0.97])
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(FIGDIR, f"nmss_transfer_asymmetry_similarity.{ext}"),
                dpi=200, bbox_inches="tight")
print("wrote figures/nmss_transfer_asymmetry_similarity.{pdf,png}")
print("A donor↔cent:", round(spearman(xc, yo), 2),
      "| B out↔cent:", round(spearman([cent[g] for g in order], xo), 2),
      "in↔cent:", round(spearman([cent[g] for g in order], yi), 2),
      "out↔in:", round(spearman(xo, yi), 2),
      "| C antisym↔hg:", round(spearman(xh, ya), 2))
