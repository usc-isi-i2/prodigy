#!/usr/bin/env python3
"""Regret version of the generalist-vs-specialist scatter, EQUAL span on both axes.
x = home-turf regret (mean Δ to best on trained graphs), y = overall regret
(mean Δ to best over all 8). Both axes cover the same range so the scale is
identical and y=x is a true 45 deg line. Home-turf regret is ~0 for everyone, so
the points collapse onto a thin strip near x=0 -- that collapse IS the finding.
Saves PNG + PDF to ~/Downloads."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

LADDER = [
    ("L1", 0.00, -8.13, 1), ("L2", -0.18, -7.89, 2), ("L3", -0.66, -7.94, 3), ("L4", -0.95, -6.93, 4),
    ("L5", -1.31, -5.79, 5), ("L6", -1.69, -3.84, 6), ("L7", -1.88, -3.88, 7), ("L8", -2.00, -2.00, 8),
]
SPEC = [  # key, home Δ, overall Δ, legible name, (dx,dy) pts
    ("covid",    0.00,  -7.89, "COVID",             (-12, 7)),
    ("ukr",     -0.10,  -8.08, "Ukraine",           (-12, -8)),
    ("twibot20", 0.00, -10.84, "TwiBot-20",         (-12, 0)),
    ("midterm",  0.00, -14.58, "Midterm",           (-12, 0)),
    ("ukr_susp", 0.00, -17.53, "Ukraine-suspended", (-12, 0)),
    ("cp_hk",    0.00, -22.09, "CP-HK",             (-12, 0)),
    ("cov_pol",  0.00, -24.89, "COVID-political",   (-12, 5)),
    ("elec20",   0.00, -25.53, "Election 2020",     (-12, -6)),
]

g1, g2 = np.array([181, 224, 134]) / 255, np.array([40, 86, 14]) / 255
ramp = lambda k: tuple(g1 + (g2 - g1) * (k - 1) / 7)
CORAL, CORAL_TXT = "#D85A30", "#8a3517"
LIM = (-27.5, 3.0)          # same range on BOTH axes -> identical scale

fig, ax = plt.subplots(figsize=(6.6, 6.6))
ax.plot(LIM, LIM, "--", color="#9a988e", lw=1.3, zorder=1)                      # y = x
ax.axvline(0, ls=":", color="#c7c5bc", lw=1.1, zorder=1)
ax.axhline(0, ls=":", color="#c7c5bc", lw=1.1, zorder=1)
ax.annotate("y = x  (home regret = overall regret)", (-20, -20), rotation=45,
            rotation_mode="anchor", ha="center", va="bottom", fontsize=9, color="#8a887e")
ax.annotate("ideal", (0, 0), xytext=(-6, 6), textcoords="offset points",
            ha="right", va="bottom", fontsize=9.5, color="#8a887e")

LAB_OFF = {1: (9, -5, "left"), 2: (9, 9, "left"), 3: (10, -2, "left"),
           4: (-10, 3, "right"), 5: (-10, -1, "right"), 6: (-10, 8, "right"), 7: (-11, -9, "right")}
lx = [m[1] for m in LADDER]
ly = [m[2] for m in LADDER]
ax.plot(lx, ly, "-", color="#5b9a2f", lw=1.7, alpha=0.55, zorder=2)
for lb, x, y, k in LADDER:
    big = k == 8
    ax.scatter(x, y, s=150 if big else 78, color=ramp(k),
               edgecolor="white" if big else "none", linewidth=1.4, zorder=4)
    if not big:
        dx, dy, ha = LAB_OFF[k]
        ax.annotate(lb, (x, y), xytext=(dx, dy), textcoords="offset points",
                    fontsize=8.5, color="#2f5a10", ha=ha, va="center", zorder=6,
                    arrowprops=dict(arrowstyle="-", color="#bcccab", lw=0.7))
ax.annotate("all-8", (-2.00, -2.00), xytext=(-11, 7), textcoords="offset points",
            fontsize=9.5, color="#1b3a09", fontweight="bold", ha="right")

for key, x, y, name, off in SPEC:
    ax.scatter(x, y, s=78, color=CORAL, edgecolor="white", linewidth=0.8, zorder=4)
    ax.annotate(name, (x, y), xytext=off, textcoords="offset points",
                fontsize=9.5, color=CORAL_TXT, ha="right", va="center", zorder=6)

ax.set_xlim(LIM); ax.set_ylim(LIM); ax.set_aspect("equal", adjustable="box")
ax.set_xlabel("home-turf regret   (mean Δ to best, AUC pts)", fontsize=10.5)
ax.set_ylabel("overall regret   (mean Δ to best over all 8, AUC pts)", fontsize=10.5)
ax.set_title("Generalist vs specialist — regret, equal scale (NM, matched-40k)", fontsize=11.5, pad=10)
ax.tick_params(labelsize=9)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.grid(True, color="#efeee9", lw=0.6, zorder=0)

handles = [
    Line2D([0], [0], marker="o", color="none", markerfacecolor=ramp(4),
           markersize=9, label="merged ladder  (L1 Ukraine → L8 all-8)"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor=CORAL,
           markersize=9, label="single-source specialist"),
    Line2D([0], [0], ls="--", color="#9a988e", label="y = x"),
]
ax.legend(handles=handles, loc="lower left", fontsize=9, frameon=True,
          facecolor="white", edgecolor="#ddddd5")
fig.tight_layout()

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures", "nm_ladder_generalist_scatter_regret")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out + ".pdf", bbox_inches="tight")
print("wrote", out + ".pdf")
