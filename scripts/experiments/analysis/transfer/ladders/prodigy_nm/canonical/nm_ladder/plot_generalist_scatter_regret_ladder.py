#!/usr/bin/env python3
"""Ladder-ONLY zoom of the regret scatter (companion to plot_generalist_scatter_regret.py).
Drops the single-source specialists and zooms onto the 8 merged-ladder rungs. Axes have
EQUAL length (same span on x and y) so the scale is identical and y=x is a true 45 deg
line. x = home-turf regret, y = overall regret (both mean Δ to best, AUC points; 0 = best).
Writes PNG (300 dpi) + PDF into figures/2d_scatter/."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (label, home-turf regret, overall regret, rung) -- same values as the full regret scatter
LADDER = [
    ("L1", 0.00, -8.13, 1), ("L2", -0.18, -7.89, 2), ("L3", -0.66, -7.94, 3), ("L4", -0.95, -6.93, 4),
    ("L5", -1.31, -5.79, 5), ("L6", -1.69, -3.84, 6), ("L7", -1.88, -3.88, 7), ("L8", -2.00, -2.00, 8),
]
g1, g2 = np.array([181, 224, 134]) / 255, np.array([40, 86, 14]) / 255
ramp = lambda k: tuple(g1 + (g2 - g1) * (k - 1) / 7)

xs = [m[1] for m in LADDER]
ys = [m[2] for m in LADDER]
cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
half = max(max(xs) - min(xs), max(ys) - min(ys)) * 1.18 / 2      # equal span, ~18% margin
XLIM, YLIM = (cx - half, cx + half), (cy - half, cy + half)

fig, ax = plt.subplots(figsize=(6.4, 6.4))
d = (min(XLIM[0], YLIM[0]), max(XLIM[1], YLIM[1]))
ax.plot(d, d, "--", color="#9a988e", lw=1.3, zorder=1)                       # y = x (true 45 deg)
ax.axvline(0, ls=":", color="#c7c5bc", lw=1.1, zorder=1)                     # home regret = 0
ax.annotate("y = x", (-3.5, -3.5), rotation=45, rotation_mode="anchor",
            ha="center", va="bottom", fontsize=9, color="#8a887e")
ax.annotate("ideal (0,0)\nis up-right ↗", (XLIM[1] - 0.2, YLIM[1] - 0.2),
            ha="right", va="top", fontsize=8.5, color="#8a887e")

LAB = {1: (9, -6, "left"), 2: (9, 10, "left"), 3: (10, -2, "left"), 4: (-10, 4, "right"),
       5: (-10, -2, "right"), 6: (-10, 9, "right"), 7: (-11, -9, "right")}
ax.plot(xs, ys, "-", color="#5b9a2f", lw=1.8, alpha=0.55, zorder=2)
for lb, x, y, k in LADDER:
    big = k == 8
    ax.scatter(x, y, s=175 if big else 92, color=ramp(k),
               edgecolor="white" if big else "none", linewidth=1.5, zorder=4)
    if big:
        ax.annotate("L8 · all-8", (x, y), xytext=(-12, 10), textcoords="offset points",
                    fontsize=10, color="#1b3a09", fontweight="bold", ha="right")
    else:
        dx, dy, ha = LAB[k]
        ax.annotate(lb, (x, y), xytext=(dx, dy), textcoords="offset points",
                    fontsize=9, color="#2f5a10", ha=ha, va="center",
                    arrowprops=dict(arrowstyle="-", color="#bcccab", lw=0.7))

ax.set_xlim(XLIM)
ax.set_ylim(YLIM)
ax.set_aspect("equal", adjustable="box")
ax.set_xlabel("home-turf regret   (mean Δ to best, AUC pts)", fontsize=10.5)
ax.set_ylabel("overall regret   (mean Δ to best over all 8, AUC pts)", fontsize=10.5)
ax.set_title("Merged ladder in regret space — zoom, equal scale (NM, matched-40k)",
             fontsize=11, pad=10)
ax.tick_params(labelsize=9)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.grid(True, color="#efeee9", lw=0.6, zorder=0)
fig.tight_layout()

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures", "2d_scatter",
                   "nm_ladder_generalist_scatter_regret_ladder")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out + ".png", dpi=300, bbox_inches="tight")
fig.savefig(out + ".pdf", bbox_inches="tight")
print("wrote", out + ".png")
print("wrote", out + ".pdf")
