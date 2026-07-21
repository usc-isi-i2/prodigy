#!/usr/bin/env python3
"""Generalist-vs-specialist scatter for the NM ladder + single-source matrix.
Equal-scale axes (home-turf AUC vs breadth); y=x is the perfect-generalist line.
Saves PNG + PDF to ~/Downloads."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# (label, home-turf AUC, breadth AUC, rung)  rung 0 = single-source specialist
LADDER = [
    ("L1", .948, .861, 1), ("L2", .962, .863, 2), ("L3", .945, .863, 3), ("L4", .933, .873, 4),
    ("L5", .931, .884, 5), ("L6", .930, .904, 6), ("L7", .929, .903, 7), ("L8", .922, .922, 8),
]
SPEC = [  # key, home-turf(diagonal), breadth, legible name, (dx,dy) label offset in pts, ha
    ("ukr",      .947, .861, "Ukraine",            (-9, -13), "right"),
    ("covid",    .981, .863, "COVID",              (0, 11),   "center"),
    ("midterm",  .925, .797, "Midterm",            (8, -2),   "left"),
    ("cov_pol",  .914, .693, "COVID-political",    (8, -3),   "left"),
    ("elec20",   .952, .687, "Election 2020",      (8, -3),   "left"),
    ("ukr_susp", .964, .767, "Ukraine-suspended",  (8, 3),    "left"),
    ("twibot20", .949, .834, "TwiBot-20",          (8, 6),    "left"),
    ("cp_hk",    .905, .721, "CP-HK",              (-8, 2),   "right"),
]

g1, g2 = np.array([181, 224, 134]) / 255, np.array([40, 86, 14]) / 255
ramp = lambda k: tuple(g1 + (g2 - g1) * (k - 1) / 7)
CORAL, CORAL_TXT = "#D85A30", "#8a3517"
LIM = (0.66, 0.99)

fig, ax = plt.subplots(figsize=(7.4, 6.2))
ax.plot(LIM, LIM, "--", color="#9a988e", lw=1.3, zorder=1)
ax.annotate("y = x  (perfect generalist)", (0.734, 0.734), rotation=45,
            rotation_mode="anchor", ha="center", va="bottom", fontsize=9.5, color="#8a887e")

# ladder path + points
lx = [m[1] for m in LADDER]
ly = [m[2] for m in LADDER]
ax.plot(lx, ly, "-", color="#5b9a2f", lw=1.7, alpha=0.55, zorder=2)
for lb, x, y, k in LADDER:
    big = k == 8
    ax.scatter(x, y, s=150 if big else 78, color=ramp(k),
               edgecolor="white" if big else "none", linewidth=1.4, zorder=4)
ax.annotate("all-8", (0.922, 0.922), xytext=(-10, 8), textcoords="offset points",
            fontsize=9.5, color="#1b3a09", fontweight="bold", ha="right")

# specialists + legible labels
for key, x, y, name, off, ha in SPEC:
    ax.scatter(x, y, s=78, color=CORAL, edgecolor="white", linewidth=0.8, zorder=4)
    ax.annotate(name, (x, y), xytext=off, textcoords="offset points",
                fontsize=9.5, color=CORAL_TXT, ha=ha, va="center", zorder=6)

ax.set_xlim(LIM); ax.set_ylim(LIM); ax.set_aspect("equal", adjustable="box")
ax.set_xlabel("home-turf AUC  (mean over the graphs the model trained on)", fontsize=11)
ax.set_ylabel("breadth  (mean AUC over all 8 graphs)", fontsize=11)
ax.set_title("Generalist vs specialist — NM (3-shot / 30-way, matched-40k)", fontsize=12.5, pad=10)
ax.tick_params(labelsize=9)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.grid(True, color="#ecece6", lw=0.7, zorder=0)

handles = [
    Line2D([0], [0], marker="o", color="none", markerfacecolor=ramp(4),
           markersize=9, label="merged ladder  (L1 Ukraine → L8 all-8, light→dark)"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor=CORAL,
           markersize=9, label="single-source specialist"),
    Line2D([0], [0], ls="--", color="#9a988e", label="y = x  (breadth = home-turf)"),
]
ax.legend(handles=handles, loc="upper left", fontsize=9, frameon=True,
          facecolor="white", edgecolor="#ddddd5")
fig.tight_layout()

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures", "nm_ladder_generalist_scatter")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out + ".pdf", bbox_inches="tight")
print("wrote", out + ".pdf")
