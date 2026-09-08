#!/usr/bin/env python3
"""Plot public aggregate cluster diagnostics; no raw bios/ids are read."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
LABELS = ["C0 · General / conversational", "C1 · Family roles",
          "C2 · Professional / research", "C3 · Sports",
          "C4 · Trump / patriot vocabulary*", "C5 · Creative / hobbies",
          "C6 · Democrat / Resist vocabulary", "C7 · Journalism / media"]


def main():
    data = json.loads((HERE/"data/query_bio_cluster_summary.json").read_text())
    clusters = data["runs"]["8"]["clusters"]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    y = np.arange(8)
    for ax, stream in zip(axes, ["original", "fresh"]):
        ukr = np.array([c["streams"][stream]["ukr_error"]*100 for c in clusters])
        hk = np.array([c["streams"][stream]["hk_error"]*100 for c in clusters])
        ax.hlines(y, ukr, hk, color=".8", lw=2)
        ax.scatter(ukr, y, color="#1673a6", marker="o", label="Ukraine-trained")
        ax.scatter(hk, y, color="#c85321", marker="s", label="Hong Kong-trained")
        for j, (u, h) in enumerate(zip(ukr, hk)):
            ax.annotate(f"{u:.1f}", (u,j), xytext=(0,-13), textcoords="offset points", ha="center", fontsize=9)
            ax.annotate(f"{h:.1f}", (h,j), xytext=(0,7), textcoords="offset points", ha="center", fontsize=9)
        ax.set_xlim(0, 50)
        ax.set_ylim(7.6, -.6)
        ax.set_yticks(y, LABELS)
        ax.set_xlabel("Query error rate (%)")
        ax.set_title(f"{stream.capitalize()} episodes · 3,072 queries")
        ax.grid(axis="x", alpha=.2)
    axes[1].legend(frameon=False, loc="upper right", bbox_to_anchor=(1.02,1.18))
    fig.suptitle("Bio embedding clusters predict different error rates", y=.99, fontsize=14)
    fig.text(.30,.015,"*C4 mixes supportive and anti-Trump bios. Clusters fit on original unique nodes only.", fontsize=10)
    fig.subplots_adjust(left=.30, right=.98, bottom=.12, top=.85, wspace=.15)
    fig.savefig(HERE/"figures/query_bio_cluster_errors.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
