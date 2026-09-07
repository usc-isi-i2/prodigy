"""Plot verified fixed support-centered readouts; no pooled task averaging.

Public: publickg_pretraining_control_fresh_20260907/summary.json.
Social: social_centered_readout_20260907/summary.json, native blocked,
fresh fixed-four-target mean. Cora: cora_readout_breadth_retry1_20260907,
blocked checkpoint. Exact values are transcribed from these completed outputs.
Panels differ in task/protocol; compare stages within panels, not task difficulty.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    rows = [
        ("Public KG", "Wiki → FB15K-237 · fresh 500 episodes",
         [.716100, .802800, .735350]),
        ("Social transfer", "Blocked training · fresh four-target mean",
         [.731363932, .808756510, .786946615]),
        ("Citation transfer", "Social → Cora · blocked · 128 episodes",
         [.571149554, .438895089, .407924107]),
    ]
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)
    colors = ["#707A89", "#187F89", "#D3753D"]
    for ax, (title, subtitle, values) in zip(axes, rows):
        values = 100 * np.array(values)
        ax.bar(range(3), values, color=colors, width=.65)
        for x, value in enumerate(values):
            ax.text(x, value + 1.0, f"{value:.1f}", ha="center", fontsize=11)
        ax.set_xticks(range(3), ["Text-only", "Intermediate", "Native"])
        ax.set_ylim(0, 100)
        ax.set_title(title, loc="left", fontweight="bold", pad=28)
        ax.text(0, 1.025, subtitle, transform=ax.transAxes, fontsize=9)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=.15)
    axes[0].set_ylabel("Accuracy (%)")
    fig.text(.075, .025,
             "Text and intermediate: fixed support-centered ridge. "
             "Compare stages within each panel; protocols differ. No seed-level uncertainty shown.",
             fontsize=9)
    fig.subplots_adjust(left=.075, right=.99, top=.79, bottom=.16, wspace=.24)
    output = Path(__file__).parent / "figures"
    output.mkdir(exist_ok=True)
    fig.savefig(output / "transfer_stages.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
