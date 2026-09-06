"""Paired cross-task changes, preserving both declared checkpoint rules."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NAMES = {
    "baseline": "Baseline",
    "exposure": "Proportional exposure",
    "schedule": "Blocked schedule",
    "composition": "Cross-source classes",
    "centers": "Degree-balanced centers",
    "eligibility": "Low-degree eligibility",
    "positives": "Uniform-neighbor positives",
    "negatives": "Degree-matched classes",
    "context": "One-hop training context",
    "optimization": "Gradient normalization",
    "objective": "Auxiliary reconstruction",
    "region_adaptive": "Adaptive degree sampling",
    "coverage": "Cyclic center coverage",
    "budget": "Budget: baseline repeat",
    "combined": "Recipe: objective repeat",
}


def main():
    root = Path(__file__).parent
    receipt = json.loads((root / "data/campaign_cls_validation.json").read_text())
    if receipt["rows"] != 5100 or not receipt["selected_nm_hashes_and_steps_match"]:
        raise ValueError("complete validated cross-task analysis required")
    deltas = pd.read_csv(root / "data/campaign_cls_deltas.csv")
    nm = pd.read_csv(root / "data/campaign_nm_cls_selected_deltas.csv")
    models = [f"nmi_{name}_r8_s0" for name in NAMES]
    target = deltas[(deltas.dataset == "twibot20") & (deltas.decoder == "full_model")]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 8.2), sharey=True)
    fig.subplots_adjust(left=.23, right=.985, top=.86, bottom=.17, wspace=.13)
    ypos = np.arange(len(models))
    for ax, role, title in zip(axes, ("common6000", "source_val_selected"),
                               ("Common 6,000 updates", "Original source-validation selection")):
        values = target[target.checkpoint_role == role].pivot(
            index="original_model_id", columns="stream", values="delta_roc_auc").loc[models]
        if values.shape != (15, 2) or not np.isfinite(values.to_numpy()).all():
            raise ValueError("incomplete figure grid")
        ax.axvline(0, color="#8a9299", lw=1)
        for stream, color, marker, offset in (("original", "#2166ac", "o", -.10),
                                               ("fresh", "#d97823", "D", .10)):
            ax.scatter(100 * values[stream], ypos + offset, color=color, marker=marker,
                       s=34, label=f"Classification: {stream} episodes", zorder=3)
        if role == "source_val_selected":
            selected = nm[(nm.dataset == "twibot20") & (nm.stream == "original")].set_index("original_model_id")
            ax.scatter(100 * selected.loc[models, "delta_nm_auc"], ypos, marker="x", color="#333333",
                       s=35, label="Neighbor matching: original test", zorder=4)
        ax.set_title(title, loc="left", fontsize=12, pad=13)
        ax.set_xlim(-10.5, 3.3)
        ax.set_xticks([-10, -7.5, -5, -2.5, 0, 2.5])
        ax.set_xlabel("ΔAUC from the same-rule baseline (percentage points)", labelpad=10)
        ax.set_yticks(ypos, NAMES.values())
        ax.grid(axis="y", color="#edf0f2")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].invert_yaxis()
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.62, .071), ncol=3,
               frameon=False, fontsize=9)
    fig.suptitle("An NM improvement is not necessarily a bot-classification improvement", fontsize=17, x=.52, y=.965)
    fig.text(.23, .911, "TwiBot20 was excluded from all training sources and checkpoint selection.", fontsize=11)
    fig.text(.23, .025,
             "One training seed; fixed inputs within each classification stream. Points are outcomes, not confidence intervals.\n"
             "NM comparisons use identical selected checkpoint files only (right). Repeat runs are not extra seeds; tasks are not averaged.",
             fontsize=9, color="#424a52")
    (root / "figures").mkdir(exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(root / f"figures/campaign_nm_cls_comparison.{suffix}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
