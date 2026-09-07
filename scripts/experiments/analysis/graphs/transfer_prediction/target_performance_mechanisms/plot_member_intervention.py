"""Show the declared manipulation and primary contrast without selecting outcomes."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    root = Path(__file__).parent
    receipt = json.loads((root / "data/member_intervention_validation.json").read_text())
    if receipt["models"] != 24 or receipt["replay_rows"] != 4080:
        raise ValueError("complete validated factorial required")
    primary = pd.read_csv(root / "data/member_primary_contrast.csv")
    exposure = pd.read_csv(root / "data/member_consumed_exposure.csv")
    expected = {(s, seed) for s in ("original", "fresh") for seed in range(3)}
    if len(primary) != 6 or set(map(tuple, primary[["stream", "seed"]].to_numpy())) != expected:
        raise ValueError("primary seed/stream grid incomplete")
    exposure["retention"] = exposure.policy.str.split("_").str[0]
    grouped = exposure.groupby(["source", "seed", "retention"]).cross_class_duplicate_fraction
    if not grouped.size().eq(2).all() or (grouped.max() - grouped.min()).abs().max() > 1e-10:
        raise ValueError("matched role policies disagree about repeated-member exposure")
    repeated = grouped.mean().reset_index()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 5.4))
    fig.subplots_adjust(left=.065, right=.985, top=.76, bottom=.23, wspace=.42)
    sources = ("ukr_rus", "cp_hk")
    for source_idx, source in enumerate(sources):
        for retention, color, offset in (("lowest", "#777777", -.15), ("uniform", "#258575", .15)):
            rows = repeated[(repeated.source == source) & (repeated.retention == retention)].sort_values("seed")
            if len(rows) != 3:
                raise ValueError("missing exposure seed")
            axes[0].scatter(source_idx + offset + .035 * (rows.seed - 1), 100 * rows.cross_class_duplicate_fraction,
                            color=color, s=38, label=retention if source_idx == 0 else None, zorder=3)
    axes[0].set_title("Did the manipulation change exposure?", fontsize=11, loc="left", pad=12)
    axes[0].set_ylabel("Repeated member positions within episodes (%)")
    axes[0].set_ylim(bottom=0)
    axes[0].legend(labels=["Lowest IDs", "Uniform endpoints"], frameon=False, fontsize=9)
    for stream, color, offset in (("original", "#2166ac", -.12), ("fresh", "#d97823", .12)):
        rows = primary[primary.stream == stream].sort_values("seed")
        for source_idx, source in enumerate(sources):
            axes[1].scatter(source_idx + offset + .035 * (rows.seed - 1), 100 * rows[source],
                            color=color, s=38, label=stream if source_idx == 0 else None, zorder=3)
        axes[2].scatter(rows.seed + offset, 100 * rows.hong_kong_minus_ukraine_retention_effect,
                        color=color, s=42, label=stream, zorder=3)
    axes[1].set_title("Retention effect on target performance", fontsize=11, loc="left", pad=12)
    axes[1].set_ylabel("Mean primary-panel ΔAUC (percentage points)")
    axes[2].set_title("Primary: Hong Kong minus Ukraine", fontsize=11, loc="left", pad=12)
    axes[2].set_ylabel("Difference in retention effects (percentage points)")
    axes[2].set_xticks([0, 1, 2], ["Seed 0", "Seed 1", "Seed 2"])
    axes[2].set_xlim(-.45, 2.45)
    axes[2].legend(title="Test episode stream", frameon=False, fontsize=9, title_fontsize=9)
    for ax in axes[:2]:
        ax.set_xticks([0, 1], ["Ukraine", "Hong Kong"])
        ax.set_xlim(-.45, 1.45)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#edf0f2")
    for ax in axes[1:]:
        ax.axhline(0, color="#92999f", lw=1)
    fig.suptitle("Does reducing repeated training members improve transfer?", fontsize=17, y=.96)
    fig.text(.065, .86, "24 matched CPU models · two source graphs · four retention/role policies · three training seeds", fontsize=11)
    fig.text(.065, .075,
             "Each point is one training seed, not a confidence interval. Retention effects average both role orders and the fixed\n"
             "COVID Political / Facebook / TwiBot panel. A positive source difference need not mean an absolute performance gain.\n"
             "Both streams use the same terminal weights at update 2,500; fresh episodes are not independent domains.", fontsize=9)
    (root / "figures").mkdir(exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(root / f"figures/member_policy_primary.{suffix}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
