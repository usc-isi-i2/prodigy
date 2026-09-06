"""Political support-path replication and its bot-detection counterexample."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot(data, output):
    seeds = pd.read_csv(data / "support_path_seed_effects.csv")
    roles = pd.read_csv(data / "role_context_cells.csv")
    bot = roles[roles.target.eq("twibot20") & roles.variant.eq("edges_support") & roles.foreign_source]
    if len(seeds) != 12 or len(bot) != 16:
        raise ValueError("complete political seed and bot foreign-source panels required")
    names = {"cp_hk": "hongkong", "covid": "covid", "covid_political": "covid-political",
        "election2020": "election2020-political", "facebook_page_reference": "facebook-page-reference",
        "midterm": "midterm", "ukr_rus": "ukraine", "ukr_rus_suspended": "ukraine-suspended"}
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.7), gridspec_kw={"width_ratios": [1, 1.5]})
    for source, color, offset in (("cp_hk", "#B54A46", -.12), ("ukr_rus", "#197A75", .12)):
        for stream, marker, shift in (("original", "o", -.025), ("fresh", "x", .025)):
            d = seeds[seeds.source.eq(source) & seeds.stream.eq(stream)].sort_values("seed")
            axes[0].scatter(d.seed + offset + shift, 100*d.delta_roc_auc,
                color=color, marker=marker, s=48, label=f"{names[source]} · {'original' if stream == 'original' else 'second'}")
    axes[0].set_xticks(range(3), ["Seed 0", "Seed 1", "Seed 2"])
    axes[0].set_title("covid-political: the hongkong penalty replicates", fontsize=11)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(.02, .13))
    sources = sorted(bot.source.unique())
    for stream, marker, offset in (("original", "o", -.1), ("fresh", "x", .1)):
        d = bot[bot.stream.eq(stream)].set_index("source").loc[sources]
        axes[1].scatter(np.arange(len(sources)) + offset, 100*d.delta_roc_auc,
            color="#344DB1", marker=marker, s=42,
            label="Original stream" if stream == "original" else "Second stream")
    axes[1].set_xticks(range(len(sources)), [names[s] for s in sources], rotation=35, ha="right", fontsize=8)
    axes[1].set_title("twibot20: the same intervention harms all foreign donors", fontsize=11)
    axes[1].legend(frameon=False, fontsize=8, loc="lower left")
    for ax in axes:
        ax.axhline(0, color="#687888", linewidth=.9)
        ax.grid(axis="y", alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylabel("AUC-point change after removing support edges")
    fig.suptitle("Support-side graph processing can help or hurt without changing the query inputs", fontsize=13)
    fig.text(.5, .015, "Political controls: query vectors are bit-exact; changed label vectors reproduce every altered logit.\n"
        "Three political initialization seeds; bot panel uses nine-source historical specialists with its target-source diagonal excluded.",
        ha="center", fontsize=9, color="#50545E")
    fig.tight_layout(rect=(0, .10, 1, .93))
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(output.with_suffix(suffix), dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--data", type=Path, default=Path(__file__).with_name("data"))
    p.add_argument("--output", type=Path, default=Path(__file__).with_name("figures") / "support_path_publication")
    args = p.parse_args()
    plot(args.data, args.output)
