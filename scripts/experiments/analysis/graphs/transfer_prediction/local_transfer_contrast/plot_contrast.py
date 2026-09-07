"""Plot the fixed local-transfer contrast from committed aggregate tables."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).parent
DATA = ROOT / "data"
FIGURES = ROOT / "figures"


def main():
    FIGURES.mkdir(exist_ok=True)
    decoders = pd.read_csv(DATA / "decoder_summary.csv")
    heuristics = pd.read_csv(DATA / "heuristic_results.csv")
    fresh = decoders[decoders.stream.eq("fresh")]
    model_names = {"ss_twibot20": "TwiBot source", "ss_election2020": "Election source"}
    stages = ["raw_joint/ridge", "U1_pre_meta/ridge", "M2_post_meta/ridge", "full_model"]
    stage_names = ["Raw + support\nridge", "Encoder +\nsupport ridge", "Metagraph +\nsupport ridge", "Full model"]

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.5), gridspec_kw={"width_ratios": [1.08, 1, 1]})
    x = np.arange(len(stages))
    for model, label in model_names.items():
        rows = fresh[fresh.model.eq(model)].set_index("decoder").loc[stages]
        axes[0].plot(x, rows.accuracy, marker="o", linewidth=2, label=label)
    axes[0].set_xticks(x, stage_names)
    axes[0].set_ylim(.55, .86)
    axes[0].set_ylabel("Validation accuracy")
    axes[0].set_title("A. Source ranking emerges after the shared input", loc="left", weight="bold")
    axes[0].legend(frameon=False)

    methods = ["fixed_b", "u1_alignment_tie_b", "learned_stage_consensus_tie_b",
               "raw_alignment_tie_b", "raw_joint_ridge_direct"]
    labels = ["Best fixed source", "U1 agreement", "Stage consensus", "Raw-anchor routing", "Raw readout"]
    rows = heuristics[heuristics.stream.eq("fresh")].set_index("method").loc[methods]
    colors = ["#59636e", "#89a8c7", "#4f83b6", "#d4743c", "#c9a227"]
    baseline = float(rows.loc["fixed_b", "accuracy"])
    lower = rows.accuracy - (baseline + rows.episode_bootstrap_gain_ci_low)
    upper = (baseline + rows.episode_bootstrap_gain_ci_high) - rows.accuracy
    axes[1].barh(
        np.arange(len(rows)), rows.accuracy, color=colors,
        xerr=np.vstack((lower.clip(lower=0), upper.clip(lower=0))),
        error_kw={"ecolor": "#252525", "capsize": 2, "linewidth": 1},
    )
    axes[1].set_yticks(np.arange(len(rows)), labels)
    axes[1].invert_yaxis()
    axes[1].set_xlim(.64, .85)
    axes[1].set_xlabel("Validation accuracy")
    axes[1].set_title("B. Transfer health enables label-free recovery", loc="left", weight="bold")
    for index, value in enumerate(rows.accuracy):
        axes[1].text(value + .004, index, f"{value:.3f}", va="center")

    multi = pd.read_csv(DATA / "multi_health_results.csv")
    multi = multi[multi.method.eq("u1_agreement")].copy()
    target_labels = {
        "covid_political": "COVID-pol.", "election2020": "Election",
        "facebook_page_reference": "Facebook", "twibot20": "TwiBot-20",
        "ukr_rus_suspended": "Suspended",
    }
    order = list(target_labels)
    multi = multi.set_index("target").loc[order]
    gain = 100 * multi.accuracy_gain_over_fixed_best
    lower = gain - 100 * multi.episode_bootstrap_gain_ci_low
    upper = 100 * multi.episode_bootstrap_gain_ci_high - gain
    y = np.arange(len(multi))
    axes[2].errorbar(gain, y, xerr=np.vstack((lower, upper)), fmt="o", color="#d4743c",
                     ecolor="#59636e", capsize=3, markersize=6)
    axes[2].axvline(0, color="#a8a8a8", linewidth=1)
    axes[2].set_yticks(y, [target_labels[item] for item in order])
    axes[2].invert_yaxis()
    axes[2].set_xlabel("Accuracy gain (points)")
    axes[2].set_title("C. U1 health routing replicates", loc="left", weight="bold")
    axes[2].set_xlim(-2.8, 8.4)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURES / f"local_transfer_contrast.{suffix}", dpi=220, bbox_inches="tight")


if __name__ == "__main__":
    main()
