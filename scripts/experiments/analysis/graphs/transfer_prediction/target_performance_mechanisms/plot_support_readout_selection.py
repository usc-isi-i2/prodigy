"""All-target view of the fixed support-only readout test, including failures."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot(root, output):
    data = pd.read_csv(root / "target_means.csv")
    targets = ["covid_political", "election2020", "facebook_page_reference", "twibot20", "ukr_rus_suspended"]
    labels = ["covid-\npolitical", "election2020-\npolitical", "facebook-page-\nreference", "twibot20", "ukraine-\nsuspended"]
    styles = {"raw_scaled": ("Raw prototype", "#687888", "o"),
        "prototype_scaled": ("Encoder prototype", "#178A72", "s"),
        "full_scaled": ("Learned inference", "#AD6684", "^"),
        "selected_scaled": ("Support-only selection", "#183BBA", "D")}
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.2), sharex=True, sharey=True)
    for row, family in enumerate(("prodigy", "samgpt")):
        for col, stream in enumerate(("original", "fresh")):
            ax = axes[row, col]
            subset = data[(data.family == family) & (data.stream == stream)]
            for condition, (label, color, marker) in styles.items():
                values = subset[subset.condition == condition].set_index("target").roc_auc
                if values.empty:
                    continue
                ax.plot(np.arange(5), values.reindex(targets), color=color, marker=marker,
                    linewidth=2.2 if condition == "selected_scaled" else 1.2,
                    markersize=5.5, alpha=1 if condition == "selected_scaled" else .85, label=label)
            ax.axhline(.5, color="#B8BEC8", linewidth=.8, linestyle="--")
            name = "PRODIGY" if family == "prodigy" else "SAMGPT frozen base"
            ax.set_title(f"{name} · {'original' if col == 0 else 'second'} episode stream", fontsize=11)
            ax.set_xticks(range(5), labels, fontsize=9)
            ax.set_ylim(.46, 1.01)
            ax.set_yticks(np.arange(.5, 1.01, .1))
            ax.grid(axis="y", alpha=.18)
            ax.spines[["top", "right"]].set_visible(False)
            if col == 0:
                ax.set_ylabel("Mean AUC · eight foreign sources")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(.5, .94), fontsize=10)
    fig.suptitle("Support-only selection adapts across tasks, but misses the best rule within tasks", fontsize=13, y=.985)
    fig.text(.5, .015, "Same 20 supports; all fixed rules receive identical support-only score scaling.\n"
        "Connected points show task profiles, not a continuous axis. Two streams are not additional training seeds.",
        ha="center", fontsize=9, color="#50545E")
    fig.tight_layout(rect=(0, .06, 1, .89))
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(output.with_suffix(suffix), dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--input", type=Path, default=Path(__file__).with_name("data") / "support_readout_selection")
    p.add_argument("--output", type=Path, default=Path(__file__).with_name("figures") / "support_readout_selection")
    args = p.parse_args()
    plot(args.input, args.output)
