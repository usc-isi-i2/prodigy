"""Publication-facing source contrasts from the focused matched-family replay."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FAMILIES = ["prodigy_full", "prodigy_prototype", "samgpt_prototype"]
TITLES = ["PRODIGY: learned readout", "PRODIGY: prototype readout", "SAMGPT: prototype readout"]
TARGETS = ["covid_political", "election2020", "facebook_page_reference", "twibot20", "ukr_rus_suspended"]
SHORT = ["Political", "Election", "Facebook", "TwiBot", "Suspension"]


def analyze(root, figures):
    done = json.loads((root / "DONE.json").read_text())
    if not done["complete"] or done["smoke"] or done["metric_cells"] != 290:
        raise ValueError("complete matched replay required")
    cells = pd.read_csv(root / "cells.csv")
    if len(cells) != 290 or cells.duplicated(["target", "stream", "source", "decoder"]).any():
        raise ValueError("duplicated or missing result cells")
    if set(cells.target) != set(TARGETS) or set(cells.stream) != {"original", "fresh"}:
        raise ValueError("incomplete target/stream panel")
    untrained = {}
    for stream, name in (("original", "replay_cells.csv"), ("fresh", "fresh_replay_cells.csv")):
        reference = pd.read_csv(root.parent / name)
        reference = reference[(reference.model_id == "random_init") & (reference.variant == "baseline")]
        for target in TARGETS:
            group = reference[reference.dataset == target]
            expected = cells[(cells.target == target) & (cells.stream == stream)].episode_fingerprint.unique()
            if len(expected) != 1 or not group.episode_fingerprint.eq(expected[0]).all():
                raise ValueError("untrained reference uses different episodes")
            untrained[(target, stream)] = group.set_index("decoder").roc_auc.to_dict()
    contrasts, controls = [], []
    for (target, stream), group in cells.groupby(["target", "stream"]):
        if group.episode_fingerprint.nunique() != 1:
            raise ValueError("model-family episode mismatch")
        table = group[group.source != "none"].pivot(index="source", columns="decoder", values="roc_auc")
        if table.shape != (9, 3) or table.isna().any().any():
            raise ValueError("nine source specialists per family required")
        for source_a, source_b in (("ukr_rus", "cp_hk"), ("covid", "cp_hk"), ("ukr_rus", "twibot20")):
            for family in FAMILIES:
                contrasts.append({"target": target, "stream": stream, "source_a": source_a, "source_b": source_b,
                    "family": family, "auc_a": table.loc[source_a, family], "auc_b": table.loc[source_b, family],
                    "auc_gap": table.loc[source_a, family] - table.loc[source_b, family]})
        raw = group[group.source == "none"].set_index("decoder").roc_auc
        for family in FAMILIES:
            foreign = table.drop(index=target, errors="ignore")[family]
            controls.append({"target": target, "stream": stream, "family": family,
                "foreign_mean_auc": foreign.mean(), "foreign_max_auc": foreign.max(),
                "raw_768_auc": raw.raw_768_prototype, "raw_50_auc": raw.raw_50_prototype,
                "prodigy_untrained_U_auc": untrained[(target, stream)]["U1_pre_meta/prototype"],
                "prodigy_untrained_full_auc": untrained[(target, stream)]["full_model"],
                "foreign_mean_minus_raw768": foreign.mean()-raw.raw_768_prototype,
                "foreign_mean_minus_raw50": foreign.mean()-raw.raw_50_prototype})
    contrasts = pd.DataFrame(contrasts)
    contrasts.to_csv(root / "source_contrasts.csv", index=False)
    pd.DataFrame(controls).to_csv(root / "raw_controls.csv", index=False)
    figures.mkdir(parents=True, exist_ok=True)
    order = ["ukr_rus", "covid", "midterm", "covid_political", "election2020",
             "ukr_rus_suspended", "twibot20", "cp_hk", "facebook_page_reference"]
    names = ["Ukraine", "COVID", "Midterm", "Political", "Election", "Suspension", "TwiBot", "Hong Kong", "Facebook"]
    for stream in ("original", "fresh"):
        fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.8), sharey=True, layout="constrained")
        part = cells[(cells.stream == stream) & (cells.source != "none")]
        limit = max(abs(part.assign(centered=part.roc_auc - part.groupby(["decoder", "target"]).roc_auc.transform("mean")).centered))
        for ax, family, title in zip(axes, FAMILIES, TITLES):
            raw = part[part.decoder == family].pivot(index="source", columns="target", values="roc_auc").loc[order, TARGETS]
            centered = raw - raw.mean(axis=0)
            im = ax.imshow(centered, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
            for i in range(9):
                for j in range(5):
                    ax.text(j, i, f"{raw.iloc[i,j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(centered.iloc[i,j]) > .62*limit else "black")
            ax.set_title(title, fontsize=11)
            ax.set_xticks(range(5), SHORT, rotation=35, ha="right")
            ax.set_yticks(range(9), names)
        fig.colorbar(im, ax=axes, shrink=.8, label="AUC relative to this family's nine-source mean")
        fig.suptitle(f"Identical target episodes; source rankings depend on learner/readout ({stream})", fontsize=12)
        fig.savefig(figures / f"cross_model_sources_{stream}.png", dpi=180)
        plt.close(fig)
    fig, axes = plt.subplots(2, 5, figsize=(13, 6), sharey=True, layout="constrained")
    for i, stream in enumerate(("original", "fresh")):
        for j, (target, short) in enumerate(zip(TARGETS, SHORT)):
            ax = axes[i, j]
            part = cells[(cells.stream == stream) & (cells.target == target)]
            raw = float(part[part.decoder == "raw_768_prototype"].roc_auc.iloc[0])
            table = part[part.source != "none"].pivot(index="source", columns="decoder", values="roc_auc")
            table = table.drop(index=target, errors="ignore")
            values = np.column_stack([np.full(len(table), raw), table.prodigy_prototype, table.prodigy_full])
            for row in values:
                ax.plot(range(3), row, color="#a9adb2", alpha=.7, linewidth=.8)
            mean = values.mean(axis=0)
            ax.plot(range(3), mean, "o-", color="#126782", linewidth=2.5, markersize=5)
            ax.scatter([1], [untrained[(target, stream)]["U1_pre_meta/prototype"]],
                marker="x", s=60, linewidths=2, color="#b65c1c", zorder=5)
            for k, value in enumerate(mean):
                ax.annotate(f"{value:.3f}", (k, value), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=8, color="#08455c")
            ax.axhline(.5, color="#c3c3c3", linestyle=":", linewidth=.7)
            ax.set_ylim(.44, 1.025)
            ax.set_xticks(range(3), ["Raw", "Encoded", "Full"])
            ax.set_title(short if i == 0 else "", fontsize=11)
            ax.spines[["top", "right"]].set_visible(False)
            if j == 0:
                ax.set_ylabel(f"{stream.capitalize()} episodes\nROC-AUC")
    fig.suptitle("Available signal versus final decisions: same supports, same queries, fixed checkpoints", fontsize=12)
    fig.supxlabel("Raw/Encoded: prototype rule; Full: learned inference. Gray: foreign sources; blue: mean; orange ×: untrained encoder.", fontsize=9)
    fig.savefig(figures / "cross_model_transfer_stages.png", dpi=180)
    plt.close(fig)
    print(contrasts[contrasts.source_b == "cp_hk"].pivot_table(
        index=["target", "stream", "source_a"], columns="family", values="auc_gap").round(4).to_string())
    print(pd.DataFrame(controls).round(4).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--input", type=Path, default=Path(__file__).with_name("data") / "cross_model_matching")
    parser.add_argument("--figures", type=Path, default=Path(__file__).with_name("figures"))
    args = parser.parse_args()
    analyze(args.input, args.figures)
