"""
Plot overperformer sweep results from wandb.
Pulls all runs whose name starts with 'eval_overperformer_' from the graph-clip project.

Usage:
    python scripts/plot_overperformer_sweep.py
    python scripts/plot_overperformer_sweep.py --entity my_entity
"""

import argparse
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import wandb

MODELS = [
    "pretrain_instagram_nm_11_03_2026_15_42_26",
    "pretrain_instagram_nm_11_03_2026_14_44_15",
    "pretrain_instagram_nm_11_03_2026_14_09_51",
]
MODEL_SHORT = {m: f"model_{i+1}" for i, m in enumerate(MODELS)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity",  default=None, help="wandb entity (username or team)")
    parser.add_argument("--project", default="graph-clip")
    parser.add_argument("--out",     default="overperformer_sweep.pdf")
    args = parser.parse_args()

    api = wandb.Api()
    path = f"{args.entity}/{args.project}" if args.entity else args.project
    runs = api.runs(path, filters={"display_name": {"$regex": "^eval_overperformer_"}})

    records = []
    for run in runs:
        name = run.name  # e.g. eval_overperformer_pretrain_..._5shot_3way_timestamp
        m = re.search(r"_(\d+)shot_(\d+)way", name)
        if not m:
            continue
        shot, way = int(m.group(1)), int(m.group(2))

        # identify which pretrained model
        model_key = next((k for k in MODELS if k in name), None)
        if model_key is None:
            continue

        acc     = run.summary.get("best_test_acc") or run.summary.get("test_acc_on_best_val")
        acc_std = run.summary.get("test_acc_std", 0)
        if acc is None:
            continue

        records.append({
            "model": MODEL_SHORT[model_key],
            "shot":  shot,
            "way":   way,
            "acc":   acc,
            "std":   acc_std,
        })

    if not records:
        print("No matching runs found.")
        return

    df = pd.DataFrame(records)
    print(df.sort_values(["model", "way", "shot"]).to_string(index=False))

    ways    = sorted(df["way"].unique())
    models  = sorted(df["model"].unique())
    shots   = sorted(df["shot"].unique())
    colors  = {m: c for m, c in zip(models, cm.tab10.colors)}

    fig, axes = plt.subplots(1, len(ways), figsize=(5 * len(ways), 4), sharey=True)
    if len(ways) == 1:
        axes = [axes]

    for ax, way in zip(axes, ways):
        sub = df[df["way"] == way]
        for model in models:
            msub = sub[sub["model"] == model].sort_values("shot")
            if msub.empty:
                continue
            ax.errorbar(
                msub["shot"], msub["acc"], yerr=msub["std"],
                marker="o", label=model, color=colors[model], capsize=4,
            )
        ax.set_title(f"{way}-way")
        ax.set_xlabel("# shots")
        ax.set_xticks(shots)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Test Accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(1.0, 1.0))
    fig.suptitle("Overperformer — Instagram NM pretrain sweep", fontsize=13)
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
