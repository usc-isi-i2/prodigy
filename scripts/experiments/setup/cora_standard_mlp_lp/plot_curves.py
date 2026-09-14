#!/usr/bin/env python3
"""Render unsmoothed training and validation curves from a completed run."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--title", default="Classic Cora feature-only link prediction — seed 0")
    args = parser.parse_args()
    records = json.loads(args.selection.read_text())
    colors = {"linear_cosine": "#2878B5", "nonlinear_mlp_cosine": "#D95319"}
    labels = {"linear_cosine": "Linear 1433→128", "nonlinear_mlp_cosine": "Nonlinear 1433→256→128"}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    fields = (("train_loss", "Training BCE", True), ("val_roc_auc", "Validation ROC-AUC", False),
              ("val_average_precision", "Validation AP", False))
    for record in records:
        arm = record["arm"]
        epochs = [row["epoch"] for row in record["history"]]
        for axis, (field, title, log_scale) in zip(axes, fields):
            axis.plot(epochs, [row[field] for row in record["history"]], color=colors[arm],
                      linewidth=1.5, alpha=0.9, label=labels[arm])
            axis.axvline(record["best_epoch"], color=colors[arm], linestyle="--", linewidth=0.9, alpha=0.7)
            axis.set_title(title)
            axis.set_xlabel("Epoch")
            if log_scale:
                axis.set_yscale("log")
            axis.grid(alpha=0.2)
    axes[0].set_ylabel("Raw unsmoothed value")
    axes[2].legend(frameon=False, loc="lower right")
    fig.suptitle(args.title)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()
