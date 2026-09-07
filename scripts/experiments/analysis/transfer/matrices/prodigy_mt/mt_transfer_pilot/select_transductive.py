#!/usr/bin/env python3
"""Select transductive hyperparameters on validation AUC, then report test metrics."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


RUN_RE = re.compile(
    r"^eval_TRANS_T(?P<threshold>\d+)_A(?P<alpha>\d+)_I(?P<iterations>\d+)_EXCL_"
    r"(?P<excluded>.+)_to_(?P<target>.+)_pl_3shot_"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--variable-way-cells", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "data")
    args = parser.parse_args()
    rows = []
    for run_dir in sorted(args.log_root.glob("eval_TRANS_*_to_*_pl_3shot_*")):
        match = RUN_RE.match(run_dir.name)
        if not match:
            continue
        values = match.groupdict()
        row = {
            "threshold": int(values["threshold"]) / 10,
            "alpha": {"025": 0.25, "05": 0.5}[values["alpha"]],
            "iterations": int(values["iterations"]),
            "excluded": values["excluded"], "target": values["target"],
            "run_dir": str(run_dir),
        }
        complete = True
        for split in ("val", "test"):
            path = run_dir / "data" / f"metrics_{split}_step0.json"
            if not path.exists():
                complete = False
                break
            metrics = json.loads(path.read_text())
            row[f"{split}_accuracy"] = metrics[f"{split}_accuracy"]
            row[f"{split}_roc_auc"] = metrics[f"{split}_roc_auc"]
        if complete:
            rows.append(row)
    frame = pd.DataFrame(rows)
    if len(frame) != 150:
        raise RuntimeError(f"Expected 150 complete grid cells, found {len(frame)}")
    # Each held-out model is scored on all five validation targets. This chooses one
    # global configuration without consulting any test metric.
    grid = frame.groupby(["threshold", "alpha", "iterations"], as_index=False).agg(
        val_accuracy=("val_accuracy", "mean"), val_roc_auc=("val_roc_auc", "mean")
    ).sort_values(["val_roc_auc", "val_accuracy"], ascending=False)
    best = grid.iloc[0]
    selected = frame[
        (frame.threshold == best.threshold) & (frame.alpha == best.alpha)
        & (frame.iterations == best.iterations)
    ]
    heldout = selected[selected.excluded == selected.target]
    baseline = pd.read_csv(args.variable_way_cells)
    baseline = baseline[
        (baseline.condition == "variable_way") & (baseline.excluded == baseline.target)
    ][["target", "accuracy", "roc_auc"]].rename(columns={
        "accuracy": "baseline_accuracy", "roc_auc": "baseline_roc_auc"
    })
    comparison = heldout.merge(baseline, on="target", validate="one_to_one")
    comparison["accuracy_delta"] = comparison.test_accuracy - comparison.baseline_accuracy
    comparison["roc_auc_delta"] = comparison.test_roc_auc - comparison.baseline_roc_auc
    summary = {
        "selection_metric": "mean validation ROC-AUC across all 25 model-target cells",
        "threshold": float(best.threshold), "alpha": float(best.alpha),
        "iterations": int(best.iterations), "selected_validation_auc": float(best.val_roc_auc),
        "heldout_test_accuracy": float(heldout.test_accuracy.mean()),
        "heldout_test_roc_auc": float(heldout.test_roc_auc.mean()),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "transductive_grid_cells.csv", index=False)
    grid.to_csv(args.output_dir / "transductive_validation_selection.csv", index=False)
    comparison.to_csv(args.output_dir / "transductive_selected_heldout.csv", index=False)
    (args.output_dir / "transductive_selected_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    labels = ["COVID", "Election", "Facebook", "TwiBot", "UKR-RUS"]
    values = comparison.set_index("target").loc[
        ["covid_political", "election2020", "facebook_page_reference", "twibot20", "ukr_rus_suspended"],
        "roc_auc_delta",
    ].to_numpy() * 100
    fig, ax = plt.subplots(figsize=(7.3, 4.3), constrained_layout=True)
    colors = ["#0072B2" if value >= 0 else "#D55E00" for value in values]
    ax.bar(labels, values, color=colors)
    ax.axhline(0, color="#555555", lw=1)
    ax.axhline(values.mean(), color="#0072B2", ls="--", lw=1.2,
               label=f"Mean {values.mean():+.2f} pp")
    ax.set(ylabel="Held-out ROC-AUC change (pp)",
           title="Validation-selected transductive refinement")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig_dir = args.output_dir.parent / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "transductive_selected_auc_delta.png", dpi=220)
    fig.savefig(fig_dir / "transductive_selected_auc_delta.pdf")
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
