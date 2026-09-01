#!/usr/bin/env python3
"""Select transductive hyperparameters on validation AUC, then report test metrics."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


RUN_RE = re.compile(
    r"^eval_TRANS_T(?P<threshold>\d+)_A(?P<alpha>\d+)_I(?P<iterations>\d+)_EXCL_"
    r"(?P<excluded>.+)_to_(?P<target>.+)_pl_3shot_"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-root", type=Path, required=True)
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
            "alpha": int(values["alpha"]) / 100,
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
    (args.output_dir / "transductive_selected_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
