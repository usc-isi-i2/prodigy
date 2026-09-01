#!/usr/bin/env python3
"""Compare fixed-way, variable-way, and support-prototype relation models."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from analyze_variable_way import GRAPHS, SHORT, collect as collect_prior, coordinates


SUPPORT_RE = re.compile(r"^eval_SUPPORTREL_EXCL_(.+)_to_(.+)_pl_3shot_")


def collect(support_root: Path, varway_root: Path, baseline_root: Path) -> pd.DataFrame:
    frame = collect_prior(varway_root, baseline_root)
    rows = []
    for run_dir in sorted(support_root.glob("eval_SUPPORTREL_EXCL_*_to_*_pl_3shot_*")):
        match = SUPPORT_RE.match(run_dir.name)
        path = run_dir / "data" / "metrics_test_step0.json"
        if not match or not path.exists():
            continue
        metrics = json.loads(path.read_text())
        rows.append({
            "condition": "support_relation", "excluded": match.group(1),
            "target": match.group(2), "accuracy": metrics["test_accuracy"],
            "roc_auc": metrics["test_roc_auc"], "run_dir": str(run_dir),
        })
    extra = pd.DataFrame(rows).drop_duplicates(["excluded", "target"], keep="last")
    expected = {(e, t) for e in GRAPHS for t in GRAPHS}
    observed = set(extra[["excluded", "target"]].itertuples(index=False, name=None))
    if expected - observed:
        raise RuntimeError(f"Missing support-relation cells: {sorted(expected - observed)}")
    return pd.concat([frame, extra], ignore_index=True).sort_values(
        ["condition", "excluded", "target"]
    )


def all_coordinates(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    prior = coordinates(frame[frame.condition != "support_relation"], metric)
    rows = []
    for excluded in GRAPHS:
        subset = frame[(frame.condition == "support_relation") & (frame.excluded == excluded)]
        rows.append({
            "condition": "support_relation", "excluded": excluded,
            "seen": subset[subset.target != excluded][metric].mean(),
            "unseen": subset[subset.target == excluded][metric].iloc[0],
        })
    return pd.concat([prior, pd.DataFrame(rows)], ignore_index=True)


def plot(coords: pd.DataFrame, metric: str, fig_dir: Path):
    label = "Accuracy" if metric == "accuracy" else "ROC-AUC"
    values = coords[["seen", "unseen"]].to_numpy() * 100
    low = max(0, int(values.min() // 10 * 10 - 5))
    fig, ax = plt.subplots(figsize=(7.7, 6.4), constrained_layout=True)
    for excluded in GRAPHS:
        points = {}
        for condition in ("fixed_30way", "variable_way", "support_relation"):
            row = coords[(coords.condition == condition) & (coords.excluded == excluded)].iloc[0]
            points[condition] = (100 * row.seen, 100 * row.unseen)
        ax.annotate("", xy=points["variable_way"], xytext=points["fixed_30way"],
                    arrowprops=dict(arrowstyle="->", color="#aaaaaa", lw=1.1))
        ax.annotate("", xy=points["support_relation"], xytext=points["variable_way"],
                    arrowprops=dict(arrowstyle="->", color="#777777", lw=1.3))
        ax.scatter(*points["fixed_30way"], s=85, marker="D", facecolors="white",
                   edgecolors="#D55E00", linewidths=1.5, zorder=3)
        ax.scatter(*points["variable_way"], s=90, marker="o", facecolors="white",
                   edgecolors="#0072B2", linewidths=1.5, zorder=4)
        ax.scatter(*points["support_relation"], s=105, marker="o", color="#0072B2",
                   edgecolors="white", linewidths=.8, zorder=5)
        offset = {"facebook_page_reference": (8, -15), "twibot20": (-52, 8)}.get(excluded, (6, 4))
        ax.annotate(SHORT[excluded], points["support_relation"], xytext=offset,
                    textcoords="offset points", fontsize=8)
    ax.plot([low, 100], [low, 100], "--", color="#999999", lw=1)
    ax.set(xlim=(low, 101), ylim=(low, 101), xlabel=f"Seen / in-mixture {label} (%)",
           ylabel=f"Held-out {label} (%)", title=f"Support prototypes + relation scorer — {label}")
    ax.grid(alpha=.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.scatter([], [], s=75, marker="D", facecolors="white", edgecolors="#D55E00",
               linewidths=1.5, label="Fixed 30-way")
    ax.scatter([], [], s=80, marker="o", facecolors="white", edgecolors="#0072B2",
               linewidths=1.5, label="Variable-way")
    ax.scatter([], [], s=85, marker="o", color="#0072B2", label="+ prototypes/relation")
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, .5))
    suffix = "accuracy" if metric == "accuracy" else "auc"
    fig.savefig(fig_dir / f"support_relation_seen_unseen_{suffix}_plane.png", dpi=220)
    fig.savefig(fig_dir / f"support_relation_seen_unseen_{suffix}_plane.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--support-log-root", type=Path, required=True)
    parser.add_argument("--varway-log-root", type=Path, required=True)
    parser.add_argument("--baseline-log-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    data_dir, fig_dir = args.output_dir / "data", args.output_dir / "figures"
    data_dir.mkdir(parents=True, exist_ok=True); fig_dir.mkdir(parents=True, exist_ok=True)
    frame = collect(args.support_log_root, args.varway_log_root, args.baseline_log_root)
    frame.to_csv(data_dir / "support_relation_comparison_cells.csv", index=False)
    summary = {"cells": int(len(frame))}
    for metric in ("accuracy", "roc_auc"):
        coords = all_coordinates(frame, metric)
        coords.to_csv(data_dir / f"support_relation_seen_unseen_{metric}.csv", index=False)
        plot(coords, metric, fig_dir)
        for condition in ("fixed_30way", "variable_way", "support_relation"):
            subset = coords[coords.condition == condition]
            for split in ("seen", "unseen"):
                summary[f"{condition}_{split}_{metric}"] = float(subset[split].mean())
        for reference in ("fixed_30way", "variable_way"):
            for split in ("seen", "unseen"):
                summary[f"support_minus_{reference}_{split}_{metric}"] = (
                    summary[f"support_relation_{split}_{metric}"] - summary[f"{reference}_{split}_{metric}"]
                )
    (data_dir / "support_relation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
