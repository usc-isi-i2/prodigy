#!/usr/bin/env python3
"""Compare variable-way NM+MT leave-one-out mixtures with fixed 30-way NM+MT."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


GRAPHS = [
    "covid_political", "election2020", "facebook_page_reference",
    "twibot20", "ukr_rus_suspended",
]
SHORT = {
    "covid_political": "COVID", "election2020": "Election",
    "facebook_page_reference": "Facebook", "twibot20": "TwiBot",
    "ukr_rus_suspended": "UKR-RUS",
}
VARWAY_RE = re.compile(r"^eval_VARWAY_EXCL_(.+)_to_(.+)_pl_3shot_")
BASE_INMIX_RE = re.compile(r"^eval_INMIX_NM_MT_EXCL_(.+)_to_(.+)_pl_3shot_")
BASE_HELDOUT_RE = re.compile(r"^eval_HELDOUT_NM_MT_to_(.+)_pl_3shot_")


def metric_row(run_dir: Path, condition: str, excluded: str, target: str):
    path = run_dir / "data" / "metrics_test_step0.json"
    if not path.exists():
        return None
    metrics = json.loads(path.read_text())
    return {
        "condition": condition, "excluded": excluded, "target": target,
        "accuracy": metrics["test_accuracy"], "roc_auc": metrics["test_roc_auc"],
        "run_dir": str(run_dir),
    }


def collect(varway_root: Path, baseline_root: Path) -> pd.DataFrame:
    rows = []
    for run_dir in sorted(varway_root.glob("eval_VARWAY_EXCL_*_to_*_pl_3shot_*")):
        match = VARWAY_RE.match(run_dir.name)
        if match:
            row = metric_row(run_dir, "variable_way", *match.groups())
            if row:
                rows.append(row)
    for run_dir in sorted(baseline_root.glob("eval_INMIX_NM_MT_EXCL_*_to_*_pl_3shot_*")):
        match = BASE_INMIX_RE.match(run_dir.name)
        if match:
            row = metric_row(run_dir, "fixed_30way", *match.groups())
            if row:
                rows.append(row)
    for run_dir in sorted(baseline_root.glob("eval_HELDOUT_NM_MT_to_*_pl_3shot_*")):
        match = BASE_HELDOUT_RE.match(run_dir.name)
        if match:
            target = match.group(1)
            row = metric_row(run_dir, "fixed_30way", target, target)
            if row:
                rows.append(row)
    frame = pd.DataFrame(rows).drop_duplicates(
        ["condition", "excluded", "target"], keep="last"
    )
    expected = {(c, e, t) for c in ("fixed_30way", "variable_way") for e in GRAPHS for t in GRAPHS}
    observed = set(frame[["condition", "excluded", "target"]].itertuples(index=False, name=None))
    missing = sorted(expected - observed)
    if missing:
        raise RuntimeError(f"Missing comparison cells: {missing}")
    return frame.sort_values(["condition", "excluded", "target"])


def coordinates(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows = []
    for condition in ("fixed_30way", "variable_way"):
        for excluded in GRAPHS:
            subset = frame[(frame.condition == condition) & (frame.excluded == excluded)]
            rows.append({
                "condition": condition, "excluded": excluded,
                "seen": subset[subset.target != excluded][metric].mean(),
                "unseen": subset[subset.target == excluded][metric].iloc[0],
            })
    return pd.DataFrame(rows)


def plot_plane(coords: pd.DataFrame, metric: str, fig_dir: Path):
    label = "Accuracy" if metric == "accuracy" else "ROC-AUC"
    values = coords[["seen", "unseen"]].to_numpy() * 100
    low = max(0, int(values.min() // 10 * 10 - 5))
    fig, ax = plt.subplots(figsize=(7.4, 6.4), constrained_layout=True)
    for excluded in GRAPHS:
        base = coords[(coords.condition == "fixed_30way") & (coords.excluded == excluded)].iloc[0]
        varied = coords[(coords.condition == "variable_way") & (coords.excluded == excluded)].iloc[0]
        bx, by = 100 * base.seen, 100 * base.unseen
        vx, vy = 100 * varied.seen, 100 * varied.unseen
        ax.annotate("", xy=(vx, vy), xytext=(bx, by),
                    arrowprops=dict(arrowstyle="->", color="#888888", lw=1.3))
        ax.scatter(bx, by, s=90, marker="D", facecolors="white", edgecolors="#D55E00",
                   linewidths=1.5, zorder=3)
        ax.scatter(vx, vy, s=105, marker="o", color="#0072B2", edgecolors="white",
                   linewidths=.8, zorder=4)
        offset = {"facebook_page_reference": (8, -15), "twibot20": (8, 7)}.get(excluded, (6, 4))
        ax.annotate(SHORT[excluded], (vx, vy), xytext=offset, textcoords="offset points", fontsize=8)
    ax.plot([low, 100], [low, 100], "--", color="#999999", lw=1)
    ax.set(xlim=(low, 101), ylim=(low, 101), xlabel=f"Seen / in-mixture {label} (%)",
           ylabel=f"Held-out {label} (%)", title=f"Variable-way NM+MT — {label}")
    ax.grid(alpha=.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.scatter([], [], s=80, marker="D", facecolors="white", edgecolors="#D55E00",
               linewidths=1.5, label="Fixed 30-way NM")
    ax.scatter([], [], s=80, marker="o", color="#0072B2", label="Variable-way NM")
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, .5))
    suffix = "accuracy" if metric == "accuracy" else "auc"
    fig.savefig(fig_dir / f"variable_way_seen_unseen_{suffix}_plane.png", dpi=220)
    fig.savefig(fig_dir / f"variable_way_seen_unseen_{suffix}_plane.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--varway-log-root", type=Path, required=True)
    parser.add_argument("--baseline-log-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    data_dir, fig_dir = args.output_dir / "data", args.output_dir / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    frame = collect(args.varway_log_root, args.baseline_log_root)
    frame.to_csv(data_dir / "variable_way_comparison_cells.csv", index=False)
    summary = {"cells": int(len(frame))}
    for metric in ("accuracy", "roc_auc"):
        coords = coordinates(frame, metric)
        coords.to_csv(data_dir / f"variable_way_seen_unseen_{metric}.csv", index=False)
        plot_plane(coords, metric, fig_dir)
        for condition in ("fixed_30way", "variable_way"):
            subset = coords[coords.condition == condition]
            summary[f"{condition}_seen_{metric}"] = float(subset.seen.mean())
            summary[f"{condition}_unseen_{metric}"] = float(subset.unseen.mean())
        for split in ("seen", "unseen"):
            summary[f"variable_minus_fixed_{split}_{metric}"] = (
                summary[f"variable_way_{split}_{metric}"] - summary[f"fixed_30way_{split}_{metric}"]
            )
    (data_dir / "variable_way_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
