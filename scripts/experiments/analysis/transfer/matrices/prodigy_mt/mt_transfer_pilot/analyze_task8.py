#!/usr/bin/env python3
"""Compare additive/FiLM 8D task conditioning with the dimension-0 baseline."""

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
TASK8_RE = re.compile(r"^eval_TASK8_EXCL_(.+)_to_(.+)_pl_3shot_")
FILM8_RE = re.compile(r"^eval_FILM8_EXCL_(.+)_to_(.+)_pl_3shot_")
BASE_INMIX_RE = re.compile(r"^eval_INMIX_NM_MT_EXCL_(.+)_to_(.+)_pl_3shot_")
BASE_HELDOUT_RE = re.compile(r"^eval_HELDOUT_NM_MT_to_(.+)_pl_3shot_")


def metric_row(run_dir: Path, condition: str, excluded: str, target: str):
    path = run_dir / "data" / "metrics_test_step0.json"
    if not path.exists():
        return None
    metrics = json.loads(path.read_text())
    return {
        "condition": condition,
        "excluded": excluded,
        "target": target,
        "accuracy": metrics["test_accuracy"],
        "roc_auc": metrics["test_roc_auc"],
        "run_dir": str(run_dir),
    }


def collect(task8_root: Path, film8_root: Path | None, baseline_root: Path) -> pd.DataFrame:
    rows = []
    for run_dir in sorted(task8_root.glob("eval_TASK8_EXCL_*_to_*_pl_3shot_*")):
        match = TASK8_RE.match(run_dir.name)
        if match:
            row = metric_row(run_dir, "task8", *match.groups())
            if row:
                rows.append(row)
    if film8_root is not None:
        for run_dir in sorted(film8_root.glob("eval_FILM8_EXCL_*_to_*_pl_3shot_*")):
            match = FILM8_RE.match(run_dir.name)
            if match:
                row = metric_row(run_dir, "film8", *match.groups())
                if row:
                    rows.append(row)
    for run_dir in sorted(baseline_root.glob("eval_INMIX_NM_MT_EXCL_*_to_*_pl_3shot_*")):
        match = BASE_INMIX_RE.match(run_dir.name)
        if match:
            row = metric_row(run_dir, "baseline", *match.groups())
            if row:
                rows.append(row)
    for run_dir in sorted(baseline_root.glob("eval_HELDOUT_NM_MT_to_*_pl_3shot_*")):
        match = BASE_HELDOUT_RE.match(run_dir.name)
        if match:
            target = match.group(1)
            row = metric_row(run_dir, "baseline", target, target)
            if row:
                rows.append(row)
    frame = pd.DataFrame(rows).drop_duplicates(
        ["condition", "excluded", "target"], keep="last"
    )
    conditions = ("baseline", "task8", "film8") if film8_root is not None else ("baseline", "task8")
    expected = {(c, e, t) for c in conditions for e in GRAPHS for t in GRAPHS}
    observed = set(frame[["condition", "excluded", "target"]].itertuples(index=False, name=None))
    missing = sorted(expected - observed)
    if missing:
        raise RuntimeError(f"Missing comparison cells: {missing}")
    return frame.sort_values(["condition", "excluded", "target"])


def coordinates(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows = []
    for condition in frame.condition.drop_duplicates():
        for excluded in GRAPHS:
            subset = frame[(frame.condition == condition) & (frame.excluded == excluded)]
            rows.append({
                "condition": condition,
                "excluded": excluded,
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
        base = coords[(coords.condition == "baseline") & (coords.excluded == excluded)].iloc[0]
        task8 = coords[(coords.condition == "task8") & (coords.excluded == excluded)].iloc[0]
        film8_rows = coords[(coords.condition == "film8") & (coords.excluded == excluded)]
        bx, by = 100 * base.seen, 100 * base.unseen
        tx, ty = 100 * task8.seen, 100 * task8.unseen
        ax.annotate("", xy=(tx, ty), xytext=(bx, by),
                    arrowprops=dict(arrowstyle="->", color="#888888", lw=1.3))
        ax.scatter(bx, by, s=90, marker="D", facecolors="white", edgecolors="#D55E00",
                   linewidths=1.5, zorder=3)
        ax.scatter(tx, ty, s=95, marker="D", color="#D55E00", edgecolors="white",
                   linewidths=.8, zorder=4)
        label_x, label_y = tx, ty
        if not film8_rows.empty:
            film8 = film8_rows.iloc[0]
            fx, fy = 100 * film8.seen, 100 * film8.unseen
            ax.annotate("", xy=(fx, fy), xytext=(tx, ty),
                        arrowprops=dict(arrowstyle="->", color="#777777", lw=1.3))
            ax.scatter(fx, fy, s=105, marker="o", color="#0072B2", edgecolors="white",
                       linewidths=.8, zorder=5)
            label_x, label_y = fx, fy
        ax.annotate(SHORT[excluded], (label_x, label_y), xytext=(6, 4),
                    textcoords="offset points", fontsize=8)
    ax.plot([low, 100], [low, 100], "--", color="#999999", lw=1)
    ax.set(xlim=(low, 101), ylim=(low, 101),
           xlabel=f"Seen / in-mixture {label} (%)", ylabel=f"Held-out {label} (%)",
           title=f"NM+MT task conditioning — {label}")
    ax.grid(alpha=.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.scatter([], [], s=80, marker="D", facecolors="white", edgecolors="#D55E00",
               linewidths=1.5, label="Baseline (0D)")
    ax.scatter([], [], s=80, marker="D", color="#D55E00", label="Task conditioned (8D)")
    if "film8" in set(coords.condition):
        ax.scatter([], [], s=80, marker="o", color="#0072B2", label="FiLM conditioned (8D)")
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    suffix = "accuracy" if metric == "accuracy" else "auc"
    fig.savefig(fig_dir / f"task8_seen_unseen_{suffix}_plane.png", dpi=220)
    fig.savefig(fig_dir / f"task8_seen_unseen_{suffix}_plane.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task8-log-root", type=Path, required=True)
    parser.add_argument("--film8-log-root", type=Path)
    parser.add_argument("--baseline-log-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    data_dir, fig_dir = args.output_dir / "data", args.output_dir / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    frame = collect(args.task8_log_root, args.film8_log_root, args.baseline_log_root)
    frame.to_csv(data_dir / "task8_comparison_cells.csv", index=False)
    summary = {"cells": int(len(frame))}
    for metric in ("accuracy", "roc_auc"):
        coords = coordinates(frame, metric)
        coords.to_csv(data_dir / f"task8_seen_unseen_{metric}.csv", index=False)
        plot_plane(coords, metric, fig_dir)
        for condition in frame.condition.drop_duplicates():
            subset = coords[coords.condition == condition]
            summary[f"{condition}_seen_{metric}"] = float(subset.seen.mean())
            summary[f"{condition}_unseen_{metric}"] = float(subset.unseen.mean())
            summary[f"{condition}_gap_{metric}"] = float((subset.unseen - subset.seen).mean())
        summary[f"task8_minus_baseline_seen_{metric}"] = (
            summary[f"task8_seen_{metric}"] - summary[f"baseline_seen_{metric}"]
        )
        summary[f"task8_minus_baseline_unseen_{metric}"] = (
            summary[f"task8_unseen_{metric}"] - summary[f"baseline_unseen_{metric}"]
        )
        if "film8" in set(frame.condition):
            for reference in ("baseline", "task8"):
                for split in ("seen", "unseen"):
                    summary[f"film8_minus_{reference}_{split}_{metric}"] = (
                        summary[f"film8_{split}_{metric}"] - summary[f"{reference}_{split}_{metric}"]
                    )
    (data_dir / "task8_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
