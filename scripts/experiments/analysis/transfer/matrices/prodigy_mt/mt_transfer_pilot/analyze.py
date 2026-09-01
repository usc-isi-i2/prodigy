#!/usr/bin/env python3
"""Aggregate and plot the matched MT versus NM+MT transfer pilot."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


GRAPHS = [
    "covid_political",
    "election2020",
    "facebook_page_reference",
    "twibot20",
    "ukr_rus_suspended",
]
SHORT = {
    "covid_political": "COVID",
    "election2020": "Election",
    "facebook_page_reference": "Facebook",
    "twibot20": "TwiBot",
    "ukr_rus_suspended": "UKR-RUS",
}
RUN_RE = re.compile(
    r"^eval_(NM_MT|MT)_(.+)_to_(.+)_pl_3shot_\d{2}_\d{2}_\d{4}_\d{2}_\d{2}_\d{2}$"
)


def collect(log_root: Path) -> pd.DataFrame:
    rows = []
    for run_dir in sorted(log_root.glob("eval_*_to_*_pl_3shot_*")):
        match = RUN_RE.match(run_dir.name)
        if not match:
            continue
        arm, source, target = match.groups()
        if source not in GRAPHS or target not in GRAPHS:
            continue
        metric_path = run_dir / "data" / "metrics_test_step0.json"
        if not metric_path.exists():
            continue
        metrics = json.loads(metric_path.read_text())
        rows.append(
            {
                "arm": arm,
                "source": source,
                "target": target,
                "accuracy": metrics["test_accuracy"],
                "f1": metrics["test_f1"],
                "roc_auc": metrics["test_roc_auc"],
                "run_dir": str(run_dir),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError(f"No completed pilot metrics found beneath {log_root}")
    # Failed attempts and reruns can share a logical cell; retain the newest result.
    frame = frame.drop_duplicates(["arm", "source", "target"], keep="last")
    expected = {(a, s, t) for a in ("MT", "NM_MT") for s in GRAPHS for t in GRAPHS}
    observed = set(frame[["arm", "source", "target"]].itertuples(index=False, name=None))
    missing = sorted(expected - observed)
    if missing:
        raise RuntimeError(f"Incomplete matrix: missing {missing}")
    return frame.sort_values(["arm", "source", "target"])


def plot_matrix(ax, matrix: pd.DataFrame, title: str, *, delta: bool = False) -> None:
    values = matrix.loc[GRAPHS, GRAPHS].to_numpy()
    if delta:
        bound = max(0.01, float(np.nanmax(np.abs(values))))
        image = ax.imshow(values, cmap="RdBu", vmin=-bound, vmax=bound)
        fmt, scale = "+.3f", 1.0
    else:
        image = ax.imshow(values, cmap="viridis", vmin=0, vmax=1)
        fmt, scale = ".1f", 100.0
    ax.set_title(title)
    ax.set_xticks(range(5), [SHORT[g] for g in GRAPHS], rotation=35, ha="right")
    ax.set_yticks(range(5), [SHORT[g] for g in GRAPHS])
    ax.set_xlabel("evaluation target")
    ax.set_ylabel("training source")
    for i in range(5):
        for j in range(5):
            ax.text(j, i, format(values[i, j] * scale, fmt), ha="center", va="center", fontsize=7,
                    color="white" if (not delta and values[i, j] < 0.45) else "black")
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    data_dir = args.output_dir / "data"
    fig_dir = args.output_dir / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    results = collect(args.log_root)
    results.to_csv(data_dir / "matched_results.csv", index=False)
    matrices = {}
    for arm in ("MT", "NM_MT"):
        matrix = results[results.arm == arm].pivot(index="source", columns="target", values="accuracy")
        matrix.loc[GRAPHS, GRAPHS].to_csv(data_dir / f"{arm.lower()}_accuracy_matrix.csv")
        matrices[arm] = matrix
    delta = matrices["NM_MT"] - matrices["MT"]
    delta.loc[GRAPHS, GRAPHS].to_csv(data_dir / "nm_mt_minus_mt_accuracy.csv")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    plot_matrix(axes[0], matrices["MT"], "MT accuracy (%)")
    plot_matrix(axes[1], matrices["NM_MT"], "NM+MT accuracy (%)")
    plot_matrix(axes[2], delta, "NM+MT − MT accuracy", delta=True)
    fig.savefig(fig_dir / "matched_transfer_matrices.png", dpi=220)
    fig.savefig(fig_dir / "matched_transfer_matrices.pdf")

    summary = {
        "cells": int(len(results)),
        "mt_mean_accuracy": float(matrices["MT"].to_numpy().mean()),
        "nm_mt_mean_accuracy": float(matrices["NM_MT"].to_numpy().mean()),
        "mean_delta": float(delta.to_numpy().mean()),
        "median_delta": float(np.median(delta.to_numpy())),
        "nm_mt_wins": int((delta.to_numpy() > 0).sum()),
        "mt_wins": int((delta.to_numpy() < 0).sum()),
        "ties": int((delta.to_numpy() == 0).sum()),
        "diagonal_mean_delta": float(np.diag(delta.loc[GRAPHS, GRAPHS]).mean()),
        "off_diagonal_mean_delta": float(
            delta.loc[GRAPHS, GRAPHS].to_numpy()[~np.eye(5, dtype=bool)].mean()
        ),
    }
    (data_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
