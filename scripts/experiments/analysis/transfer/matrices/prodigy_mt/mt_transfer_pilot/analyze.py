#!/usr/bin/env python3
"""Aggregate and plot the matched MT, NM, and NM+MT transfer pilot."""

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
    "heldout_mixture": "Held-out mix",
}
ROWS = GRAPHS + ["heldout_mixture"]
RUN_RE = re.compile(
    r"^eval_(NM_MT|NM|MT)_(.+)_to_(.+)_pl_3shot_\d{2}_\d{2}_\d{4}_\d{2}_\d{2}_\d{2}$"
)
HELDOUT_RE = re.compile(
    r"^eval_HELDOUT_(NM_MT|NM|MT)_to_(.+)_pl_3shot_\d{2}_\d{2}_\d{4}_\d{2}_\d{2}_\d{2}$"
)


def collect(log_root: Path) -> pd.DataFrame:
    rows = []
    for run_dir in sorted(log_root.glob("eval_*_to_*_pl_3shot_*")):
        match = RUN_RE.match(run_dir.name)
        heldout_match = HELDOUT_RE.match(run_dir.name)
        if heldout_match:
            arm, target = heldout_match.groups()
            source = "heldout_mixture"
        elif match:
            arm, source, target = match.groups()
        else:
            continue
        if source not in ROWS or target not in GRAPHS:
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
    expected = {(a, s, t) for a in ("MT", "NM", "NM_MT") for s in GRAPHS for t in GRAPHS}
    expected |= {(a, "heldout_mixture", t) for a in ("MT", "NM", "NM_MT") for t in GRAPHS}
    observed = set(frame[["arm", "source", "target"]].itertuples(index=False, name=None))
    missing = sorted(expected - observed)
    if missing:
        raise RuntimeError(f"Incomplete matrix: missing {missing}")
    return frame.sort_values(["arm", "source", "target"])


def plot_matrix(ax, matrix: pd.DataFrame, title: str, *, delta: bool = False) -> None:
    values = matrix.loc[ROWS, GRAPHS].to_numpy()
    if delta:
        bound = max(0.01, float(np.nanmax(np.abs(values))))
        image = ax.imshow(values, cmap="RdBu", vmin=-bound, vmax=bound)
        fmt, scale = "+.3f", 1.0
    else:
        image = ax.imshow(values, cmap="viridis", vmin=0, vmax=1)
        fmt, scale = ".1f", 100.0
    ax.set_title(title)
    ax.set_xticks(range(5), [SHORT[g] for g in GRAPHS], rotation=35, ha="right")
    ax.set_yticks(range(len(ROWS)), [SHORT[g] for g in ROWS])
    ax.set_xlabel("evaluation target")
    ax.set_ylabel("training source")
    for i in range(len(ROWS)):
        for j in range(5):
            ax.text(j, i, format(values[i, j] * scale, fmt), ha="center", va="center", fontsize=7,
                    color="white" if (not delta and values[i, j] < 0.45) else "black")
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def build_figure(results, metric, data_dir, fig_dir):
    matrices = {}
    for arm in ("MT", "NM", "NM_MT"):
        matrix = results[results.arm == arm].pivot(index="source", columns="target", values=metric)
        matrix.loc[ROWS, GRAPHS].to_csv(data_dir / f"{arm.lower()}_{metric}_matrix.csv")
        matrices[arm] = matrix
    delta_mt = matrices["NM_MT"] - matrices["MT"]
    delta_nm = matrices["NM_MT"] - matrices["NM"]
    delta_mt.loc[ROWS, GRAPHS].to_csv(data_dir / f"nm_mt_minus_mt_{metric}.csv")
    delta_nm.loc[ROWS, GRAPHS].to_csv(data_dir / f"nm_mt_minus_nm_{metric}.csv")
    label = "accuracy" if metric == "accuracy" else "ROC-AUC"
    fig, axes = plt.subplots(1, 5, figsize=(24, 5.2), constrained_layout=True)
    plot_matrix(axes[0], matrices["MT"], f"MT {label} (%)")
    plot_matrix(axes[1], matrices["NM"], f"NM {label} (%)")
    plot_matrix(axes[2], matrices["NM_MT"], f"NM+MT {label} (%)")
    plot_matrix(axes[3], delta_mt, f"NM+MT − MT {label}", delta=True)
    plot_matrix(axes[4], delta_nm, f"NM+MT − NM {label}", delta=True)
    suffix = "matched_transfer_matrices" if metric == "accuracy" else "matched_transfer_matrices_auc"
    fig.savefig(fig_dir / f"{suffix}.png", dpi=220)
    fig.savefig(fig_dir / f"{suffix}.pdf")
    plt.close(fig)
    return matrices, delta_mt, delta_nm


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
    matrices, delta_mt, delta_nm = build_figure(results, "accuracy", data_dir, fig_dir)
    auc_matrices, auc_delta_mt, auc_delta_nm = build_figure(results, "roc_auc", data_dir, fig_dir)

    summary = {
        "cells": int(len(results)),
        "mt_mean_accuracy": float(matrices["MT"].loc[GRAPHS, GRAPHS].to_numpy().mean()),
        "nm_mean_accuracy": float(matrices["NM"].loc[GRAPHS, GRAPHS].to_numpy().mean()),
        "nm_mt_mean_accuracy": float(matrices["NM_MT"].loc[GRAPHS, GRAPHS].to_numpy().mean()),
        "nm_mt_minus_mt_mean": float(delta_mt.loc[GRAPHS, GRAPHS].to_numpy().mean()),
        "nm_mt_minus_nm_mean": float(delta_nm.loc[GRAPHS, GRAPHS].to_numpy().mean()),
        "nm_mt_minus_nm_median": float(np.median(delta_nm.loc[GRAPHS, GRAPHS].to_numpy())),
        "nm_mt_beats_nm_cells": int((delta_nm.loc[GRAPHS, GRAPHS].to_numpy() > 0).sum()),
        "nm_beats_nm_mt_cells": int((delta_nm.loc[GRAPHS, GRAPHS].to_numpy() < 0).sum()),
        "nm_mt_minus_nm_diagonal": float(np.diag(delta_nm.loc[GRAPHS, GRAPHS]).mean()),
        "nm_mt_minus_nm_off_diagonal": float(
            delta_nm.loc[GRAPHS, GRAPHS].to_numpy()[~np.eye(5, dtype=bool)].mean()
        ),
        "mt_mean_roc_auc": float(auc_matrices["MT"].loc[GRAPHS, GRAPHS].to_numpy().mean()),
        "nm_mean_roc_auc": float(auc_matrices["NM"].loc[GRAPHS, GRAPHS].to_numpy().mean()),
        "nm_mt_mean_roc_auc": float(auc_matrices["NM_MT"].loc[GRAPHS, GRAPHS].to_numpy().mean()),
        "nm_mt_minus_mt_auc_mean": float(auc_delta_mt.loc[GRAPHS, GRAPHS].to_numpy().mean()),
        "nm_mt_minus_nm_auc_mean": float(auc_delta_nm.loc[GRAPHS, GRAPHS].to_numpy().mean()),
        "heldout_mt_mean_accuracy": float(matrices["MT"].loc["heldout_mixture"].mean()),
        "heldout_nm_mean_accuracy": float(matrices["NM"].loc["heldout_mixture"].mean()),
        "heldout_nm_mt_mean_accuracy": float(matrices["NM_MT"].loc["heldout_mixture"].mean()),
        "heldout_mt_mean_roc_auc": float(auc_matrices["MT"].loc["heldout_mixture"].mean()),
        "heldout_nm_mean_roc_auc": float(auc_matrices["NM"].loc["heldout_mixture"].mean()),
        "heldout_nm_mt_mean_roc_auc": float(auc_matrices["NM_MT"].loc["heldout_mixture"].mean()),
    }
    (data_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
