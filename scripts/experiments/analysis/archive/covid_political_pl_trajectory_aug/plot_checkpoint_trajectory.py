#!/usr/bin/env python3
"""Plot the augmented merged checkpoint trajectory exported by the eval wrapper."""

from __future__ import annotations

from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
CSV_PATH = SCRIPT_DIR / "eval_results.csv"
OUT_PATH = SCRIPT_DIR / "checkpoint_trajectory.png"
METRICS = ("accuracy", "f1", "roc_auc")


def checkpoint_step(model_name: str) -> int:
    match = re.search(r"_step([0-9]+)$", model_name)
    if match is None:
        raise ValueError(f"Cannot parse checkpoint step from model name: {model_name}")
    return int(match.group(1))


def main() -> int:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Export eval results first: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    df = df[(df["dataset"] == "covid_political") & (df["task"] == "pl") & (df["split"] == "test")].copy()
    if df.empty:
        raise ValueError("No covid_political/pl/test rows found in eval_results.csv")

    df["step"] = df["model"].map(checkpoint_step)
    df = df.sort_values("step")

    fig, axes = plt.subplots(1, len(METRICS), figsize=(15, 4), sharex=True)
    for ax, metric in zip(axes, METRICS):
        if metric not in df.columns:
            ax.set_visible(False)
            continue
        ax.plot(df["step"], df[metric], marker="o", linewidth=2)
        ax.set_title(metric)
        ax.set_xlabel("checkpoint step")
        ax.set_ylabel(metric)
        ax.set_ylim(0, 1.02)
        ax.set_xscale("symlog", linthresh=1000)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Merged UKR+COVID NM Aug checkpoint trajectory on COVID political PL, 3-shot")
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=200)
    print(f"[done] wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
