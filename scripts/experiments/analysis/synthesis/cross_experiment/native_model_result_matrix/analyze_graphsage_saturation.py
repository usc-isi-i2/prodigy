#!/usr/bin/env python3
"""Validate and plot the narrow GraphSAGE pilot-v1 SSL-to-CLS trajectory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
STEPS = (0, 20, 60, 100, 300, 900, 2000)


def load_result(path: Path) -> tuple[pd.DataFrame, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("task") != "twibot20_bot_classification":
        raise ValueError("GraphSAGE trajectory is not downstream TwiBot classification")
    if payload.get("seed") != 20260817:
        raise ValueError("GraphSAGE trajectory seed changed")
    cache = payload["cache"]
    if cache["splits"] != {"train": 8278, "val": 2365, "test": 1183}:
        raise ValueError(f"official TwiBot split changed: {cache['splits']}")
    frame = pd.DataFrame(payload["models"])
    learned = frame[frame.run.str.startswith("step-")].copy()
    learned["pretraining_updates"] = learned.run.str.removeprefix("step-").astype(int)
    learned = learned.sort_values("pretraining_updates")
    if tuple(learned.pretraining_updates) != STEPS or len(learned) != len(STEPS):
        raise ValueError("GraphSAGE fixed-budget trajectory is incomplete")
    if set(learned.feature_dim) != {64}:
        raise ValueError("GraphSAGE representation width changed")
    return learned, cache


def plot(frame: pd.DataFrame, output: Path) -> None:
    positions = range(len(frame))
    labels = [f"{value:,}" for value in frame.pretraining_updates]
    figure, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    axes[0].plot(positions, frame.test_roc_auc, marker="o", color="#4477AA", linewidth=2)
    axes[0].set_ylabel("test ROC-AUC")
    axes[0].set_ylim(0.755, 0.765)
    axes[1].plot(positions, frame.test_accuracy, marker="o", label="accuracy", color="#228833")
    axes[1].plot(positions, frame.test_f1, marker="s", label="binary F1", color="#CC6677")
    axes[1].set_ylabel("test score")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.set_xticks(list(positions), labels, rotation=35)
        axis.set_xlabel("native link-prediction updates")
        axis.grid(alpha=0.25)
    figure.suptitle("GraphSAGE pilot-v1 → TwiBot CLS saturation\nfull-label official split; one training seed")
    figure.tight_layout()
    figure.savefig(output.with_suffix(".png"), dpi=180, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        default=ROOT / "data" / "graphsage_pilot_v1_twibot_cls_trajectory_raw" / "results.json",
    )
    args = parser.parse_args()
    frame, _ = load_result(args.input)
    frame.to_csv(ROOT / "data" / "graphsage_pilot_v1_twibot_cls_trajectory.csv", index=False)
    plot(frame, ROOT / "figures" / "graphsage_pilot_v1_twibot_cls_saturation")
    delta = frame.iloc[-1].test_roc_auc - frame.iloc[0].test_roc_auc
    print(f"GRAPHSAGE_SATURATION_OK cells={len(frame)} terminal_minus_init_auc={delta:+.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
