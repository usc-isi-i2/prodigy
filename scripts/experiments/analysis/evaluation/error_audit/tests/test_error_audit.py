#!/usr/bin/env python3
"""Offline gates for correct/error labelling and balanced card selection."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = next(p for p in Path(__file__).resolve().parents if (p / "AGENTS.md").is_file())
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.eval.pair_link_eval import lock_decision_threshold  # noqa: E402
from scripts.eval.regression_probe import (  # noqa: E402
    EpisodeSet, probe_prediction_records, probe_spearman, score_prediction_records,
)
from scripts.experiments.analysis.evaluation.error_audit.build_error_report import (  # noqa: E402
    assign_group, balanced_sample,
)


def main() -> int:
    failures = []

    def check(condition, label):
        print(f"  {'OK  ' if condition else 'FAIL'} {label}")
        if not condition:
            failures.append(label)

    labels = np.array([0, 0, 1, 1, 0, 1])
    scores = np.array([-0.9, -0.7, 0.7, 0.9, -0.8, 0.8])
    val = np.array([1, 1, 1, 1, 0, 0], dtype=bool)
    threshold, bal = lock_decision_threshold(labels, scores, val, orientation=1)
    pred = (scores[~val] >= threshold).astype(int)
    check(bal == 1.0, "LP threshold is selected on validation")
    check(np.array_equal(pred, labels[~val]), "locked LP threshold classifies held-out pairs")

    features = np.arange(20, dtype=float).reshape(10, 2)
    episode = EpisodeSet(
        support=np.array([[0, 1, 2]]),
        query=np.array([[3, 4]]),
        nodes=np.arange(10) * 10,
        target=np.arange(10, dtype=float),
    )
    reg = probe_prediction_records(features, episode, alpha=1.0)
    check(len(reg) == 2, "regression exports every query")
    check(reg[0]["query_node_id"] == 30, "regression export preserves graph node id")
    check(reg[0]["support_node_ids"] == [0, 10, 20], "regression export preserves supports")
    direct = probe_spearman(features, episode, alpha=1.0)
    rescored = score_prediction_records(reg, alpha=1.0)
    check(abs(direct["rmse"] - rescored["rmse"]) < 1e-12,
          "exported regression rows reproduce aggregate metrics")

    rows = [
        {"task": "regression", "model": "m", "dataset": "d", "target": "t",
         "alpha": 1.0, "absolute_error": float(i)}
        for i in range(10)
    ]
    assign_group(rows)
    counts = {group: sum(row["audit_group"] == group for row in rows)
              for group in ("low_error", "middle_error", "high_error")}
    check(counts == {"low_error": 2, "middle_error": 6, "high_error": 2},
          "regression error quintiles are deterministic")

    binary = [
        {"task": "classification", "correct": bool(i % 2), "confidence": i / 10}
        for i in range(20)
    ]
    assign_group(binary)
    sample = balanced_sample(binary, per_group=4, seed=0)
    sample_counts = {group: sum(row["audit_group"] == group for row in sample)
                     for group in ("correct", "incorrect")}
    check(sample_counts == {"correct": 4, "incorrect": 4},
          "report sample contains equal correct and incorrect counts")

    print(f"\n{len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
