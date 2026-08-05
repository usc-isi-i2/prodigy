#!/usr/bin/env python3
"""Decode gate for NM/classification episodic prediction rows."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

REPO = Path(__file__).resolve().parents[5]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.trainer import TrainerFS  # noqa: E402


def main() -> int:
    trainer = TrainerFS.__new__(TrainerFS)
    trainer.parameter = {
        "export_predictions": True,
        "task_name": "neighbor_matching",
        "prediction_context_neighbors": 3,
        "prediction_support_per_label": 1,
    }
    trainer.dataset_name = "toy"

    graph = SimpleNamespace(
        center_node_idx=torch.tensor([10, 11, 20, 21]),
        task_id_per_sample=torch.zeros(4, dtype=torch.long),
        task_label_map=torch.tensor([[100, 200]]),
    )
    labels = torch.tensor([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=torch.float)
    query_per_class_edge = torch.tensor(
        [False, False, True, True, False, False, True, True]
    )
    batch = [graph, None, labels, None, None, query_per_class_edge]
    yt = torch.tensor([[1, 0], [0, 1]], dtype=torch.float)
    yp = torch.tensor([[0.1, 0.9], [0.2, 0.8]], dtype=torch.float)
    raw = {
        "ptr": torch.tensor([0, 3, 6, 9, 12]),
        "global_node_ids": torch.tensor([
            10, 12, -1, 11, 13, -1, 20, 22, -1, 21, 23, -1,
        ]),
    }
    records = trainer._prediction_records_for_batch(
        batch, yt, yp, "test", 0, raw
    )
    assert len(records) == 2
    assert records[0]["query_node_id"] == 11
    assert records[0]["context_node_ids"] == [13]
    assert records[0]["gt"] == 100 and records[0]["prediction"] == 200
    assert records[0]["correct"] is False
    assert records[1]["correct"] is True
    assert {row["node_id"] for row in records[0]["supports"]} == {10, 20}
    print("episode export gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
