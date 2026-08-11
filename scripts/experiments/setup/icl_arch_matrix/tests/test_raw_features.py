import json
import sys

import pandas as pd
import torch

from scripts.experiments.setup.icl_arch_matrix.aggregate_raw_features import main
from scripts.experiments.setup.icl_arch_matrix.aggregate_results import (
    ARCHITECTURES,
    TARGETS,
)
from scripts.experiments.setup.icl_arch_matrix.evaluate_raw_features import (
    BASELINES,
    _prototype_logits,
)


def test_prototype_logits_use_support_class_means():
    support_x = torch.tensor([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.2, 0.8]])
    support_y = torch.tensor([0, 0, 1, 1])
    query_x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    logits = _prototype_logits(support_x, support_y, query_x)
    assert logits.argmax(1).tolist() == [0, 1]


def test_raw_feature_aggregate_requires_frozen_episode_grid(tmp_path, monkeypatch):
    fingerprints = {target: f"fingerprint-{target}" for target in TARGETS}
    rows = []
    for baseline_index, baseline in enumerate(BASELINES):
        for target_index, target in enumerate(TARGETS):
            rows.append(
                {
                    "baseline": baseline,
                    "model_id": baseline,
                    "sources": [],
                    "seed": 0,
                    "training_updates": 0,
                    "feature_view": "l2_normalized_raw_768d_center",
                    "topology_used": False,
                    "support_fit": "none" if baseline == "raw_cosine_prototype" else "logistic_c1",
                    "task": "classification",
                    "dataset": target,
                    "n_way": 2,
                    "n_shot": 10,
                    "n_query": 1,
                    "episodes": 128,
                    "queries": 256,
                    "episode_fingerprint": fingerprints[target],
                    "roc_auc": 0.50 + 0.02 * baseline_index + 0.01 * target_index,
                    "accuracy": 0.5,
                    "f1": 0.5,
                }
            )
    raw_path = tmp_path / "raw.jsonl"
    raw_path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    trained_rows = []
    for architecture in ARCHITECTURES:
        for model_index in range(31):
            for target in TARGETS:
                trained_rows.append(
                    {
                        "architecture": architecture,
                        "model_id": f"model-{model_index}",
                        "dataset": target,
                        "episode_fingerprint": fingerprints[target],
                        "roc_auc": 0.75,
                    }
                )
    trained_path = tmp_path / "trained.csv"
    pd.DataFrame(trained_rows).to_csv(trained_path, index=False)
    output_root = tmp_path / "summary"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "aggregate_raw_features.py",
            "--results", str(raw_path),
            "--trained-reference", str(trained_path),
            "--output-root", str(output_root),
        ],
    )

    assert main() == 0
    summary = json.loads((output_root / "summary.json").read_text())
    assert summary["rows"] == 8
    assert summary["mean_roc_auc"] == {
        "raw_cosine_prototype": 0.515,
        "raw_logistic": 0.535,
    }
    assert len(pd.read_csv(output_root / "classification_long.csv")) == 8
    assert len(pd.read_csv(output_root / "architecture_comparison.csv")) == 6
