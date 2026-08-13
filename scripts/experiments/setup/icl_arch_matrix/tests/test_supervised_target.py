import json
import sys

import pandas as pd
import torch

from scripts.experiments.setup.icl_arch_matrix.aggregate_results import ARCHITECTURES, TARGETS
from scripts.experiments.setup.icl_arch_matrix.aggregate_supervised_target import main
from scripts.experiments.setup.icl_arch_matrix.evaluate_supervised_target import (
    BASELINES,
    SupervisedMLP,
)


def test_supervised_mlp_prediction_does_not_use_edges():
    torch.manual_seed(0)
    model = SupervisedMLP(input_dim=3, hidden_dim=4).eval()
    x = torch.randn(5, 3)
    centers = torch.tensor([0, 3])
    logits_a = model(x, torch.tensor([[0], [1]]), centers)
    logits_b = model(x, torch.tensor([[4, 2], [1, 0]]), centers)
    assert torch.equal(logits_a, logits_b)


def test_supervised_aggregate_requires_frozen_episode_grid(tmp_path, monkeypatch):
    fingerprints = {target: f"fingerprint-{target}" for target in TARGETS}
    paths = {}
    for baseline_index, baseline in enumerate(BASELINES):
        rows = []
        for target_index, target in enumerate(TARGETS):
            rows.append(
                {
                    "baseline": baseline,
                    "model_id": baseline,
                    "sources": [],
                    "seed": 0,
                    "training_updates": 100,
                    "training_label_scope": "target_train_split",
                    "validation_selection": "best_of_two_fixed_lrs_at_update100",
                    "lr_grid": [0.001, 0.0003],
                    "selected_lr": 0.001,
                    "selected_val_roc_auc": 0.7,
                    "tuning_results": [{"lr": 0.001, "val_roc_auc": 0.7}],
                    "train_loss_final": 0.5,
                    "labeled_centers_seen": 4400,
                    "node_split_sizes": {"train": 60, "val": 20, "test": 20},
                    "parameters": 100,
                    "raw_features_used": True,
                    "topology_used": baseline == "supervised_graphsage",
                    "test_support_labels_used": False,
                    "query_labels_used_for_selection": False,
                    "task": "classification",
                    "dataset": target,
                    "n_way": 2,
                    "n_shot": 10,
                    "n_query": 1,
                    "episodes": 128,
                    "queries": 256,
                    "episode_fingerprint": fingerprints[target],
                    "roc_auc": 0.65 + 0.02 * baseline_index + 0.01 * target_index,
                    "accuracy": 0.6,
                    "f1": 0.6,
                }
            )
        path = tmp_path / f"{baseline}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        paths[baseline] = path

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
            "aggregate_supervised_target.py",
            "--mlp", str(paths["supervised_mlp"]),
            "--graphsage", str(paths["supervised_graphsage"]),
            "--trained-reference", str(trained_path),
            "--output-root", str(output_root),
        ],
    )

    assert main() == 0
    summary = json.loads((output_root / "summary.json").read_text())
    assert summary["rows"] == 8
    assert summary["test_support_labels_used"] is False
    assert len(pd.read_csv(output_root / "classification_long.csv")) == 8
    assert len(pd.read_csv(output_root / "architecture_comparison.csv")) == 6
