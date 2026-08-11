import json
import sys
from types import SimpleNamespace

import pandas as pd
import torch

from scripts.experiments.setup.icl_arch_matrix.aggregate_results import (
    ARCHITECTURES,
    TARGETS,
)
from scripts.experiments.setup.icl_arch_matrix.aggregate_topology_features import main
from scripts.experiments.setup.icl_arch_matrix.evaluate_topology_features import (
    BASELINES,
    FEATURE_NAMES,
    FEATURE_VIEW,
    _topology_features,
)


def test_topology_features_use_only_directed_graph_degrees():
    graph = SimpleNamespace(
        edge_index=torch.tensor([[0, 0, 1, 2], [1, 2, 2, 0]]),
        num_nodes=3,
    )
    features = _topology_features(SimpleNamespace(graph=graph))
    assert features.shape == (3, 3)
    assert torch.isfinite(features).all()
    assert torch.allclose(features.mean(0), torch.zeros(3), atol=1e-6)


def test_topology_aggregate_requires_frozen_episode_grid(tmp_path, monkeypatch):
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
                    "feature_view": FEATURE_VIEW,
                    "feature_names": list(FEATURE_NAMES),
                    "topology_scope": "entire_loaded_target_graph",
                    "topology_used": True,
                    "raw_features_used": False,
                    "support_fit": "none" if baseline.endswith("prototype") else "logistic_c1",
                    "task": "classification",
                    "dataset": target,
                    "n_way": 2,
                    "n_shot": 10,
                    "n_query": 1,
                    "episodes": 128,
                    "queries": 256,
                    "episode_fingerprint": fingerprints[target],
                    "roc_auc": 0.48 + 0.02 * baseline_index + 0.01 * target_index,
                    "accuracy": 0.5,
                    "f1": 0.5,
                }
            )
    result_path = tmp_path / "topology.jsonl"
    result_path.write_text("".join(json.dumps(row) + "\n" for row in rows))

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
            "aggregate_topology_features.py",
            "--results", str(result_path),
            "--trained-reference", str(trained_path),
            "--output-root", str(output_root),
        ],
    )

    assert main() == 0
    summary = json.loads((output_root / "summary.json").read_text())
    assert summary["rows"] == 8
    assert summary["raw_features_used"] is False
    assert summary["mean_roc_auc"] == {
        "topology_degree_prototype": 0.495,
        "topology_degree_logistic": 0.515,
    }
    assert len(pd.read_csv(output_root / "classification_long.csv")) == 8
    assert len(pd.read_csv(output_root / "architecture_comparison.csv")) == 6
