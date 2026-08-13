import json
import sys

import pandas as pd

from scripts.experiments.setup.icl_arch_matrix.aggregate_random_init import main
from scripts.experiments.setup.icl_arch_matrix.aggregate_results import (
    ARCHITECTURES,
    TARGETS,
)


def test_random_init_aggregate_requires_and_compares_exact_episode_grid(tmp_path, monkeypatch):
    fingerprints = {target: f"fingerprint-{target}" for target in TARGETS}
    raw_paths = {}
    for architecture_index, architecture in enumerate(ARCHITECTURES):
        path = tmp_path / f"{architecture}.jsonl"
        rows = []
        for target_index, target in enumerate(TARGETS):
            score = 0.45 + 0.01 * architecture_index + 0.02 * target_index
            rows.append(
                {
                    "architecture": architecture,
                    "model_id": "random_init",
                    "sources": [],
                    "seed": 0,
                    "checkpoint_step": 0,
                    "baseline": "random_init",
                    "task": "classification",
                    "dataset": target,
                    "n_way": 2,
                    "n_shot": 10,
                    "n_query": 1,
                    "episodes": 128,
                    "queries": 256,
                    "episode_fingerprint": fingerprints[target],
                    "roc_auc": score,
                    "accuracy": 0.5,
                    "f1": 0.5,
                }
            )
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        raw_paths[architecture] = path

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
            "aggregate_random_init.py",
            "--prodigy", str(raw_paths["prodigy"]),
            "--vision", str(raw_paths["vision"]),
            "--gilt", str(raw_paths["gilt"]),
            "--trained-reference", str(trained_path),
            "--output-root", str(output_root),
        ],
    )

    assert main() == 0
    summary = json.loads((output_root / "summary.json").read_text())
    assert summary["rows"] == 12
    assert summary["checkpoint_step"] == 0
    assert summary["update100_mean_roc_auc"] == {
        architecture: 0.75 for architecture in ARCHITECTURES
    }
    assert len(pd.read_csv(output_root / "classification_long.csv")) == 12
    assert len(pd.read_csv(output_root / "target_summary.csv")) == 12
