import json

from scripts.experiments.analysis.synthesis.cross_experiment.native_model_result_matrix.analyze_vision_cross_ssl import (
    CHECKPOINTS,
    TARGETS,
    load_cells,
)


def test_complete_native_cross_ssl_grid(tmp_path):
    path = tmp_path / "cells.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for source in TARGETS:
            for target in TARGETS:
                for step in CHECKPOINTS:
                    handle.write(json.dumps({
                        "task": "native_feature_similarity_ssl",
                        "source": source,
                        "target": target,
                        "checkpoint_step": step,
                        "training_seed": 0,
                        "episode_fingerprint": f"fixed-{target}",
                        "pseudo_classification_accuracy": 0.2,
                        "native_ssl_loss": 2.0,
                    }) + "\n")
    frame = load_cells(tmp_path)
    assert len(frame) == 125
    assert frame.groupby("target").episode_fingerprint.nunique().eq(1).all()
