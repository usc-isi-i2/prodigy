import json

from scripts.experiments.analysis.synthesis.cross_experiment.native_model_result_matrix.analyze_vision_mixture import (
    CHECKPOINTS,
    TARGETS,
    expand_orders,
    load_cells,
)
from scripts.experiments.setup.vision_native_mixture_finalcore.mixture_plan import (
    build_mixture_models,
)


def write_rows(path, models):
    with path.open("w", encoding="utf-8") as handle:
        for model in models:
            for step in CHECKPOINTS:
                for target in TARGETS:
                    handle.write(json.dumps({
                        "architecture": "vision",
                        "task": "classification",
                        "model_id": model.model_id,
                        "sources": list(model.sources),
                        "training_seed": 0,
                        "checkpoint_step": step,
                        "dataset": target,
                        "episode_fingerprint": f"fixed-{target}",
                        "roc_auc": 0.6,
                        "accuracy": 0.55,
                        "f1": 0.5,
                    }) + "\n")


def test_complete_mixture_grid_and_order_expansion(tmp_path):
    new_root = tmp_path / "new"
    all9_root = tmp_path / "all9"
    new_root.mkdir()
    all9_root.mkdir()
    models = build_mixture_models()
    write_rows(new_root / "new.jsonl", [model for model in models if model.model_id != "all9"])
    write_rows(all9_root / "all9.jsonl", [model for model in models if model.model_id == "all9"])
    frame = load_cells(new_root, all9_root)
    assert len(frame) == 260
    expanded = expand_orders(frame)
    assert len(expanded) == 300
    assert set(expanded.order) == {"A", "B", "C"}
