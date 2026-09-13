import json

import pytest

from scripts.experiments.analysis.synthesis.cross_experiment.native_model_result_matrix.analyze_vision_mixture_three_seed import (
    CHECKPOINTS,
    EXPECTED_EXPANDED_CELLS,
    EXPECTED_PHYSICAL_CELLS,
    SEEDS,
    TARGETS,
    expand_orders,
    load_cells,
    main,
    summarize,
)
from scripts.experiments.setup.vision_native_mixture_finalcore.mixture_plan import (
    build_mixture_models,
)


def write_rows(path, models, seeds):
    with path.open("w", encoding="utf-8") as handle:
        for model in models:
            for seed in seeds:
                for step in CHECKPOINTS:
                    for target in TARGETS:
                        handle.write(json.dumps({
                            "architecture": "vision",
                            "task": "classification",
                            "model_id": model.model_id,
                            "sources": list(model.sources),
                            "training_seed": seed,
                            "checkpoint_step": step,
                            "dataset": target,
                            "episode_fingerprint": f"fixed-{target}",
                            "roc_auc": 0.6 + seed / 100,
                            "accuracy": 0.55 + seed / 100,
                            "f1": 0.5 + seed / 100,
                        }) + "\n")


def complete_inputs(tmp_path):
    models = build_mixture_models()
    mixture_roots = []
    for seed in SEEDS:
        root = tmp_path / f"seed{seed}"
        root.mkdir()
        write_rows(
            root / "mixtures.jsonl",
            [model for model in models if model.model_id != "all9"],
            [seed],
        )
        mixture_roots.append(root)
    all9 = tmp_path / "all9"
    all9.mkdir()
    write_rows(
        all9 / "all9.jsonl",
        [model for model in models if model.model_id == "all9"],
        SEEDS,
    )
    return mixture_roots, all9


def test_complete_three_seed_grid_and_order_expansion(tmp_path):
    mixture_roots, all9 = complete_inputs(tmp_path)
    frame = load_cells(mixture_roots, all9)
    assert len(frame) == EXPECTED_PHYSICAL_CELLS == 780
    expanded = expand_orders(frame)
    assert len(expanded) == EXPECTED_EXPANDED_CELLS == 900
    per_target, summary = summarize(expanded)
    assert len(per_target) == 900
    assert summary.training_seeds.eq(3).all()
    assert summary.roc_auc_min.eq(0.60).all()
    assert summary.roc_auc_max.eq(0.62).all()


def test_three_seed_analysis_rejects_missing_seed(tmp_path):
    mixture_roots, all9 = complete_inputs(tmp_path)
    with pytest.raises(ValueError, match="expected training seeds"):
        load_cells(mixture_roots[:2], all9)


def test_three_seed_analysis_writes_tables_and_figures(tmp_path, monkeypatch):
    mixture_roots, all9 = complete_inputs(tmp_path)
    output = tmp_path / "analysis"
    argv = ["analyze_vision_mixture_three_seed.py"]
    for root in mixture_roots:
        argv.extend(["--mixture-root", str(root)])
    argv.extend(["--all9-root", str(all9), "--output", str(output)])
    monkeypatch.setattr("sys.argv", argv)
    assert main() == 0
    assert (output / "data" / "vision_native_mixture_three_seed_summary.csv").is_file()
    assert (output / "figures" / "vision_mixture_diversity_three_seed_trajectory.png").is_file()
    assert (output / "figures" / "vision_mixture_diversity_three_seed_terminal_targets.pdf").is_file()
