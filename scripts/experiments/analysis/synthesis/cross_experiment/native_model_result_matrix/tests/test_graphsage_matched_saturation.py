import pandas as pd
import pytest

from scripts.experiments.analysis.synthesis.cross_experiment.native_model_result_matrix.analyze_graphsage_matched_saturation import (
    EXPECTED_ROWS,
    LABEL_SEEDS,
    STEPS,
    TARGETS,
    main,
    validate_cells,
)


def complete_cells():
    rows = []
    for step in STEPS:
        for target in TARGETS:
            for seed in LABEL_SEEDS:
                for budget in (0, 1, 10, 100):
                    updates = (0,) if budget == 0 else (0, 1, 10, 100)
                    for update in updates:
                        for split in ("val", "test"):
                            rows.append({
                                "model_id": f"graphsage_pilot_v1_step{step}",
                                "target": target,
                                "label_seed": seed,
                                "label_budget_per_class": budget,
                                "head_updates": update,
                                "split": split,
                                "selected_nodes_fingerprint": f"sample-{target}-{seed}-{budget}",
                                "split_fingerprint": f"split-{target}",
                                "head_initialization_fingerprint": f"head-{target}-{seed}",
                                "optimizer": "none" if budget == 0 else "AdamW",
                                "learning_rate": 0.0 if budget == 0 else 0.01,
                                "weight_decay": 0.0,
                                "roc_auc": 0.55 + step / 100_000,
                                "accuracy": 0.52 + step / 100_000,
                                "macro_f1": 0.50 + step / 100_000,
                            })
    return pd.DataFrame(rows)


def test_matched_graphsage_grid_is_exact():
    cells = complete_cells()
    assert len(cells) == EXPECTED_ROWS == 2184
    validate_cells(cells)
    with pytest.raises(ValueError, match="expected 2184"):
        validate_cells(cells.iloc[:-1])

    mismatched = cells.copy()
    mismatched.loc[mismatched.model_id.str.endswith("step2000"), "split_fingerprint"] = "changed"
    with pytest.raises(ValueError, match="split fingerprints"):
        validate_cells(mismatched)


def test_matched_graphsage_analysis_preserves_full_grid(tmp_path, monkeypatch):
    cells_path = tmp_path / "cells.csv"
    output = tmp_path / "analysis"
    complete_cells().to_csv(cells_path, index=False)
    monkeypatch.setattr(
        "sys.argv",
        ["analyze_graphsage_matched_saturation.py", "--cells", str(cells_path), "--output", str(output)],
    )
    assert main() == 0
    full = pd.read_csv(output / "data" / "graphsage_matched_saturation_cells.csv")
    assert len(full) == EXPECTED_ROWS
    assert (output / "figures" / "graphsage_matched_saturation_full_grid.png").is_file()
    assert (output / "figures" / "graphsage_matched_saturation_by_target.pdf").is_file()
