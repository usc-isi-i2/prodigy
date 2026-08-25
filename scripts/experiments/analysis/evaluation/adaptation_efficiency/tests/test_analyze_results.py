import pandas as pd
import pytest

from scripts.experiments.analysis.evaluation.adaptation_efficiency.analyze_results import (
    EXPECTED_MODELS,
    EXPECTED_ROWS,
    EXPECTED_TARGETS,
    main,
    validate_shared_protocol,
)


def complete_cells():
    rows = []
    for model in sorted(EXPECTED_MODELS):
        for target in sorted(EXPECTED_TARGETS):
            for label_seed in (0, 1, 2):
                for budget in (0, 1, 10, 100):
                    updates = (0,) if budget == 0 else (0, 1, 10, 100)
                    for update in updates:
                        for split in ("val", "test"):
                            rows.append({
                                "model_id": model,
                                "target": target,
                                "label_seed": label_seed,
                                "label_budget_per_class": budget,
                                "head_updates": update,
                                "split": split,
                                "selected_nodes_fingerprint": f"sample-{target}-{label_seed}-{budget}",
                                "split_fingerprint": f"split-{target}",
                                "head_initialization_fingerprint": (
                                    f"mlp-{target}-{label_seed}"
                                    if model == "raw_mlp"
                                    else f"linear-{target}-{label_seed}"
                                ),
                                "optimizer": "none" if budget == 0 else "AdamW",
                                "learning_rate": 0.0 if budget == 0 else 0.01,
                                "weight_decay": 0.0,
                                "roc_auc": 0.6,
                                "accuracy": 0.55,
                                "macro_f1": 0.5,
                            })
    return pd.DataFrame(rows)


def test_exact_grid_and_raw_logistic_head_match_are_enforced():
    cells = complete_cells()
    assert len(cells) == EXPECTED_ROWS == 3744
    validate_shared_protocol(cells)

    incomplete = cells.iloc[:-1]
    with pytest.raises(ValueError, match="expected 3744"):
        validate_shared_protocol(incomplete)

    mismatched = cells.copy()
    mask = mismatched.model_id == "raw_logistic"
    mismatched.loc[mask, "head_initialization_fingerprint"] = "different"
    with pytest.raises(ValueError, match="raw logistic"):
        validate_shared_protocol(mismatched)


def test_complete_grid_generates_full_and_efficiency_outputs(tmp_path, monkeypatch):
    cells_path = tmp_path / "cells.csv"
    output = tmp_path / "analysis"
    complete_cells().to_csv(cells_path, index=False)
    monkeypatch.setattr(
        "sys.argv", ["analyze_results.py", "--cells", str(cells_path), "--output", str(output)]
    )
    assert main() == 0
    assert len(pd.read_csv(output / "data" / "adaptation_cells_full.csv")) == EXPECTED_ROWS
    assert (output / "data" / "label_efficiency_auc.csv").is_file()
    assert (output / "data" / "updates_to_95pct_summary.csv").is_file()
    findings = (output / "FINDINGS.md").read_text()
    assert "Label-efficiency summary" in findings
    assert "Optimization-efficiency summary" in findings
