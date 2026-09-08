import pandas as pd
import pytest

from scripts.experiments.analysis.synthesis.cross_experiment.native_model_result_matrix.plot_native_mixture_ladder_three_seed import (
    MODEL_RUNGS,
    TARGETS,
    summarize,
    validate_cells,
)


def complete_cells():
    rows = []
    for model_index, (model, rungs) in enumerate(MODEL_RUNGS.items()):
        for order in ("A", "B", "C"):
            for rung in rungs:
                for seed_index, seed in enumerate((10, 20, 30)):
                    for target_index, target in enumerate(TARGETS):
                        rows.append(
                            {
                                "model": model,
                                "order": order,
                                "rung": rung,
                                "training_seed": seed,
                                "target": target,
                                "roc_auc": 0.5 + model_index / 10 + seed_index / 100 + target_index / 1000,
                                "fingerprint": f"{model}-{target}",
                            }
                        )
    return pd.DataFrame(rows)


def test_cross_family_grid_keeps_training_seed_as_replication_unit():
    cells = complete_cells()
    validate_cells(cells)
    assert len(cells) == 1035
    per_seed, summary = summarize(cells)
    assert len(per_seed) == 207
    assert len(summary) == 69
    row = summary[
        summary.model.eq("PRODIGY") & summary.order.eq("A") & summary.rung.eq(1)
    ].iloc[0]
    assert row.training_seeds == 3
    assert row.roc_auc_min == pytest.approx(0.502)
    assert row.roc_auc_max == pytest.approx(0.522)


def test_cross_family_grid_rejects_episode_drift():
    cells = complete_cells()
    cells.loc[
        cells.model.eq("VISION") & cells.target.eq(TARGETS[0]) & cells.training_seed.eq(20),
        "fingerprint",
    ] = "changed"
    with pytest.raises(ValueError, match="episode fingerprint drift"):
        validate_cells(cells)
