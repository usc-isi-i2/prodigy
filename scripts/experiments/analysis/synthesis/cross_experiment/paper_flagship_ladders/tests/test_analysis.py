import pandas as pd
import pytest

from scripts.experiments.analysis.synthesis.cross_experiment.paper_flagship_ladders.analyze import (
    ARMS,
    CAPACITY_ARMS,
    RUNGS,
    SEEDS,
    capacity_per_seed,
    seed_summary,
)


def test_seed_summary_uses_observed_seed_range():
    rows = []
    for arm_index, arm in enumerate(ARMS):
        for rung in RUNGS:
            for seed in SEEDS:
                rows.append(
                    {
                        "arm": arm,
                        "rung": rung,
                        "training_seed": seed,
                        "nm_fixed_panel": 0.5 + 0.01 * arm_index + 0.001 * seed,
                        "cls_fixed_panel": 0.6 + 0.01 * arm_index + 0.002 * seed,
                    }
                )
    summary = seed_summary(pd.DataFrame(rows))
    row = summary[
        (summary.arm == "baseline")
        & (summary.rung == 1)
        & (summary.metric == "nm_fixed_panel")
    ].iloc[0]
    assert row.seed_count == 3
    assert row["min"] == 0.5
    assert row["max"] == 0.502
    assert row["mean"] == 0.501


def test_capacity_per_seed_keeps_fixed_target_macro_separate():
    rows = []
    for arm_index, arm in enumerate(CAPACITY_ARMS):
        for rung in RUNGS:
            for seed in SEEDS:
                for target in range(9):
                    rows.append(
                        {
                            "arm": arm,
                            "rung": rung,
                            "training_seed": seed,
                            "roc_auc": 0.5 + 0.01 * arm_index + 0.001 * target,
                        }
                    )
    result = capacity_per_seed(pd.DataFrame(rows))
    assert len(result) == 2 * 8 * 3
    wide = result[
        (result.arm == "capacity") & (result.rung == 8) & (result.training_seed == 2)
    ].iloc[0]
    assert wide.nm_fixed_panel == pytest.approx(0.514)
