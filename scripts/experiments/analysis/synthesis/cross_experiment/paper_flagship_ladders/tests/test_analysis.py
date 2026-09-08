import pandas as pd

from scripts.experiments.analysis.synthesis.cross_experiment.paper_flagship_ladders.analyze import (
    ARMS,
    RUNGS,
    SEEDS,
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
