import pandas as pd

from scripts.experiments.analysis.graphs.transfer_prediction.trace_schedule_scaling.analyze_results import (
    interaction_rows,
    paired_contrasts,
)


def synthetic_cells():
    rows = []
    for stream in ("original", "fresh"):
        for target in ("a", "b"):
            for rung in (2, 3, 4):
                for seed in (0, 1, 2):
                    for schedule in ("blocked", "replay100", "interleaved"):
                        delta = 0.0
                        if schedule == "blocked":
                            delta = 0.03 if rung == 2 else -0.01
                        elif schedule == "replay100":
                            delta = 0.01
                        rows.append(
                            {
                                "target": target,
                                "stream": stream,
                                "rung": rung,
                                "seed": seed,
                                "schedule": schedule,
                                "accuracy": 0.7 + delta,
                                "roc_auc": 0.8 + delta,
                                "nll": 0.5 - delta,
                                "u1_accuracy": 0.7 + delta,
                                "u1_agreement": 0.75 + delta,
                                "support_competence": 0.65 + delta,
                            }
                        )
    return pd.DataFrame(rows)


def test_paired_contrasts_and_scaling_interaction() -> None:
    contrasts = paired_contrasts(synthetic_cells())
    assert len(contrasts) == 72
    interactions = interaction_rows(contrasts)
    blocked = interactions[
        interactions.schedule.eq("blocked")
        & interactions.stream.eq("fresh")
        & interactions.scope.eq("all_targets")
        & interactions.metric.eq("roc_auc")
    ].iloc[0]
    assert abs(blocked.mean_delta_rung2 - 0.03) < 1e-12
    assert abs(blocked.mean_delta_rungs3_4 + 0.01) < 1e-12
    assert abs(blocked.rung2_minus_larger_interaction - 0.04) < 1e-12
