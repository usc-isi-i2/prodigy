import numpy as np
import pandas as pd

from scripts.experiments.analysis.graphs.transfer_prediction.trace_schedule_scaling.analyze_results import (
    health_probability_fusion,
    interaction_rows,
    paired_contrasts,
    selector_contrasts,
    selector_gain_rows,
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


def test_health_probability_fusion_masks_and_falls_back() -> None:
    record = {
        "models": {
            "a": {"logits": {"full_model": np.log([[.8, .2], [.9, .1]])}},
            "b": {"logits": {"full_model": np.log([[.4, .6], [.2, .8]])}},
            "c": {"logits": {"full_model": np.log([[.6, .4], [.3, .7]])}},
        }
    }
    agreement = {
        "a": np.array([True, False]),
        "b": np.array([False, False]),
        "c": np.array([True, False]),
    }
    fused = np.exp(
        health_probability_fusion(record, ["a", "b", "c"], agreement)
    )
    np.testing.assert_allclose(fused[0], [.7, .3])
    np.testing.assert_allclose(fused[1], [1.4 / 3, 1.6 / 3])


def test_selector_gain_summary_uses_fixed_reference() -> None:
    rows = []
    for target in ("a", "b"):
        for rung in (2, 3, 4):
            for seed in (0, 1, 2):
                for method, delta in (
                    ("discovery_auc_fixed", 0.0),
                    ("trace_health_fusion", 0.02),
                ):
                    rows.append(
                        {
                            "target": target,
                            "rung": rung,
                            "seed": seed,
                            "method": method,
                            "target_supported_at_0_55": target == "a",
                            "accuracy": 0.7 + delta,
                            "roc_auc": 0.8 + delta,
                            "nll": 0.5 - delta,
                        }
                    )
    contrasts = selector_contrasts(pd.DataFrame(rows))
    assert len(contrasts) == 18
    summary = selector_gain_rows(contrasts, draws=100)
    supported = summary[
        summary.scope.eq("supported_targets")
        & summary.method.eq("trace_health_fusion")
        & summary.metric.eq("accuracy")
    ].iloc[0]
    assert abs(supported.mean_delta - 0.02) < 1e-12
    assert supported.cell_wins == 9
