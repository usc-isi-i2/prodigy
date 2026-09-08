import pandas as pd
import pytest

from scripts.experiments.analysis.synthesis.cross_experiment.paper_flagship_ladders.analyze import (
    ARMS,
    CAPACITY_ARMS,
    RUNGS,
    SEEDS,
    capacity_per_seed,
    ladder_area_per_seed,
    ladder_area_summary,
    scientific_decision,
    seed_summary,
    validate_grid,
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


def test_capacity_grid_uses_nm_episode_count_not_task_label_heuristics():
    rows = []
    for arm in CAPACITY_ARMS:
        for rung in RUNGS:
            for seed in SEEDS:
                for target in ("a", "b"):
                    rows.append(
                        {
                            "arm": arm,
                            "rung": rung,
                            "training_seed": seed,
                            "target": target,
                            "roc_auc": 0.5,
                            "fingerprint": f"fixed-{target}",
                            "episodes": 512,
                        }
                    )
    frame = pd.DataFrame(rows)
    validate_grid(
        frame,
        ("a", "b"),
        "capacity NM",
        CAPACITY_ARMS,
        expected_episodes=512,
    )
    with pytest.raises(ValueError, match="expected=128"):
        validate_grid(
            frame,
            ("a", "b"),
            "capacity NM",
            CAPACITY_ARMS,
            expected_episodes=128,
        )


def complete_per_seed(winner="baseline"):
    rows = []
    for arm_index, arm in enumerate(ARMS):
        for rung in RUNGS:
            for seed in SEEDS:
                bonus = 0.02 if arm == winner else 0.0
                rows.append(
                    {
                        "arm": arm,
                        "rung": rung,
                        "training_seed": seed,
                        "nm_fixed_panel": 0.5 + bonus + 0.001 * rung + 0.0001 * seed,
                        "cls_fixed_panel": 0.6 + bonus + 0.001 * rung + 0.0001 * seed,
                        "nm_included": 0.55 + bonus + 0.001 * rung,
                        "nm_future_sources": float("nan") if rung == 8 else 0.45 + bonus,
                        "nm_heldout_twibot": 0.48 + bonus + 0.001 * rung,
                    }
                )
    return pd.DataFrame(rows)


def test_ladder_area_is_within_seed_and_uses_seven_finite_future_rungs():
    area = ladder_area_per_seed(complete_per_seed())
    assert len(area) == len(ARMS) * len(SEEDS) * 5
    assert set(area.loc[area.metric.eq("nm_fixed_panel"), "rungs_in_area"]) == {8}
    assert set(area.loc[area.metric.eq("nm_future_sources"), "rungs_in_area"]) == {7}
    baseline = area[
        area.arm.eq("baseline")
        & area.training_seed.eq(0)
        & area.metric.eq("nm_fixed_panel")
    ].iloc[0]
    assert baseline.normalized_ladder_area == pytest.approx(0.5245)


def test_scientific_gate_requires_endpoint_and_whole_curve_on_both_panels():
    baseline = complete_per_seed("baseline")
    baseline_area = ladder_area_summary(ladder_area_per_seed(baseline))
    decision = scientific_decision(baseline, baseline_area)
    assert decision["universal_winner_gate_passed"]
    assert decision["universal_winner"] == "baseline"
    assert decision["headline"] == "baseline_universal_winner"

    objective = complete_per_seed("objective")
    objective_area = ladder_area_summary(ladder_area_per_seed(objective))
    decision = scientific_decision(objective, objective_area)
    assert decision["universal_winner_gate_passed"]
    assert decision["universal_winner"] == "objective"
    assert decision["headline"] == "objective_universal_winner"
    assert decision["whole_ladder_area_winners"]["cls_fixed_panel"] == "objective"


def test_scientific_gate_rejects_a_subpractical_common_lead():
    tiny = complete_per_seed("baseline")
    tiny.loc[tiny.arm.eq("baseline"), ["nm_fixed_panel", "cls_fixed_panel"]] -= 0.0195
    area = ladder_area_summary(ladder_area_per_seed(tiny))
    decision = scientific_decision(tiny, area)
    assert set(decision["endpoint_winners"].values()) == {"baseline"}
    assert not decision["practical_margins_passed"]
    assert decision["universal_winner"] is None
    assert decision["headline"] == "target_dependent_or_pareto_tradeoff"
