import json

import pandas as pd
import pytest

from scripts.experiments.analysis.synthesis.cross_experiment.paper_flagship_ladders.analyze import (
    ARMS,
    CAPACITY_ARMS,
    RUNGS,
    SEEDS,
    SOURCE_ORDER,
    annotate_models,
    capacity_per_seed,
    ladder_area_per_seed,
    ladder_area_summary,
    load_nm,
    scientific_decision,
    seed_summary,
    target_diagnostics,
    validate_cross_task_models,
    validate_grid,
)


def test_fresh_seed0_cells_replace_historical_seed0(tmp_path):
    seed0_csv = tmp_path / "historical.csv"
    pd.DataFrame(
        [{"model_id": "nmi_baseline_r1_s0", "roc_auc": 0.1, "sources": "ukr_rus"}]
    ).to_csv(seed0_csv, index=False)
    refresh = tmp_path / "refresh" / "nmi_baseline_r1_s0"
    refresh.mkdir(parents=True)
    (refresh / "ukr_rus.json").write_text(
        json.dumps(
            {"model_id": "nmi_baseline_r1_s0", "roc_auc": 0.7, "sources": ["ukr_rus"]}
        )
    )
    replicas = tmp_path / "replicas" / "nmi_baseline_r1_s1"
    replicas.mkdir(parents=True)
    (replicas / "ukr_rus.json").write_text(
        json.dumps(
            {"model_id": "nmi_baseline_r1_s1", "roc_auc": 0.6, "sources": ["ukr_rus"]}
        )
    )

    result = load_nm(seed0_csv, replicas.parent, refresh.parent)

    assert len(result) == 2
    assert result.loc[result.training_seed.eq(0), "roc_auc"].item() == 0.7


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
                            "model_id": f"nmi_{arm}_r{rung}_s{seed}",
                            "target": target,
                            "roc_auc": 0.5,
                            "fingerprint": f"fixed-{target}",
                            "episodes": 512,
                            "sources": ",".join(SOURCE_ORDER[:rung]),
                            "checkpoint": "/tmp/state_dict.ckpt",
                            "checkpoint_step": 2000,
                            "checkpoint_sha256": "a" * 64,
                            "training_revision": "deadbeef",
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


def test_declared_training_seed_must_match_model_id():
    frame = pd.DataFrame(
        [{"model_id": "nmi_baseline_r1_s2", "training_seed": 1}]
    )
    with pytest.raises(ValueError, match="training_seed disagrees"):
        annotate_models(frame)


def test_source_set_must_match_registered_rung():
    rows = []
    for arm in CAPACITY_ARMS:
        for rung in RUNGS:
            for seed in SEEDS:
                for target in ("a", "b"):
                    sources = SOURCE_ORDER[:rung]
                    if arm == "capacity" and rung == 4 and seed == 1:
                        sources = SOURCE_ORDER[:3]
                    rows.append(
                        {
                            "arm": arm,
                            "rung": rung,
                            "training_seed": seed,
                            "model_id": f"nmi_{arm}_r{rung}_s{seed}",
                            "target": target,
                            "roc_auc": 0.5,
                            "fingerprint": f"fixed-{target}",
                            "episodes": 512,
                            "sources": ",".join(sources),
                            "checkpoint": "/tmp/state_dict.ckpt",
                            "checkpoint_step": 2000,
                            "checkpoint_sha256": "a" * 64,
                            "training_revision": "deadbeef",
                        }
                    )
    with pytest.raises(ValueError, match="source-set mismatch"):
        validate_grid(
            pd.DataFrame(rows),
            ("a", "b"),
            "capacity NM",
            CAPACITY_ARMS,
            expected_episodes=512,
        )


def test_cross_task_model_state_must_match():
    rows = []
    for arm in ARMS:
        for rung in RUNGS:
            for seed in SEEDS:
                rows.append(
                    {
                        "model_id": f"nmi_{arm}_r{rung}_s{seed}",
                        "checkpoint_sha256": "a" * 64,
                        "checkpoint_step": 2000,
                        "training_revision": "deadbeef",
                        "sources": ",".join(SOURCE_ORDER[:rung]),
                    }
                )
    nm = pd.DataFrame(rows)
    cls = pd.DataFrame(rows)
    paired = validate_cross_task_models(nm, cls)
    assert len(paired) == len(ARMS) * len(RUNGS) * len(SEEDS)

    cls = cls.copy()
    cls.loc[cls.model_id.eq("nmi_objective_r8_s2"), "checkpoint_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="cross-task checkpoint_sha256 mismatch"):
        validate_cross_task_models(nm, cls)


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


def test_target_diagnostics_and_worst_target_gate_are_paired():
    nm_rows = []
    cls_rows = []
    for arm in ARMS:
        for rung in RUNGS:
            for seed in SEEDS:
                for target in ("n0", "n1"):
                    bonus = 0.02 if arm == "objective" else 0.0
                    if arm == "objective" and rung == 8 and target == "n1":
                        bonus = -0.01
                    nm_rows.append(
                        dict(
                            arm=arm, rung=rung, training_seed=seed, target=target,
                            roc_auc=0.5 + 0.001 * rung + bonus,
                        )
                    )
                for target in ("c0", "c1"):
                    bonus = 0.02 if arm == "objective" else 0.0
                    cls_rows.append(
                        dict(
                            arm=arm, rung=rung, training_seed=seed, target=target,
                            roc_auc=0.6 + 0.001 * rung + bonus,
                        )
                    )
    _, target_summary, endpoint = target_diagnostics(
        pd.DataFrame(nm_rows), pd.DataFrame(cls_rows)
    )
    assert len(target_summary) == (2 + 2) * len(ARMS) * len(RUNGS)
    objective_nm = endpoint[
        endpoint.task.eq("NM") & endpoint.arm.eq("objective")
    ].iloc[0]
    assert objective_nm.targets_improved_vs_baseline == 1
    assert objective_nm.worst_target_vs_baseline == "n1"
    assert objective_nm.worst_target_delta_vs_baseline_mean == pytest.approx(-0.01)

    curves = complete_per_seed("objective")
    area = ladder_area_summary(ladder_area_per_seed(curves))
    decision = scientific_decision(curves, area, endpoint)
    assert decision["target_safety"]["nm_fixed_panel"]["worst_target"] == "n1"
    assert not decision["target_safety_passed"]
    assert decision["universal_winner"] is None
