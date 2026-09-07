from scripts.experiments.setup.trace_schedule_scaling.make_plan import build_plan


def test_full_plan_is_complete_and_matched() -> None:
    arms = build_plan()
    assert len(arms) == 27
    assert len({arm.model_id for arm in arms}) == 27
    for rung in (2, 3, 4):
        for seed in (0, 1, 2):
            group = [arm for arm in arms if (arm.rung, arm.seed) == (rung, seed)]
            assert {arm.schedule for arm in group} == {
                "blocked", "replay100", "interleaved"
            }
            assert len({arm.sources for arm in group}) == 1
            assert len({arm.source_counts for arm in group}) == 1


def test_schedule_granularity_and_rotated_order() -> None:
    arms = build_plan(total_steps=20, replay_block=4, rungs=(2,), seeds=(1,))
    by_schedule = {arm.schedule: arm for arm in arms}
    assert by_schedule["blocked"].order == ("covid", "ukr_rus")
    assert by_schedule["blocked"].segment_steps == (10, 10)
    assert max(by_schedule["replay100"].segment_steps) == 4
    assert set(by_schedule["interleaved"].segment_steps) == {1}


def test_rung_three_remainder_is_schedule_invariant() -> None:
    arms = build_plan(total_steps=2500, rungs=(3,), seeds=(0,))
    assert {arm.source_counts for arm in arms} == {(834, 833, 833)}
