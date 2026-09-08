from scripts.experiments.setup.trace_health_guided.make_stage1_plan import (
    SOURCES,
    build_plan,
)
from scripts.experiments.setup.trace_health_guided.derive_health_schedule import (
    allocate_rank_counts,
    compress,
    weighted_fair_sequence,
)


def test_full_plan_contains_four_specialists_and_merge_per_seed() -> None:
    arms = build_plan()
    assert len(arms) == 15
    assert len({arm.model_id for arm in arms}) == 15
    for seed in (0, 1, 2):
        group = [arm for arm in arms if arm.seed == seed]
        singles = [arm for arm in group if arm.condition == "single"]
        merged = [arm for arm in group if arm.condition == "merged"]
        assert {arm.sources[0] for arm in singles} == set(SOURCES)
        assert len(merged) == 1
        assert merged[0].sources == SOURCES


def test_subset_plan_is_still_complete() -> None:
    arms = build_plan(sources=SOURCES[:2], seeds=(4,))
    assert [arm.condition for arm in arms] == ["single", "single", "merged"]
    assert arms[-1].sources == SOURCES[:2]


def test_weighted_fair_schedule_has_exact_preregistered_allocation() -> None:
    counts = allocate_rank_counts(2500)
    assert counts == (1000, 750, 500, 250)
    sequence = weighted_fair_sequence(SOURCES, counts)
    assert len(sequence) == 2500
    assert tuple(sequence.count(source) for source in SOURCES) == counts
    segment_sources, segment_steps = compress(sequence)
    assert sum(segment_steps) == 2500
    assert set(segment_sources) == set(SOURCES)


def test_rank_allocation_handles_small_smoke_budget() -> None:
    assert allocate_rank_counts(20) == (8, 6, 4, 2)
