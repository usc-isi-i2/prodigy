from collections import Counter

from mixture_scaling.node_mlp_ladder import ladder_rows, resident_sources, source_schedule


def test_ladder_has_nested_prefixes_and_equal_total_budget():
    rows = ladder_rows()
    assert len(rows) == 9
    for i, (_, sources) in enumerate(rows, 1):
        assert len(sources) == i
        if i > 1:
            assert sources[:-1] == rows[i-2][1]
        counts = Counter(source_schedule(sources, 2500))
        assert sum(counts.values()) == 2500
        assert max(counts.values()) - min(counts.values()) <= 1


def test_residency_respects_budget_and_keeps_smaller_sources():
    sizes = {"large": 66, "small_a": 8, "small_b": 7}
    assert resident_sources(sizes, 70) == {"small_a", "small_b"}
    assert resident_sources(sizes, 81) == set(sizes)
    assert resident_sources(sizes, 0) == set()
