"""Unit tests for --neighbor_sampling_source_subset (data/covid19_twitter.py).

The subset knob lets one merged graph stand in for every rung of a curriculum ladder:
because merges are disjoint block-concats, restricting which sources episodes may be
drawn from is equivalent to training on the merge of just those sources. That claim is
what makes scripts/experiments/setup/nm_ladder_order_robustness-jul_23/ affordable, so
it is pinned here rather than left to the launch scripts.

Pure-python/torch, no cluster data needed.
Run: python -m pytest data/tests/test_source_subset.py
"""
import numpy as np
import pytest

from data.covid19_twitter import resolve_source_subset
from data.dataloader import NeighborTask

# The all8 ladder graph, in graph_id order (see scripts/graph_construction/
# merge_ukr_rus_covid_midterm_all8.yaml).
ALL8 = [
    "ukr_rus", "covid", "midterm", "covid_political",
    "election2020", "ukr_rus_suspended", "twibot20", "cp_hk",
]
IDS8 = list(range(8))


def test_empty_spec_means_all_sources():
    assert resolve_source_subset("", IDS8, ALL8) is None
    assert resolve_source_subset(None, IDS8, ALL8) is None
    assert resolve_source_subset("  ,  ", IDS8, ALL8) is None


def test_resolves_names():
    assert resolve_source_subset("covid,ukr_rus", IDS8, ALL8) == {0, 1}


def test_resolves_integer_ids_and_mixed_tokens():
    assert resolve_source_subset("0,1,6", IDS8, ALL8) == {0, 1, 6}
    assert resolve_source_subset("covid, 0 ,twibot20", IDS8, ALL8) == {0, 1, 6}


def test_whitespace_and_duplicates_are_tolerated():
    assert resolve_source_subset(" covid , covid ,1 ", IDS8, ALL8) == {1}


def test_unknown_name_raises_and_lists_available():
    with pytest.raises(ValueError) as err:
        resolve_source_subset("covid,not_a_graph", IDS8, ALL8)
    assert "not_a_graph" in str(err.value)
    assert "ukr_rus" in str(err.value)  # the available-sources list is shown


def test_id_absent_from_graph_raises():
    # A 3-source merge asked for graph_id 5.
    with pytest.raises(ValueError):
        resolve_source_subset("5", [0, 1, 2], ALL8[:3])


def test_subset_strata_match_the_equivalent_submerge():
    """Filtering strata of the all8 graph == strata of a graph built from those sources.

    This is the equivalence the ladder rests on: same node sets, same stratum count, so
    'balanced' episode sampling sees exactly the sub-merge.
    """
    rng = np.random.default_rng(0)
    graph_ids = np.repeat(np.arange(8), 5)          # 5 nodes per source
    rng.shuffle(graph_ids)

    subset = resolve_source_subset("ukr_rus,covid,twibot20", IDS8, ALL8)
    kept = [gid for gid in sorted(set(graph_ids.tolist())) if gid in subset]
    strata = [np.where(graph_ids == gid)[0].tolist() for gid in kept]

    assert kept == [0, 1, 6]
    assert [len(s) for s in strata] == [5, 5, 5]
    # every selected node really belongs to a selected source, and none are dropped
    assert set().union(*map(set, strata)) == set(np.where(np.isin(graph_ids, [0, 1, 6]))[0].tolist())


def test_balanced_weighting_is_uniform_over_the_subset_not_all_sources():
    """After subsetting, 'balanced' must be 1/k over the k kept sources.

    If the full stratum list leaked through, weights would be 1/8 and the small sources
    would get a different episode share than the equivalent sub-merge run.
    """
    graph_ids = np.repeat(np.arange(8), 5)
    subset = resolve_source_subset("ukr_rus,covid,twibot20", IDS8, ALL8)
    kept = [gid for gid in sorted(set(graph_ids.tolist())) if gid in subset]
    strata = [np.where(graph_ids == gid)[0].tolist() for gid in kept]

    task = NeighborTask(
        neighbor_sampler=None, size=40, direction="inout",
        strata=strata, confine_to_single_stratum=True, stratum_weighting="balanced",
    )
    assert task.stratum_weights == pytest.approx([1 / 3, 1 / 3, 1 / 3])


def test_proportional_weighting_follows_subset_node_counts():
    graph_ids = np.concatenate([np.zeros(10), np.ones(30), np.full(60, 7)]).astype(int)
    subset = resolve_source_subset("ukr_rus,covid", [0, 1, 7], ALL8)
    kept = [gid for gid in sorted(set(graph_ids.tolist())) if gid in subset]
    strata = [np.where(graph_ids == gid)[0].tolist() for gid in kept]

    task = NeighborTask(
        neighbor_sampler=None, size=100, direction="inout",
        strata=strata, confine_to_single_stratum=True, stratum_weighting="proportional",
    )
    # 10 and 30 nodes -> 0.25 / 0.75 within the subset; the dropped 60-node source
    # must not appear in the denominator.
    assert task.stratum_weights == pytest.approx([0.25, 0.75])
