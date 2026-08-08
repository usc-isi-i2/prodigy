import random

import pytest
import torch
from torch_geometric.data import Data

from data.covid19_twitter import (
    _parse_center_radii,
    _parse_center_radius_weights,
)
from data.dataloader import NeighborTask
from experiments.sampler import NeighborSampler


def disconnected_cliques(size=14):
    edges = []
    for offset in (0, size):
        for source in range(offset, offset + size):
            for target in range(source + 1, offset + size):
                edges.append((source, target))
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return Data(edge_index=edge_index, num_nodes=size * 2), size


def make_task(radii, weights=None):
    graph, component_size = disconnected_cliques()
    sampler = NeighborSampler(graph, num_hops=1, walk_hops=1)
    return (
        NeighborTask(
            sampler,
            graph.num_nodes,
            "inout",
            filter_min_degree=True,
            center_radii=radii,
            center_radius_weights=weights,
            center_region_fanout=64,
            center_region_node_limit=128,
            center_region_candidate_limit=32,
        ),
        component_size,
    )


def assert_collision_free(episode):
    centers = set(episode)
    members = [node for values in episode.values() for node in values]
    assert len(members) == len(set(members))
    assert centers.isdisjoint(members)


def test_parses_integer_and_global_radii():
    radii = _parse_center_radii("2, 3, global")
    assert radii == [2, 3, None]
    assert _parse_center_radius_weights("1,2,3", len(radii)) == [1.0, 2.0, 3.0]


def test_radius_parser_rejects_misaligned_weights():
    with pytest.raises(ValueError, match="one value per radius"):
        _parse_center_radius_weights("1,2", 3)


def test_finite_radius_keeps_centers_in_one_component_and_targets_disjoint():
    task, component_size = make_task([1])
    torch.manual_seed(4)
    episode = task.sample(4, 2, 1, 1, random.Random(9))

    assert len(episode) == 4
    assert len({center // component_size for center in episode}) == 1
    assert task.last_sampled_center_radius == 1
    assert_collision_free(episode)


def test_global_radius_can_cross_disconnected_components():
    task, component_size = make_task([None])
    rng = random.Random(2)
    torch.manual_seed(3)

    observed_cross_component = False
    for _ in range(12):
        episode = task.sample(8, 1, 1, 0, rng)
        assert_collision_free(episode)
        if len({center // component_size for center in episode}) > 1:
            observed_cross_component = True
            break
    assert observed_cross_component
    assert task.last_sampled_center_radius is None


def test_radius_mode_rejects_graph_id_strata():
    graph, component_size = disconnected_cliques()
    sampler = NeighborSampler(graph, num_hops=1, walk_hops=1)
    with pytest.raises(ValueError, match="source-unaware"):
        NeighborTask(
            sampler,
            graph.num_nodes,
            "inout",
            strata=[list(range(component_size)), list(range(component_size, graph.num_nodes))],
            confine_to_single_stratum=True,
            center_radii=[2, None],
        )


def test_empty_radius_configuration_preserves_historical_stream():
    graph, _ = disconnected_cliques()
    sampler = NeighborSampler(graph, num_hops=1, walk_hops=1)
    baseline = NeighborTask(sampler, graph.num_nodes, "inout")
    disabled = NeighborTask(sampler, graph.num_nodes, "inout", center_radii=None)

    torch.manual_seed(17)
    expected = baseline.sample(3, 2, 1, 1, random.Random(23))
    torch.manual_seed(17)
    actual = disabled.sample(3, 2, 1, 1, random.Random(23))
    assert actual == expected
