from __future__ import annotations

import torch
from torch_geometric.data import Data

from experiments.sampler import NeighborSampler, parse_hop_sizes


def two_level_tree(first_degree: int = 30, second_degree: int = 30) -> Data:
    src = []
    dst = []
    next_node = 1 + first_degree
    for first in range(1, 1 + first_degree):
        src.append(0); dst.append(first)
        for _ in range(second_degree):
            src.append(first); dst.append(next_node)
            next_node += 1
    return Data(
        edge_index=torch.tensor([src, dst], dtype=torch.long),
        num_nodes=next_node,
    )


def test_compute_matched_two_hop_sampler_keeps_budget_and_reaches_hop_two() -> None:
    graph = two_level_tree()
    sampler = NeighborSampler(
        graph,
        num_hops=2,
        hop_sizes=[9, 9],
        limit=101,
        walk_hops=1,
    )
    node_ids, edge_index, _ = sampler.sample_node(0)
    assert node_ids.numel() <= 101
    assert edge_index.shape[1] <= 100
    assert (node_ids > 30).any(), "the matched budget must retain genuine hop-2 nodes"


def test_walk_hops_decouples_nm_positives_from_context_radius() -> None:
    graph = Data(
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        num_nodes=3,
    )
    starts = torch.zeros(32, dtype=torch.long)
    matched = NeighborSampler(graph, num_hops=2, hop_sizes=[1, 1],
                              limit=3, walk_hops=1)
    literal = NeighborSampler(graph, num_hops=2, hop_sizes=[1, 1], limit=3)
    assert set(matched.random_walk(starts, "inout").tolist()) == {1}
    assert set(literal.random_walk(starts, "inout").tolist()).issubset({0, 2})


def test_hop_size_parser_is_strict_and_defaults_off() -> None:
    assert parse_hop_sizes("", 2) is None
    assert parse_hop_sizes("9,9", 2) == [9, 9]
    try:
        parse_hop_sizes("10", 2)
    except ValueError as exc:
        assert "contain 2 values" in str(exc)
    else:
        raise AssertionError("one fanout for two hops should fail")


def test_empty_options_preserve_historical_sampling() -> None:
    graph = two_level_tree(first_degree=20, second_degree=20)
    historical = NeighborSampler(graph, num_hops=2)
    explicit = NeighborSampler(graph, num_hops=2, hop_sizes=[100, 100],
                               limit=2000, walk_hops=2)
    torch.manual_seed(7)
    historical_result = historical.sample_node(0)
    torch.manual_seed(7)
    explicit_result = explicit.sample_node(0)
    for left, right in zip(historical_result, explicit_result):
        assert torch.equal(left, right)
