import random

import torch

from scripts.experiments.setup.vision_all9_finalcore.train_vision_all9 import (
    first_neighbors,
    sampled_source_nodes,
    source_node_sets,
    vision_subgraph_from_csr,
)


class ToyGraph:
    source_graph_names = ["ukr_rus", "covid", "midterm", "covid_political", "election2020", "ukr_rus_suspended", "twibot20", "cp_hk", "facebook_page_reference"]
    graph_id = torch.arange(9).repeat_interleave(300)


def test_source_node_sets_select_requested_mixture_order():
    names, nodes = source_node_sets(ToyGraph(), ("twibot20", "covid"))
    assert names == ["twibot20", "covid"]
    assert set(nodes[0].tolist()) == set(range(6 * 300, 7 * 300))
    assert set(nodes[1].tolist()) == set(range(300, 600))


def test_sampled_source_nodes_are_unique_and_confined():
    random.seed(7)
    source = torch.arange(100, 200)
    sampled = sampled_source_nodes(source, 40)
    assert sampled.unique().numel() == 40
    assert set(sampled.tolist()) <= set(source.tolist())


def test_csr_subgraph_matches_padded_first_neighbor_semantics():
    # 0:{1,2}, 1:{0,2,3}, 2:{0,1}, 3:{1}
    rowptr = torch.tensor([0, 2, 5, 7, 8])
    col = torch.tensor([1, 2, 0, 2, 3, 0, 1, 1])
    assert first_neighbors(rowptr, col, torch.tensor([0, 1]), 2).tolist() == [[1, 2], [0, 2]]

    x = torch.arange(12, dtype=torch.float32).view(4, 3)
    sub_x, sub_adj, support, query = vision_subgraph_from_csr(
        x, rowptr, col, torch.tensor([0]), torch.tensor([1]), 2
    )
    assert torch.equal(sub_x, x[:3])
    assert support.tolist() == [0]
    assert query.tolist() == [1]
    assert sub_adj.tolist() == [[1, 2], [0, 2], [0, 1]]
