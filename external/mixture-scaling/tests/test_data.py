import torch

from mixture_scaling.data import load_graph, split_edges


def test_edge_split_is_pair_disjoint_and_bidirectional() -> None:
    edges = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    background, train, validation = split_edges(edges, 1 / 3, 0)
    train_pairs = set(map(tuple, train.T.tolist()))
    validation_pairs = set(map(tuple, validation.T.tolist()))
    assert train_pairs.isdisjoint(validation_pairs)
    assert background.shape[1] == 2 * train.shape[1]


def test_full_edge_loading_restores_all_observed_pairs(tmp_path) -> None:
    path = tmp_path / "graph.pt"
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
    torch.save({"x": torch.randn(4, 768), "edge_index": edge_index, "y": torch.arange(4)}, path)
    graph = load_graph("toy", path, validation_fraction=0.25, use_full_edges=True)
    observed = set(map(tuple, graph.data.edge_index.T.tolist()))
    expected = {(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2), (0, 3), (3, 0)}
    assert observed == expected
