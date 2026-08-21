import torch

from mixture_scaling.data import split_edges


def test_edge_split_is_pair_disjoint_and_bidirectional() -> None:
    edges = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    background, train, validation = split_edges(edges, 1 / 3, 0)
    train_pairs = set(map(tuple, train.T.tolist()))
    validation_pairs = set(map(tuple, validation.T.tolist()))
    assert train_pairs.isdisjoint(validation_pairs)
    assert background.shape[1] == 2 * train.shape[1]

