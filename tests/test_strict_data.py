import torch
from torch_geometric.data import Data

from mixture_scaling.strict_data import split_hash, ssl_train_edge_partition, stratified_node_split


def test_stratified_split_is_disjoint_exhaustive_and_deterministic():
    y = torch.tensor([0] * 20 + [1] * 20 + [2] * 20)
    first = stratified_node_split(y, len(y), seed=7)
    second = stratified_node_split(y, len(y), seed=7)
    assert split_hash(first) == split_hash(second)
    partitions = [set(first.train.tolist()), set(first.validation.tolist()), set(first.test.tolist())]
    assert not partitions[0] & partitions[1]
    assert not partitions[0] & partitions[2]
    assert not partitions[1] & partitions[2]
    assert set.union(*partitions) == set(range(len(y)))
    for indices in (first.train, first.validation, first.test):
        assert set(y[indices].tolist()) == {0, 1, 2}


def test_stratified_split_is_seventy_fifteen_fifteen_per_class():
    y = torch.tensor([0] * 100 + [1] * 100)
    split = stratified_node_split(y, len(y), seed=0)
    for label in (0, 1):
        assert int((y[split.train] == label).sum()) == 70
        assert int((y[split.validation] == label).sum()) == 15
        assert int((y[split.test] == label).sum()) == 15


def test_ssl_edge_validation_is_disjoint_deterministic_and_hidden_from_message_passing(tmp_path):
    x = torch.randn(20, 4)
    y = torch.tensor([0] * 10 + [1] * 10)
    pairs = [(i, j) for i in range(20) for j in range(i + 1, 20)]
    edges = torch.tensor([[i for i, j in pairs] + [j for i, j in pairs],
                          [j for i, j in pairs] + [i for i, j in pairs]])
    path = tmp_path / "graph.pt"
    torch.save(Data(x=x, y=y, edge_index=edges), path)
    split = stratified_node_split(y, len(y), seed=7)
    first = ssl_train_edge_partition("toy", path, split, validation_fraction=0.25, seed=9)
    second = ssl_train_edge_partition("toy", path, split, validation_fraction=0.25, seed=9)
    train = {tuple(edge) for edge in first.train_edges.T.tolist()}
    validation = {tuple(edge) for edge in first.validation_edges.T.tolist()}
    background = {tuple(edge) for edge in first.data.edge_index.T.tolist()}
    assert train and validation and not train & validation
    assert all(edge not in background and edge[::-1] not in background for edge in validation)
    assert torch.equal(first.train_edges, second.train_edges)
    assert torch.equal(first.validation_edges, second.validation_edges)
