import torch

from mixture_scaling.strict_data import split_hash, stratified_node_split


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
