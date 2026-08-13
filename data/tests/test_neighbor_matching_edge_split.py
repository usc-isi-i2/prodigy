from types import SimpleNamespace

import pytest
import torch

from data.dataloader import NeighborTask
from data import neighbor_matching_split as nms


class _FakeAdj:
    def __init__(self, rowptr):
        self.rowptr = torch.tensor(rowptr, dtype=torch.long)

    def csr(self):
        return self.rowptr, torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)


class _FakeSampler:
    def __init__(self, degrees):
        rowptr = [0]
        for degree in degrees:
            rowptr.append(rowptr[-1] + degree)
        self.whole_adj = _FakeAdj(rowptr)

    def random_walk(self, node_idx, direction):
        del direction
        center = int(node_idx[0])
        degree = int(self.whole_adj.rowptr[center + 1] - self.whole_adj.rowptr[center])
        return torch.arange(degree, dtype=torch.long) + 100 * center


def test_min_degree_filter_fails_before_rejection_loop():
    task = NeighborTask(
        _FakeSampler([2, 2, 7, 8]),
        size=4,
        direction="inout",
        filter_min_degree=True,
    )
    with pytest.raises(RuntimeError, match="Only 2 eligible NM centers"):
        task.sample(num_label=3, num_member=7, num_shot=3, num_query=4, rng=__import__("random").Random(0))


def test_min_degree_filter_samples_only_eligible_centers():
    task = NeighborTask(
        _FakeSampler([1, 7, 8]),
        size=3,
        direction="inout",
        filter_min_degree=True,
    )
    episode = task.sample(
        num_label=2,
        num_member=7,
        num_shot=3,
        num_query=4,
        rng=__import__("random").Random(0),
    )
    assert set(episode) == {1, 2}
    assert all(len(members) == 7 for members in episode.values())


def test_positive_sampler_routes_train_to_background_and_test_to_holdout():
    background = object()
    holdout = object()
    dataset = SimpleNamespace(
        neighbor_sampler=background,
        nm_holdout_neighbor_sampler=holdout,
    )
    kwargs = {"neighbor_matching_edge_split": True}
    assert nms.positive_sampler_for_split(dataset, "train", kwargs) is background
    assert nms.positive_sampler_for_split(dataset, "val", kwargs) is holdout
    assert nms.positive_sampler_for_split(dataset, "test", kwargs) is holdout


def test_positive_sampler_routes_three_way_validation_and_test_separately():
    background, validation, test = object(), object(), object()
    dataset = SimpleNamespace(
        neighbor_sampler=background,
        nm_validation_neighbor_sampler=validation,
        nm_test_neighbor_sampler=test,
    )
    kwargs = {"neighbor_matching_edge_split": True}
    assert nms.positive_sampler_for_split(dataset, "train", kwargs) is background
    assert nms.positive_sampler_for_split(dataset, "val", kwargs) is validation
    assert nms.positive_sampler_for_split(dataset, "test", kwargs) is test


def test_split_is_opt_in_and_missing_holdout_fails_closed():
    background = object()
    dataset = SimpleNamespace(neighbor_sampler=background)
    assert nms.positive_sampler_for_split(dataset, "test", {}) is background
    with pytest.raises(ValueError, match="no holdout sampler"):
        nms.positive_sampler_for_split(
            dataset, "test", {"neighbor_matching_edge_split": True}
        )


def test_configure_requires_named_background_and_holdout_views(monkeypatch):
    class _ConstructedSampler:
        def __init__(self, graph, num_hops, **kwargs):
            self.graph = graph
            self.num_hops = num_hops
            self.kwargs = kwargs

    monkeypatch.setattr(nms, "NeighborSampler", _ConstructedSampler)
    dataset = SimpleNamespace(
        graph=SimpleNamespace(num_nodes=4, edge_index=torch.tensor([[0], [1]])),
        neighbor_sampler=object(),
    )
    kwargs = {
        "task_name": "neighbor_matching",
        "neighbor_matching_edge_split": True,
        "edge_view": "static_background",
        "target_edge_view": "static_holdout",
    }
    raw = {
        "target_edge_index_views": {
            "static_holdout": torch.tensor([[2], [3]], dtype=torch.long)
        }
    }
    nms.configure_edge_split(raw, dataset, kwargs, n_hop=2, sampler_kwargs={"walk_hops": 1})
    assert dataset.nm_holdout_neighbor_sampler.graph.edge_index.tolist() == [[2], [3]]

    with pytest.raises(ValueError, match="no.*static_holdout"):
        nms.configure_edge_split({}, dataset, kwargs, n_hop=2, sampler_kwargs={})

    wrong = {**kwargs, "edge_view": "default"}
    with pytest.raises(ValueError, match="edge_view=static_background"):
        nms.configure_edge_split(raw, dataset, wrong, n_hop=2, sampler_kwargs={})
