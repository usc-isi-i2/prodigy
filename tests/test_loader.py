from pathlib import Path

import torch
import pytest
from torch_geometric.data import Data
from torch_geometric import typing

from mixture_scaling.data import GraphArtifact
from mixture_scaling.train import make_loader


def test_link_neighbor_loader_returns_local_labeled_pairs() -> None:
    if not (typing.WITH_PYG_LIB or typing.WITH_TORCH_SPARSE):
        pytest.skip("local PyG build has no neighbor-sampling backend; Tucker smoke is authoritative")
    features = torch.randn(20, 768)
    positive = torch.tensor([list(range(19)), list(range(1, 20))])
    background = torch.cat((positive, positive.flip(0)), dim=1)
    graph = GraphArtifact(
        "tiny", Path("tiny.pt"), Data(x=features, edge_index=background),
        positive, positive[:, :5],
    )
    protocol = {
        "fanout": 5, "layers": 1, "negatives_per_positive": 2,
        "ssl_batch_size": 4,
    }
    batch = next(iter(make_loader(graph, protocol, validation=False)))
    assert batch.edge_label_index.shape == (2, 12)
    assert batch.edge_label.shape == (12,)
    assert int(batch.edge_label_index.max()) < batch.x.shape[0]
