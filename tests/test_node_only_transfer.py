import torch

from mixture_scaling.evaluate_node_only import FP_TARGETS
from mixture_scaling.evaluate_lattice import LP_TARGETS
from mixture_scaling.model import MaskedFeatureMLP, NodeMLP
from mixture_scaling.node_only_transfer import (
    DirectLinkLoader, masked_feature_loss, selected_rows, singleton_rows,
)


def test_transfer_contract_is_nine_specialists_by_eligible_targets():
    assert len(singleton_rows()) == 9
    assert len(FP_TARGETS) == 9
    assert len(LP_TARGETS) == 6
    assert len(selected_rows("gate")) == 1


def test_node_mlp_has_no_topology_argument():
    model = NodeMLP(4, 3, 2)
    x = torch.randn(5, 4)
    assert model(x).shape == (5, 2)


def test_coordinate_masked_feature_loss_is_finite_and_replayable():
    torch.manual_seed(0)
    model = MaskedFeatureMLP(6, 4, 3)
    x = torch.randn(8, 6)
    first = masked_feature_loss(
        model, x, 0.5, 2.0, torch.Generator().manual_seed(7), torch.device("cpu")
    )
    second = masked_feature_loss(
        model, x, 0.5, 2.0, torch.Generator().manual_seed(7), torch.device("cpu")
    )
    assert torch.isfinite(first)
    assert torch.equal(first, second)


def test_direct_link_loader_builds_endpoint_only_batches_deterministically():
    x = torch.arange(40, dtype=torch.float).reshape(10, 4)
    edges = torch.tensor([[0, 2, 4], [1, 3, 5]])
    first = next(iter(DirectLinkLoader(x, edges, 2, 5, False, 7)))
    second = next(iter(DirectLinkLoader(x, edges, 2, 5, False, 7)))
    assert first.x.shape == (24, 4)
    assert first.edge_label_index.shape == (2, 12)
    assert first.edge_label.tolist() == [1.0, 1.0] + [0.0] * 10
    assert torch.equal(first.x, second.x)
    assert not torch.any(first.edge_label_index[0] == first.edge_label_index[1])
