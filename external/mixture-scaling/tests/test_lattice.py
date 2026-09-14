import torch
from torch_geometric.data import Data

from mixture_scaling.lattice import SOURCE_ORDER, lattice_rows, lp_loss, mae_loss, selected_rows, update_selection, validate
from mixture_scaling.model import GraphMAE


def test_lattice_is_9_plus_36_plus_9():
    rows = lattice_rows()
    assert len(SOURCE_ORDER) == 9
    assert len(rows) == 54
    assert len({name for name, _ in rows}) == 54
    assert sorted(map(lambda row: len(row[1]), rows)).count(1) == 9
    assert sorted(map(lambda row: len(row[1]), rows)).count(2) == 36
    assert sorted(map(lambda row: len(row[1]), rows)).count(8) == 9


def test_gate_has_three_shape_representatives():
    assert sorted(len(sources) for _, sources in selected_rows("gate")) == [1, 2, 8]


def test_absolute_best_is_saved_without_patience_reset():
    result = update_selection(0.999, 250, 1.0, 0, 1.0, 0, 0.0025)
    best_loss, best_step, reference, patience, save_best = result
    assert (best_loss, best_step, save_best) == (0.999, 250, True)
    assert reference == 1.0
    assert patience == 1


def test_graphmae_loss_masks_and_reconstructs_roots():
    class Batch:
        x = torch.randn(6, 4)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
        batch_size = 3

        def to(self, _device):
            return self

    model = GraphMAE(4, 3, 3)
    loss = mae_loss(model, Batch(), torch.device("cpu"), 0.5, 2.0, torch.Generator().manual_seed(0))
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_lp_validation_replays_identical_negative_samples():
    class Batch:
        x = torch.randn(8, 4)
        edge_index = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6]])

        def __init__(self):
            self.edge_label_index = torch.stack((torch.arange(6), torch.randperm(8)[:6]))
            self.edge_label = torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.float)

        def to(self, _device):
            return self

    class RandomLoader:
        def __iter__(self):
            return iter([Batch()])

    model = GraphMAE(4, 3, 3).encoder
    protocol = {"validation_batches": 1, "mask_rate": 0.5, "sce_alpha": 2.0}
    first = validate(model, {"tiny": RandomLoader()}, "lp", torch.device("cpu"), protocol, 0)
    torch.rand(20)
    second = validate(model, {"tiny": RandomLoader()}, "lp", torch.device("cpu"), protocol, 0)
    assert first == second
