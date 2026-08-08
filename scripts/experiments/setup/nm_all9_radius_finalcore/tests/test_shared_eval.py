from pathlib import Path
import sys

import pytest
import torch


HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from shared_eval import score_models_on_shared_batches  # noqa: E402


CHECKPOINT_STEPS = (100, 300, 900, 2500)


class FakeGraph:
    def __init__(self, value: float):
        self.x = torch.tensor([[value]], dtype=torch.float)
        self.edge_attr = torch.tensor([[value + 10]], dtype=torch.float)

    def to(self, device):
        self.x = self.x.to(device)
        self.edge_attr = self.edge_attr.to(device)
        return self


class FakeModel(torch.nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(scale, dtype=torch.float))

    def forward(self, graph, y_true):
        y_pred = y_true + graph.x.mean() * self.scale
        graph.x = graph.x + 1000
        graph.edge_attr = graph.edge_attr + 1000
        return y_true, y_pred, graph


def get_loss_and_acc(y_true, y_pred):
    loss = torch.mean((y_pred - y_true) ** 2)
    return loss, 1.0 / (1.0 + loss)


def get_aux_loss(_graph):
    return torch.tensor(0.0)


def test_shared_checkpoint_eval_restores_graph_and_matches_each_model():
    batches = [
        [FakeGraph(1.0), torch.zeros((2, 2))],
        [FakeGraph(2.0), torch.zeros((2, 2))],
    ]
    models = {
        step: FakeModel(step / 100) for step in CHECKPOINT_STEPS
    }
    rows = score_models_on_shared_batches(
        models=models,
        steps=CHECKPOINT_STEPS,
        dataloader=batches,
        device=torch.device("cpu"),
        get_loss_and_score=get_loss_and_acc,
        get_aux_loss=get_aux_loss,
    )

    assert list(rows) == list(CHECKPOINT_STEPS)
    for step, row in rows.items():
        scale = step / 100
        expected_loss = scale**2 * (1**2 + 2**2) / 2
        assert row["loss"] == pytest.approx(expected_loss)
        assert row["score"] == pytest.approx(1.0 / (1.0 + expected_loss))
        assert row["aux_loss"] == 0.0

    assert [float(batch[0].x.item()) for batch in batches] == [1.0, 2.0]
    assert [float(batch[0].edge_attr.item()) for batch in batches] == [11.0, 12.0]
