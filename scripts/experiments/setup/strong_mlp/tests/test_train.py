import numpy as np
import torch

from scripts.experiments.setup.strong_mlp.train import MLP, metrics


def test_mlp_shape_and_metrics():
    model = MLP(8, (16, 8), 0.2)
    logits = model(torch.randn(6, 8))
    assert logits.shape == (6, 2)
    values = metrics(torch.tensor([[4.0, -4.0], [-4.0, 4.0]]), torch.tensor([0, 1]))
    assert values["roc_auc"] == 1.0
    assert values["accuracy"] == 1.0
