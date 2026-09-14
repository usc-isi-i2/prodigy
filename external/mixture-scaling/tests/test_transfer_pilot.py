import torch
from torch_geometric.data import Data

from mixture_scaling.model import GraphSAGE
from mixture_scaling.train_transfer_pilot import graphmae_loss


def test_graphmae_loss_is_finite_and_differentiable():
    batch = Data(
        x=torch.randn(6, 4),
        edge_index=torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]]),
    )
    batch.batch_size = 4
    encoder = GraphSAGE(4, 5, 3)
    decoder = torch.nn.Linear(3, 4)
    loss = graphmae_loss(encoder, decoder, batch, torch.device("cpu"), 0.5, 2.0)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None for parameter in encoder.parameters())
