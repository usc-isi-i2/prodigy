from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import SAGEConv


class GraphSAGE(nn.Module):
    """Plain node encoder used by both scratch and SSL GraphSAGE rows."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be positive")
        dims = [input_dim] + [hidden_dim] * (layers - 1) + [output_dim]
        self.convs = nn.ModuleList(
            SAGEConv(dims[index], dims[index + 1]) for index in range(layers)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for index, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if index + 1 < len(self.convs):
                x = self.dropout(torch.relu(x))
        return x


class GraphMAE(nn.Module):
    """GraphSAGE encoder with a lightweight masked-feature decoder."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = GraphSAGE(input_dim, hidden_dim, output_dim, layers, dropout)
        self.mask_token = nn.Parameter(torch.zeros(input_dim))
        self.decoder = nn.Linear(output_dim, input_dim)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encode(x, edge_index))
