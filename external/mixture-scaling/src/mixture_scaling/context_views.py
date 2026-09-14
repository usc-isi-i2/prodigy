from __future__ import annotations

import torch


VIEWS = ("neighborhood", "node_neighborhood")


def neighbor_mean(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Mean incoming-neighbor features, excluding explicit self loops."""
    src, dst = edge_index
    keep = src != dst
    src, dst = src[keep], dst[keep]
    total = x.new_zeros(x.shape)
    total.index_add_(0, dst, x[src])
    degree = x.new_zeros((len(x), 1))
    degree.index_add_(0, dst, torch.ones((len(dst), 1), device=x.device, dtype=x.dtype))
    return total / degree.clamp_min(1)


def fixed_view(x: torch.Tensor, edge_index: torch.Tensor, view: str) -> torch.Tensor:
    if view not in VIEWS:
        raise ValueError(f"unknown context view: {view}")
    neighborhood = neighbor_mean(x, edge_index)
    if view == "neighborhood":
        return neighborhood
    return torch.cat((x, neighborhood), dim=-1)


def view_dim(feature_dim: int, view: str) -> int:
    if view not in VIEWS:
        raise ValueError(f"unknown context view: {view}")
    return feature_dim if view == "neighborhood" else 2 * feature_dim
