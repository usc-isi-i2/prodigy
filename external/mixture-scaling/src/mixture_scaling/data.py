from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch_geometric.data import Data


@dataclass(frozen=True)
class GraphArtifact:
    name: str
    path: Path
    data: Data
    train_edges: torch.Tensor
    validation_edges: torch.Tensor


def _undirected_pairs(edge_index: torch.Tensor) -> torch.Tensor:
    src, dst = edge_index.long()
    keep = src != dst
    lo = torch.minimum(src[keep], dst[keep])
    hi = torch.maximum(src[keep], dst[keep])
    pairs = torch.stack((lo, hi), dim=1)
    return torch.unique(pairs, dim=0)


def split_edges(
    edge_index: torch.Tensor,
    validation_fraction: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pairs = _undirected_pairs(edge_index)
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(pairs), generator=generator)
    n_validation = max(1, round(len(pairs) * validation_fraction))
    validation = pairs[order[:n_validation]].T.contiguous()
    train = pairs[order[n_validation:]].T.contiguous()
    background = torch.cat((train, train.flip(0)), dim=1)
    return background, train, validation


def load_graph(
    name: str,
    path: str | Path,
    *,
    validation_fraction: float = 0.15,
    seed: int = 0,
    use_full_edges: bool = False,
) -> GraphArtifact:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"missing graph artifact: {path}")
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(raw, dict):
        raw = raw.to_dict()
    x, edge_index, y = raw.get("x"), raw.get("edge_index"), raw.get("y")
    if x is None or edge_index is None:
        raise ValueError(f"{name}: artifact requires x and edge_index")
    if x.ndim != 2 or x.shape[1] != 768:
        raise ValueError(f"{name}: expected [N,768] features, got {tuple(x.shape)}")
    background, train, validation = split_edges(edge_index, validation_fraction, seed)
    if use_full_edges:
        full_pairs = _undirected_pairs(edge_index).T.contiguous()
        background = torch.cat((full_pairs, full_pairs.flip(0)), dim=1)
    data = Data(x=x.float(), edge_index=background, y=None if y is None else y.long())
    return GraphArtifact(name, path, data, train, validation)


def labeled_nodes(data: Data) -> tuple[torch.Tensor, torch.Tensor]:
    if data.y is None:
        raise ValueError("graph has no node labels")
    y = data.y.reshape(-1)
    mask = torch.isfinite(y.float()) & (y >= 0)
    return mask.nonzero(as_tuple=False).reshape(-1), y[mask]
