#!/usr/bin/env python3
"""Thin adapters around the pinned official VISION and GILT implementations."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor

from .common_protocol import Episode, N_QUERY, N_SHOT, N_WAY


HERE = Path(__file__).resolve().parent
PINS = json.loads((HERE / "upstream_pins.json").read_text(encoding="utf-8"))


def validate_upstream(name: str, root: str | Path) -> Path:
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"{name} upstream checkout not found: {root}")
    actual = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    expected = PINS[name]["commit"]
    if actual != expected:
        raise RuntimeError(f"{name} upstream must be pinned at {expected}, got {actual}")
    return root


def _sym_adj(edge_index: torch.Tensor, num_nodes: int) -> SparseTensor:
    return SparseTensor.from_edge_index(
        edge_index, sparse_sizes=(num_nodes, num_nodes)
    ).to_symmetric().coalesce()


def _prototype_mean(
    context_x: torch.Tensor, context_y: torch.Tensor, num_classes: int
) -> torch.Tensor:
    result = torch.zeros(
        num_classes, context_x.size(1), device=context_x.device, dtype=context_x.dtype
    )
    return torch.scatter_reduce(
        result,
        0,
        context_y.view(-1, 1).expand(-1, context_x.size(1)),
        context_x,
        reduce="mean",
        include_self=False,
    )


class GILTAdapter(nn.Module):
    """Official PureGCN_v1 + PFNPredictorNodeCls on common NM episodes."""

    architecture = "gilt"
    learning_rate = 1e-5
    weight_decay = 1e-4
    optimizer_name = "adam"

    def __init__(self, upstream_root: str | Path, input_dim: int = 768, hidden: int = 128):
        super().__init__()
        root = validate_upstream("gilt", upstream_root)
        sys.path.insert(0, str(root))
        try:
            from src.model import PFNPredictorNodeCls, PureGCN_v1
        finally:
            sys.path.pop(0)

        generator = torch.Generator(device="cpu").manual_seed(0)
        random_matrix = torch.randn(input_dim, hidden, generator=generator)
        projection, _ = torch.linalg.qr(random_matrix, mode="reduced")
        self.register_buffer("feature_projection", projection)

        self.encoder = PureGCN_v1(
            hidden,
            num_layers=6,
            hidden=hidden,
            dp=0.2,
            norm=True,
            res=False,
            relu=False,
            norm_affine=True,
        )
        self.predictor = PFNPredictorNodeCls(
            hidden_dim=hidden,
            nhead=4,
            num_layers=2,
            mlp_layers=2,
            dropout=0.2,
            norm=True,
            separate_att=True,
            sim="dot",
            padding="zero",
            norm_affine=True,
            normalize=True,
            norm_type="pre",
            ffn_expansion_ratio=4,
            nc_sim="dot",
            nc_proto_pooling="mean",
            head_num_layers=0,
            nc_head_num_layers=0,
            lp_head_type="standard",
        )

    def episode_logits(self, episode: Episode) -> torch.Tensor:
        x = episode.x.float() @ self.feature_projection
        x = F.normalize(x, p=2, dim=1)
        adj_t = _sym_adj(episode.edge_index, x.size(0)).to(x.device)
        node_h = self.encoder(x, adj_t)
        item_h = node_h[episode.centers]
        context_x = item_h[episode.support_mask]
        target_x = item_h[episode.query_mask]
        context_y = episode.labels[episode.support_mask]
        class_x = _prototype_mean(context_x, context_y, episode.n_way)
        pfn_data = SimpleNamespace(
            y=context_y,
            context_sample=torch.arange(context_y.numel(), device=context_y.device),
        )
        logits, _ = self.predictor(
            pfn_data, context_x, target_x, context_y, class_x,
            task_type="node_classification"
        )
        return logits

    def episode_loss_and_accuracy(self, episode: Episode):
        logits = self.episode_logits(episode)
        target = episode.labels[episode.query_mask]
        return F.cross_entropy(logits, target), (logits.argmax(1) == target).float().mean()


def _load_vision_class(root: Path):
    spec = importlib.util.spec_from_file_location("vision_upstream_models", root / "models.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load VISION models from {root}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.LabelInjectedTransformer


def padded_adjacency(edge_index: torch.Tensor, num_nodes: int, max_neighbors: int = 30):
    adj = _sym_adj(edge_index, num_nodes)
    row, col, _ = adj.coo()
    rowptr, _, _ = adj.csr()
    degrees = rowptr[1:] - rowptr[:-1]
    starts = torch.repeat_interleave(rowptr[:-1], degrees)
    positions = torch.arange(col.numel(), device=col.device) - starts
    keep = positions < max_neighbors
    padded = torch.full(
        (num_nodes, max_neighbors), -1, dtype=torch.long, device=edge_index.device
    )
    padded[row[keep], positions[keep]] = col[keep]
    return padded


def vision_contrastive_loss(
    q_feats, s_feats, query_labels, support_labels, temperature: float = 0.1
):
    logits = q_feats @ s_feats.t() / temperature
    positive = query_labels[:, None] == support_labels[None, :]
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    return -(log_prob * positive).sum(1).div(positive.sum(1)).mean()


class VISIONAdapter(nn.Module):
    """Official LabelInjectedTransformer on common NM episodes."""

    architecture = "vision"
    learning_rate = 2e-4
    weight_decay = 1e-4
    optimizer_name = "adamw"

    def __init__(self, upstream_root: str | Path, input_dim: int = 768):
        super().__init__()
        root = validate_upstream("vision", upstream_root)
        cls = _load_vision_class(root)
        self.model = cls(
            feature_dim=input_dim,
            hidden_dim=256,
            n_way=N_WAY,
            k_shot=N_SHOT,
            m_qry=N_QUERY,
            dropout=0.1,
            nhead=4,
            nlayers=2,
            num_ensembles=3,
        )
        self.max_neighbors = 30
        self.drop_edge = 0.2
        self.drop_label = 0.2
        self.input_noise = 0.02
        self.contrastive_weight = 0.5

    def episode_logits(self, episode: Episode):
        x = episode.x.float()
        x = x - x.mean(dim=0, keepdim=True)
        adj = padded_adjacency(episode.edge_index, x.size(0), self.max_neighbors)
        if self.training and self.drop_edge > 0:
            drop = torch.rand(adj.shape, device=adj.device) < self.drop_edge
            adj = adj.masked_fill(drop & (adj >= 0), -1)
        support = episode.centers[episode.support_mask]
        query = episode.centers[episode.query_mask]
        old_n_way, old_k_shot = self.model.n_way, self.model.k_shot
        self.model.n_way, self.model.k_shot = episode.n_way, episode.n_shot
        try:
            return self.model(
                x,
                adj,
                support,
                query,
                drop_label_prob=self.drop_label if self.training else 0.0,
                input_noise_std=self.input_noise if self.training else 0.0,
            )
        finally:
            self.model.n_way, self.model.k_shot = old_n_way, old_k_shot

    def episode_loss_and_accuracy(self, episode: Episode):
        logits, contrastive = self.episode_logits(episode)
        target = episode.labels[episode.query_mask]
        ce = F.cross_entropy(logits, target, label_smoothing=0.1)
        query_labels = episode.labels[episode.query_mask]
        support_labels = episode.labels[episode.support_mask]
        con = torch.stack([
            vision_contrastive_loss(q, s, query_labels, support_labels)
            for q, s in contrastive
        ]).mean()
        loss = ce + self.contrastive_weight * con
        return loss, (logits.argmax(1) == target).float().mean()


def build_adapter(architecture: str, upstream_root: str | Path) -> nn.Module:
    if architecture == "gilt":
        return GILTAdapter(upstream_root)
    if architecture == "vision":
        return VISIONAdapter(upstream_root)
    raise ValueError(f"unknown architecture: {architecture}")


def build_optimizer(model: nn.Module):
    if model.optimizer_name == "adam":
        return torch.optim.Adam(
            model.parameters(), lr=model.learning_rate, weight_decay=model.weight_decay
        )
    return torch.optim.AdamW(
        model.parameters(), lr=model.learning_rate, weight_decay=model.weight_decay
    )
