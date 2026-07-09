"""Directed per-node structural features for the topology_feature_ssl experiment.

Shared by:
  * E1 (inject these as node input features so topology becomes *representable*),
  * the leakage-control baseline (linear-probe these raw onto regression targets,
    the passthrough ceiling E1/E2 must beat before claiming "learned structure"),
  * diagnostics.

The retweet edge_index is directed (retweeter -> retweeted): in-degree ≈ influence,
out-degree ≈ activity. We compute six columns, in this fixed order:

    [in_deg, out_deg, log_deg, k_core, pagerank, clustering]

Degrees are exact/cheap (torch scatter). k-core / PageRank / clustering are computed
with networkx on the (directed for PageRank, undirected for k-core/clustering) graph;
they are a one-time offline cost per graph and are cached by the callers.
"""

from __future__ import annotations

import math

import torch

STRUCTURAL_FEATURE_NAMES = [
    "in_deg", "out_deg", "log_deg", "k_core", "pagerank", "clustering",
]


def _degrees(edge_index: torch.Tensor, num_nodes: int) -> tuple[torch.Tensor, torch.Tensor]:
    """in-degree (as edge target) and out-degree (as edge source), directed."""
    out_deg = torch.zeros(num_nodes, dtype=torch.float)
    in_deg = torch.zeros(num_nodes, dtype=torch.float)
    if edge_index.numel():
        out_deg.scatter_add_(0, edge_index[0].long(), torch.ones(edge_index.size(1)))
        in_deg.scatter_add_(0, edge_index[1].long(), torch.ones(edge_index.size(1)))
    return in_deg, out_deg


def compute_structural_features(
    edge_index: torch.Tensor,
    num_nodes: int,
    *,
    standardize: bool = True,
    pagerank_alpha: float = 0.85,
    max_nx_nodes: int | None = None,
) -> torch.Tensor:
    """Return a [num_nodes, 6] float tensor of directed structural features.

    standardize: z-score each column (recommended when used as GNN input features
    so no single structural column dominates the 768-dim bio space by scale).
    max_nx_nodes: if set and num_nodes exceeds it, skip the networkx features
    (k_core/pagerank/clustering) and return zeros for them (degrees still exact),
    so very large graphs stay tractable. None = always compute.
    """
    in_deg, out_deg = _degrees(edge_index, num_nodes)
    log_deg = torch.log1p(in_deg + out_deg)

    k_core = torch.zeros(num_nodes, dtype=torch.float)
    pagerank = torch.zeros(num_nodes, dtype=torch.float)
    clustering = torch.zeros(num_nodes, dtype=torch.float)

    do_nx = edge_index.numel() > 0 and (max_nx_nodes is None or num_nodes <= max_nx_nodes)
    if do_nx:
        try:
            import networkx as nx
        except ImportError as exc:  # pragma: no cover - env dependent
            raise RuntimeError(
                "networkx is required for k_core/pagerank/clustering structural "
                "features. Install it, or pass max_nx_nodes to skip them."
            ) from exc

        ei = edge_index.cpu().numpy()
        edges = list(zip(ei[0].tolist(), ei[1].tolist()))

        digraph = nx.DiGraph()
        digraph.add_nodes_from(range(num_nodes))
        digraph.add_edges_from(edges)
        pr = nx.pagerank(digraph, alpha=pagerank_alpha) if digraph.number_of_edges() else {}
        for node, value in pr.items():
            pagerank[node] = value

        undirected = nx.Graph()
        undirected.add_nodes_from(range(num_nodes))
        undirected.add_edges_from(edges)
        undirected.remove_edges_from(nx.selfloop_edges(undirected))
        for node, core in nx.core_number(undirected).items():
            k_core[node] = core
        for node, coef in nx.clustering(undirected).items():
            clustering[node] = coef

    feats = torch.stack([in_deg, out_deg, log_deg, k_core, pagerank, clustering], dim=1)
    if standardize:
        mean = feats.mean(dim=0, keepdim=True)
        std = feats.std(dim=0, keepdim=True).clamp_min(1e-6)
        feats = (feats - mean) / std
    return feats
