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

import os

import torch

STRUCTURAL_FEATURE_NAMES = [
    "in_deg", "out_deg", "log_deg", "k_core", "pagerank", "clustering",
]
# directed3 = degree-only (exact + O(E), scales to the 34M-node merged graph);
# directed6 adds the networkx features (k_core/pagerank/clustering), only tractable
# on small graphs. The mode MUST match at pretrain and eval (defines input_dim).
STRUCTURAL_FEATURE_MODES = {"directed3": 3, "directed3_log": 3, "directed6": 6}


def structural_feature_names(mode: str = "directed6") -> list[str]:
    return STRUCTURAL_FEATURE_NAMES[: STRUCTURAL_FEATURE_MODES[mode]]


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
    mode: str = "directed6",
    standardize: bool = True,
    pagerank_alpha: float = 0.85,
    max_nx_nodes: int | None = None,
) -> torch.Tensor:
    """Return a [num_nodes, k] float tensor of directed structural features.

    mode: 'directed3' = [in_deg, out_deg, log_deg] only (exact, O(E), scales to
    tens of millions of nodes); 'directed6' additionally computes the networkx
    features (k_core/pagerank/clustering), tractable only on small graphs.
    standardize: z-score each column (recommended when used as GNN input features
    so no single structural column dominates the 768-dim bio space by scale).
    max_nx_nodes: with mode='directed6', if num_nodes exceeds this, skip networkx
    (those columns stay zero, degrees still exact). None = always compute.
    """
    if mode not in STRUCTURAL_FEATURE_MODES:
        raise ValueError(f"mode must be one of {list(STRUCTURAL_FEATURE_MODES)}, got {mode}")
    in_deg, out_deg = _degrees(edge_index, num_nodes)
    log_deg = torch.log1p(in_deg + out_deg)

    if mode == "directed3":
        feats = torch.stack([in_deg, out_deg, log_deg], dim=1)
        if standardize:
            mean = feats.mean(dim=0, keepdim=True)
            std = feats.std(dim=0, keepdim=True).clamp_min(1e-6)
            feats = (feats - mean) / std
        return feats

    if mode == "directed3_log":
        # Scaling FIX for directed3. The raw in/out degree counts are power-law, so
        # z-scoring them directly leaves the bulk crushed (IQR ~ 0) while hubs blow up
        # to |z| ~ 1300 — a heavy tail no linear input weight can undo (see
        # scripts/experiments/analysis/topology_feature_ssl/check_directed3_distribution.py). We
        # log1p the two raw counts BEFORE z-scoring, exactly as log_deg already is, so
        # all three columns are well-scaled (max |z| drops to the low teens, like
        # log_deg). Kept as a SEPARATE mode so directed3 stays byte-stable and old E1/E2
        # checkpoints remain evaluable; the cache lives at *.structural_directed3_log.pt.
        feats = torch.stack([torch.log1p(in_deg), torch.log1p(out_deg), log_deg], dim=1)
        if standardize:
            mean = feats.mean(dim=0, keepdim=True)
            std = feats.std(dim=0, keepdim=True).clamp_min(1e-6)
            feats = (feats - mean) / std
        return feats

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


def load_or_compute_structural(
    edge_index: torch.Tensor,
    num_nodes: int,
    cache_path: str | None = None,
    *,
    mode: str = "directed6",
    standardize: bool = True,
    max_nx_nodes: int | None = 2_000_000,
) -> torch.Tensor:
    """compute_structural_features with an on-disk cache keyed by the graph path.

    The features are identical across the many eval jobs that reload the same
    graph, so we cache the [num_nodes, k] tensor next to the graph
    (``<graph>.structural_<mode>.pt``) and validate it against (mode, num_nodes,
    edge_count) before reuse.
    """
    n_edges = int(edge_index.size(1))
    if cache_path and os.path.exists(cache_path):
        try:
            blob = torch.load(cache_path, map_location="cpu")
            if (blob.get("mode") == mode and blob.get("num_nodes") == num_nodes
                    and blob.get("n_edges") == n_edges
                    and blob.get("standardize") == standardize):
                print(f"[structural] loaded cached {mode} features from {cache_path}", flush=True)
                return blob["feats"]
            print(f"[structural] cache {cache_path} stale — recomputing", flush=True)
        except Exception as exc:  # pragma: no cover - corrupt cache
            print(f"[structural] failed to read cache {cache_path} ({exc}) — recomputing",
                  flush=True)

    print(f"[structural] computing {mode} structural features "
          f"({num_nodes} nodes, {n_edges} edges)...", flush=True)
    feats = compute_structural_features(
        edge_index, num_nodes, mode=mode, standardize=standardize, max_nx_nodes=max_nx_nodes
    )
    if cache_path:
        try:
            torch.save({"feats": feats, "mode": mode, "num_nodes": num_nodes,
                        "n_edges": n_edges, "standardize": standardize}, cache_path)
            print(f"[structural] cached features to {cache_path}", flush=True)
        except Exception as exc:  # pragma: no cover - read-only fs
            print(f"[structural] could not write cache {cache_path} ({exc})", flush=True)
    return feats
