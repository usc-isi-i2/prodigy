"""Readouts and role interventions on one cached native KG episode.

The input is the original nine-argument model tuple, before forward mutates
the PyG Batch.  The public protocol is 20-way, 3-shot, 4-query, batch size one.
These functions neither sample episodes nor run or change a model.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch_geometric.data import Batch


def _require(condition: bool | Tensor, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _finite(value: Tensor, name: str) -> None:
    _require(torch.isfinite(value).all(), f"{name} contains nonfinite values")


def validate_episode(args, expected_ways=20, shots=3, queries=4) -> dict:
    """Validate the native batch-one schema and recover graph-level roles.

    ``is_query`` and ``labels`` have one entry per subgraph / one-hot label
    row.  Labels are *episode-local* class indices, not global KG relations.
    The cached query mask instead has one entry per metagraph edge.
    """
    _require(isinstance(args, (tuple, list)) and len(args) == 9,
             "expected the native nine-argument cached tuple")
    graph, x_label, y, meta_index, meta_attr, query_mask, *_ = args
    _require(isinstance(graph, Batch), "args[0] must be a PyG Batch")
    _require(all(isinstance(value, Tensor) for value in args[1:]),
             "all non-graph native arguments must be tensors")
    _require(isinstance(expected_ways, int) and expected_ways >= 2,
             "expected_ways must be an integer >= 2")
    _require(isinstance(shots, int) and shots > 0 and
             isinstance(queries, int) and queries > 0,
             "shots and queries must be positive integers")
    _require(y.ndim == 2 and y.shape[1] == expected_ways,
             f"expected {expected_ways}-way one-hot labels")
    graph_count, ways = y.shape
    _require(graph_count == ways * (shots + queries),
             "graph count does not match ways * (shots + queries)")
    _finite(y, "one-hot labels")
    _require(((y == 0) | (y == 1)).all() and (y.sum(dim=1) == 1).all(),
             "labels must be exactly one-hot")
    labels = y.argmax(dim=1).long()
    _require(x_label.shape == (ways, 768), "expected [ways, 768] label embeddings")
    _finite(x_label, "label embeddings")

    x, ptr, batch = graph.x, graph.ptr, graph.batch
    _require(isinstance(x, Tensor) and x.ndim == 2 and x.shape[1] == 770,
             "expected 768 node features plus two head/tail flags")
    _require(x.is_floating_point(), "node features must be floating point")
    _finite(x, "node features")
    _require(ptr.dtype == torch.long and ptr.shape == (graph_count + 1,),
             "ptr must have one boundary per subgraph")
    _require(ptr[0] == 0 and ptr[-1] == len(x), "ptr does not span node features")
    _require(((ptr[1:] - ptr[:-1]) >= 3).all(),
             "each KG subgraph must contain head, tail and pooling supernode")
    _require(batch.dtype == torch.long and batch.shape == (len(x),),
             "batch must map each node to a subgraph")
    _require(all(value.device == x.device for value in
                 (ptr, batch, x_label, y, meta_index, meta_attr, query_mask)),
             "episode graph and metagraph tensors must share a device")
    expected_batch = torch.arange(graph_count, device=x.device).repeat_interleave(
        ptr[1:] - ptr[:-1])
    _require(torch.equal(batch, expected_batch), "batch and ptr disagree")

    meta_count = graph_count * ways
    _require(meta_index.dtype == torch.long and meta_index.shape == (2, meta_count),
             "unexpected metagraph edge index shape or dtype")
    expected_source = torch.arange(graph_count, device=x.device).repeat_interleave(ways)
    expected_target = torch.arange(ways, device=x.device).repeat(graph_count) + graph_count
    _require(torch.equal(meta_index[0], expected_source) and
             torch.equal(meta_index[1], expected_target),
             "metagraph must be the canonical single-episode graph-to-label ordering")
    _require(query_mask.dtype == torch.bool and query_mask.shape == (meta_count,),
             "query_set_mask must be boolean with one entry per metagraph edge")
    row_mask = query_mask.reshape(graph_count, ways)
    is_query = row_mask[:, 0].clone()
    _require(torch.equal(row_mask, is_query[:, None].expand_as(row_mask)),
             "query_set_mask has inconsistent roles within a subgraph")
    _require(meta_attr.shape == (meta_count, 2),
             "metagraph edge attributes must have two aligned columns")
    _finite(meta_attr, "metagraph edge attributes")
    # Query rows are zero regardless of their held-out labels.  They cannot
    # provide class membership to support-only prototype construction.
    expected_sign = (2 * y - 1) * (~is_query[:, None])
    _require(torch.equal(meta_attr[:, 0], query_mask.to(meta_attr.dtype)) and
             torch.equal(meta_attr[:, 1], expected_sign.reshape(-1).to(meta_attr.dtype)),
             "metagraph attributes disagree with roles or support labels")
    support_counts = torch.bincount(labels[~is_query], minlength=ways)
    query_counts = torch.bincount(labels[is_query], minlength=ways)
    _require((support_counts == shots).all(), f"expected exactly {shots} supports per class")
    _require((query_counts == queries).all(), f"expected exactly {queries} queries per class")

    edge_index, edge_attr = graph.edge_index, graph.edge_attr
    _require(isinstance(edge_index, Tensor) and edge_index.dtype == torch.long and
             edge_index.ndim == 2 and edge_index.shape[0] == 2,
             "background edge_index must be long [2, E]")
    _require(isinstance(edge_attr, Tensor) and edge_attr.ndim == 2 and
             edge_attr.shape[0] == edge_index.shape[1],
             "background edge attributes are not aligned with edge_index")
    _require(edge_index.device == x.device and edge_attr.device == x.device,
             "background edges and graph features must share a device")
    _finite(edge_attr, "background edge attributes")
    _require(((edge_index >= 0) & (edge_index < len(x))).all(),
             "background edge index outside node range")
    _require(torch.equal(batch[edge_index[0]], batch[edge_index[1]]),
             "background edge crosses subgraphs")
    return {"is_query": is_query, "labels": labels, "ways": ways}


def _inferred_episode(args) -> dict:
    """Allow small synthetic fixtures while retaining the fixed 3/4 protocol."""
    _require(isinstance(args, (tuple, list)) and len(args) == 9 and
             isinstance(args[2], Tensor) and args[2].ndim == 2,
             "expected the native nine-argument tuple with two-dimensional labels")
    return validate_episode(args, expected_ways=args[2].shape[1])


def raw_descriptors(args) -> Tensor:
    """Concatenate ordered head/tail MPNet features, excluding both flags."""
    _inferred_episode(args)
    graph = args[0]
    heads, tails = graph.ptr[:-1], graph.ptr[:-1] + 1
    expected_flags = torch.zeros_like(graph.x[:, -2:])
    expected_flags[heads, 1] = 1
    expected_flags[tails, 0] = 1
    _require(torch.equal(graph.x[:, -2:], expected_flags),
             "head/tail flags do not match first-two-node KG ordering")
    return torch.cat((graph.x[heads, :768], graph.x[tails, :768]), dim=1)


def _unit_vectors(vectors: Tensor) -> tuple[Tensor, Tensor]:
    norms = torch.linalg.vector_norm(vectors, dim=1, keepdim=True)
    _finite(norms, "vector norms")
    zero = norms[:, 0] == 0
    # There is no tunable epsilon or replacement direction: exact zeros stay
    # zero and are explicitly counted in the returned diagnostics.
    normalized = vectors / torch.where(norms > 0, norms, torch.ones_like(norms))
    _finite(normalized, "normalized vectors")
    return normalized, zero


def cosine_prototype_logits(descriptors: Tensor, args) -> tuple[Tensor, Tensor, dict]:
    """Normalize examples, mean supports per class, normalize prototypes.

    Returns query one-hot truth and [queries, ways] cosine logits. Logits
    retain the descriptors' dtype/device; truth retains its original dtype
    on that device. Query labels are used only for returned truth
    and schema validation, never to construct a prototype.
    """
    episode = _inferred_episode(args)
    _require(isinstance(descriptors, Tensor) and descriptors.ndim == 2 and
             descriptors.shape[0] == len(episode["labels"]) and descriptors.shape[1] > 0,
             "descriptors must have one nonempty vector per subgraph")
    _require(descriptors.is_floating_point(), "descriptors must be floating point")
    _finite(descriptors, "descriptors")
    is_query = episode["is_query"].to(descriptors.device)
    labels = episode["labels"].to(descriptors.device)
    vectors, zero_examples = _unit_vectors(descriptors)
    support_vectors, support_labels = vectors[~is_query], labels[~is_query]
    means = torch.stack([support_vectors[support_labels == label].mean(dim=0)
                         for label in range(episode["ways"])])
    prototypes, zero_prototypes = _unit_vectors(means)
    logits = vectors[is_query] @ prototypes.T
    _finite(logits, "cosine logits")
    query_y_true = args[2].to(descriptors.device)[is_query].clone()
    diagnostics = {
        "ways": episode["ways"],
        "supports": int((~is_query).sum()),
        "queries": int(is_query.sum()),
        "descriptor_dim": descriptors.shape[1],
        "zero_descriptor_count": int(zero_examples.sum()),
        "zero_support_descriptor_count": int(zero_examples[~is_query].sum()),
        "zero_query_descriptor_count": int(zero_examples[is_query].sum()),
        "zero_prototype_count": int(zero_prototypes.sum()),
        "zero_norm_policy": "exact zero remains zero; positive norms divided without epsilon",
    }
    return query_y_true, logits, diagnostics


def suppress_background(args, role="support") -> tuple[tuple, dict]:
    """Clone the cache and remove only the nominated role's background edges.

    Node features, pooling edges, metagraphs and all other input fields are
    unchanged. PyG private slice metadata is cloned, not reconstructed: the
    returned Batch is for direct native forward, not ``to_data_list()``.
    """
    _require(role in ("support", "query", "intact"), "unknown suppression role")
    episode = _inferred_episode(args)
    graph = args[0]
    edge_is_query = episode["is_query"][graph.batch[graph.edge_index[0]]]
    remove = (edge_is_query if role == "query" else ~edge_is_query)
    if role == "intact":
        remove = torch.zeros_like(edge_is_query)
    keep = ~remove
    cloned = tuple(value.clone() for value in args)
    cloned[0].edge_index = graph.edge_index[:, keep].clone()
    cloned[0].edge_attr = graph.edge_attr[keep].clone()
    diagnostics = {
        "role": role,
        "edges_before": graph.edge_index.shape[1],
        "edges_after": int(keep.sum()),
        "edges_removed": int(remove.sum()),
        "support_edges_before": int((~edge_is_query).sum()),
        "support_edges_after": int((~edge_is_query & keep).sum()),
        "query_edges_before": int(edge_is_query.sum()),
        "query_edges_after": int((edge_is_query & keep).sum()),
    }
    return cloned, diagnostics
