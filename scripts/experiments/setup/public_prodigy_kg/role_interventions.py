"""Role-specific native KG context donor, without changing relation endpoints.

This changes background message-passing edges, not sampled nodes/features,
head/tail flags, pooling edges, support labels, or metagraph topology. Native
batch normalization can transmit the effect to query embeddings; no fixed-query
claim is made, even before the two metagraph layers.
"""
from contextlib import contextmanager

import torch


def query_rows(arguments):
    graph, _, labels, _, _, mask, *_ = arguments
    rows, classes = labels.shape
    if rows != len(graph.ptr) - 1 or mask.numel() != rows * classes:
        raise ValueError("Expected native multiway example-by-class mask")
    mask = mask.reshape(rows, classes)
    if not ((mask == 0) | (mask == 1)).all() or not (mask == mask[:, :1]).all():
        raise ValueError("Query role must be identical across each example's classes")
    return mask[:, 0].bool()


def context_donor(arguments, role="support"):
    if role not in ("support", "query"):
        raise ValueError("Role must be support or query")
    cloned = tuple(value.clone() for value in arguments)
    graph = cloned[0]
    queries = query_rows(cloned)
    chosen = ~queries if role == "support" else queries
    src, dst = graph.edge_index
    if not torch.equal(graph.batch[src], graph.batch[dst]):
        raise ValueError("Background edges cross example boundaries")
    keep = ~chosen[graph.batch[src]]
    # In the pinned KG loader only edge_attr is read by the encoder. Refuse
    # additional edge payloads instead of silently leaving them misaligned.
    auxiliary = {"edge_index_supernode", "edge_index_from_supernode"}
    edge_payloads = [name for name, _ in graph
                     if name not in auxiliary | {"edge_index"} and graph.is_edge_attr(name)]
    unknown = set(edge_payloads) - {"edge_attr"}
    if unknown:
        raise ValueError(f"Unrecognized background edge payloads: {sorted(unknown)}")
    if "edge_attr" in graph and graph.edge_attr is not None:
        if graph.edge_attr.shape[0] != keep.numel():
            raise ValueError("Background edge attributes are not aligned")
        graph.edge_attr = graph.edge_attr[keep]
    graph.edge_index = graph.edge_index[:, keep]
    return cloned, {"role": role, "removed_edges": int((~keep).sum()),
                    "retained_edges": int(keep.sum()), "selected_examples": int(chosen.sum())}


@contextmanager
def transplant_projection(projection, donor, support_rows, component):
    """Replace support K/V blocks at ONE named native QKV projection.

    Native ordering is Q,K,V despite the mlp_kqv name. Multi-layer callers must
    nominate a layer and capture its donor separately. Non-support rows and Q
    are untouched at the hook, not guaranteed unchanged downstream.
    """
    if component not in ("keys", "values", "joint"):
        raise ValueError("Unknown projection component")
    if donor.ndim != 2 or donor.shape[1] % 3:
        raise ValueError("Expected concatenated Q,K,V donor")
    if support_rows.dtype != torch.bool or support_rows.shape != donor.shape[:1]:
        raise ValueError("Expected explicit support-only row mask including label rows")
    calls = []

    def replace(_module, _inputs, output):
        if output.shape != donor.shape or output.device != donor.device or output.dtype != donor.dtype:
            raise ValueError("Donor projection differs from native contract")
        if calls:
            raise ValueError("Nominated projection invoked more than once")
        calls.append(True)
        width = output.shape[1] // 3
        changed = output.clone()
        blocks = {"keys": (1,), "values": (2,), "joint": (1, 2)}[component]
        for block in blocks:
            changed[support_rows, block * width:(block + 1) * width] = donor[support_rows, block * width:(block + 1) * width]
        return changed

    handle = projection.register_forward_hook(replace)
    try:
        yield
        if len(calls) != 1:
            raise ValueError("Nominated projection was not invoked")
    finally:
        handle.remove()
