"""Nested, label-blind support-edge suppression; endpoints match role removal."""
import hashlib
import random

import torch

from .replay import clone_batch
from .role_context import query_mask
from .role_topology import stable_seed

DOSES = (0, 25, 50, 75, 100)


def dose_masks(batch, *, seed):
    g, q = batch[0], query_mask(batch)
    edges = g.edge_index
    if edges.device.type != "cpu" or not torch.equal(g.batch[edges[0]], g.batch[edges[1]]):
        raise ValueError("disjoint CPU background subgraphs required")
    if torch.any(g.global_node_ids[edges] < 0):
        raise ValueError("background edges must not touch pooling nodes")
    masks = {dose: torch.ones(edges.shape[1], dtype=torch.bool) for dose in DOSES}
    stats = {dose: {"support_edges": 0, "retained_support_edges": 0, "support_units": 0,
                   "deleted_support_units": 0, "support_subgraphs": 0, "nonempty_support_subgraphs": 0,
                   "emptied_support_subgraphs": 0, "support_self_loops": 0} for dose in DOSES}
    for gi in (~q).nonzero().flatten().tolist():
        indices = (g.batch[edges[0]] == gi).nonzero().flatten()
        pairs = [tuple(e) for e in edges[:, indices].T.tolist()]
        unique = set(pairs)
        if len(unique) != len(pairs):
            raise ValueError("parallel background edges need an explicit sampling policy")
        reciprocal = all((v, u) in unique for u, v in pairs)
        units = sorted((u, v) for u, v in unique if not reciprocal or u <= v)
        random.Random(stable_seed(seed, gi)).shuffle(units)
        rank = {pair: i for i, pair in enumerate(units)}
        edge_ranks = [rank[(min(u, v), max(u, v)) if reciprocal else (u, v)] for u, v in pairs]
        for dose in DOSES:
            # Round half up, per subgraph. Directed edges or reciprocal pairs are
            # deletion units; self loops are singleton units and vanish at 100%.
            count = (len(units)*dose + 50)//100
            kept = torch.tensor([r >= count for r in edge_ranks], dtype=torch.bool)
            masks[dose][indices] = kept
            s = stats[dose]
            s["support_edges"] += len(pairs)
            s["retained_support_edges"] += int(kept.sum())
            s["support_units"] += len(units)
            s["deleted_support_units"] += count
            s["support_subgraphs"] += 1
            s["nonempty_support_subgraphs"] += bool(pairs)
            s["emptied_support_subgraphs"] += bool(pairs) and not bool(kept.any())
            s["support_self_loops"] += sum(u == v for u, v in pairs)
    query_edges = q[g.batch[edges[0]]]
    for dose, mask in masks.items():
        if not mask[query_edges].all():
            raise ValueError("query edge changed")
        stats[dose].update({"suppression_percent": dose, "query_edges": int(query_edges.sum()),
                           "mask_sha256": hashlib.sha256(mask.numpy().tobytes()).hexdigest()})
    if not masks[0].all() or not torch.equal(masks[100], query_edges):
        raise ValueError("incorrect intact/removal endpoints")
    if any(torch.any(masks[b] & ~masks[a]) for a, b in zip(DOSES, DOSES[1:])):
        raise ValueError("suppression masks are not nested")
    return masks, stats


def masked_batch(batch, mask):
    out = clone_batch(batch)
    g = out[0]
    if mask.dtype != torch.bool or mask.shape != (g.edge_index.shape[1],):
        raise ValueError("one Boolean per background edge required")
    g.edge_index = g.edge_index[:, mask]
    if g.edge_attr is not None:
        g.edge_attr = g.edge_attr[mask]
    return out
