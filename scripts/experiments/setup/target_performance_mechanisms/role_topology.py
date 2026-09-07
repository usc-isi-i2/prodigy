"""Label-blind, within-subgraph topology controls for the role-factorial replay.

Rewiring is a finite double-edge-swap perturbation, not a uniform graph sample.
Simple reciprocal graphs keep reciprocity; other graphs keep directed in/out
degrees. Self loops, attributes, nodes and pooling edges are never rewired.
"""
import hashlib
import random

import torch
import torch.nn.functional as F

from .replay import clone_batch, trace_stages
from .role_context import query_mask
from .episode_cardinality import cached_meta_forward
from models.gnn_with_edge_attr import SAGEConvSelfLoops


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256(repr(parts).encode()).digest()[:8], "little")


def swap_edges(edge_index, *, seed, swaps_per_edge=5, attempt_factor=20):
    """Return a degree-exact perturbation, preserving the original edge slots."""
    if edge_index.device.type != "cpu" or swaps_per_edge < 1 or attempt_factor < 1:
        raise ValueError("CPU edges and positive swap/attempt budgets required")
    edges = [tuple(e) for e in edge_index.T.tolist()]
    slots = {e: i for i, e in enumerate(edges)}
    if len(slots) != len(edges):
        raise ValueError("parallel background edges need a different null model")
    original = set(edges)
    reciprocal = all((v, u) in original for u, v in edges)
    movable = [(u, v) for u, v in edges if u != v and (not reciprocal or u < v)]
    rng = random.Random(seed)
    requested = swaps_per_edge * len(movable)
    accepted = attempts = 0
    # No swap is possible with fewer than two non-loop edges. Rigid stars etc.
    # are retained and explicitly counted, never silently called randomized.
    while len(movable) >= 2 and accepted < requested and attempts < requested * attempt_factor:
        attempts += 1
        i, j = rng.sample(range(len(movable)), 2)
        a, b = movable[i]
        c, d = movable[j]
        if reciprocal and rng.random() < .5:
            c, d = d, c
        new = [(a, d), (c, b)]
        old = [(a, b), (c, d)]
        if a == c or b == d or a == d or c == b:
            continue
        if reciprocal:
            if len({a, b, c, d}) != 4:
                continue
            old += [(b, a), (d, c)]
            new += [(d, a), (b, c)]
        if len(set(new)) != len(new) or any(e in slots for e in new):
            continue
        indices = [slots.pop(e) for e in old]
        for index, edge in zip(indices, new):
            edges[index] = edge
            slots[edge] = index
        movable[i] = (min(a, d), max(a, d)) if reciprocal else (a, d)
        movable[j] = (min(c, b), max(c, b)) if reciprocal else (c, b)
        accepted += 1
    result = torch.tensor(edges, dtype=edge_index.dtype).T.contiguous() if edges else edge_index.clone()
    for axis in (0, 1):
        torch.testing.assert_close(result[axis].sort().values, edge_index[axis].sort().values, rtol=0, atol=0)
    after = set(edges)
    if len(after) != len(original) or {(u, v) for u, v in after if u == v} != {(u, v) for u, v in original if u == v}:
        raise ValueError("edge count, simplicity or self loops changed")
    if reciprocal and not all((v, u) in after for u, v in after):
        raise ValueError("reciprocity changed")
    return result, {"reciprocal": reciprocal, "edges": len(edges), "requested_swaps": requested,
                    "accepted_swaps": accepted, "attempts": attempts,
                    "changed_edges": len(original - after), "degree_exact": True}


def assert_background_attributes_unused(model):
    layers = model.layer_list[0].module_list
    if not layers or any(not isinstance(layer, SAGEConvSelfLoops) or layer.lin_edge_attr is not None for layer in layers):
        raise ValueError("this null requires the verified attribute-blind SAGE encoder")


def topology_batch(batch, condition, *, seed, role="both", swaps_per_edge=5, edge_attributes_unused=False):
    """Never reads class labels; masks only use the existing support/query roles."""
    if role not in ("both", "query", "support"):
        raise ValueError("unknown role")
    if condition not in ("intact", "removed", "rewired", "drop25", "drop50", "drop75"):
        raise ValueError("unknown topology condition")
    out = clone_batch(batch)
    graph = out[0]
    edge = graph.edge_index
    if not torch.equal(graph.batch[edge[0]], graph.batch[edge[1]]):
        raise ValueError("cross-subgraph background edge")
    if torch.any(graph.global_node_ids[edge] < 0):
        raise ValueError("background edges touch pooling/virtual nodes")
    attrs = graph.edge_attr
    if attrs is not None and len(attrs) and not torch.all(attrs == attrs[0]) and not edge_attributes_unused:
        raise ValueError("nonconstant edge attributes require a typed rewiring null")
    q = query_mask(batch)
    selected = q if role == "query" else ~q if role == "support" else torch.ones_like(q)
    keep = torch.ones(edge.shape[1], dtype=torch.bool)
    audits = []
    for gi in range(len(q)):
        indices = (graph.batch[edge[0]] == gi).nonzero().flatten()
        original = edge[:, indices].clone()
        record = {"subgraph": gi, "query": bool(q[gi]), "edges": len(indices),
                  "condition": condition, "selected": bool(selected[gi])}
        if selected[gi] and condition == "rewired":
            changed, audit = swap_edges(original, seed=stable_seed(seed, gi), swaps_per_edge=swaps_per_edge)
            edge[:, indices] = changed
            record.update(audit)
        elif selected[gi] and condition == "removed":
            keep[indices] = False
        elif selected[gi] and condition.startswith("drop"):
            # Pair reciprocal edges when present; leave existing self loops fixed.
            pairs = [tuple(e) for e in original.T.tolist()]
            pair_set = set(pairs)
            reciprocal = all((v, u) in pair_set for u, v in pairs)
            units = [(u, v) for u, v in pairs if u != v and (not reciprocal or u < v)]
            random.Random(stable_seed(seed, gi)).shuffle(units)
            count = round(len(units) * int(condition[4:]) / 100)
            deleted = set(units[:count])
            if reciprocal:
                deleted |= {(v, u) for u, v in deleted}
            for index, pair in zip(indices.tolist(), pairs):
                if pair in deleted:
                    keep[index] = False
        record["retained_edges"] = int(keep[indices].sum())
        audits.append(record)
    graph.edge_index = edge[:, keep]
    if attrs is not None:
        graph.edge_attr = attrs[keep]
    return out, audits


def encode(model, batch):
    working = clone_batch(batch)
    with trace_stages(model, working[0]) as trace:
        _, logits, _ = model(*working)
    return trace["U1_pre_meta"], logits


def factorial_predictions(model, batch, encodings):
    if model.training or any(m.training for m in model.modules()):
        raise ValueError("frozen evaluation required")
    if any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) and not m.track_running_stats for m in model.modules()):
        raise ValueError("batch-dependent normalization invalidates factorization")
    q = query_mask(batch)
    outputs, geometry = {}, []
    for query_condition in ("intact", "removed", "rewired"):
        for support_condition in ("intact", "removed", "rewired"):
            mixed = torch.where(q[:, None], encodings[query_condition], encodings[support_condition])
            _, logits = cached_meta_forward(model, batch, mixed)
            if not torch.isfinite(logits).all():
                raise ValueError("nonfinite logits")
            outputs[(query_condition, support_condition)] = logits
    # Support-only geometry; no query labels enter these descriptors.
    tasks, labels = batch[0].task_id_per_sample, batch[2].argmax(1)
    base = F.normalize(encodings["intact"], dim=1)
    label_input = model.initial_label_mlp(batch[1])
    _, base_labels = model.forward_metagraph(model.layer_list[2], encodings["intact"], label_input, *batch[3:9])
    base_labels = F.normalize(base_labels, dim=1).reshape(-1, batch[2].shape[1], base_labels.shape[-1])
    for condition, values in encodings.items():
        z = F.normalize(values, dim=1)
        mixed = torch.where(q[:, None], encodings["intact"], values)
        _, learned_labels = model.forward_metagraph(model.layer_list[2], mixed, label_input, *batch[3:9])
        learned_labels = F.normalize(learned_labels, dim=1).reshape_as(base_labels)
        for task in tasks.unique(sorted=True).tolist():
            support = (tasks == task) & ~q
            means = torch.stack([z[support & (labels == c)].mean(0) for c in range(batch[2].shape[1])])
            class_vectors = F.normalize(means, dim=1)
            within = ((z[support] - means[labels[support]]) ** 2).sum(1).mean()
            geometry.append({"episode": task, "condition": condition,
                             "support_unit_dispersion": float(within),
                             "support_class_cosine": float(class_vectors[0] @ class_vectors[1]),
                             "support_movement_cosine": float((base[support] * z[support]).sum(1).mean()),
                             "learned_label_pair_cosine": float(learned_labels[task, 0] @ learned_labels[task, 1]),
                             "learned_label_movement_cosine": float((base_labels[task] * learned_labels[task]).sum(1).mean())})
    return outputs, geometry


def input_geometry(batch):
    """Target-input descriptors, shared across every source model by design."""
    g, rows = batch[0], []
    q = query_mask(batch)
    unit = F.normalize(g.x, dim=1)
    for gi, (start, end) in enumerate(zip(g.ptr[:-1].tolist(), g.ptr[1:].tolist())):
        e = g.edge_index[:, g.batch[g.edge_index[0]] == gi]
        real = g.global_node_ids[start:end] >= 0
        rows.append({"subgraph": gi, "query": bool(q[gi]), "real_nodes": int(real.sum()),
                     "edges": e.shape[1], "center_in_degree": int((e[1] == start).sum()),
                     "center_out_degree": int((e[0] == start).sum()),
                     "edge_feature_cosine": float((unit[e[0]] * unit[e[1]]).sum(1).mean()) if e.shape[1] else None})
    return rows
