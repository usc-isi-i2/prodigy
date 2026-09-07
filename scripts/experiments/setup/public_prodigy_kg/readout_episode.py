"""Support-only ridge at the actual input to the first metagraph block."""
import torch
from contextlib import nullcontext
from torch.nn import functional as F

from .paired_replay import replay_native


def ridge_tasks(embeddings, labels, edges, query_mask, *, ridge=1.0):
    """Global bipartite edges define tasks and their ordered local class columns.

    Query labels are deliberately never read. Return query scores in the same
    row order as native inference, plus per-task indices for separate scoring.
    """
    n = len(embeddings)
    if ridge <= 0 or not torch.isfinite(torch.tensor(ridge)):
        raise ValueError("Ridge must be positive and finite")
    if labels.ndim != 2 or labels.shape[0] != n or labels.shape[1] < 2:
        raise ValueError("Invalid example/class inventory")
    ways = labels.shape[1]
    if edges.shape != (2, n * ways) or query_mask.numel() != n * ways:
        raise ValueError("Expected one ordered edge per example/class")
    source, destination = edges.reshape(2, n, ways)
    if not torch.equal(source, torch.arange(n, device=edges.device)[:, None].expand(n, ways)):
        raise ValueError("Edges must follow native example-major decoding order")
    if (destination < n).any():
        raise ValueError("Label destinations must use global, not local, indices")
    mask = query_mask.reshape(n, ways)
    if not torch.all((mask == 0) | (mask == 1)) or not torch.all(mask == mask[:, :1]):
        raise ValueError("Query membership must be constant across class edges")
    query = mask[:, 0].bool()
    features = F.normalize(embeddings, dim=1)
    if features.ndim != 2 or not torch.isfinite(features).all():
        raise ValueError("Invalid pre-metagraph embeddings")
    groups = {}
    used = set()
    for row, destinations in enumerate(destination.tolist()):
        key = tuple(destinations)
        if len(set(key)) != ways:
            raise ValueError("Duplicate class destination")
        if key not in groups:
            if used.intersection(key):
                raise ValueError("Task labels overlap or class order is inconsistent")
            used.update(key)
            groups[key] = []
        groups[key].append(row)
    output = features.new_empty((int(query.sum()), ways))
    query_order = torch.full((n,), -1, dtype=torch.long, device=features.device)
    query_order[query] = torch.arange(int(query.sum()), device=features.device)
    tasks = []
    for key, rows in groups.items():
        rows = torch.tensor(rows, device=features.device)
        support, queries = rows[~query[rows]], rows[query[rows]]
        y = labels[support].to(features)
        if not len(support) or not len(queries):
            raise ValueError("Every task needs support and query examples")
        if not torch.all((y == 0) | (y == 1)) or not torch.all(y.sum(1) == 1) or not torch.all(y.sum(0) > 0):
            raise ValueError("Every class needs one-hot labeled support")
        x = features[support]
        weights = torch.linalg.solve(x @ x.T + ridge * torch.eye(len(x), device=x.device, dtype=x.dtype), y)
        indices = query_order[queries]
        output[indices] = features[queries] @ x.T @ weights
        tasks.append({"query_indices": indices.cpu(), "class_destinations": list(key),
                      "support_rows": support.cpu(), "query_rows": queries.cpu()})
    return output, tasks


def run_episode(model, artifacts, device, *, atol=0.0, rtol=0.0, compare_post=False):
    from .episode_mechanism import capture_decoder
    modules = [module for module in model.layer_list if hasattr(module, "gnn_layers")]
    if not modules:
        raise ValueError("No metagraph block found")
    captured = []

    def before(_module, _args, kwargs):
        captured.append(kwargs["x"][:kwargs["start_right"]].detach().clone())

    handle = modules[0].register_forward_pre_hook(before, with_kwargs=True)
    try:
        with capture_decoder(model) if compare_post else nullcontext() as decoded:
            native = replay_native(model, artifacts, device, atol=atol, rtol=rtol)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise ValueError("Expected exactly one first-metagraph invocation")
    arguments = artifacts["input"]
    with torch.no_grad():
        logits, tasks = ridge_tasks(captured[0], arguments[2].to(device),
                                   arguments[3].to(device), arguments[5].to(device))
    if logits.shape != native["logits"].shape:
        raise ValueError("Readout and native query/class inventories differ")
    result = {"native": native, "ridge_logits": logits.cpu(), "tasks": tasks,
            "protocol": {"ridge_lambda": 1.0, "row_l2_normalization": True,
                         "intercept": False, "logit_scale": 1.0,
                         "stage": "input to first metagraph block"}}
    if compare_post:
        with torch.no_grad():
            post, _ = ridge_tasks(decoded[0]["input_x"], arguments[2].to(device),
                                  arguments[3].to(device), arguments[5].to(device))
        result["post_ridge_logits"] = post.cpu()
        result["geometry"] = {"pre": captured[0].cpu(),
                              "post": decoded[0]["input_x"].cpu(),
                              "references": decoded[0]["label_x"].cpu()}
        result["protocol"]["post_stage"] = "actual input embeddings consumed by native decoder"
    return result
