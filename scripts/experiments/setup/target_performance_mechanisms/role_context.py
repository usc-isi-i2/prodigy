"""Factor query/support input roles while retaining the production model."""
import torch

from .replay import clone_batch, trace_stages
from .episode_cardinality import cached_meta_forward


VARIANTS = ("baseline", "features_query", "features_support", "features_both",
            "edges_query", "edges_support", "edges_both", "center_zero_query")


def query_mask(batch):
    return batch[5].reshape(-1, batch[2].shape[1])[:, 0].bool()


def intervene_roles(batch, variant):
    if variant not in VARIANTS:
        raise ValueError(variant)
    out = clone_batch(batch)
    if variant == "baseline":
        return out
    q = query_mask(out)
    role = variant.rsplit("_", 1)[-1]
    selected = q if role == "query" else ~q if role == "support" else torch.ones_like(q)
    g = out[0]
    centers = g.ptr[:-1]
    if variant.startswith("features_"):
        mask = selected[g.batch] & (g.global_node_ids >= 0)
        mask[centers] = False
        g.x[mask] = g.x[centers[g.batch[mask]]]
    elif variant.startswith("edges_"):
        keep = ~selected[g.batch[g.edge_index[0]]]
        g.edge_index = g.edge_index[:, keep]
        if g.edge_attr is not None:
            g.edge_attr = g.edge_attr[keep]
    else:
        g.x[centers[selected]] = 0
    return out


def changed_embedding(model, batch, variant):
    altered = intervene_roles(batch, variant)
    with trace_stages(model, altered[0]) as trace:
        _, logits, _ = model(*altered)
    return trace["U1_pre_meta"], logits


def role_predictions(model, batch, baseline_pre, *, verify_direct=False):
    """Reuse per-subgraph S/U outputs, verifying actual whole-forward equivalence.

    Requires eval-mode BN with frozen buffers and disjoint background subgraphs.
    Every suffix is the original metagraph and decoder, not an approximation.
    """
    if model.training or any(m.training for m in model.modules()):
        raise ValueError("all layers must be in evaluation mode")
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm) and not m.track_running_stats:
            raise ValueError("transductive batch statistics invalidate role factorization")
    g = batch[0]
    if not torch.equal(g.batch[g.edge_index[0]], g.batch[g.edge_index[1]]):
        raise ValueError("background edge joins different subgraphs")
    q = query_mask(batch)
    _, base = cached_meta_forward(model, batch, baseline_pre)
    outputs = {"baseline": base}
    checks = {}
    for family in ("features", "edges"):
        both = family + "_both"
        pre, direct_both = changed_embedding(model, batch, both)
        for role in ("query", "support", "both"):
            variant = family + "_" + role
            mask = q if role == "query" else ~q if role == "support" else torch.ones_like(q)
            mixed = torch.where(mask[:, None], pre, baseline_pre)
            _, outputs[variant] = cached_meta_forward(model, batch, mixed)
            if role == "both" or verify_direct:
                if role == "both":
                    direct = direct_both
                else:
                    _, direct, _ = model(*intervene_roles(batch, variant))
                error = float((direct - outputs[variant]).abs().max())
                torch.testing.assert_close(direct, outputs[variant], rtol=0, atol=1e-5)
                checks[variant] = error
    # This condition is already role-localized, so it needs just one full pass.
    _, outputs["center_zero_query"] = changed_embedding(model, batch, "center_zero_query")
    if verify_direct:
        _, direct, _ = model(*clone_batch(batch))
        torch.testing.assert_close(direct, base, rtol=0, atol=1e-5)
        checks["baseline_full"] = float((direct-base).abs().max())
    return outputs, checks
