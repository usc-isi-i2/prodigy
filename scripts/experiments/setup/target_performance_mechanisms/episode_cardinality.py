"""Scoped attention-multiplicity diagnostics, not synthetic new label classes."""
from contextlib import contextmanager

import torch
from torch_geometric.utils import softmax

from models.metaGNN import MetaGNN, MetaGNNLayer
from .episode_cardinality_contract import VARIANTS


def edge_counts(edges, attrs, start_right, nodes, train_ways=30, train_shots=3):
    """Validate actual production geometry without accessing query ground truth."""
    if not 1 < train_ways or not 0 < train_shots or edges.shape != (2, len(attrs)):
        raise ValueError("invalid count specification or edge shape")
    if attrs.shape != (edges.shape[1], 2) or not 0 < start_right < nodes:
        raise ValueError("invalid bipartite shape")
    if edges.min() < 0 or edges.max() >= nodes:
        raise ValueError("node index out of bounds")
    src, dst = edges
    loop = src == dst
    if not torch.equal(src[loop].sort().values, torch.arange(nodes, device=src.device)) or torch.any(attrs[loop] != 0):
        raise ValueError("exactly one zero-attribute self loop per node required")
    to_label = ~loop & (dst >= start_right) & (src < start_right)
    to_data = ~loop & (dst < start_right) & (src >= start_right)
    if not torch.all(loop | to_label | to_data):
        raise ValueError("unexpected non-bipartite edge")
    is_query = attrs[:, 0] == 1
    pos, neg = attrs[:, 1] == 1, attrs[:, 1] == -1
    if torch.any(is_query & to_label) or not torch.all(torch.isin(attrs[:, 0], torch.tensor([0., 1.], device=attrs.device))):
        raise ValueError("query-to-label message or invalid query marker")
    if not torch.all((pos | neg)[to_label]) or not torch.all((is_query | pos | neg | loop)):
        raise ValueError("missing signed support relation")
    if torch.any(attrs[is_query, 1] != 0):
        raise ValueError("query ground truth entered message attributes")
    if edges.T.unique(dim=0).shape[0] != edges.shape[1]:
        raise ValueError("duplicate production edge")
    counts = torch.bincount(dst[to_data], minlength=nodes)[:start_right]
    ways = int(counts[0])
    if ways < 2 or not torch.all(counts == ways):
        raise ValueError("inconsistent evaluation ways")
    pc = torch.bincount(dst[to_label & pos], minlength=nodes)[start_right:]
    nc = torch.bincount(dst[to_label & neg], minlength=nodes)[start_right:]
    shots = int(pc[0])
    if shots < 1 or not torch.all(pc == shots) or not torch.all(nc == (ways - 1) * shots):
        raise ValueError("inconsistent support counts")
    query_nodes = torch.zeros(nodes, dtype=torch.bool, device=dst.device)
    query_nodes[dst[to_data & is_query]] = True
    if not torch.all(is_query[to_data] == query_nodes[dst[to_data]]):
        raise ValueError("mixed query/support role for one data point")
    support_pos = torch.bincount(dst[to_data & pos], minlength=nodes)[:start_right]
    if not torch.all(support_pos[~query_nodes[:start_right]] == 1):
        raise ValueError("support point must have exactly one positive label")
    roles = {"label_positive": to_label & pos, "label_negative": to_label & neg,
             "label_self": loop & (dst >= start_right),
             "query_labels": to_data & is_query, "query_self": loop & query_nodes[dst],
             "support_positive": to_data & pos, "support_negative": to_data & neg,
             "support_self": loop & (dst < start_right) & ~query_nodes[dst]}
    if not torch.all(torch.stack(list(roles.values())).sum(0) == 1):
        raise ValueError("message roles are not a partition")
    way = torch.ones(len(attrs), dtype=attrs.dtype, device=attrs.device)
    shot = way.clone()
    way[roles["label_negative"] | roles["support_negative"]] = (train_ways - 1) / (ways - 1)
    way[roles["query_labels"]] = train_ways / ways
    shot[to_label] = train_shots / shots
    return way, shot, roles, {"eval_ways": ways, "eval_shots": shots,
        "train_ways": train_ways, "train_shots": train_shots,
        "label_nodes": nodes - start_right, "query_nodes": int(query_nodes.sum()),
        "support_nodes": start_right - int(query_nodes.sum())}


@contextmanager
def multiplicity_hooks(layer, weights, *, adjust_attention=True, adjust_bias=False):
    """Weights are per original message; virtual mode equals edge replication.

    Adding log(w) aggregates replicated softmax value messages. The affine bias
    is outside that softmax in production, so virtual replication also needs
    (w - 1) * bias for each retained message. Attention-only deliberately omits it.
    """
    if layer.training or weights.ndim != 1 or not torch.isfinite(weights).all() or torch.any(weights <= 0):
        raise ValueError("positive finite weights and frozen evaluation required")
    handles = []
    def attention(module, args, output):
        if output.shape[0] != len(weights):
            raise ValueError("attention/message ordering changed")
        return output + weights.log().reshape(-1, 1, 1)
    def bias(module, args, output):
        if len(output) != len(weights):
            raise ValueError("output/message ordering changed")
        return output + (weights - 1)[:, None] * module.bias
    try:
        if adjust_attention:
            handles.append(layer.att_mlp.register_forward_hook(attention))
        if adjust_bias:
            handles.append(layer.out_proj.register_forward_hook(bias))
        yield
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def cardinality_mode(meta, variant, train_ways=30, train_shots=3):
    """Wrap the actual production attention, preserving its remaining algebra."""
    if variant not in VARIANTS or not isinstance(meta, MetaGNN) or meta.num_gnn_layers != 1:
        raise ValueError("known condition and single-layer production metagraph required")
    if meta.training or meta.msg_pos_only or not meta.self_loops or meta.gnn_layers_back is not None:
        raise ValueError("unexpected metagraph recipe")
    layer = meta.gnn_layers[0]
    if not isinstance(layer, MetaGNNLayer):
        raise ValueError("expected original attention layer")
    records, handles, context = [], [], {}
    def before(module, args, kwargs):
        x = args[0] if args else kwargs["x"]
        edges = args[1] if len(args) > 1 else kwargs["edge_index"]
        attrs = kwargs["edge_attr"]
        way, shot, roles, audit = edge_counts(edges, attrs, kwargs["start_right"], len(x), train_ways, train_shots)
        weights = {"baseline": torch.ones_like(way), "count_way_attention": way,
                   "count_shot_attention": shot, "count_inverse_attention": 1 / (way * shot)}.get(variant, way * shot)
        context.update(weights=weights, roles=roles, audit=audit, index=edges[1], nodes=len(x))
    def attention(module, args, output):
        weights = context["weights"]
        adjusted = output if variant in {"baseline", "count_bias_only"} else output + weights.log()[:, None, None]
        alpha = softmax(adjusted, context["index"], num_nodes=context["nodes"]).detach()
        means = {}
        for role, mask in context["roles"].items():
            group = "label" if role.startswith("label_") else "query" if role.startswith("query_") else "support"
            count = context["audit"][f"{group}_nodes"]
            means[role] = alpha[mask].sum(0).reshape(-1).cpu().tolist() if count == 0 else (alpha[mask].sum(0) / count).reshape(-1).cpu().tolist()
        records.append(context["audit"] | {"attention_mass_per_head": means,
            "min_multiplicity": float(weights.min()), "max_multiplicity": float(weights.max())})
        return adjusted
    def bias(module, args, output):
        if variant in {"count_joint_virtual", "count_bias_only"}:
            return output + (context["weights"] - 1)[:, None] * module.bias
        return output
    try:
        handles.append(layer.register_forward_pre_hook(before, with_kwargs=True))
        handles.append(layer.att_mlp.register_forward_hook(attention))
        handles.append(layer.out_proj.register_forward_hook(bias))
        yield records
    finally:
        for handle in handles:
            handle.remove()


def cached_meta_forward(model, batch, pre_meta):
    """Replay only the suffix using production metagraph and cosine decoding."""
    if len(model.layer_list) != 3 or model.params["skip_path"] or model.params["zero_shot"]:
        raise ValueError("unexpected full-model recipe")
    if model.params["ignore_label_embeddings"] or model.params["zero_label_embeddings"]:
        raise ValueError("expected production classification label interface")
    if not isinstance(model.final_input_mlp, torch.nn.Identity) or not isinstance(model.final_label_mlp, torch.nn.Identity):
        raise ValueError("non-identity final projection not supported")
    x_label = model.initial_label_mlp(batch[1])
    x, labels = model.forward_metagraph(model.layer_list[2], pre_meta, x_label, *batch[3:9])
    logits = model.decode(x, labels, batch[3]).reshape(batch[2].shape)
    query = batch[5].reshape(-1, batch[2].shape[1])[:, 0].bool()
    return x, logits[query]
