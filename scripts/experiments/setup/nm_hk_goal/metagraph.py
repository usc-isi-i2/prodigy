"""Actual multiclass metagraph K/V interventions on cached embeddings."""
import torch
from torch_geometric.utils import softmax


def radial_value_donors(original, replacement, rows, dim):
    """Keep actual projected-value direction or radius at its original endpoint.

    The radius/direction factorial need not be additive after aggregation or
    normalization. It is a controlled tensor intervention, not a support rule.
    """
    if original.ndim != 2 or original.shape != replacement.shape or original.shape[1] != 3 * dim:
        raise ValueError('Actual Q/K/V layout required')
    a = original[list(rows), 2 * dim:]
    b = replacement[list(rows), 2 * dim:]
    ra, rb = a.norm(dim=1, keepdim=True), b.norm(dim=1, keepdim=True)
    if (ra <= 1e-12).any() or (rb <= 1e-12).any():
        raise ValueError('Cannot separate direction of a zero projected value')
    direction, radius = original.clone(), original.clone()
    direction[list(rows), 2 * dim:] = b / rb * ra
    radius[list(rows), 2 * dim:] = a / ra * rb
    return direction, radius


@torch.no_grad()
def replay(model, args, changed_rows=(), key_donor=None, value_donor=None):
    pre, labels, edges, attrs, roles = args
    module = model.layer_list[2]
    if any(m.training for m in model.modules()):
        raise ValueError('Frozen evaluation model required')
    if module.num_gnn_layers != 1 or len(module.gnn_layers) != 1 or module.gnn_layers_back is not None:
        raise ValueError('Exactly one M layer without final-back required')
    if module.msg_pos_only or not module.self_loops:
        raise ValueError('Signed support relations and self-loops required')
    layer = module.gnn_layers[0]
    if layer.dropout != 0 or not isinstance(layer.bn, torch.nn.BatchNorm1d) or not layer.bn.track_running_stats:
        raise ValueError('Zero dropout and fixed BatchNorm required')
    n, dim = pre.shape
    total = n + len(labels)
    mask = torch.zeros(total, dtype=torch.bool, device=pre.device)
    mask[list(changed_rows)] = True
    query = roles.reshape(n, -1)[:, 0].bool()
    if mask[:n][query].any() or mask[n:].any():
        raise ValueError('Only support rows may be transplanted')
    for donor in [key_donor, value_donor]:
        if donor is not None and (donor.shape != (total, 3 * dim) or donor.device != pre.device or donor.dtype != pre.dtype or not torch.isfinite(donor).all()):
            raise ValueError('Donor layout, dtype, device or finiteness mismatch')
    audit, handles = {}, []

    def before_layer(_module, positional, kwargs):
        audit['edges'] = positional[1].detach().clone()
        audit['attrs'] = kwargs['edge_attr'].detach().clone()

    def projected(_module, positional, output):
        audit['native_kqv'] = output.detach().clone()
        used = output.clone()
        for donor, start in [(key_donor, dim), (value_donor, 2 * dim)]:
            if donor is not None:
                used[mask, start:start + dim] = donor[mask, start:start + dim]
        assert torch.equal(used[:, :dim], output[:, :dim])
        assert torch.equal(used[~mask], output[~mask])
        audit['used_kqv'] = used.detach().clone()
        return used

    def attention(_module, positional, output):
        audit['attention'] = softmax(output, audit['edges'][1], num_nodes=total).detach().clone()

    def before_norm(_module, positional):
        audit['before_norm'] = positional[0].detach().clone()

    try:
        handles.append(layer.register_forward_pre_hook(before_layer, with_kwargs=True))
        handles.append(layer.mlp_kqv.register_forward_hook(projected))
        handles.append(layer.att_mlp.register_forward_hook(attention))
        handles.append(layer.bn.register_forward_pre_hook(before_norm))
        with torch.no_grad():
            x, label_out = model.forward_metagraph(module, pre, labels, edges, attrs, roles, None, None, None)
            logits = model.decode(model.final_input_mlp(x), model.final_label_mlp(label_out), edges).reshape(n, -1)
    finally:
        for handle in handles:
            handle.remove()
    audit['queries'] = x[query].detach().clone()
    audit['inputs'] = x.detach().clone()
    audit['labels'] = label_out.detach().clone()
    audit['label_before_norm'] = audit.pop('before_norm')[n:]
    audit['query_mask'] = query
    # Attention mass and observed additive label terms. The output projection's
    # bias is applied per edge in production; keep that separate from values.
    source, destination = audit['edges']
    incoming = destination >= n
    values = audit['used_kqv'][source, 2 * dim:].reshape(-1, layer.heads, dim // layer.heads)
    weighted = (audit['attention'] * values).reshape(-1, dim)
    projected_values = torch.nn.functional.linear(weighted, layer.out_proj.weight, None)
    self_mask = incoming & (source == destination)
    positive = incoming & (source < n) & (audit['attrs'][:, -1] > 0)
    negative = incoming & (source < n) & (audit['attrs'][:, -1] < 0)
    assert torch.equal(incoming, self_mask | positive | negative)
    parts = {}
    masses = {}
    for name, selected in [('positive', positive), ('negative', negative), ('self', self_mask)]:
        result = pre.new_zeros(len(labels), dim)
        result.index_add_(0, destination[selected] - n, projected_values[selected])
        parts[name] = result
        mass = pre.new_zeros(len(labels), layer.heads)
        mass.index_add_(0, destination[selected] - n, audit['attention'][selected, :, 0])
        masses[name] = mass
    counts = torch.bincount(destination[incoming] - n, minlength=len(labels)).to(pre.dtype)
    parts['output_bias'] = counts[:, None] * layer.out_proj.bias[None]
    parts['label_residual'] = labels
    before = sum(parts.values())
    error = float((before - audit['label_before_norm']).abs().max())
    assert error < 1e-4, ('label_sum_error', error)
    scale = layer.bn.weight / torch.sqrt(layer.bn.running_var + layer.bn.eps)
    parts = {k: v * scale for k, v in parts.items()}
    parts['bn_offset'] = (layer.bn.bias - layer.bn.running_mean * scale)[None].expand_as(label_out)
    final_error = float((sum(parts.values()) - label_out).abs().max())
    assert final_error < 1e-4, ('final_label_sum_error', final_error)
    audit['parts'] = {k: v.detach().clone() for k, v in parts.items()}
    audit['attention_mass'] = masses
    audit['max_accounting_error'] = max(error, final_error)
    return logits.detach(), audit


def compact_trace(audit, local, truth, logit_scale):
    query = torch.nn.functional.normalize(audit['inputs'][local], dim=0)
    labels = audit['labels']
    norms = labels.norm(dim=1)
    assert (norms > 0).all()
    terms = {k: (v @ query / norms * logit_scale).cpu() for k, v in audit['parts'].items()}
    return dict(query=audit['inputs'][local].cpu(), labels=labels.cpu(),
        label_before_norm=audit['label_before_norm'].cpu(),
        terms=terms, attention_mass={k: v.cpu() for k, v in audit['attention_mass'].items()},
        true_attention=audit['attention'][audit['edges'][1] == len(audit['inputs']) + truth].cpu(),
        max_accounting_error=audit['max_accounting_error'])
