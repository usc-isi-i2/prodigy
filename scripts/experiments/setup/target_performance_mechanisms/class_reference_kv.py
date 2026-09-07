"""One frozen, support-local key/value factorial on the actual metagraph.

Keys control the learned attention routing; values control what is aggregated.
This is not a new decoder or a target-fitted intervention. Full K/Q/V tensors
are donor containers, but only support K and V blocks may be transplanted.
"""
import torch
from torch_geometric.utils import softmax

from models.layer_classes import BackgroundGNNLayer, SupernodeAggrLayer
from models.metaGNN import MetaGNN, MetaGNNLayer
from .episode_cardinality import edge_counts
from .run_class_reference_contrast import final_forward


def _contract(model, batch):
    if any(m.training for m in model.modules()):
        raise ValueError('all modules must be frozen in evaluation mode')
    if len(model.layer_list) != 3 or not isinstance(model.layer_list[0], BackgroundGNNLayer) or not isinstance(model.layer_list[1], SupernodeAggrLayer) or not isinstance(model.layer_list[2], MetaGNN):
        raise ValueError('exact S/U/M module order required')
    meta = model.layer_list[2]
    if meta.num_gnn_layers != 1 or len(meta.gnn_layers) != 1 or meta.gnn_layers_back is not None or meta.msg_pos_only or not meta.self_loops:
        raise ValueError('single self-loop metagraph layer, all support relations, and no final-back required')
    layer = meta.gnn_layers[0]
    if not isinstance(layer, MetaGNNLayer) or layer.dropout != 0:
        raise ValueError('original attention layer with zero dropout required')
    if not isinstance(model.final_input_mlp, torch.nn.Identity) or not isinstance(model.final_label_mlp, torch.nn.Identity):
        raise ValueError('identity final projections required')
    if model.params.get('zero_shot') or model.params.get('skip_path'):
        raise ValueError('ordinary nonzero-shot forward without outer skip path required')
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout) and module.p != 0:
            raise ValueError('zero dropout required throughout the model')
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            if not module.track_running_stats or module.running_mean is None or module.running_var is None:
                raise ValueError('saved evaluation BatchNorm statistics required')
    if not isinstance(layer.bn, torch.nn.BatchNorm1d):
        raise ValueError('saved metagraph BatchNorm required')
    if len(model.layer_list[0].module_list) != 1 or not isinstance(model.layer_list[0].module_list[0].bn, torch.nn.BatchNorm1d):
        raise ValueError('one background layer with saved BatchNorm required')
    if batch[2].ndim != 2 or batch[2].shape[1] != 2:
        raise ValueError('binary episodes required')
    n = len(batch[2])
    mask = batch[5]
    if mask.ndim != 1 or mask.numel() != n * 2 or not torch.all((mask == 0) | (mask == 1)):
        raise ValueError('one binary role-mask entry per decoder edge required')
    roles = mask.reshape(n, 2).bool()
    if not torch.equal(roles[:, 0], roles[:, 1]) or not roles[:, 0].any() or roles[:, 0].all():
        raise ValueError('consistent per-example roles with supports and queries required')
    if len(batch[0].ptr) - 1 != n or batch[1].ndim != 2:
        raise ValueError('data-point/label-node shapes differ from batch contract')
    if batch[0].x.device.type != 'cpu' or any(p.device.type != 'cpu' for p in model.parameters()):
        raise ValueError('CPU-only audited decoder replay required')
    return layer, roles[:, 0].clone(), n + len(batch[1])


def kv_forward(model, batch, *, key_donor=None, value_donor=None, **forward_options):
    """Return ``logits, trace, contrast_parts, decoder_error, audit``.

    Donors are full ``[data points + labels, 3 * embedding dimension]`` K/Q/V
    tensors from ``audit['kqv_native']`` of a paired forward on the same batch.
    Production orders their blocks Q, K, V. Only support-node K/V blocks are
    copied; all Q blocks and every query/label row remain native. ``alpha=0``
    captures the support-message-removed endpoint using the existing helper.

    Audit tensors are detached copies and remain private experimental state.
    In particular, ``attention`` is per edge and head, while ``attention_sum``
    is per destination and head. Attention is reconstructed from actual MLP
    logits and checked against the actual input to the output projection.
    """
    layer, q, total = _contract(model, batch)
    dim, heads = layer.emb_dim, layer.heads
    support = torch.zeros(total, dtype=torch.bool)
    support[:len(q)] = ~q
    donors = {'key': key_donor, 'value': value_donor}
    for name, donor in donors.items():
        if donor is not None and (not isinstance(donor, torch.Tensor) or donor.shape != (total, 3 * dim) or not torch.isfinite(donor).all()):
            raise ValueError(f'{name} donor must be a finite full paired K/Q/V tensor')
    audit = {'query_mask': q, 'support_mask': support,
             'key_transplanted': key_donor is not None,
             'value_transplanted': value_donor is not None}
    counts = {'meta': 0, 'kqv': 0, 'attention': 0, 'weighted_values': 0, 'labels': 0}
    handles = []

    def before_meta(module, args, kwargs):
        counts['meta'] += 1
        x = args[0] if args else kwargs['x']
        edges = args[1] if len(args) > 1 else kwargs['edge_index']
        attrs = kwargs['edge_attr']
        if counts['meta'] != 1 or x.shape != (total, dim) or kwargs['start_right'] != len(q):
            raise ValueError('one metagraph call with the original data/label layout required')
        _, _, roles, geometry = edge_counts(edges, attrs, len(q), total, train_ways=2, train_shots=1)
        geometry = {k: v for k, v in geometry.items() if not k.startswith('train_')}
        audit.update(meta_input=x.detach().clone(), edge_index=edges.detach().clone(),
                     edge_attr=attrs.detach().clone(), edge_roles=roles, edge_geometry=geometry)

    def transplant(module, args, output):
        counts['kqv'] += 1
        if counts['kqv'] != 1 or output.shape != (total, 3 * dim):
            raise ValueError('one full native K/Q/V projection required')
        audit['kqv_native'] = output.detach().clone()
        result = output.clone()
        for name, start in (('key', dim), ('value', 2 * dim)):
            donor = donors[name]
            if donor is not None:
                if donor.dtype != output.dtype or donor.device != output.device:
                    raise ValueError(f'{name} donor dtype/device differs from native projection')
                result[support, start:start + dim] = donor[support, start:start + dim]
        torch.testing.assert_close(result[:, :dim], output[:, :dim], rtol=0, atol=0)
        torch.testing.assert_close(result[~support], output[~support], rtol=0, atol=0)
        audit['kqv_used'] = result.detach().clone()
        return result

    def attention(module, args, output):
        counts['attention'] += 1
        edges = audit['edge_index']
        if counts['attention'] != 1 or output.shape != (edges.shape[1], heads, 1):
            raise ValueError('unexpected attention-score layout')
        weights = softmax(output, edges[1], num_nodes=total)
        sums = output.new_zeros((total, heads, 1))
        sums.index_add_(0, edges[1], weights)
        torch.testing.assert_close(sums, torch.ones_like(sums), rtol=1e-6, atol=1e-6)
        audit.update(attention_logits=output.detach().clone(), attention=weights.detach().clone(),
                     attention_sum=sums.detach().clone())

    def weighted_values(module, args):
        counts['weighted_values'] += 1
        edges = audit['edge_index']
        values = audit['kqv_used'][edges[0], 2 * dim:].reshape(-1, heads, dim // heads)
        expected = (audit['attention'] * values).reshape(-1, dim)
        actual = args[0]
        if counts['weighted_values'] != 1 or actual.shape != expected.shape:
            raise ValueError('unexpected attention-weighted value layout')
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        audit['max_attention_value_error'] = float((actual - expected).abs().max())
        audit['weighted_values'] = actual.detach().clone()

    def final_labels(module, args, output):
        counts['labels'] += 1
        audit['final_labels'] = output.detach().clone()

    try:
        handles.append(layer.register_forward_pre_hook(before_meta, with_kwargs=True))
        handles.append(layer.mlp_kqv.register_forward_hook(transplant))
        handles.append(layer.att_mlp.register_forward_hook(attention))
        handles.append(layer.out_proj.register_forward_pre_hook(weighted_values))
        handles.append(model.final_label_mlp.register_forward_hook(final_labels))
        with torch.no_grad():
            logits, trace, parts, error = final_forward(model, batch, **forward_options)
    finally:
        for handle in handles:
            handle.remove()
    if any(value != 1 for value in counts.values()):
        raise ValueError(f'incomplete single-layer capture: {counts}')
    audit['final_inputs'] = trace['final_input'].detach().clone()
    audit['capture_calls'] = counts
    return logits, trace, parts, error, audit
