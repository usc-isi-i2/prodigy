"""Additive accounting of saved class-reference activations, not interventions.

No model forwards or outcome labels are used. First replay the saved native
float32 affine suffix; only then split it in float64. Component contrasts use
the ORIGINAL final-label norms, never component-wise normalization. Omitting
one such component is not the result of removing it from the actual model.
"""
import math

import torch
import torch.nn.functional as F


PREFIX = 'layer_list.2.gnn_layers.0.'
COMPONENTS = ('support_values', 'label_self_values', 'label_residual',
              'output_bias', 'bn_offset')
NATIVE_RTOL = 2e-6
NATIVE_ATOL = 2e-6
ACCOUNTING_RTOL = 5e-5
ACCOUNTING_ATOL = 5e-5


def _tensor(value, name, *, dtype=None):
    if not isinstance(value, torch.Tensor) or value.device.type != 'cpu':
        raise ValueError(f'{name}: saved CPU tensor required')
    if dtype is not None and value.dtype != dtype:
        raise ValueError(f'{name}: expected {dtype}, found {value.dtype}')
    if not torch.isfinite(value).all():
        raise ValueError(f'{name}: nonfinite saved tensor')
    return value.detach()


def _maximum_error(actual, expected):
    return float((actual.double() - expected.double()).abs().max())


def decompose_class_reference(audit, state, *, label_pairs, bn_eps=1e-5):
    """Account for saved single-M labels with explicit within-episode pairing.

    ``label_pairs[E,2]`` contains LOCAL final-label-row indices (not metagraph
    indices), in desired class-0/class-1 order. Every label must occur once;
    paired labels must share their supports and have opposite signed relations.
    The capture must come from ``kv_forward`` with identity final projections.
    ``bn_eps=1e-5`` is explicit because epsilon is not stored in a state dict.

    Native replay is F.linear -> index_add -> residual -> F.batch_norm, all
    float32. Exactness is reported, with rtol=atol=2e-6 permitting documented
    CPU summation/kernel rounding. Float64 accounting is separately checked
    at rtol=atol=5e-5; its discrepancy is not hidden or attributed to a component.
    Returned tensors contain activations and contrasts, not predictions or
    query ground truth. No contribution here is a causal effect or a variance
    partition, and component contrasts can cancel one another.
    """
    if not math.isfinite(bn_eps) or bn_eps != 1e-5:
        raise ValueError('this saved model contract requires explicit BN epsilon 1e-5')
    calls = {'meta': 1, 'kqv': 1, 'attention': 1, 'weighted_values': 1, 'labels': 1}
    if audit.get('capture_calls') != calls:
        raise ValueError('complete single-layer kv_forward capture required')
    for key in state:
        if key.startswith('layer_list.2.gnn_layers.') and not key.startswith(PREFIX):
            raise ValueError('multiple metagraph layers are unsupported')
        if key.startswith(('layer_list.2.gnn_layers_back.', 'final_input_mlp.', 'final_label_mlp.')):
            raise ValueError('final-back or nonidentity final projection is unsupported')
    required = ('meta_input', 'final_labels', 'final_inputs', 'weighted_values',
                'edge_index', 'edge_attr', 'support_mask', 'query_mask',
                'attention', 'kqv_used')
    missing = [key for key in required if key not in audit]
    if missing:
        raise ValueError(f'incomplete saved activation capture: {missing}')
    x = _tensor(audit['meta_input'], 'meta_input', dtype=torch.float32)
    labels = _tensor(audit['final_labels'], 'final_labels', dtype=torch.float32)
    inputs = _tensor(audit['final_inputs'], 'final_inputs', dtype=torch.float32)
    weighted = _tensor(audit['weighted_values'], 'weighted_values', dtype=torch.float32)
    edges = _tensor(audit['edge_index'], 'edge_index', dtype=torch.long)
    attrs = _tensor(audit['edge_attr'], 'edge_attr', dtype=torch.float32)
    support = _tensor(audit['support_mask'], 'support_mask', dtype=torch.bool)
    query = _tensor(audit['query_mask'], 'query_mask', dtype=torch.bool)
    attention = _tensor(audit['attention'], 'attention', dtype=torch.float32)
    kqv = _tensor(audit['kqv_used'], 'kqv_used', dtype=torch.float32)
    pairs = _tensor(label_pairs, 'label_pairs', dtype=torch.long)
    if x.ndim != 2 or x.shape[1] < 1 or labels.ndim != 2 or inputs.ndim != 2:
        raise ValueError('matrix-valued native label/input captures required')
    total, dim = x.shape
    n = len(query)
    count = total - n
    if query.ndim != 1 or count < 2 or count % 2 or labels.shape != (count, dim) or inputs.shape != (n, dim):
        raise ValueError('binary class-reference node layout differs')
    if support.shape != (total,) or not torch.equal(support[:n], ~query) or support[n:].any():
        raise ValueError('support mask must select only nonquery data-point rows')
    if edges.ndim != 2 or edges.shape[0] != 2 or not edges.numel() or edges.min() < 0 or edges.max() >= total:
        raise ValueError('invalid saved metagraph edge indices')
    edge_count = edges.shape[1]
    if weighted.shape != (edge_count, dim) or attrs.shape != (edge_count, 2) or kqv.shape != (total, 3 * dim):
        raise ValueError('saved message dimensions differ')
    if attention.ndim != 3 or attention.shape[0] != edge_count or attention.shape[2] != 1 or attention.shape[1] < 1 or dim % attention.shape[1]:
        raise ValueError('per-edge/head attention shape differs')
    heads = attention.shape[1]
    expected_weighted = (attention * kqv[edges[0], 2 * dim:].reshape(-1, heads, dim // heads)).reshape(-1, dim)
    torch.testing.assert_close(weighted, expected_weighted, rtol=0, atol=0)
    if pairs.shape != (count // 2, 2) or not torch.equal(pairs.reshape(-1).sort().values, torch.arange(count)):
        raise ValueError('explicit label pairs must cover each local label row exactly once')
    src, dst = edges
    to_label = dst >= n
    support_edge = to_label & support[src]
    self_edge = to_label & (src == dst)
    if not torch.equal(to_label, support_edge | self_edge):
        raise ValueError('only support and own-self messages may enter labels')
    if not torch.equal(dst[self_edge].sort().values, torch.arange(n, total)) or torch.any(attrs[self_edge] != 0):
        raise ValueError('one zero-attribute self edge per label required')
    if torch.any(attrs[support_edge, 0] != 0) or not torch.all((attrs[support_edge, 1] == 1) | (attrs[support_edge, 1] == -1)):
        raise ValueError('support messages must carry existing signed support relations')
    degrees = torch.bincount(dst[to_label] - n, minlength=count)
    if degrees.min() < 2 or not torch.all(degrees == degrees[0]):
        raise ValueError('equal nontrivial class-reference degrees required')
    for left, right in pairs.tolist():
        a = torch.where(support_edge & (dst == n + left))[0]
        b = torch.where(support_edge & (dst == n + right))[0]
        a, b = a[src[a].argsort()], b[src[b].argsort()]
        if not torch.equal(src[a], src[b]) or not torch.equal(attrs[a, 1], -attrs[b, 1]) or src[a].unique().numel() != len(a):
            raise ValueError('paired labels must share unique supports with opposite class signs')
    names = ('out_proj.weight', 'out_proj.bias', 'bn.weight', 'bn.bias', 'bn.running_mean', 'bn.running_var')
    if any(PREFIX + key not in state for key in names):
        raise ValueError('all native output projection and affine saved-BN tensors are required')
    tensors = {key: _tensor(state[PREFIX + key], key, dtype=torch.float32) for key in names}
    weight, bias = tensors['out_proj.weight'], tensors['out_proj.bias']
    gamma, beta = tensors['bn.weight'], tensors['bn.bias']
    mean, variance = tensors['bn.running_mean'], tensors['bn.running_var']
    if weight.shape != (dim, dim) or any(v.shape != (dim,) for key, v in tensors.items() if key != 'out_proj.weight') or torch.any(variance < 0):
        raise ValueError('saved affine suffix dimensions or variances differ')

    # This is a saved-tensor arithmetic replay, not a model forward.
    with torch.no_grad():
        messages32 = F.linear(weighted, weight, bias)
        aggregated32 = torch.zeros_like(x).index_add_(0, dst, messages32)
        native32 = F.batch_norm(aggregated32 + x, mean, variance, gamma, beta,
                                training=False, momentum=0., eps=bn_eps)
        observed32 = torch.cat((inputs, labels))
        torch.testing.assert_close(native32, observed32, rtol=NATIVE_RTOL, atol=NATIVE_ATOL)

        # Per-edge output bias is excluded here and accounted for separately.
        value_messages64 = F.linear(weighted.double(), weight.double(), None)
        scale = gamma.double() / (variance.double() + bn_eps).sqrt()
        def incoming_sum(mask):
            return torch.zeros((count, dim), dtype=torch.float64).index_add_(
                0, dst[mask] - n, value_messages64[mask])
        components = {
            'support_values': incoming_sum(support_edge) * scale,
            'label_self_values': incoming_sum(self_edge) * scale,
            'label_residual': x[n:].double() * scale,
            'output_bias': degrees.double()[:, None] * bias.double() * scale,
            'bn_offset': (beta.double() - mean.double() * scale).expand(count, -1).clone(),
        }
        reconstructed = torch.stack(list(components.values())).sum(0)
        observed = labels.double()
        torch.testing.assert_close(reconstructed, observed, rtol=ACCOUNTING_RTOL, atol=ACCOUNTING_ATOL)
        norms = observed.norm(dim=1)
        if torch.any(norms <= 1e-8):
            raise ValueError('native cosine epsilon regime needs a separate explicit analysis')
        def contrast(values):
            normalized_terms = values / norms[:, None]
            return normalized_terms[pairs[:, 1]] - normalized_terms[pairs[:, 0]]
        contrast_components = {name: contrast(value) for name, value in components.items()}
        normalized_sum = torch.stack(list(contrast_components.values())).sum(0)
        normalized_observed = contrast(observed)
        normalized_accounting = contrast(reconstructed)
        torch.testing.assert_close(normalized_sum, normalized_accounting, rtol=1e-11, atol=1e-11)
    return {
        'raw_components': components, 'contrast_components': contrast_components,
        'raw_labels_reconstructed': reconstructed,
        'raw_labels_native_replay': native32[n:].detach().clone(),
        'raw_labels_observed': observed.clone(),
        'normalized_contrast_reconstructed': normalized_sum,
        'normalized_contrast_observed': normalized_observed,
        'label_norms': norms, 'label_degrees': degrees, 'label_pairs': pairs.clone(),
        'discrepancies': {
            'native_all_nodes_bit_exact': bool(torch.equal(native32, observed32)),
            'native_labels_bit_exact': bool(torch.equal(native32[n:], labels)),
            'native_all_nodes_max_abs': _maximum_error(native32, observed32),
            'native_labels_max_abs': _maximum_error(native32[n:], labels),
            'float64_raw_max_abs': _maximum_error(reconstructed, observed),
            'float64_normalized_contrast_max_abs': _maximum_error(normalized_sum, normalized_observed),
            'normalized_component_additivity_max_abs': _maximum_error(normalized_sum, normalized_accounting),
            'bn_eps': bn_eps, 'native_rtol': NATIVE_RTOL, 'native_atol': NATIVE_ATOL,
            'accounting_rtol': ACCOUNTING_RTOL, 'accounting_atol': ACCOUNTING_ATOL,
        },
    }
