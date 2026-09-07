"""The complete role/topology/message/dose inference panel; no fitted policy."""
import torch

from .digest import model_digest
from .replay import batch_hash, episode_probe
from .role_context import query_mask
from .role_topology import encode, topology_batch, stable_seed, factorial_predictions
from .episode_cardinality import cached_meta_forward
from .message_content import aggregation_audit, message_control
from .support_dose import dose_masks, masked_batch

CHANGES = ('actual_mean', 'no_message_bias', 'bias_only', 'mean_message', 'zero_messages')


def expected_conditions():
    keys = {f'topology/{q}/{s}/{draw}' for draw in range(3)
            for q in ('intact', 'removed', 'rewired') for s in ('intact', 'removed', 'rewired')
            if draw == 0 or 'rewired' in (q, s)}
    keys |= {f'message/{c}/{r}' for c in CHANGES for r in ('query', 'support', 'both')}
    keys |= {f'dose/{d}/{draw}' for d in (25, 50, 75) for draw in range(3)}
    return keys | {'prototype/raw', 'prototype/encoder'}


@torch.no_grad()
def predict_batch(model, batch, *, target, stream, batch_index, check_direct=False):
    before, weights, operator = batch_hash(batch), model_digest(model.state_dict()), aggregation_audit(model)
    if model.training or any(m.training for m in model.modules()):
        raise ValueError('Frozen evaluation required')
    q = query_mask(batch)
    original, baseline = encode(model, batch)
    original_post, original_suffix = cached_meta_forward(model, batch, original)
    torch.testing.assert_close(baseline, original_suffix, rtol=0, atol=1e-5)
    removed, _ = topology_batch(batch, 'removed', seed=0, edge_attributes_unused=True)
    removed_pre, _ = encode(model, removed)
    outputs, checks = {}, {'query_pre_exact': 0, 'query_post_exact': 0, 'direct_role_checks': 0}
    for draw in range(3):
        seed = stable_seed(target, stream, batch_index, draw)
        altered, _ = topology_batch(batch, 'rewired', seed=seed, edge_attributes_unused=True)
        changed, _ = encode(model, altered)
        values, _ = factorial_predictions(model, batch, {'intact': original, 'removed': removed_pre, 'rewired': changed})
        for (qc, sc), logits in values.items():
            if draw == 0 or 'rewired' in (qc, sc):
                outputs[f'topology/{qc}/{sc}/{draw}'] = logits
            if check_direct:
                direct, _ = topology_batch(batch, qc, seed=seed, role='query', edge_attributes_unused=True)
                direct, _ = topology_batch(direct, sc, seed=seed, role='support', edge_attributes_unused=True)
                _, pred = encode(model, direct)
                torch.testing.assert_close(logits, pred, rtol=0, atol=1e-5)
                checks['direct_role_checks'] += 1
    for condition in CHANGES:
        with message_control(model, batch[0], condition):
            changed, direct_both = encode(model, batch)
        for role in ('query', 'support', 'both'):
            selected = q if role == 'query' else ~q if role == 'support' else torch.ones_like(q)
            mixed = torch.where(selected[:, None], changed, original)
            post, logits = cached_meta_forward(model, batch, mixed)
            outputs[f'message/{condition}/{role}'] = logits
            if role == 'support':
                torch.testing.assert_close(post[q], original_post[q], rtol=0, atol=0)
                checks['query_post_exact'] += 1
            if role == 'both':
                torch.testing.assert_close(logits, direct_both, rtol=0, atol=1e-5)
            elif check_direct:
                with message_control(model, batch[0], condition, node_mask=selected[batch[0].batch]):
                    pre, direct = encode(model, batch)
                torch.testing.assert_close(pre[~selected], original[~selected], rtol=0, atol=0)
                torch.testing.assert_close(direct, logits, rtol=0, atol=1e-5)
                checks['direct_role_checks'] += 1
    for draw in range(3):
        masks, _ = dose_masks(batch, seed=stable_seed('support_dose_v1', target, stream, batch_index, draw))
        for dose in (25, 50, 75):
            pre, logits = encode(model, masked_batch(batch, masks[dose]))
            post, suffix = cached_meta_forward(model, batch, pre)
            torch.testing.assert_close(pre[q], original[q], rtol=0, atol=0)
            torch.testing.assert_close(post[q], original_post[q], rtol=0, atol=0)
            torch.testing.assert_close(logits, suffix, rtol=0, atol=1e-5)
            outputs[f'dose/{dose}/{draw}'] = logits
            checks['query_pre_exact'] += 1
            checks['query_post_exact'] += 1
    outputs['prototype/raw'] = episode_probe(batch[0].x[batch[0].ptr[:-1]], batch, 'prototype')
    outputs['prototype/encoder'] = episode_probe(original, batch, 'prototype')
    if set(outputs) != expected_conditions() or not all(torch.isfinite(v).all() for v in outputs.values()):
        raise ValueError('Incomplete or nonfinite inference panel')
    if batch_hash(batch) != before or model_digest(model.state_dict()) != weights or aggregation_audit(model) != operator:
        raise ValueError('Input, weights, buffers or operator mutated')
    return outputs, checks
