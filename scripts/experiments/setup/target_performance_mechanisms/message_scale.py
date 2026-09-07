"""Frozen, support-only message-scale and norm/direction interventions.

Scale is applied to W x + b before aggregation and the message MLP. In
particular, zero still passes through MLP(0); it does not delete affine terms
elsewhere. BatchNorm uses its saved evaluation statistics unless explicitly
bypassed. No model parameters, buffers, labels or input tensors are modified.
"""
from contextlib import contextmanager
import math

import torch

from .replay import clone_batch, trace_stages
from .role_context import query_mask
from .role_topology import assert_background_attributes_unused


SCALES = (0., 1e-6, 1e-4, .01, .1, .3, 1.)


def radial_swap(direction, norm_donor):
    """Preserve direction, taking per-vector L2 norms from a paired condition.

    A zero direction remains zero, even if the donor is nonzero. Report these
    impossible radial assignments; never invent a direction using outcomes.
    """
    if direction.shape != norm_donor.shape or direction.ndim != 2:
        raise ValueError('matched two-dimensional stage tensors required')
    norms = direction.double().norm(dim=1, keepdim=True)
    wanted = norm_donor.double().norm(dim=1, keepdim=True)
    ratio = torch.where(norms > 0, wanted / norms.clamp_min(1e-300), 0.)
    result = direction * ratio.to(direction.dtype)
    if not torch.isfinite(result).all():
        raise ValueError('nonfinite norm transplant')
    return result, {'vectors':len(direction), 'zero_direction':int((norms == 0).sum()),
                    'nonzero_norm_unassignable':int(((norms == 0) & (wanted > 0)).sum())}


@contextmanager
def scale_control(model, batch, *, alpha=1., bias_only=False, bypass_bn=False,
                  replace_stage=None, replacement=None):
    assert_background_attributes_unused(model)
    if any(m.training for m in model.modules()) or len(model.layer_list[0].module_list) != 1:
        raise ValueError('one frozen background layer required')
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError('finite message multiplier in [0,1] required')
    if replace_stage not in (None,'post_relu','pre_meta') or (replace_stage is None) != (replacement is None):
        raise ValueError('stage and replacement must be supplied together')
    q = query_mask(batch)
    node_mask = ~q[batch[0].batch]
    stage_mask = ~q if replace_stage == 'pre_meta' else node_mask
    if replacement is not None and (replacement.ndim != 2 or len(replacement) != len(stage_mask)):
        raise ValueError('replacement stage size differs')
    bg, layer = model.layer_list[0], model.layer_list[0].module_list[0]
    if bypass_bn and (not isinstance(layer.bn, torch.nn.BatchNorm1d) or not layer.bn.track_running_stats):
        raise ValueError('saved-statistics BatchNorm required')
    handles, restore = [], None
    def alter_message(module, args, values):
        proposed = (module.bias.expand_as(values) if bias_only else values) * alpha
        return torch.where(node_mask[:,None], proposed, values)
    handles.append(layer.lin_x.register_forward_hook(alter_message))
    if bypass_bn:
        handles.append(layer.bn.register_forward_hook(lambda m,a,v: torch.where(node_mask[:,None],a[0],v)))
    act_calls = 0
    if replace_stage == 'post_relu':
        def alter_nodes(module,args,values):
            nonlocal act_calls
            act_calls += 1
            if act_calls != 1:return values
            if values.shape != replacement.shape:raise ValueError('wrong post-ReLU shape')
            return torch.where(node_mask[:,None],replacement,values)
        handles.append(bg.act.register_forward_hook(alter_nodes))
    if replace_stage == 'pre_meta':
        layer_u = model.layer_list[1]
        original = layer_u.forward
        had_instance = 'forward' in layer_u.__dict__
        def altered_u(*args,**kwargs):
            values=original(*args,**kwargs)
            if values.shape != replacement.shape:raise ValueError('wrong pre-metagraph shape')
            return torch.where((~q)[:,None],replacement,values)
        layer_u.forward=altered_u
        restore=(layer_u,original,had_instance)
    try:
        yield
        if replace_stage == 'post_relu' and act_calls != 2:
            raise ValueError('unexpected background ReLU call contract')
    finally:
        for handle in handles:handle.remove()
        if restore is not None:
            layer_u,original,had_instance=restore
            if had_instance:layer_u.forward=original
            else:del layer_u.forward


def forward_scaled(model,batch,**options):
    working=clone_batch(batch)
    layer=model.layer_list[0].module_list[0]
    extra,handles={},[]
    def save(key,value):extra[key]=value.detach().clone()
    def post_relu(module,args,value):
        if 'post_relu' not in extra:save('post_relu',value)
    with scale_control(model,working,**options):
        handles.append(layer.lin_x.register_forward_hook(lambda m,a,v:save('messages',v)))
        handles.append(layer.mlp.register_forward_pre_hook(lambda m,a:save('aggregated',a[0])))
        handles.append(layer.bn.register_forward_pre_hook(lambda m,a:save('pre_bn',a[0])))
        handles.append(layer.bn.register_forward_hook(lambda m,a,v:save('post_bn',v)))
        handles.append(model.layer_list[0].act.register_forward_hook(post_relu))
        try:
            with trace_stages(model,working[0]) as trace:
                _,logits,_=model(*working)
        finally:
            for handle in handles:handle.remove()
    if not torch.isfinite(logits).all() or any(not torch.isfinite(v).all() for v in extra.values()):
        raise ValueError('nonfinite forward')
    return logits,{**trace,**extra}


def stage_summary(values,original,zero,mask):
    """Additive occurrence-weighted summaries, not independent-node inference."""
    x,a,z=(v[mask].double() for v in (values,original,zero))
    if not len(x):return {'count':0}
    norms=x.norm(dim=1)
    an,zn=a.norm(dim=1),z.norm(dim=1)
    cos_a=(x*a).sum(1)/(norms*an).clamp_min(1e-300)
    cos_z=(x*z).sum(1)/(norms*zn).clamp_min(1e-300)
    return {'count':len(x),'norm_sum':float(norms.sum()),'norm_sq_sum':float(norms.square().sum()),
            'norm_min':float(norms.min()),'norm_max':float(norms.max()),'zero_vectors':int((norms==0).sum()),
            'cos_intact_sum':float(cos_a.sum()),'cos_zero_sum':float(cos_z.sum()),
            'l2_intact_sum':float((x-a).norm(dim=1).sum()),'l2_zero_sum':float((x-z).norm(dim=1).sum()),
            'intact_norm_sum':float(an.sum()),'zero_norm_sum':float(zn.sum()),
            'max_abs_intact':float((x-a).abs().max()),'max_abs_zero':float((x-z).abs().max()),
            'positive_gate_changes_from_intact':int(((x>0)!=(a>0)).sum()),'coordinates':x.numel()}


def summarize_trace(batch,trace,intact,zero):
    g=batch[0]; q=query_mask(batch); real=g.global_node_ids>=0
    degree=torch.bincount(g.edge_index[1],minlength=len(g.x))
    center=torch.zeros_like(real);center[g.ptr[:-1]]=True
    masks={'support_real':~q[g.batch]&real,'support_receives':~q[g.batch]&real&(degree>0),
           'support_no_incoming':~q[g.batch]&real&(degree==0),'support_center':~q[g.batch]&center,
           'query_real':q[g.batch]&real}
    rows=[]
    for stage in ('messages','aggregated','pre_bn','post_bn','post_relu','S0_pool','U1_pre_meta','M2_post_meta'):
        stage_masks=masks if stage in ('messages','aggregated','pre_bn','post_bn','post_relu') else {'support':~q,'query':q}
        for cohort,mask in stage_masks.items():
            rows.append({'stage':stage,'cohort':cohort,**stage_summary(trace[stage],intact[stage],zero[stage],mask)})
    return rows
