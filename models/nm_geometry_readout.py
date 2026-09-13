"""Experimental, parameter-free NM readout retaining pre-metagraph geometry.

This is a bounded inference intervention, not a trained or calibrated head.
Only observed support signs define class evidence; query labels are not inputs.
"""
import torch
import torch.nn.functional as F


def mean_support_cosine(pre, edges, attrs, query_roles, n_way):
    """Mean cosine to positive supports for every input/candidate-label edge.

    Handles multiple disconnected episodes and unequal support counts. The
    input-row-major edge order is the same contract used by PRODIGY's decoder.
    """
    n = len(pre)
    if n_way < 2 or edges.shape != (2, n * n_way):
        raise ValueError('Multiclass row-major metagraph required')
    if attrs.shape != (n * n_way, 2) or query_roles.shape != (n * n_way,):
        raise ValueError('Expected two edge attributes and one role per edge')
    if not torch.equal(edges[0], torch.arange(n, device=pre.device).repeat_interleave(n_way)):
        raise ValueError('Input rows and decoder edges must align')
    roles = query_roles.bool().reshape(n, n_way)
    if not torch.equal(roles, roles[:, :1].expand_as(roles)):
        raise ValueError('A row cannot have mixed query/support roles')
    label = edges[1] - n
    if (label < 0).any():
        raise ValueError('Expected input-to-label edges')
    labels = int(label.max()) + 1
    positive = (~query_roles.bool()) & (attrs[:, -1] > 0)
    units = F.normalize(pre, dim=1)
    sums = pre.new_zeros(labels, pre.shape[1])
    counts = pre.new_zeros(labels)
    sums.index_add_(0, label[positive], units[edges[0, positive]])
    counts.index_add_(0, label[positive], torch.ones_like(label[positive], dtype=pre.dtype))
    if (counts == 0).any():
        raise ValueError('Every candidate label needs positive support')
    means = sums / counts[:, None]
    return (units[edges[0]] * means[label]).sum(1).reshape(n, n_way)


def standardize_scores(scores):
    """Remove query-specific offset and positive scale across candidate classes."""
    if scores.ndim != 2 or scores.shape[1] < 2 or not torch.isfinite(scores).all():
        raise ValueError('Finite multiclass score matrix required')
    centered = scores - scores.mean(1, keepdim=True)
    scale = centered.square().mean(1, keepdim=True).sqrt()
    return centered / scale.clamp_min(1e-8)


class GeometryResidualReadout(torch.nn.Module):
    """Equal mean of standardized native scores and mean support cosine.

    No mixing weight is fitted or swept. Output scores do not have a calibrated
    probability interpretation. Ties use the caller's ordinary argmax policy.
    """
    def forward(self, native_scores, geometry_scores):
        if native_scores.shape != geometry_scores.shape:
            raise ValueError('The two heads must score identical candidates')
        return (standardize_scores(native_scores) + standardize_scores(geometry_scores)) / 2
