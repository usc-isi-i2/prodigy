"""Balanced support-only cross-validation and a fixed readout selector."""
import torch
import torch.nn.functional as F

from .natural_support import meta_logits


def fold_pairs(y, tasks, episodes):
    pairs = []
    for ep in range(episodes):
        groups = [torch.where((tasks == ep) & (y == c))[0] for c in range(2)]
        if any(len(g) != 10 for g in groups):
            raise ValueError("ten supports per class required")
        pairs.append(torch.stack(groups, 1))
    result = torch.stack(pairs, 1)  # fold, episode, class
    if sorted(result.flatten().tolist()) != list(range(len(y))):
        raise ValueError("folds must hold each support out once")
    return result


def prototype_oof(x, y, tasks, pairs):
    """Normalize each node, average nine supports per class, predict the pair."""
    x = F.normalize(x, dim=1)
    episodes = pairs.shape[1]
    sums = x.new_zeros((2 * episodes, x.shape[1])).index_add_(0, 2 * tasks + y, x)
    prototypes = F.normalize(sums.reshape(episodes, 2, -1)[None] - x[pairs], dim=-1)
    heldout = x[pairs]
    z = torch.einsum("fecj,fekj->feck", heldout, prototypes)
    result = x.new_empty((len(x), 2))
    result[pairs.flatten()] = z.reshape(-1, 2)
    return result


def full_oof(model, pre_support, label_inputs, y, tasks, pairs):
    shots, episodes, _ = pairs.shape
    heldout = torch.zeros(shots, len(y), dtype=torch.bool)
    for fold in range(shots):
        heldout[fold, pairs[fold].flatten()] = True
    all_tasks = torch.cat([tasks + fold * episodes for fold in range(shots)])
    _, logits = meta_logits(model, pre_support.repeat(shots, 1), label_inputs.repeat(shots, 1),
        y.repeat(shots), heldout.flatten(), all_tasks)
    rows = torch.arange(len(y)).repeat(shots)[heldout.flatten()]
    result = logits.new_empty((len(y), 2))
    result[rows] = logits[heldout.flatten()]
    return result


def support_scores(oof, y, tasks, episodes):
    margins = (oof[:, 1] - oof[:, 0]).double()
    aucs, scales = [], []
    for ep in range(episodes):
        mask = tasks == ep
        positive, negative = margins[mask & (y == 1)], margins[mask & (y == 0)]
        comparison = positive[:, None] - negative[None]
        aucs.append(((comparison > 0).double() + .5 * (comparison == 0).double()).mean())
        rms = margins[mask].square().mean().sqrt()
        scales.append(rms if rms > 0 else rms.new_tensor(1.))
    return torch.stack(aucs), torch.stack(scales)


def select_predictions(candidates, oof, y, support_tasks, query_tasks):
    """Ordered dictionaries define the outcome-independent tie preference."""
    if list(candidates) != list(oof):
        raise ValueError("candidate order differs")
    episodes = int(support_tasks.max()) + 1
    results = [support_scores(z, y, support_tasks, episodes) for z in oof.values()]
    aucs = torch.stack([r[0] for r in results])
    scales = torch.stack([r[1] for r in results])
    if not torch.isfinite(aucs).all() or not torch.isfinite(scales).all():
        raise ValueError("nonfinite support score")
    choice = aucs.argmax(0)
    original = torch.stack(list(candidates.values()))
    margin = (original[:, :, 1] - original[:, :, 0]).double() / scales[:, query_tasks]
    normalized = torch.stack((torch.zeros_like(margin), margin), dim=-1)
    rows = torch.arange(len(query_tasks))
    return {"choice": choice, "cv_auc": aucs, "cv_rms": scales,
        "fixed_original": original, "fixed_scaled": normalized,
        "selected_original": original[choice[query_tasks], rows],
        "selected_scaled": normalized[choice[query_tasks], rows]}
