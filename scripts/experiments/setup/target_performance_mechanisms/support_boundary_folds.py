"""Label-hidden support folds for the frozen binary metagraph experiment."""
import torch


def hide_support_labels(batch, heldout):
    """Return an independent batch with nominated support rows acting as queries.

    All query labels (including original queries) become dummy class zero in the
    model input. Truth must be kept separately by the scorer. Every incident
    decoder edge gets query role and zero label relation. No graph is deleted.
    """
    n, ways = batch[2].shape
    if ways != 2 or batch[5].numel() != n * ways:
        raise ValueError("binary per-example decoder layout required")
    roles = batch[5].reshape(n, ways).bool()
    if not torch.equal(roles[:, 0], roles[:, 1]):
        raise ValueError("inconsistent query roles")
    heldout = torch.as_tensor(heldout, dtype=torch.long)
    if heldout.ndim != 1 or heldout.numel() == 0 or heldout.unique().numel() != heldout.numel():
        raise ValueError("nonempty unique held-out indices required")
    if (heldout < 0).any() or (heldout >= n).any() or roles[heldout, 0].any():
        raise ValueError("held-out rows must be existing supports")
    if batch[4].shape != (n * ways, 2):
        raise ValueError("two metagraph edge attributes required")
    expected = torch.arange(n).repeat_interleave(ways)
    if not torch.equal(batch[3][0].cpu(), expected):
        raise ValueError("decoder edges must be grouped by example")
    out = [value.clone() if value is not None else None for value in batch]
    query = roles[:, 0].clone()
    query[heldout] = True
    if query.all():
        raise ValueError("fold must retain supports")
    out[5] = query.repeat_interleave(ways).to(batch[5].dtype)
    out[4][:, 0] = out[5].to(out[4].dtype)
    out[4][out[5].bool(), 1] = 0
    out[2][query] = 0
    out[2][query, 0] = 1
    return tuple(out)


def balanced_support_folds(batch):
    """Hold one support per class per episode out in each fixed-order fold.

    Preserve balanced reference construction; every support is held out once.
    Target query labels are never used to define folds.
    """
    q = batch[5].reshape(-1, 2)[:, 0].bool()
    tasks = batch[0].task_id_per_sample
    groups = []
    for ep in tasks.unique(sorted=True):
        support = torch.where((tasks == ep) & ~q)[0]
        y = batch[2][support].argmax(1)
        classes = [support[y == c] for c in range(2)]
        if len(classes[0]) != len(classes[1]) or len(classes[0]) < 2:
            raise ValueError("balanced supports with at least two per class required")
        groups.append(torch.stack(classes, 1))
    if len({len(g) for g in groups}) != 1:
        raise ValueError("shared shot count required")
    folds = torch.stack(groups, 1).reshape(len(groups[0]), -1)
    if not torch.equal(folds.flatten().sort().values, torch.where(~q)[0]):
        raise ValueError("each support must be withheld exactly once")
    return folds
