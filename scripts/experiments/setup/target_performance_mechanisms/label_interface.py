"""Frozen-weight label-interface diagnostics; no target-label fitting."""
from contextlib import contextmanager

import torch


LABEL_VARIANTS = {"train_label_table", "permuted_train_label_table",
                  "train_norm_label_text", "zero_projected_labels"}
PERMUTATION_SEED = 839113


@contextmanager
def label_interface_mode(model, variant, training_label_count=120):
    audit = {}
    if variant not in LABEL_VARIANTS:
        yield audit
        return
    table = model.learned_label_embedding.weight.detach()
    if not 1 <= training_label_count <= len(table):
        raise ValueError("training label count outside frozen table")
    old_ignore = model.params["ignore_label_embeddings"]
    use_table = variant in {"train_label_table", "permuted_train_label_table"}
    target_norm = table[:training_label_count].norm(dim=1).mean()
    permutation = torch.randperm(training_label_count, generator=torch.Generator().manual_seed(PERMUTATION_SEED)).to(table.device)
    audit.update(variant=variant, training_label_count=training_label_count,
                 training_table_mean_norm=float(target_norm), original_ignore_label_embeddings=old_ignore,
                 permutation_seed=PERMUTATION_SEED if variant == "permuted_train_label_table" else None,
                 forward_calls=0, last_projected_mean_norm=None, last_interface_mean_norm=None)
    handles = []

    def project_hook(module, args, out):
        if not 1 <= len(out) <= training_label_count:
            raise ValueError("target label-node count exceeds declared training range")
        before = out.norm(dim=1).mean()
        audit["forward_calls"] += 1
        audit["last_projected_mean_norm"] = float(before)
        if variant == "train_norm_label_text":
            if not torch.isfinite(before) or before <= 0:
                raise ValueError("invalid projected-label magnitude")
            out = out * (target_norm / before)
        elif variant == "zero_projected_labels":
            out = torch.zeros_like(out)
        if not use_table:
            audit["last_interface_mean_norm"] = float(out.norm(dim=1).mean())
        return out

    def table_hook(module, args, out):
        index = args[0]
        if torch.any(index < 0) or torch.any(index >= training_label_count):
            raise ValueError("label index outside declared training range")
        if variant == "permuted_train_label_table":
            out = table[permutation[index]]
        audit["last_interface_mean_norm"] = float(out.norm(dim=1).mean())
        return out

    try:
        model.params["ignore_label_embeddings"] = use_table
        handles.append(model.initial_label_mlp.register_forward_hook(project_hook))
        if use_table:
            handles.append(model.learned_label_embedding.register_forward_hook(table_hook))
        yield audit
    finally:
        model.params["ignore_label_embeddings"] = old_ignore
        for handle in handles:
            handle.remove()
