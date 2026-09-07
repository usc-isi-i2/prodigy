"""Support-only, mass-matched identity-label interventions and gradient probes."""
from collections import defaultdict
from contextlib import contextmanager

import torch
import torch.nn.functional as F

from .replay import batch_hash, clone_batch
from .verify_member_training import model_digest


VARIANTS = ("baseline", "identity_soft", "permuted_soft")
MODES = ("training", "meta_frozen")


def support_plan(batch, seed):
    g, nway = batch[0], batch[2].shape[1]
    ids = g.global_node_ids[g.ptr[:-1]].long()
    query = batch[5].reshape(-1, nway)[:, 0].bool()
    attrs = batch[4].reshape(-1, nway, 2)
    tasks = g.task_id_per_sample
    if len(ids) != len(query) or len(tasks) != len(ids):
        raise ValueError("invalid episode metadata")
    if not torch.equal(attrs[:, :, 0], query[:, None].expand(-1, nway).float()) or attrs[query, :, 1].abs().any():
        raise ValueError("invalid query edge attributes")
    classes = attrs[:, :, 1].argmax(1)
    expected = F.one_hot(classes[~query], nway).float()*2-1
    if not torch.equal(attrs[~query, :, 1], expected):
        raise ValueError("supports must start with hard single-class relations")
    rng = torch.Generator().manual_seed(seed)
    groups, null_groups, permutations = [], [], []
    for task in tasks.unique(sorted=True):
        support = torch.where((tasks == task) & ~query)[0]
        mapping = torch.arange(len(ids))
        for cls in range(nway):
            positions = support[classes[support] == cls]
            if len(positions) != 3:
                raise ValueError("requires exactly three supports per pseudo-class")
            shift = int(torch.randint(1, 3, (1,), generator=rng))
            other = positions.roll(shift)
            mapping[positions] = other
            permutations.append((positions, other))
        by_id = defaultdict(list)
        for i in support.tolist():
            by_id[int(ids[i])].append(i)
        for positions in by_id.values():
            if len(positions) > 1:
                group = torch.tensor(positions)
                if len(classes[group].unique()) != len(group):
                    raise ValueError("within-class repeated support center")
                groups.append(group)
                null_groups.append(mapping[group])
    return {"groups": groups, "null_groups": null_groups, "permutations": permutations,
            "ids": ids, "query": query, "tasks": tasks, "classes": classes, "nway": nway, "seed": seed}


def label_intervention(batch, plan, variant):
    if variant not in VARIANTS:
        raise ValueError(variant)
    out = clone_batch(batch)
    if variant == "baseline":
        return out
    attrs = out[4].reshape(-1, plan["nway"], 2)
    for group in plan["groups"]:
        attrs[group] = attrs[group].mean(0, keepdim=True)
    if variant == "permuted_soft":
        original = attrs.clone()
        for positions, other in plan["permutations"]:
            attrs[other] = original[positions]
    return out


def verify_interventions(batch, plan):
    actual, null = [label_intervention(batch, plan, v) for v in VARIANTS[1:]]
    q = plan["query"]
    attrs, a, n = [b[4].reshape(-1, plan["nway"], 2) for b in (batch, actual, null)]
    for changed in (actual, null):
        # Full hashes with the sole allowed field restored establish that no
        # other tensor, including any query input/truth, was changed.
        saved = changed[4]
        changed[4] = batch[4]
        if batch_hash(changed) != batch_hash(batch):
            raise ValueError("intervention changed another input")
        changed[4] = saved
        torch.testing.assert_close(saved.reshape_as(attrs)[q], attrs[q], rtol=0, atol=0)
    errors = []
    for task in plan["tasks"].unique():
        indices = torch.where((plan["tasks"] == task) & ~q)[0]
        for changed in (a, n):
            error = float((changed[indices].double().sum(0)-attrs[indices].double().sum(0)).abs().max())
            if error > 1e-5:
                raise ValueError("label mass not conserved")
            errors.append(error)
    for positions, _ in plan["permutations"]:
        # Per-class row-vector multiset, not just a global scalar, is matched.
        rows_a = sorted(map(tuple, a[positions].reshape(3, -1).tolist()))
        rows_n = sorted(map(tuple, n[positions].reshape(3, -1).tolist()))
        if rows_a != rows_n:
            raise ValueError("null does not match per-class label vectors")
    size_a = (a.double()-attrs.double()).square().sum()
    size_n = (n.double()-attrs.double()).square().sum()
    torch.testing.assert_close(size_a, size_n, rtol=1e-14, atol=1e-12)
    altered_truth = clone_batch(batch)
    altered_truth[2][q] = altered_truth[2][q].roll(1, dims=1)
    other_plan = support_plan(altered_truth, plan["seed"])
    for kind in ("groups", "null_groups"):
        if [v.tolist() for v in other_plan[kind]] != [v.tolist() for v in plan[kind]]:
            raise ValueError("query truth influenced identity grouping")
    for variant, expected in zip(VARIANTS[1:], (actual, null)):
        changed = label_intervention(altered_truth, other_plan, variant)
        torch.testing.assert_close(changed[4], expected[4], rtol=0, atol=0)
    null_coherent = sum(len(g) for g in plan["null_groups"] if len(plan["ids"][g].unique()) == 1)
    return {"groups": len(plan["groups"]), "affected_support_positions": sum(map(len, plan["groups"])),
        "total_support_positions": int((~q).sum()), "null_fully_identity_coherent_positions": null_coherent,
        "maximum_label_mass_roundoff": max(errors, default=0), "squared_perturbation": float(size_a),
        "null_squared_perturbation": float(size_n), "null_matches_every_class_vector_multiset": True,
        "query_truth_blind": True, "only_support_label_edge_values_changed": True}


@contextmanager
def capture_gradients(model):
    layer = model.layer_list[1]
    old = layer.forward
    had_instance = "forward" in layer.__dict__
    captured = {}
    def forward(*args, **kwargs):
        out = old(*args, **kwargs)
        out.retain_grad()
        captured["pre"] = out
        return out
    layer.forward = forward
    handle = model.layer_list[2].register_forward_hook(lambda m, a, out: captured.update(post=out))
    try:
        yield captured
    finally:
        handle.remove()
        if had_instance:
            layer.forward = old
        else:
            del layer.forward


def gradient_probe(trainer, state, batch, mode, rng_state):
    if mode not in MODES or not torch.are_deterministic_algorithms_enabled():
        raise ValueError("unknown mode or nondeterministic gradient execution")
    model = trainer.model
    model.load_state_dict(state, strict=True)
    model.train()
    if mode == "meta_frozen":
        for module in model.layer_list[2].modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()
    trainer._restore_rng_state(rng_state)
    model.zero_grad(set_to_none=True)
    with capture_gradients(model) as trace:
        truth, logits, graph = model(*clone_batch(batch))
        loss, _ = trainer.get_loss_and_acc(truth, logits)
        aux = trainer.get_aux_loss(graph)*trainer.parameter["attr_regression_weight"]
        if float(aux.detach() if isinstance(aux, torch.Tensor) else aux) != 0:
            raise ValueError("unexpected auxiliary training loss")
        loss.backward()
        tensors = {"logits": logits.detach().clone(), "truth": truth.detach().clone(),
            "pre": trace["pre"].detach().clone(), "pre_gradient": trace["pre"].grad.detach().clone(),
            "post": trace["post"].detach().clone(),
            "gradients": {k: p.grad.detach().clone() for k, p in model.named_parameters() if p.grad is not None},
            "buffers": {k: v.detach().clone() for k, v in model.named_buffers()}}
    if any(not torch.isfinite(v).all() for v in [tensors["logits"], tensors["pre_gradient"], *tensors["gradients"].values()]):
        raise ValueError("nonfinite gradient probe")
    return tensors


def tensor_receipt(tensors):
    return {key: model_digest(value if isinstance(value, dict) else {key: value}) for key, value in tensors.items()}


def gradient_comparison(reference, changed):
    if reference.keys() != changed.keys():
        raise ValueError("gradient inventory differs")
    groups = {
        "all": list(reference),
        "encoder": [k for k in reference if k.startswith("layer_list.0.module_list")],
        "readout": [k for k in reference if k.startswith("layer_list.0.reset")],
        "metagraph": [k for k in reference if k.startswith("layer_list.2")],
        "label_input": [k for k in reference if k.startswith("initial_label_mlp")],
        "logit_scale": [k for k in reference if k == "logit_scale"]}
    results = {}
    for name, keys in groups.items():
        a = torch.cat([reference[k].double().reshape(-1) for k in keys]) if keys else torch.empty(0, dtype=torch.float64)
        b = torch.cat([changed[k].double().reshape(-1) for k in keys]) if keys else torch.empty(0, dtype=torch.float64)
        na, nb = float(a.norm()), float(b.norm())
        results[name] = {"active_parameter_tensors": len(keys), "active_parameter_elements": a.numel(),
            "baseline_norm": na, "changed_norm": nb,
            "difference_norm": float((a-b).norm()),
            "relative_difference": float((a-b).norm())/na if na else None,
            "cosine": float(a.dot(b))/(na*nb) if na*nb else None}
    return results


def cancellation(gradients, groups):
    result = []
    for group in groups:
        g = gradients[group].double()
        norms = g.norm(dim=1)
        ratio = float(g.sum(0).norm()/norms.sum()) if float(norms.sum()) else None
        valid = norms > 0
        v = F.normalize(g[valid], dim=1)
        pair = (v@v.T)[torch.triu(torch.ones((len(v), len(v)), dtype=torch.bool), diagonal=1)]
        result.append({"positions": group.tolist(), "members": len(group), "shared_gradient_ratio": ratio,
            "pair_cosine_mean": float(pair.mean()) if len(pair) else None})
    return result


def probe_summary(tensors, baseline, plan):
    labels = tensors["truth"].argmax(1)
    loss = F.cross_entropy(tensors["logits"], labels, reduction="none")
    correct = tensors["logits"].argmax(1) == labels
    q = plan["query"]
    qconflict = torch.zeros_like(q)
    for task in plan["tasks"].unique():
        support_ids = plan["ids"][(plan["tasks"] == task) & ~q]
        qconflict |= (plan["tasks"] == task) & q & torch.isin(plan["ids"], support_ids)
    masks = {"all": torch.ones(len(loss), dtype=torch.bool), "query_support_conflict": qconflict[q],
             "no_query_support_conflict": ~qconflict[q]}
    cohorts = {k: {"queries": int(m.sum()), "nll": float(loss[m].mean()) if m.any() else None,
        "accuracy": float(correct[m].float().mean()) if m.any() else None} for k, m in masks.items()}
    pre_equal = torch.equal(tensors["pre"], baseline["pre"])
    if not pre_equal:
        raise ValueError("support-label mutation changed pre-metagraph embeddings")
    n = len(q)
    difference = tensors["post"]-baseline["post"]
    return {"cohorts": cohorts, "pre_bit_exact": pre_equal,
        "post_query_max_difference": float(difference[:n][q].abs().max()),
        "post_label_max_difference": float(difference[n:].abs().max()),
        "gradient_comparison": gradient_comparison(baseline["gradients"], tensors["gradients"]),
        "identity_group_gradients": cancellation(tensors["pre_gradient"], plan["groups"]),
        "null_group_gradients": cancellation(tensors["pre_gradient"], plan["null_groups"]),
        "tensor_digests": tensor_receipt(tensors)}
