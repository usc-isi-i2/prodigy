"""Natural context draws at fixed support identities, with immutable query inputs."""
import hashlib
import json

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch

from .analyze_mixture_predictions import evaluate_logits, ensemble_scores
from .replay import batch_hash
from .role_context import query_mask


def draw_seed(stream, target, batch, draw):
    if stream not in ("original", "fresh") or batch < 0 or draw < 1:
        raise ValueError("invalid fresh context draw")
    return 96060000 + (10000000 if stream == "fresh" else 0) + sum((i+1)*ord(c) for i, c in enumerate(target))*1000 + batch*10 + draw


def graph_hash(graph):
    return batch_hash([graph])


def support_draw(dataset, batch, seed):
    # No labels, query features or query IDs participate in the sampling decision.
    ids = batch[0].global_node_ids[batch[0].ptr[:-1]][~query_mask(batch)]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return Batch.from_data_list([dataset[int(center)] for center in ids])


def replace_support_contexts(batch, supports):
    q = query_mask(batch)
    original = batch[0]
    graphs = original.to_data_list()
    indices = torch.where(~q)[0].tolist()
    replacements = supports.to_data_list()
    if len(indices) != len(replacements):
        raise ValueError("support count changed")
    for i, graph in zip(indices, replacements):
        graphs[i] = graph.clone()
    out = [Batch.from_data_list(graphs), *[x.clone() for x in batch[1:]]]
    for k, v in original:
        if k not in original._slice_dict and k not in ("batch", "ptr"):
            out[0][k] = v.clone() if isinstance(v, torch.Tensor) else v
    return out


def verify_replacement(batch, changed):
    q = query_mask(batch)
    for a, b in zip(batch[1:], changed[1:]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    a, b = batch[0], changed[0]
    torch.testing.assert_close(a.global_node_ids[a.ptr[:-1]], b.global_node_ids[b.ptr[:-1]], rtol=0, atol=0)
    torch.testing.assert_close(a.x[a.ptr[:-1]], b.x[b.ptr[:-1]], rtol=0, atol=0)
    left, right = a.to_data_list(), b.to_data_list()
    for i in torch.where(q)[0].tolist():
        if graph_hash(left[i]) != graph_hash(right[i]):
            raise ValueError("query graph changed")
    return {"query_graphs_bit_exact": True, "support_centers_and_center_features_bit_exact": True,
        "all_metagraph_and_truth_tensors_bit_exact": True}


def context_comparison(original, changed):
    rows = []
    for a, b in zip(original.to_data_list(), changed.to_data_list()):
        if int(a.global_node_ids[0]) != int(b.global_node_ids[0]):
            raise ValueError("center changed")
        ai, bi = set(a.global_node_ids[1:-1].tolist()), set(b.global_node_ids[1:-1].tolist())
        union = ai | bi
        rows.append({"center": int(a.global_node_ids[0]), "original_context_nodes": len(ai),
            "draw_context_nodes": len(bi), "context_jaccard": len(ai & bi)/len(union) if union else 1.,
            "node_set_changed": ai != bi, "graph_tensors_changed": graph_hash(a) != graph_hash(b)})
    return rows


def seal_plan(plan):
    return hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()


def score_context_draws(predictions, labels, context, common):
    if len(predictions) not in (2, 8) or predictions.shape[1:] != (len(labels["local_y"]), 2):
        raise ValueError("invalid complete draw tensor")
    rows = [{**common, "condition": "draw", "draw": d, **evaluate_logits(z, labels)}
        for d, z in enumerate(predictions)]
    rows.append({**common, "condition": "probability_ensemble", "draw": -1,
        **ensemble_scores(predictions, labels)["probability"][0]})
    truth, ep = labels["local_y"], labels["episode_ids"]
    loss = F.cross_entropy(predictions.double().reshape(-1, 2), truth.repeat(len(predictions)), reduction="none").reshape(len(predictions), -1)
    correct = predictions.argmax(2).eq(truth[None])
    probs = predictions.softmax(2).double()[:, :, 1]
    episodes, cohorts = [], []
    for e in ep.unique(sorted=True).tolist():
        mask = ep.eq(e)
        for draw in range(len(predictions)):
            episodes.append({**common, "episode": e, "draw": draw, "queries": int(mask.sum()),
                "query_nll": float(loss[draw, mask].mean()), "query_correct": int(correct[draw, mask].sum()),
                "prediction_changes_vs_original": int((predictions[draw, mask].argmax(1) != predictions[0, mask].argmax(1)).sum())})
    for name, mask in (("all", torch.ones_like(context, dtype=torch.bool)), ("no_context", context.eq(0)), ("has_context", context.gt(0))):
        if not mask.any():
            continue
        hits = correct[:, mask].sum(0)
        cohorts.append({**common, "cohort": name, "queries": int(mask.sum()),
            "mean_probability_variance": float(probs[:, mask].var(0, unbiased=False).mean()),
            "correctness_flips": int(((hits > 0) & (hits < len(predictions))).sum()),
            "always_correct": int(hits.eq(len(predictions)).sum()), "always_wrong": int(hits.eq(0).sum())})
    return rows, episodes, cohorts
