"""Natural support substitution and query-blind support cross-validation."""
import hashlib
import json
import random

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch

from models.metaGNN import MetaGNN
from .role_context import query_mask


def frozen_contract(model):
    if (model.training or any(m.training for m in model.modules()) or len(model.layer_list) != 3
            or not isinstance(model.layer_list[2], MetaGNN) or model.layer_list[2].num_gnn_layers != 1
            or model.layer_list[2].gnn_layers_back is not None):
        raise ValueError("frozen single-layer non-reverse metagraph required")
    if any(model.params[k] for k in ("skip_path", "zero_shot", "ignore_label_embeddings", "zero_label_embeddings")):
        raise ValueError("unsupported model recipe")
    if not isinstance(model.final_input_mlp, torch.nn.Identity) or not isinstance(model.final_label_mlp, torch.nn.Identity):
        raise ValueError("identity final transforms required")
    if any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) and not m.track_running_stats for m in model.modules()):
        raise ValueError("transductive batch statistics invalidate independent subgraphs")


def meta_inputs(labels, query, tasks, label_count):
    """Query labels never enter edge attributes; only support labels are used."""
    n = len(labels)
    if labels.shape != query.shape or tasks.shape != labels.shape or label_count != 2 * (int(tasks.max()) + 1):
        raise ValueError("complete binary episode geometry required")
    edges = torch.stack((torch.arange(n).repeat_interleave(2),
                         n + 2 * tasks.repeat_interleave(2) + torch.arange(2).repeat(n)))
    relation = F.one_hot(labels, 2).float() * 2 - 1
    relation[query] = 0
    mask = query.repeat_interleave(2)
    attrs = torch.stack((mask.float(), relation.flatten()), 1)
    return edges, attrs, mask


def meta_logits(model, pre, label_inputs, labels, query, tasks):
    edges, attrs, mask = meta_inputs(labels, query, tasks, len(label_inputs))
    x, z = model.forward_metagraph(model.layer_list[2], pre, model.initial_label_mlp(label_inputs),
                                   edges, attrs, mask, None, None, None)
    return x, model.decode(x, z, edges).reshape(-1, 2)


def support_cv_loss(model, pre, label_inputs, labels, query, tasks):
    """Ten balanced leave-one-pair-out folds, using no target query vectors."""
    support = (~query).nonzero().flatten()
    y, task = labels[support], tasks[support]
    episodes = len(label_inputs) // 2
    pairs = []
    for ep in range(episodes):
        groups = [(task.eq(ep) & y.eq(c)).nonzero().flatten() for c in range(2)]
        if len(groups[0]) != len(groups[1]) or len(groups[0]) < 2:
            raise ValueError("balanced supports with at least two shots required")
        pairs.append(torch.stack(groups, 1))
    shots = len(pairs[0])
    if any(len(p) != shots for p in pairs):
        raise ValueError("shared shot count required")
    heldout = torch.zeros(shots, len(support), dtype=torch.bool)
    for fold in range(shots):
        heldout[fold, torch.cat([p[fold] for p in pairs])] = True
    all_tasks = torch.cat([task + fold * episodes for fold in range(shots)])
    all_labels = y.repeat(shots)
    _, predictions = meta_logits(model, pre[support].repeat(shots, 1), label_inputs.repeat(shots, 1),
                                  all_labels, heldout.flatten(), all_tasks)
    loss = F.cross_entropy(predictions[heldout.flatten()], all_labels[heldout.flatten()], reduction="none")
    ep = all_tasks[heldout.flatten()] % episodes
    total = torch.zeros(episodes).scatter_add_(0, ep, loss)
    count = torch.bincount(ep, minlength=episodes)
    if not count.eq(2 * shots).all():
        raise ValueError("every support must be held out exactly once")
    return total / count


def support_plan(batches, draws, seed):
    """Uniform unique identities then occurrences; exclude recipient queries."""
    if draws < 1:
        raise ValueError("positive number of draws required")
    rng = random.Random(seed)
    pool, specs, offset, episode_offset = [], [], 0, 0
    identity_labels = {}
    for bi, batch in enumerate(batches):
        g, q = batch[0], query_mask(batch)
        tasks = g.task_id_per_sample.long()
        centers = g.global_node_ids[g.ptr[:-1]].long()
        y = batch[2].argmax(1).long()
        classes = g.task_label_map.long()
        if classes.shape[1] != 2 or (centers < 0).any():
            raise ValueError("binary tasks and valid center identities required")
        for row in (~q).nonzero().flatten().tolist():
            label = int(classes[tasks[row], y[row]])
            center = int(centers[row])
            if center in identity_labels and identity_labels[center] != label:
                raise ValueError("support identity has conflicting dataset labels")
            identity_labels[center] = label
            pool.append(dict(flat=offset + row, batch=bi, row=row, episode=episode_offset + int(tasks[row]),
                             center=center, global_label=label))
        specs.append(dict(offset=offset, episode_offset=episode_offset, tasks=tasks, centers=centers,
                          query=q, support_labels=y.masked_fill(q, -1), classes=classes))
        offset += len(q)
        episode_offset += len(classes)
    mappings, audits = [], []
    for bi, spec in enumerate(specs):
        mapping = torch.arange(spec["offset"], spec["offset"] + len(spec["query"])).repeat(draws, 1)
        for ep in range(len(spec["classes"])):
            is_episode = spec["tasks"].eq(ep)
            forbidden = set(spec["centers"][is_episode & spec["query"]].tolist())
            origin = spec["episode_offset"] + ep
            for local_class, global_class in enumerate(spec["classes"][ep].tolist()):
                slots = (is_episode & ~spec["query"] & spec["support_labels"].eq(local_class)).nonzero().flatten()
                eligible = {}
                for record in pool:
                    if record["global_label"] == global_class and record["episode"] != origin and record["center"] not in forbidden:
                        eligible.setdefault(record["center"], []).append(record)
                if len(eligible) < len(slots):
                    raise ValueError(f"insufficient unique eligible supports: batch={bi} episode={ep} class={global_class}")
                original_ids = set(spec["centers"][slots].tolist())
                for draw in range(draws):
                    chosen_ids = rng.sample(sorted(eligible), len(slots))
                    chosen = [rng.choice(eligible[identity]) for identity in chosen_ids]
                    mapping[draw, slots] = torch.tensor([r["flat"] for r in chosen])
                    audits.append(dict(batch=bi, episode=origin, draw=draw, global_class=global_class,
                                       shots=len(slots), eligible_unique_ids=len(eligible),
                                       overlap_with_original=len(original_ids.intersection(chosen_ids)),
                                       selected=chosen))
        mappings.append(mapping)
    packed = {"seed": seed, "draws": draws, "mappings": [m.tolist() for m in mappings], "audits": audits,
              "pool_unique_identities": len(identity_labels), "pool_occurrences": len(pool)}
    packed["sha256"] = hashlib.sha256(json.dumps(packed, sort_keys=True).encode()).hexdigest()
    return mappings, packed


def replace_graphs(batch, source_graphs, mapping):
    out = list(batch)
    original = batch[0]
    out[0] = Batch.from_data_list([source_graphs[int(index)].clone() for index in mapping])
    out[0].task_id_per_sample = original.task_id_per_sample.clone()
    out[0].task_label_map = original.task_label_map.clone()
    return out
