"""Cache new neighborhoods for exact existing episode identities; no model calls."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch, Data

from data.dataset import SubgraphDataset
from experiments.sampler import NeighborSampler
from .replay import batch_hash
from .probe_cached_inputs import standardized_probe
from scripts.experiments.setup.icl_arch_matrix.common_protocol import (
    new_fingerprint, update_episode_fingerprint, iter_episodes)

TASK_FIELDS = {"task_id_per_sample", "task_id_per_query", "source_id_per_task", "task_label_map"}


def rebuild_batch(old, dataset):
    g = old[0]
    ids = g.global_node_ids[g.ptr[:-1]]
    new = Batch.from_data_list([dataset[int(i)] for i in ids])
    extras = {k for k, _ in g} - {k for k, _ in new}
    if extras != TASK_FIELDS:
        raise ValueError(f"Unexpected batch metadata: {extras}")
    for key in extras:
        new[key] = g[key].clone()
    result = [new] + [v.clone() for v in old[1:]]
    if not torch.equal(new.global_node_ids[new.ptr[:-1]], ids):
        raise ValueError("Episode centers/order changed")
    for key in ("x", "y"):
        if not torch.equal(new[key][new.ptr[:-1]], g[key][g.ptr[:-1]]):
            raise ValueError("Episode center features/labels changed")
    for a, b in zip(old[1:], result[1:]):
        if not torch.equal(a, b):
            raise ValueError("Support/query labels or task tensors changed")
    return result


def describe(batch):
    from sklearn.metrics import roc_auc_score
    g = batch[0]
    centers = g.ptr[:-1]
    degree = torch.bincount(g.edge_index[1], minlength=len(g.x))[centers]
    outdegree = torch.bincount(g.edge_index[0], minlength=len(g.x))[centers]
    query = batch[5].reshape(-1, 2)[:, 0].bool()
    logits = standardized_probe(degree.float().log1p()[:, None], batch)
    local_y = batch[2][query].argmax(1)
    tasks = g.task_id_per_sample[query]
    mapping = g.task_label_map[tasks]
    p = logits.softmax(1)
    semantic_y = mapping[torch.arange(len(local_y)), local_y]
    semantic_p = p[torch.arange(len(local_y)), (mapping == 1).long().argmax(1)]
    episode_aucs = [roc_auc_score(local_y[tasks == ep].numpy(), p[tasks == ep, 1].numpy())
                    for ep in tasks.unique(sorted=True)]
    cov = []
    for role, mask in (("support", ~query), ("query", query)):
        for label in (0, 1):
            selected = mask & (g.y[centers] == label)
            cov.append(dict(role=role, label=label, count=int(selected.sum()),
                isolated=int(((degree+outdegree == 0) & selected).sum()),
                incoming_zero=int(((degree == 0) & selected).sum()),
                context_nodes=(g.ptr[1:]-g.ptr[:-1]-2)[selected].tolist(),
                indegrees=degree[selected].tolist()))
    return cov, episode_aucs, semantic_y, semantic_p


def main():
    import yaml
    from sklearn.metrics import roc_auc_score
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--graph", type=Path, required=True)
    p.add_argument("--reference-root", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists() or torch.cuda.is_available():
        raise ValueError("New output and hidden GPUs required")
    torch.set_num_threads(2)
    config = yaml.safe_load(args.config.read_text())
    if (config["n_hop"], str(config["neighbor_sampling_hop_sizes"]),
        config["neighbor_sampling_node_limit"]) != (2,"9,9",101):
        raise ValueError("Nominated sampler differs from recorded production config")
    payload = torch.load(args.graph, map_location="cpu", weights_only=False)
    if payload["matched_relation_provenance"]["same_node_order_features_labels_splits"] is not True:
        raise ValueError("Unverified matched graph")
    g = Data(x=payload["x"], y=payload["y"], edge_index=payload["edge_index"], edge_attr=payload["edge_attr"])
    dataset = SubgraphDataset(g, NeighborSampler(g, num_hops=2, hop_sizes=[9,9], limit=101), bidirectional=False)
    args.output.mkdir(parents=True)
    summaries, receipts = [], []
    for stream, offset in (("original",0),("fresh",100003)):
        reference = args.reference_root/stream/"twibot20"
        cache = json.loads((reference/"cache.json").read_text())
        if cache["episodes"] != 128 or len(cache["batch_sha256"]) != 32:
            raise ValueError("Complete reference stream required")
        out = args.output/stream/"twibot20"
        (out/"batches").mkdir(parents=True)
        hashes, stats = [], {name:dict(coverage=[], aucs=[], y=[], p=[]) for name in ("retweet","follow")}
        episode_fingerprint = new_fingerprint()
        for bi, fingerprint in enumerate(cache["batch_sha256"]):
            old = torch.load(reference/"batches"/f"batch_{bi:03d}.pt", map_location="cpu", weights_only=False)
            if batch_hash(old) != fingerprint:
                raise ValueError("Original cached input changed")
            torch.manual_seed(710000+offset+bi)
            new = rebuild_batch(old, dataset)
            real = new[0].global_node_ids >= 0
            if not torch.equal(new[0].x[real], payload["x"][new[0].global_node_ids[real]]):
                raise ValueError("Sampled features differ from exact graph rows")
            hashes.append(batch_hash(new))
            for episode in iter_episodes(new, n_way=2, n_shot=10, n_query=12, equal_query_counts=True):
                update_episode_fingerprint(episode_fingerprint, episode)
            torch.save(new, out/"batches"/f"batch_{bi:03d}.pt")
            for name, batch in (("retweet", old),("follow",new)):
                cov, aucs, y, prob = describe(batch)
                stats[name]["coverage"].extend(cov); stats[name]["aucs"].extend(aucs)
                stats[name]["y"].append(y); stats[name]["p"].append(prob)
        # The existing fingerprint includes neighborhood identities and edges too.
        # Preserve its provenance separately; never claim unchanged full inputs.
        new_cache = dict(cache, episode_fingerprint=episode_fingerprint.hexdigest(),
            reference_episode_fingerprint=cache["episode_fingerprint"],
            batch_sha256=hashes, graph_path=str(args.graph),
            reference_cache=str(reference/"cache.json"), matched_center_tasks=True,
            sampling_seed_base=710000+offset)
        (out/"cache.json").write_text(json.dumps(new_cache,indent=2)+"\n")
        for name, values in stats.items():
            row=dict(stream=stream, graph=name, degree_cue_mean_episode_auc=float(np.mean(values["aucs"])),
                degree_cue_pooled_auc=float(roc_auc_score(torch.cat(values["y"]),torch.cat(values["p"]))),
                coverage=[])
            for role in ("support","query"):
                for label in (0,1):
                    selected=[r for r in values["coverage"] if r["role"]==role and r["label"]==label]
                    nodes=[v for r in selected for v in r["context_nodes"]]
                    degrees=[v for r in selected for v in r["indegrees"]]
                    row["coverage"].append(dict(role=role,label=label,
                        **{k:sum(r[k] for r in selected) for k in ("count","isolated","incoming_zero")},
                        mean_context_nodes=float(np.mean(nodes)), mean_incoming=float(np.mean(degrees))))
            summaries.append(row)
        receipts.append(dict(stream=stream, batches=32, exact_centers_features_labels_task_tensors=True,
                             reference_hashes_verified=True))
        print(json.dumps({"stream":stream,"cached_batches":32}),flush=True)
    (args.output/"coverage.json").write_text(json.dumps(summaries,indent=2)+"\n")
    (args.output/"DONE.json").write_text(json.dumps(dict(receipts=receipts,new_model_forwards=0,
        sampler={"hops":2,"fanouts":[9,9],"limit":101}, training=False),indent=2)+"\n")


if __name__ == "__main__": main()
