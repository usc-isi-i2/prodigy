"""Audit the exact training sampler distribution, not recovered historical draws.

Counterfactual member policies reuse each successful anchor's very same random
walk endpoints. This separates sorted retention from support/query role order.
Only the stored static_train relation is used; no models are trained here.
"""
import argparse
import hashlib
import inspect
import json
import random
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data.covid19_twitter import _build_covid19_twitter_graph
from data.dataloader import NeighborTask
from data.neighbor_matching_split import ensure_static_views
from experiments.params import get_params
from experiments.sampler import NeighborSampler


class RecordedWalk:
    def __init__(self, sampler):
        self.sampler = sampler
        self.whole_adj = sampler.whole_adj
        self.endpoints = {}
        self.calls = 0

    def random_walk(self, node_idx, direction):
        result = self.sampler.random_walk(node_idx, direction)
        self.endpoints[int(node_idx[0])] = result.clone()
        self.calls += 1
        return result


def member_variants(task, endpoints, rng):
    baseline, role_shuffle, uniform = [], [], []
    for anchor, members in task.items():
        unique = torch.unique(endpoints[anchor]).tolist()
        if members != unique[:7]:
            raise RuntimeError("production member selector no longer matches audited policy")
        baseline.append(members)
        role_shuffle.append(rng.sample(members, 7))
        uniform.append(rng.sample(unique, 7))
    return {"production_sorted": torch.tensor(baseline),
            "same_members_shuffled_roles": torch.tensor(role_shuffle),
            "uniform_walk_endpoints": torch.tensor(uniform)}


def raw_probe_stats(features):
    # Shapes: 30 pseudo-classes x (3 supports + 4 queries) x feature dimension.
    x = F.normalize(features, dim=-1)
    protos = F.normalize(x[:, :3].mean(1), dim=-1)
    query = x[:, 3:].reshape(-1, x.shape[-1])
    scores = query @ protos.T
    labels = torch.arange(len(x)).repeat_interleave(4)
    positive = (x[:, :3, None] * x[:, None, 3:]).sum(-1).mean()
    return {"raw_prototype_nm_accuracy": float((scores.argmax(1) == labels).float().mean()),
            "raw_support_query_cosine": float(positive),
            "raw_prototype_margin": float((scores[torch.arange(len(query)), labels] -
                scores.masked_fill(F.one_hot(labels, len(x)).bool(), float("-inf")).max(1).values).mean())}


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--config", default="scripts/experiments/setup/final_core/training.yaml")
    p.add_argument("--output", required=True)
    p.add_argument("--episodes-per-source", type=int, default=256)
    p.add_argument("--context-samples-per-source", type=int, default=128)
    p.add_argument("--seed", type=int, default=28019)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    params = get_params(["--config", args.config, "--device", "123"])
    if (params["n_way"], params["n_shots"], params["n_query"]) != (30, 3, 4):
        raise ValueError("this matched-policy audit requires the final-core 30/3/4 task")
    graph_path = Path(params["root"]) / params["graph_filename"]
    print(json.dumps({"graph": str(graph_path), **vars(args)}), flush=True)
    if args.dry_run:
        return
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    # Torch 2.0 on Tucker predates mmap loading. Newer versions can avoid eagerly
    # reading unused target-view tensors; otherwise budget RAM for the full file.
    load_options = {"mmap": True} if "mmap" in inspect.signature(torch.load).parameters else {}
    print(f"Loading source graph (mmap={bool(load_options)})", flush=True)
    raw = torch.load(graph_path, map_location="cpu", weights_only=False, **load_options)
    ensure_static_views(raw, params)
    graph, view = _build_covid19_twitter_graph(raw, **params)
    if view != "static_train":
        raise RuntimeError("not the exact static_train background")
    sampler = NeighborSampler(graph, num_hops=2, hop_sizes=[9, 9], limit=101, walk_hops=1)
    recorder = RecordedWalk(sampler)
    graph_ids = graph.graph_id.numpy()
    source_ids = np.unique(graph_ids)
    strata = [np.flatnonzero(graph_ids == source) for source in source_ids]
    task = NeighborTask(recorder, graph.num_nodes, "inout", strata=strata,
                        confine_to_single_stratum=True, stratum_weighting="balanced",
                        filter_min_degree=True)
    source_names = graph.source_graph_names
    protocol = {**vars(args), "graph_path": str(graph_path), "graph_size_bytes": graph_path.stat().st_size,
                "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "member_selector_sha256": hashlib.sha256(inspect.getsource(NeighborTask._sample_center_members).encode()).hexdigest(),
                "historical_realized_stream": False, "edge_view": view, "mmap_loading": bool(load_options),
                "feature_dim": graph.x.shape[1], "num_nodes": graph.num_nodes,
                "source_ids": source_ids.tolist(), "source_names": list(source_names)}
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    summaries = []
    with (output / "episodes.jsonl").open("w") as handle:
        for stratum, source_id in enumerate(source_ids):
            name = source_names[int(source_id)]
            rng = random.Random(args.seed + int(source_id))
            variant_rng = random.Random(args.seed + 10000 + int(source_id))
            candidates = task._eligible_candidates(strata[stratum], 7, ("stratum", stratum))
            lo, hi = int(strata[stratum].min()), int(strata[stratum].max())
            retained, anchors_seen = defaultdict(list), []
            values = defaultdict(list)
            for episode in range(args.episodes_per_source):
                recorder.endpoints.clear()
                before = recorder.calls
                sampled = task._sample_from_stratum(30, 7, rng, stratum)
                anchors = list(sampled)
                anchors_seen.extend(anchors)
                variants = member_variants(sampled, recorder.endpoints, variant_rng)
                for variant, members in variants.items():
                    if not torch.all(graph.graph_id[members] == int(source_id)):
                        raise RuntimeError("cross-source member in source-confined episode")
                    features = graph.x[members].float()
                    flat = members.reshape(-1)
                    counts = Counter(flat.tolist())
                    row = {"source": name, "episode": episode, "variant": variant,
                           "walk_calls_per_30_anchors": recorder.calls - before,
                           "distinct_members": len(counts),
                           "cross_class_duplicate_fraction": 1 - len(counts) / len(flat),
                           "support_mean_relative_index": float((members[:, :3].float() - lo).mean() / max(1, hi-lo)),
                           "query_mean_relative_index": float((members[:, 3:].float() - lo).mean() / max(1, hi-lo)),
                           "feature_norm_mean": float(features.norm(dim=-1).mean()), **raw_probe_stats(features)}
                    handle.write(json.dumps(row) + "\n")
                    for key, value in row.items():
                        if isinstance(value, (float, int)) and key != "episode":
                            values[(variant, key)].append(value)
                    retained[variant].append(members)
            context_nodes, context_edges, context_cosine = [], [], []
            context_ids = torch.cat(retained["production_sorted"]).reshape(-1).unique()
            context_rng = torch.Generator().manual_seed(args.seed + int(source_id))
            context_ids = context_ids[torch.randperm(len(context_ids), generator=context_rng)[:args.context_samples_per_source]]
            for center in context_ids.tolist():
                nodes, edges, _ = sampler.sample_node(center)
                context_nodes.append(len(nodes))
                context_edges.append(edges.shape[1])
                neighbors = nodes[nodes != center]
                if len(neighbors):
                    context_cosine.append(float(F.cosine_similarity(graph.x[center][None], graph.x[neighbors].float()).mean()))
            for variant, batches in retained.items():
                members = torch.stack(batches)
                ids, counts = members.reshape(-1).unique(return_counts=True)
                # Exact simulated examples and their original features, useful for
                # asymmetric target-to-training coverage and future qualitative audit.
                torch.save({"member_ids": members, "anchor_ids": torch.tensor(anchors_seen).reshape(-1, 30),
                            "unique_member_ids": ids, "features": graph.x[ids].float(), "counts": counts},
                           output / f"{name}__{variant}.pt")
                row = {"source": name, "variant": variant, "source_nodes": len(strata[stratum]),
                       "eligible_anchors": len(candidates), "eligible_fraction": len(candidates) / len(strata[stratum]),
                       "unique_sampled_anchors": len(set(anchors_seen)), "unique_sampled_members": len(ids),
                       "effective_member_count": float(counts.sum().double().square() / counts.double().square().sum()),
                       "context_nodes_mean": float(np.mean(context_nodes)), "context_edges_mean": float(np.mean(context_edges)),
                       "context_center_cosine": float(np.mean(context_cosine)) if context_cosine else None}
                for (v, key), nums in values.items():
                    if v == variant:
                        row[key] = float(np.mean(nums))
                summaries.append(row)
                print(json.dumps(row), flush=True)
            handle.flush()
            (output / "summary.json").write_text(json.dumps(summaries, indent=2))
    (output / "DONE").write_text("Completed simulated training-distribution audit.\n")


if __name__ == "__main__":
    main()
