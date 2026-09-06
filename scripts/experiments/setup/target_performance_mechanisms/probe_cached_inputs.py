"""Support-only topology/text-summary probes on immutable cached test episodes."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.trainer import TrainerFS, _concat_global_eval_parts
from scripts.experiments.setup.icl_arch_matrix.common_protocol import classification_targets
from scripts.experiments.setup.target_performance_mechanisms.replay import batch_hash, episode_probe, raw_stages


def detailed_summaries(batch, raw, topology, text):
    graph = batch[0]
    centers = graph.ptr[:-1]
    mask = graph.global_node_ids >= 0
    mask[centers] = False
    groups = graph.batch[mask]
    norm = graph.x[mask].norm(dim=1)
    counts = torch.bincount(groups, minlength=len(centers)).float().clamp_min(1)
    avg_norm = torch.zeros(len(centers)).scatter_add_(0, groups, norm) / counts
    nonzero = torch.zeros(len(centers)).scatter_add_(0, groups, (norm > 1e-8).float()) / counts
    unit_sum = torch.zeros(len(centers), graph.x.shape[1]).index_add_(0, groups, F.normalize(graph.x[mask], dim=1))
    valid_count = torch.zeros(len(centers)).scatter_add_(0, groups, (norm > 1e-8).float())
    pairwise = (unit_sum.square().sum(1) - valid_count) / (valid_count * (valid_count - 1)).clamp_min(1)
    pairwise[valid_count < 2] = 0
    return {
        "sample_nodes": topology[:, 0], "sample_edges": topology[:, 1],
        "center_outdegree": topology[:, 2], "center_indegree": topology[:, 3],
        "sample_density": topology[:, 4],
        "center_norm": text[:, 0], "context_mean_norm": text[:, 1],
        "center_context_cosine": text[:, 2],
        "context_nonzero_fraction": nonzero, "context_avg_norm": avg_norm,
        "context_coherence": text[:, 1] / avg_norm.clamp_min(1e-8),
        "context_pairwise_cosine": pairwise,
        "has_incoming_edge": (topology[:, 3] > 0).float(),
    }


def input_summaries(batch):
    graph = batch[0]
    centers = graph.ptr[:-1]
    count = len(centers)
    nodes = (graph.ptr[1:] - graph.ptr[:-1] - 1).float()
    edges = torch.bincount(graph.batch[graph.edge_index[0]], minlength=count).float()
    outgoing = torch.bincount(graph.edge_index[0], minlength=len(graph.x))[centers].float()
    incoming = torch.bincount(graph.edge_index[1], minlength=len(graph.x))[centers].float()
    topology = torch.stack([nodes.log1p(), edges.log1p(), outgoing.log1p(), incoming.log1p(),
                            edges / (nodes * (nodes - 1)).clamp_min(1)], dim=1)
    raw = raw_stages(graph)
    text = torch.stack([raw["raw_center"].norm(dim=1), raw["raw_context"].norm(dim=1),
                        F.cosine_similarity(raw["raw_center"], raw["raw_context"], dim=1)], dim=1)
    return raw, topology, text


def standardized_probe(x, batch, raw=None):
    """Fit feature scaling and ridge only on supports, with fixed regularization."""
    labels = batch[2].argmax(1)
    n_way = batch[2].shape[1]
    query = batch[5].reshape(-1, n_way)[:, 0].bool()
    tasks = batch[0].task_id_per_sample
    result = x.new_zeros((len(x), n_way))
    for task in tasks.unique(sorted=True):
        support = (tasks == task) & ~query
        q = (tasks == task) & query
        mean = x[support].mean(0)
        scale = x[support].std(0, unbiased=False).clamp_min(1e-4)
        sx = (x[support] - mean) / scale
        qx = (x[q] - mean) / scale
        sx = torch.cat([sx, torch.ones(len(sx), 1)], dim=1)
        qx = torch.cat([qx, torch.ones(len(qx), 1)], dim=1)
        if raw is not None:
            sx = torch.cat([F.normalize(raw[support], dim=1), F.normalize(sx, dim=1)], dim=1) / 2**0.5
            qx = torch.cat([F.normalize(raw[q], dim=1), F.normalize(qx, dim=1)], dim=1) / 2**0.5
        else:
            sx = sx / sx.shape[1]**0.5
            qx = qx / qx.shape[1]**0.5
        coef = torch.linalg.solve(sx @ sx.T + torch.eye(len(sx)), F.one_hot(labels[support], n_way).float())
        result[q] = qx @ sx.T @ coef
    return result[query]


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--replay-roots", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--catalog", default="docs/graph_catalog.json")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--single-features", action="store_true",
                   help="Exploratory follow-up: probe every scalar and export query-level summaries.")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    metric = TrainerFS.__new__(TrainerFS)
    metric.parameter = {"task_name": "classification"}
    metric.is_feature_prediction = metric.is_regression = False
    all_metrics = []
    with torch.no_grad():
        for target in classification_targets(args.catalog, include_facebook=True):
            # Later roots override earlier incomplete runs, but require all
            # random-init latent batches (written only after a completed cell).
            paths = [root / target for root in args.replay_roots if (root / target / "random_init__baseline.pt").is_file()]
            if not paths:
                raise FileNotFoundError(f"no completed cache for {target}")
            root = paths[-1]
            cache = json.loads((root / "cache.json").read_text())
            if cache["episodes"] != 128 or len(cache["batch_sha256"]) != 32:
                raise ValueError("requires complete fixed episode stream")
            collected = defaultdict(lambda: {"yt": [], "yp": [], "global": []})
            exports = []
            projection = None
            for i in range(32):
                batch = torch.load(root / "batches" / f"batch_{i:03d}.pt", weights_only=False)
                if batch_hash(batch) != cache["batch_sha256"][i]:
                    raise RuntimeError("cached input changed")
                raw, topology, text = input_summaries(batch)
                if projection is None:
                    projection = torch.randn(raw["raw_center"].shape[1], 256, generator=torch.Generator().manual_seed(891)) / 256**0.5
                predictions = {"node_count_only": standardized_probe(topology[:, :1], batch),
                               "topology_only": standardized_probe(topology, batch),
                               "text_summaries_only": standardized_probe(text, batch),
                               "topology_and_text_summaries": standardized_probe(torch.cat([topology, text], 1), batch),
                               "raw_center_plus_topology": standardized_probe(topology, batch, raw["raw_center"]),
                               "raw_joint_plus_topology": standardized_probe(topology, batch, raw["raw_joint"]),
                               "raw_center_random_projection_256": episode_probe(raw["raw_center"] @ projection, batch, "ridge")}
                detail = detailed_summaries(batch, raw, topology, text) if args.single_features else {}
                predictions.update({f"scalar_{key}": standardized_probe(value[:, None], batch)
                                    for key, value in detail.items()})
                query = batch[5].reshape(-1, 2)[:, 0].bool()
                yt = batch[2][query]
                for name, yp in predictions.items():
                    collected[name]["yt"].append(yt)
                    collected[name]["yp"].append(yp)
                    collected[name]["global"].append(metric._extract_global_classification_eval(batch, yt, yp))
                exports.append({"batch": i, "batch_sha256": cache["batch_sha256"][i],
                                "topology": topology, "text_summaries": text,
                                "detail": detail, "logits": predictions})
            for probe, part in collected.items():
                yt, yp = torch.cat(part["yt"]), torch.cat(part["yp"])
                scores = metric._compute_eval_metrics(yt, yp, global_eval=_concat_global_eval_parts(part["global"]))
                if "roc_auc" not in scores:
                    raise RuntimeError("metric computation failed")
                row = {"dataset": target, "probe": probe, "episodes": 128, "queries": len(yt),
                       "episode_fingerprint": cache["episode_fingerprint"], "cache_root": str(root), **scores}
                all_metrics.append(row)
                print(json.dumps(row), flush=True)
            torch.save(exports, args.output / f"{target}.pt")
    (args.output / "metrics.json").write_text(json.dumps(all_metrics, indent=2))
    (args.output / "DONE").write_text("Complete input probes; support-only fitting and cached input hashes verified.\n")


if __name__ == "__main__":
    main()
