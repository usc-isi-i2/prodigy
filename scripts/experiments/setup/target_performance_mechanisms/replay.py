"""Fixed-episode, stage-resolved diagnostics using the production CLS protocol.

No training or test-label fitting. Every probe fits only the episode's supports.
Stage probes are diagnostic decoders, not counterfactual full-model predictions.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import subprocess
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
import wandb
from torch_geometric.nn import global_mean_pool

from experiments.trainer import TrainerFS, _concat_global_eval_parts
from models.layer_classes import SupernodeAggrLayer, MetagraphLayer, BackgroundGNNLayer
from models.metaGNN import MetaGNNLayer
from scripts.experiments.setup.icl_arch_matrix.common_protocol import (
    build_classification_dataset, classification_targets, iter_episodes,
    new_fingerprint, reset_episode_rng, update_episode_fingerprint,
)
from scripts.experiments.setup.icl_arch_matrix.evaluate_prodigy import (
    resolved_params, load_external_models,
)


def clone_batch(batch, device="cpu"):
    # Production forward mutates graph.x; never reuse its input object.
    return [v.clone().to(device) for v in batch]


def tensor_hash(hasher, name, value):
    value = value.detach().cpu().contiguous()
    hasher.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
    hasher.update(value.numpy().tobytes())


def batch_hash(batch):
    """Hash every tensor, including features, not only the episode identities."""
    h = hashlib.sha256()
    for key, value in sorted(batch[0]):
        if isinstance(value, torch.Tensor):
            tensor_hash(h, f"graph.{key}", value)
    for i, value in enumerate(batch[1:], 1):
        tensor_hash(h, str(i), value)
    return h.hexdigest()


@contextmanager
def trace_stages(model, graph):
    """Observe actual production forwards without reimplementing model algebra.

    S/U are called through .forward directly in this architecture. Normal hooks
    cannot observe U, hence a scoped wrapper which is restored even on failure.
    """
    traces, handles, restores = {}, [], []
    centers = graph.ptr[:-1]
    n_samples = centers.numel()

    def save(name, value):
        traces[name] = value.detach().clone()

    for i, layer in enumerate(model.layer_list):
        if isinstance(layer, BackgroundGNNLayer):
            handles.append(layer.reset_mlp_c.register_forward_pre_hook(
                lambda m, a, key=f"S{i}_conv_center": save(key, a[0])))
            handles.append(layer.reset_mlp_m.register_forward_pre_hook(
                lambda m, a, key=f"S{i}_pool": save(key, a[0])))
        elif isinstance(layer, SupernodeAggrLayer):
            old = layer.forward
            had_instance = "forward" in layer.__dict__
            def wrapper(*a, _old=old, _key=f"U{i}_pre_meta", **kw):
                value = _old(*a, **kw)
                save(_key, value)
                return value
            layer.forward = wrapper
            restores.append((layer, old, had_instance))
        elif isinstance(layer, MetagraphLayer):
            handles.append(layer.register_forward_hook(
                lambda m, a, out, key=f"M{i}_post_meta": save(key, out[:n_samples])))
    handles.append(model.final_input_mlp.register_forward_hook(
        lambda m, a, out: save("final_input", out)))
    try:
        yield traces
    finally:
        for handle in handles:
            handle.remove()
        for layer, old, had_instance in restores:
            if had_instance:
                layer.forward = old
            else:
                del layer.forward


def raw_stages(graph):
    centers = graph.ptr[:-1]
    real = graph.global_node_ids >= 0
    context = real.clone()
    context[centers] = False
    mean = global_mean_pool(graph.x[context], graph.batch[context], size=len(centers))
    return {"raw_center": graph.x[centers].clone(), "raw_context": mean,
            "raw_joint": torch.cat((graph.x[centers], mean), dim=1)}


def episode_probe(embeddings, batch, method):
    """Return local query logits, keeping the production query-row order."""
    labels = batch[2].argmax(1)
    n_way = batch[2].shape[1]
    query = batch[5].reshape(-1, n_way)[:, 0].bool()
    tasks = batch[0].task_id_per_sample
    result = embeddings.new_zeros((len(labels), n_way))
    x = F.normalize(embeddings, dim=1)
    for task in tasks.unique(sorted=True):
        support = (tasks == task) & ~query
        q = (tasks == task) & query
        sx, qx = x[support], x[q]
        sy = labels[support]
        if method == "prototype":
            protos = torch.stack([sx[sy == c].mean(0) for c in range(n_way)])
            result[q] = qx @ F.normalize(protos, dim=1).T
        elif method == "ridge":
            # Fixed regularization, normalized inputs, no target-test selection.
            kernel = sx @ sx.T + torch.eye(len(sx), device=sx.device)
            coef = torch.linalg.solve(kernel, F.one_hot(sy, n_way).to(sx.dtype))
            result[q] = qx @ sx.T @ coef
        else:
            raise ValueError(method)
    return result[query]


def intervene(batch, variant, seed):
    """Fixed-context input interventions; no labels used for feature shuffling."""
    graph = batch[0]
    if variant in {"baseline", "bn_batch_encoder", "bn_batch_meta", "bn_batch_all", "meta_bias_normalized"}:
        return
    if variant == "center_features_only":
        real = graph.global_node_ids >= 0
        graph.x[real] = graph.x[graph.ptr[:-1]][graph.batch[real]]
    elif variant == "no_background_edges":
        graph.edge_index = graph.edge_index[:, :0]
        if graph.edge_attr is not None:
            graph.edge_attr = graph.edge_attr[:0]
    elif variant == "shuffle_context":
        mask = graph.global_node_ids >= 0
        mask[graph.ptr[:-1]] = False
        # Shuffle within each episode, excluding its centers and synthetic nodes.
        node_tasks = graph.task_id_per_sample[graph.batch]
        rng = torch.Generator(device="cpu").manual_seed(seed)
        for task in node_tasks.unique(sorted=True):
            idx = torch.where(mask & (node_tasks == task))[0]
            perm = torch.randperm(len(idx), generator=rng).to(idx.device)
            graph.x[idx] = graph.x[idx[perm]].clone()
    elif variant == "zero_support_relations":
        batch[4][:, 1] = 0
    elif variant == "zero_label_text":
        batch[1].zero_()
    else:
        raise ValueError(variant)


@contextmanager
def bn_mode(model, variant):
    """Explicitly transductive diagnostic, not a leakage-free adaptation result."""
    originals = []
    if variant.startswith("bn_batch_"):
        which = variant.removeprefix("bn_batch_")
        for layer in model.layer_list:
            selected = which == "all" or (
                which == "encoder" and isinstance(layer, BackgroundGNNLayer)
            ) or (which == "meta" and isinstance(layer, MetagraphLayer))
            if selected:
                for module in layer.modules():
                    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                        originals.append((module, module.training, module.track_running_stats))
                        module.training = True
                        module.track_running_stats = False
    try:
        yield
    finally:
        for module, training, tracking in originals:
            module.training, module.track_running_stats = training, tracking


@contextmanager
def meta_bias_mode(model, variant):
    """Diagnostic: apply attention output-projection bias once per destination.

    Production projects each already-attention-weighted message before summing,
    hence its bias is added indegree times. This observer subtracts only the
    excess bias, retaining every learned weight and attention coefficient.
    """
    handles = []
    if variant == "meta_bias_normalized":
        for layer in model.modules():
            if not isinstance(layer, MetaGNNLayer):
                continue
            saved = {}
            def capture(m, args, kwargs, _saved=saved):
                x = args[0] if args else kwargs["x"]
                edges = args[1] if len(args) > 1 else kwargs["edge_index"]
                index = edges[1]
                degrees = torch.bincount(index, minlength=len(x))
                _saved["factor"] = (1 - 1 / degrees[index].to(x.dtype))[:, None]
            def correct(m, args, out, _saved=saved):
                if out.shape[0] != _saved["factor"].shape[0]:
                    raise RuntimeError("unexpected metagraph message ordering")
                return out - _saved["factor"] * m.bias
            handles.append(layer.register_forward_pre_hook(capture, with_kwargs=True))
            handles.append(layer.out_proj.register_forward_hook(correct))
        if not handles:
            raise RuntimeError("no compatible metagraph layer")
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def select_models(args):
    models = load_external_models(args.model_list)
    if args.model_ids:
        requested = set(args.model_ids.split(","))
        models = [m for m in models if m.model_id in requested]
        if requested != {m.model_id for m in models}:
            raise ValueError("unresolved requested model ids")
    elif args.specialists_only:
        models = [m for m in models if len(m.sources) == 1]
    if args.include_random:
        models.append(SimpleNamespace(model_id="random_init", sources=(), checkpoint=None))
    return models


def parse_args():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--model-list", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--datasets", default="covid_political,facebook_page_reference")
    p.add_argument("--model-ids", default="")
    p.add_argument("--specialists-only", action="store_true")
    p.add_argument("--include-random", action="store_true")
    p.add_argument("--reference-tsv", default="")
    p.add_argument("--variants", default="baseline")
    p.add_argument("--device", default="3")
    p.add_argument("--batch-count", type=int, default=32)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--save-embeddings", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--config", default="scripts/experiments/setup/final_core/training.yaml")
    p.add_argument("--catalog", default="docs/graph_catalog.json")
    p.add_argument("--data-root", default="/dataMeR1/phil/data")
    p.add_argument("--training-seed", type=int, default=0)
    p.add_argument("--eval-episode-seed-offset", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    models = select_models(args)
    targets = classification_targets(args.catalog, include_facebook=True)
    names = args.datasets.split(",")
    if not set(names) <= targets.keys():
        raise ValueError("unknown target")
    variants = args.variants.split(",")
    allowed = {"baseline", "center_features_only", "no_background_edges", "shuffle_context",
               "zero_support_relations", "zero_label_text", "bn_batch_encoder", "bn_batch_meta", "bn_batch_all", "meta_bias_normalized"}
    if not set(variants) <= allowed or len(set(variants)) != len(variants):
        raise ValueError("invalid or duplicate variant")
    if not 1 <= args.batch_count <= 32:
        raise ValueError("batch count must be 1..32; 32 is the full 128-episode protocol")
    print(json.dumps({"models": [m.model_id for m in models], "targets": names,
                      "variants": variants, "episodes": args.batch_count * 4}), flush=True)
    if args.dry_run:
        return
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    args.state_root = str(output / "state")
    args.eval_state_root = args.state_root
    args.log_root = str(output / "trainer_logs")
    args.run_stamp = "mechanism_replay"
    torch.set_num_threads(args.threads)
    reference = {}
    if args.reference_tsv:
        with open(args.reference_tsv) as f:
            reference = {(r["model_id"], r["dataset"]): r for r in csv.DictReader(f, delimiter="\t")}
    metadata = vars(args) | {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                             "protocol": "fixed_cls_stage_replay_v1", "bn_batch_is_transductive": True}
    (output / "protocol.json").write_text(json.dumps(metadata, indent=2))
    for name in names:
        target = targets[name]
        dataset, _, graph_path = build_classification_dataset(
            dataset_name=name, data_root=args.data_root, target=target)
        # One architecture construction, identical to official eval; weights only
        # are changed afterwards. Also retain the genuinely random initial state.
        params = resolved_params(args, name, target, graph_path, None, "replay", 2500)
        params["test_len_cap"] = args.batch_count
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        trainer = TrainerFS(dataset, params)
        initial = {k: v.detach().cpu().clone() for k, v in trainer.model.state_dict().items()}
        target_dir = output / name
        cache_dir = target_dir / "batches"
        cache_dir.mkdir(parents=True)
        fingerprint = new_fingerprint()
        batch_paths, hashes = [], []
        reset_episode_rng()
        for index, batch in enumerate(trainer.test_dataloader):
            batch = clone_batch(batch)
            for episode in iter_episodes(batch, n_way=2, n_shot=10, n_query=target["n_query"],
                                         equal_query_counts=not target["eval_random_query"]):
                update_episode_fingerprint(fingerprint, episode)
            path = cache_dir / f"batch_{index:03d}.pt"
            hashes.append(batch_hash(batch))
            torch.save(batch, path)
            batch_paths.append(path)
        if len(batch_paths) != args.batch_count:
            raise RuntimeError("wrong number of cached batches")
        cache_info = {"episode_fingerprint": fingerprint.hexdigest(), "batch_sha256": hashes,
                      "episodes": len(batch_paths) * 4, "graph_path": str(graph_path)}
        (target_dir / "cache.json").write_text(json.dumps(cache_info, indent=2))
        with (target_dir / "metrics.jsonl").open("w") as metrics_file:
            for model in models:
                state = initial if model.checkpoint is None else torch.load(
                    model.checkpoint, map_location="cpu", weights_only=True)["model"]
                trainer.model.load_state_dict(state, strict=True)
                trainer.model.eval()
                state_hash = hashlib.sha256()
                for k, v in sorted(state.items()):
                    tensor_hash(state_hash, k, v)
                for variant in variants:
                    collected = defaultdict(lambda: {"yt": [], "yp": [], "global": []})
                    exports = []
                    with torch.no_grad(), bn_mode(trainer.model, variant), meta_bias_mode(trainer.model, variant):
                        for index, path in enumerate(batch_paths):
                            # Trusted artifacts just created in this invocation.
                            raw = torch.load(path, map_location="cpu", weights_only=False)
                            if batch_hash(raw) != hashes[index]:
                                raise RuntimeError("cached batch changed")
                            batch = clone_batch(raw, trainer.device)
                            intervene(batch, variant, 710000 + index)
                            raw_embeddings = raw_stages(batch[0]) if variant == "baseline" else {}
                            if index == 0:
                                _, plain, _ = trainer.model(*clone_batch(batch, trainer.device))
                            with trace_stages(trainer.model, batch[0]) as traces:
                                yt, yp, _ = trainer.model(*batch)
                            if index == 0:
                                torch.testing.assert_close(plain, yp, rtol=0, atol=0)
                            predictions = {"full_model": yp}
                            if variant == "baseline":
                                for stage, x in (raw_embeddings | traces).items():
                                    for method in ("prototype", "ridge"):
                                        predictions[f"{stage}/{method}"] = episode_probe(x, batch, method)
                            for decoder, logits in predictions.items():
                                part = collected[decoder]
                                part["yt"].append(yt.cpu())
                                part["yp"].append(logits.cpu())
                                glob = trainer._extract_global_classification_eval(batch, yt, logits)
                                if glob is None:
                                    raise RuntimeError("missing global class mapping")
                                part["global"].append(glob)
                            record = {"batch": index, "batch_sha256": hashes[index],
                                      "logits": {k: v.cpu() for k, v in predictions.items()}}
                            if args.save_embeddings and variant == "baseline":
                                record["embeddings"] = {k: v.cpu() for k, v in (raw_embeddings | traces).items()}
                            exports.append(record)
                    rows = []
                    for decoder, part in collected.items():
                        yt, yp = torch.cat(part["yt"]), torch.cat(part["yp"])
                        glob = _concat_global_eval_parts(part["global"])
                        metrics = trainer._compute_eval_metrics(yt, yp, global_eval=glob)
                        if "roc_auc" not in metrics:
                            raise RuntimeError("metric computation failed")
                        row = {"model_id": model.model_id, "sources": list(model.sources), "dataset": name,
                               "variant": variant, "decoder": decoder, "episodes": 4 * len(batch_paths),
                               "queries": len(yt), "episode_fingerprint": fingerprint.hexdigest(),
                               "weights_sha256": state_hash.hexdigest(), "checkpoint": str(model.checkpoint),
                               "nll": F.cross_entropy(yp, yt).item(), **metrics}
                        prior = reference.get((model.model_id, name))
                        if variant == "baseline" and decoder == "full_model" and prior and args.batch_count == 32 and args.eval_episode_seed_offset == 0:
                            if prior["episode_fingerprint"] != fingerprint.hexdigest():
                                raise RuntimeError(f"official episode fingerprint mismatch for {name}")
                            row["official_metric_max_abs_error"] = max(abs(metrics[k] - float(prior[k])) for k in ("accuracy", "f1", "roc_auc"))
                            if row["official_metric_max_abs_error"] > 1e-6:
                                raise RuntimeError(f"official metric parity failed: {row}")
                        rows.append(row)
                    torch.save(exports, target_dir / f"{model.model_id}__{variant}.pt")
                    for row in rows:
                        metrics_file.write(json.dumps(row, sort_keys=True) + "\n")
                        if row["decoder"] == "full_model":
                            print(json.dumps(row, sort_keys=True), flush=True)
                    metrics_file.flush()
        wandb.finish()
        del trainer, dataset
    (output / "DONE").write_text("All requested cells completed; parity guards passed where applicable.\n")


if __name__ == "__main__":
    main()
