from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .config import load_config
from .lattice import SOURCE_ORDER, load_shared_graph, make_lp_loader, next_batch, update_selection
from .model import MaskedFeatureMLP, NodeMLP
from .train import configure_sampling_backend


def singleton_rows() -> list[tuple[str, tuple[str, ...]]]:
    return [(f"ss_{source}", (source,)) for source in SOURCE_ORDER]


def selected_rows(selection: str) -> list[tuple[str, tuple[str, ...]]]:
    rows = singleton_rows()
    if selection == "full":
        return rows
    if selection == "gate":
        return [rows[1]]
    wanted = {value for value in selection.split(",") if value}
    selected = [row for row in rows if row[0] in wanted]
    unknown = wanted - {run_id for run_id, _ in selected}
    if unknown:
        raise ValueError(f"unknown run ids: {sorted(unknown)}")
    return selected


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def feature_split(count: int, validation: bool, seed: int, fraction: float) -> torch.Tensor:
    order = torch.randperm(count, generator=torch.Generator().manual_seed(seed + 104729))
    n_validation = max(1, round(count * fraction))
    return order[:n_validation] if validation else order[n_validation:]


def feature_batches(x: torch.Tensor, indices: torch.Tensor, batch_size: int, shuffle: bool, seed: int):
    generator = torch.Generator().manual_seed(seed)
    order = indices[torch.randperm(len(indices), generator=generator)] if shuffle else indices
    for start in range(0, len(order), batch_size):
        yield x[order[start:start + batch_size]]


@dataclass
class DirectLinkBatch:
    x: torch.Tensor
    edge_label_index: torch.Tensor
    edge_label: torch.Tensor

    def to(self, device):
        self.x = self.x.to(device, non_blocking=True)
        self.edge_label_index = self.edge_label_index.to(device, non_blocking=True)
        self.edge_label = self.edge_label.to(device, non_blocking=True)
        return self


class DirectLinkLoader:
    """Topology-free endpoint batches without neighborhood sampling or PyG collation."""

    def __init__(self, x, positive_edges, batch_size, negatives_per_positive, shuffle, seed):
        self.x = x.float()
        self.positive_edges = positive_edges.long()
        self.batch_size = int(batch_size)
        self.negatives_per_positive = int(negatives_per_positive)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        if self.positive_edges.ndim != 2 or self.positive_edges.shape[0] != 2:
            raise ValueError("positive_edges must have shape [2, E]")
        if self.negatives_per_positive < 1:
            raise ValueError("negatives_per_positive must be a positive integer")

    def endpoint_batches(self):
        """CPU-only sampler; order/RNG identical to the direct loader."""
        epoch = self.epoch if self.shuffle else 0
        self.epoch += int(self.shuffle)
        generator = torch.Generator().manual_seed(self.seed + epoch)
        count = self.positive_edges.shape[1]
        order = torch.randperm(count, generator=generator) if self.shuffle else torch.arange(count)
        n_nodes = self.x.shape[0]
        for start in range(0, count, self.batch_size):
            positive = self.positive_edges[:, order[start:start + self.batch_size]]
            n_positive = positive.shape[1]
            n_negative = n_positive * self.negatives_per_positive
            negative = torch.randint(n_nodes, (2, n_negative), generator=generator)
            self_loops = negative[0] == negative[1]
            while self_loops.any():
                negative[1, self_loops] = torch.randint(
                    n_nodes, (int(self_loops.sum()),), generator=generator
                )
                self_loops = negative[0] == negative[1]
            pairs = torch.cat((positive, negative), dim=1)
            endpoints = torch.cat((pairs[0], pairs[1]))
            n_pairs = pairs.shape[1]
            local_edges = torch.stack((torch.arange(n_pairs), torch.arange(n_pairs, 2 * n_pairs)))
            labels = torch.cat((torch.ones(n_positive), torch.zeros(n_negative)))
            yield endpoints, local_edges, labels

    def __iter__(self):
        for endpoints, local_edges, labels in self.endpoint_batches():
            if self.x.is_cuda:
                features = self.x[endpoints.to(self.x.device, non_blocking=True)]
            else:
                features = self.x[endpoints].pin_memory() if torch.cuda.is_available() else self.x[endpoints]
            yield DirectLinkBatch(features, local_edges, labels)


def make_direct_lp_loader(graph, protocol, validation: bool, seed: int, features=None):
    edges = graph.validation_edges if validation else graph.train_edges
    return DirectLinkLoader(
        graph.data.x if features is None else features, edges, int(protocol["ssl_batch_size"]),
        int(protocol["negatives_per_positive"]), not validation,
        seed + (32452843 if validation else 49979687),
    )


def masked_feature_loss(model, x, mask_rate: float, alpha: float, generator, device):
    target = x.float().to(device)
    mask = torch.rand(target.shape, generator=generator).to(device) < mask_rate
    # Guarantee that every row contributes at least one held-out coordinate.
    empty = ~mask.any(dim=1)
    if empty.any():
        mask[empty, 0] = True
    corrupted = torch.where(mask, model.mask_token.expand_as(target), target)
    prediction = model(corrupted)
    prediction = prediction.masked_fill(~mask, 0.0)
    masked_target = target.masked_fill(~mask, 0.0)
    error = 1.0 - F.cosine_similarity(prediction, masked_target, dim=-1)
    return error.clamp_min(0).pow(alpha).mean()


def lp_loss(model, batch, device, return_logits=False):
    batch = batch.to(device)
    embedding = model(batch.x.float())
    src, dst = batch.edge_label_index
    logits = (embedding[src] * embedding[dst]).sum(-1)
    loss = F.binary_cross_entropy_with_logits(logits, batch.edge_label.float())
    return (loss, logits) if return_logits else loss


def build_model(objective: str, protocol: dict, device):
    args = (
        int(protocol["input_dim"]), int(protocol["hidden_dim"]),
        int(protocol["output_dim"]), float(protocol["dropout"]),
    )
    cls = NodeMLP if objective == "lp" else MaskedFeatureMLP
    return cls(*args).to(device)


def save_checkpoint(path: Path, model, optimizer, step: int, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoder = model.encoder if isinstance(model, MaskedFeatureMLP) else model
    from .checkpointing import atomic_torch_save, runtime_state
    atomic_torch_save({
        **runtime_state(), "model": encoder.state_dict(), "pretrain_model": model.state_dict(),
        "optimizer": optimizer.state_dict(), "step": step, "metadata": metadata,
    }, path)


@torch.no_grad()
def validate(model, graph, loader, objective, device, protocol, seed):
    model.eval()
    if objective == "lp":
        cpu_state = torch.random.get_rng_state()
        torch.manual_seed(seed + 32452843)
        losses = [float(lp_loss(model, batch, device)) for _, batch in zip(
            range(int(protocol["validation_batches"])), loader
        )]
        torch.random.set_rng_state(cpu_state)
    else:
        indices = feature_split(
            int(graph.data.num_nodes), True, seed, float(protocol["node_validation_fraction"])
        )
        batches = feature_batches(
            graph.data.x, indices, int(protocol["ssl_batch_size"]), False, seed
        )
        generator = torch.Generator().manual_seed(seed + 32452843)
        losses = [float(masked_feature_loss(
            model, batch, float(protocol["mask_rate"]), float(protocol["sce_alpha"]),
            generator, device,
        )) for _, batch in zip(range(int(protocol["validation_batches"])), batches)]
    if not losses:
        raise RuntimeError("validation yielded no batches")
    model.train()
    return float(np.mean(losses))


def train_one(run_id, source, objective, graph, config, device, output_root, seed):
    protocol = dict(config["protocol"])
    run_dir = output_root / objective / run_id
    if (run_dir / "summary.json").is_file():
        return json.loads((run_dir / "summary.json").read_text())
    if run_dir.exists():
        raise FileExistsError(f"partial run exists: {run_dir}")
    (run_dir / "checkpoints").mkdir(parents=True)
    seed_everything(seed)
    model = build_model(objective, protocol, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(protocol.get(f"node_mlp_{objective}_learning_rate", protocol["learning_rate"])),
        weight_decay=float(protocol["weight_decay"]),
    )
    if objective == "lp" and protocol.get("node_mlp_link_loader", "direct") == "direct":
        residency = protocol.get("node_mlp_feature_residency", "cpu")
        if residency not in ("cpu", "gpu"):
            raise ValueError(f"unknown node_mlp_feature_residency: {residency}")
        shared_features = graph.data.x.float().to(device) if residency == "gpu" else graph.data.x
        lp_train = make_direct_lp_loader(graph, protocol, False, seed, shared_features)
        lp_val = make_direct_lp_loader(graph, protocol, True, seed, shared_features)
    else:
        lp_train = make_lp_loader(graph, protocol, False) if objective == "lp" else None
        lp_val = make_lp_loader(graph, protocol, True) if objective == "lp" else None
    feature_indices = feature_split(
        int(graph.data.num_nodes), False, seed, float(protocol["node_validation_fraction"])
    )
    feature_iterator = None
    lp_iterator = None
    metadata = {
        "run_id": run_id, "sources": [source], "seed": seed, "objective": objective,
        "architecture": "node_mlp", "input_view": "center_node_features_only",
        "uses_topology_in_encoder": False, "protocol": protocol,
        "checkpoint_selection": "source_ssl_validation_only",
        "link_loader": protocol.get("node_mlp_link_loader", "direct"),
        "feature_residency": protocol.get("node_mlp_feature_residency", "cpu"),
    }
    if objective == "lp":
        metadata["training_negative_sampling"] = "uniform_approximate_excluding_self_loops"
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    best_loss = reference = math.inf
    best_step = patience = 0
    start = time.monotonic()
    train_window = []
    mask_generator = torch.Generator().manual_seed(seed + 49979687)
    checkpoints = set(map(int, protocol["checkpoint_steps"]))
    model.train()
    for step in range(1, int(protocol["max_steps"]) + 1):
        optimizer.zero_grad(set_to_none=True)
        if objective == "lp":
            batch, lp_iterator = next_batch(lp_train, lp_iterator)
            loss = lp_loss(model, batch, device)
        else:
            if feature_iterator is None:
                feature_iterator = iter(feature_batches(
                    graph.data.x, feature_indices, int(protocol["ssl_batch_size"]), True,
                    seed + step,
                ))
            try:
                batch = next(feature_iterator)
            except StopIteration:
                feature_iterator = iter(feature_batches(
                    graph.data.x, feature_indices, int(protocol["ssl_batch_size"]), True,
                    seed + step,
                ))
                batch = next(feature_iterator)
            loss = masked_feature_loss(
                model, batch, float(protocol["mask_rate"]), float(protocol["sce_alpha"]),
                mask_generator, device,
            )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(protocol["gradient_clip_norm"])
        ))
        optimizer.step()
        train_window.append(float(loss))
        if step in checkpoints:
            save_checkpoint(run_dir / "checkpoints" / f"step_{step}.pt", model, optimizer, step, metadata)
        if step % int(protocol["validation_interval"]):
            continue
        value = validate(model, graph, lp_val, objective, device, protocol, seed)
        elapsed = time.monotonic() - start
        row = {
            "step": step, "train_loss": float(np.mean(train_window)),
            "validation_loss": value, "elapsed_seconds": elapsed,
            "updates_per_second": step / elapsed, "gradient_norm": gradient_norm,
        }
        with (run_dir / "validation.jsonl").open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps({"run_id": run_id, **row}), flush=True)
        train_window.clear()
        best_loss, best_step, reference, patience, save_best = update_selection(
            value, step, best_loss, best_step, reference, patience,
            float(protocol["min_relative_improvement"]),
        )
        if save_best:
            save_checkpoint(run_dir / "best.pt", model, optimizer, step, metadata)
        if step >= int(protocol["minimum_steps"]) and patience >= int(protocol["patience_evaluations"]):
            break
    save_checkpoint(run_dir / "checkpoints" / f"step_{step}.pt", model, optimizer, step, metadata)
    summary = {
        **metadata, "status": "complete", "best_step": best_step,
        "best_validation_loss": best_loss, "final_step": step,
        "elapsed_seconds": time.monotonic() - start,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--objective", choices=("lp", "fp"), required=True)
    parser.add_argument("--selection", default="full")
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--cache-root")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.device not in (0, 1, 2, 3):
        raise ValueError("only owned Tucker GPUs 0-3 are allowed")
    if not 0 <= args.worker_index < args.workers:
        raise ValueError("invalid worker index")
    config = load_config(args.config)
    configure_sampling_backend(config["protocol"])
    rows = selected_rows(args.selection)[args.worker_index::args.workers]
    output_root = Path(args.output_root)
    for run_id, (source,) in rows:
        graph = load_shared_graph(
            source, config["graphs"][source]["path"],
            "lp" if args.objective == "lp" else "graphmae", config["protocol"], args.seed,
            Path(args.cache_root or (output_root / "_cache")),
        )
        result = train_one(run_id, source, args.objective, graph, config,
                           torch.device(f"cuda:{args.device}"), output_root, args.seed)
        print(json.dumps({"completed": run_id, "final_step": result["final_step"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


@torch.no_grad()
def validate_lp_metrics(model, loader, device, max_batches=20, return_data=False):
    """Evaluate fixed loader pairs once; preserve the caller's train/eval mode."""
    from .binary_metrics import binary_report
    mode = model.training
    labels, logits = [], []
    try:
        model.eval()
        for _, batch in zip(range(max_batches), loader):
            _, scores = lp_loss(model, batch, device, return_logits=True)
            labels.append(batch.edge_label.cpu().numpy())
            logits.append(scores.cpu().numpy())
    finally:
        model.train(mode)
    if not labels:
        raise RuntimeError("validation yielded no batches")
    y, scores = np.concatenate(labels), np.concatenate(logits)
    report = binary_report(y, scores)
    return (report, y, scores) if return_data else report
