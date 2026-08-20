#!/usr/bin/env python3
"""Train source-confined VISION/GILT models with their native task generators."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops, degree

from scripts.experiments.setup.icl_arch_matrix.architecture_adapters import (
    PINS,
    build_adapter,
    build_optimizer,
    padded_adjacency,
    vision_contrastive_loss,
)
from scripts.experiments.setup.icl_arch_matrix.common_protocol import (
    TRAIN_BATCH_SIZE,
    build_classification_dataset,
    build_classification_loader,
    classification_targets,
    iter_episodes,
)


VISION_N_WAY = 30
VISION_N_SHOT = 3
VISION_N_QUERY = 4
VISION_POOL_SIZE = 4096
GILT_N_WAY = 2
GILT_N_SHOT = 5
GILT_N_QUERY = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=("vision", "gilt"), required=True)
    parser.add_argument("--upstream-root", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--state-root", required=True, type=Path)
    parser.add_argument("--data-root", default="/dataMeR1/phil/data")
    parser.add_argument("--catalog", default="docs/graph_catalog.json")
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--checkpoint-steps", default="20,60,100,300,900")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    return parser.parse_args()


def save_checkpoint(path: Path, model, optimizer, scheduler, args, step: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "architecture": args.architecture,
            "model_id": args.model_id,
            "sources": args.source,
            "seed": args.seed,
            "step": step,
            "upstream": PINS[args.architecture],
            "pretraining_protocol": (
                "vision_native_feature_similarity_pseudo_episodes"
                if args.architecture == "vision"
                else "gilt_native_source_classification_episodes"
            ),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
        },
        path,
    )


def adaptive_vision_task_features(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    x_raw = F.normalize(x, p=2, dim=1)
    edges, _ = add_self_loops(edge_index, num_nodes=x.size(0))
    row, col = edges
    deg = degree(row, x.size(0), dtype=x.dtype)
    inv_sqrt = deg.pow(-0.5)
    inv_sqrt.masked_fill_(torch.isinf(inv_sqrt), 0)
    weights = inv_sqrt[row] * inv_sqrt[col]
    adjacency = torch.sparse_coo_tensor(edges, weights, (x.size(0), x.size(0))).coalesce()
    x_smooth = F.normalize(torch.sparse.mm(adjacency, x_raw), p=2, dim=1)
    gate = ((x_raw * x_smooth).sum(1).clamp(-1, 1) + 1) / 2
    return (1 - gate[:, None]) * x_raw + gate[:, None] * x_smooth


def vision_pseudo_task(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    needed = VISION_N_WAY * (VISION_N_SHOT + VISION_N_QUERY)
    if features.size(0) < needed + VISION_N_WAY:
        raise ValueError(f"VISION source has only {features.size(0)} nodes; need at least {needed}")
    permutation = torch.randperm(features.size(0), device=features.device)
    anchors = permutation[:VISION_N_WAY]
    candidates = permutation[VISION_N_WAY : VISION_N_WAY + min(VISION_POOL_SIZE, features.size(0) - VISION_N_WAY)]
    similarities = features[anchors] @ features[candidates].t()
    ranked = similarities.argsort(dim=1, descending=True)
    used = torch.zeros(candidates.numel(), dtype=torch.bool, device=features.device)
    groups = []
    per_class = VISION_N_SHOT + VISION_N_QUERY
    for class_index in range(VISION_N_WAY):
        available = ranked[class_index][~used[ranked[class_index]]]
        if available.numel() < per_class:
            raise RuntimeError("VISION pseudo-task candidate pool exhausted")
        chosen = available[:per_class]
        used[chosen] = True
        groups.append(candidates[chosen])
    samples = torch.stack(groups)
    return samples[:, :VISION_N_SHOT].reshape(-1), samples[:, VISION_N_SHOT:].reshape(-1)


def vision_subgraph(
    x: torch.Tensor,
    full_adjacency: torch.Tensor,
    support: torch.Tensor,
    query: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    centers = torch.cat((support, query))
    center_adjacency = full_adjacency[centers]
    neighbors = center_adjacency.reshape(-1)
    nodes = torch.cat((centers, neighbors[neighbors >= 0])).unique(sorted=True)
    global_adjacency = full_adjacency[nodes]
    flat = global_adjacency.reshape(-1)
    valid = flat >= 0
    mapped = torch.searchsorted(nodes, flat[valid])
    present = (mapped < nodes.numel()) & (nodes[mapped.clamp_max(nodes.numel() - 1)] == flat[valid])
    local_flat = torch.full_like(flat, -1)
    local_valid = torch.full_like(flat[valid], -1)
    local_valid[present] = mapped[present]
    local_flat[valid] = local_valid
    return (
        x[nodes],
        local_flat.view_as(global_adjacency),
        torch.searchsorted(nodes, support),
        torch.searchsorted(nodes, query),
    )


def train_vision(model, dataset, optimizer, scheduler, args, checkpoints, metrics) -> None:
    device = next(model.parameters()).device
    graph = dataset.graph
    x = graph.x.float().to(device)
    edge_index = graph.edge_index.long().to(device)
    task_features = adaptive_vision_task_features(x, edge_index)
    full_adjacency = padded_adjacency(edge_index, x.size(0), model.max_neighbors)
    for step in range(1, args.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses, accuracies = [], []
        for _ in range(TRAIN_BATCH_SIZE):
            support, query = vision_pseudo_task(task_features)
            sub_x, sub_adj, local_support, local_query = vision_subgraph(
                x, full_adjacency, support, query
            )
            sub_x = sub_x - sub_x.mean(dim=0, keepdim=True)
            drop = torch.rand(sub_adj.shape, device=device) < model.drop_edge
            sub_adj = sub_adj.masked_fill(drop & (sub_adj >= 0), -1)
            logits, contrastive = model.model(
                sub_x,
                sub_adj,
                local_support,
                local_query,
                drop_label_prob=model.drop_label,
                input_noise_std=model.input_noise,
            )
            labels = torch.arange(VISION_N_WAY, device=device).repeat_interleave(VISION_N_QUERY)
            ce = F.cross_entropy(logits, labels, label_smoothing=0.1)
            con = torch.stack(
                [
                    vision_contrastive_loss(
                        q,
                        s,
                        labels,
                        torch.arange(VISION_N_WAY, device=device).repeat_interleave(VISION_N_SHOT),
                    )
                    for q, s in contrastive
                ]
            ).mean()
            loss = ce + model.contrastive_weight * con
            (loss / TRAIN_BATCH_SIZE).backward()
            losses.append(loss.detach())
            accuracies.append((logits.argmax(1) == labels).float().mean().detach())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        record_step(model, optimizer, scheduler, args, checkpoints, metrics, step, losses, accuracies)


def train_gilt(model, dataset, get_dataloader, graph_path, target, optimizer, scheduler, args, checkpoints, metrics) -> None:
    loader = build_classification_loader(
        dataset_name=args.source,
        data_root=args.data_root,
        target=target,
        dataset=dataset,
        get_dataloader=get_dataloader,
        graph_path=graph_path,
        workers=args.workers,
        split="train",
        batch_count=args.steps,
        batch_size=TRAIN_BATCH_SIZE,
        n_way=GILT_N_WAY,
        n_shot=GILT_N_SHOT,
        n_query=GILT_N_QUERY,
        random_query=False,
    )
    device = next(model.parameters()).device
    for step, batch in enumerate(loader, start=1):
        model.train()
        graphs = batch[0].to(device)
        moved = (graphs,) + tuple(
            value.to(device) if torch.is_tensor(value) else value for value in batch[1:]
        )
        optimizer.zero_grad(set_to_none=True)
        losses, accuracies = [], []
        for episode in iter_episodes(
            moved,
            n_way=GILT_N_WAY,
            n_shot=GILT_N_SHOT,
            n_query=GILT_N_QUERY,
        ):
            loss, accuracy = model.episode_loss_and_accuracy(episode)
            (loss / TRAIN_BATCH_SIZE).backward()
            losses.append(loss.detach())
            accuracies.append(accuracy.detach())
        if len(losses) != TRAIN_BATCH_SIZE:
            raise RuntimeError(f"expected {TRAIN_BATCH_SIZE} GILT episodes, got {len(losses)}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        record_step(model, optimizer, scheduler, args, checkpoints, metrics, step, losses, accuracies)


def record_step(model, optimizer, scheduler, args, checkpoints, metrics, step, losses, accuracies) -> None:
    row = {
        "step": step,
        "loss": float(torch.stack(losses).mean().cpu()),
        "accuracy": float(torch.stack(accuracies).mean().cpu()),
        "lr": float(optimizer.param_groups[0]["lr"]),
        "elapsed_seconds": time.time() - record_step.started,
    }
    metrics.write(json.dumps(row, sort_keys=True) + "\n")
    metrics.flush()
    if step == 1 or step % 20 == 0:
        print(json.dumps(row, sort_keys=True), flush=True)
    if step in checkpoints:
        save_checkpoint(
            args.run_dir / "checkpoint" / f"state_dict_{step}.pt",
            model,
            optimizer,
            scheduler,
            args,
            step,
        )


record_step.started = 0.0


def main() -> int:
    args = parse_args()
    checkpoints = {int(value) for value in args.checkpoint_steps.split(",") if value}
    if not checkpoints or min(checkpoints) < 1 or max(checkpoints) > args.steps:
        raise ValueError(f"invalid checkpoints: {sorted(checkpoints)}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    targets = classification_targets(args.catalog, include_facebook=True)
    if args.source not in targets:
        raise ValueError(f"unsupported source: {args.source}")
    target = targets[args.source]
    dataset, get_dataloader, graph_path = build_classification_dataset(
        dataset_name=args.source, data_root=args.data_root, target=target
    )
    model = build_adapter(args.architecture, args.upstream_root).to(device)
    optimizer = build_optimizer(model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=model.learning_rate * 0.05
    )
    args.run_dir = args.state_root / args.architecture / f"{args.model_id}_s{args.seed}"
    if args.run_dir.exists():
        raise FileExistsError(f"refusing ambiguous resume into {args.run_dir}")
    args.run_dir.mkdir(parents=True)
    record_step.started = time.time()
    with (args.run_dir / "train_metrics.jsonl").open("w", encoding="utf-8") as metrics:
        if args.architecture == "vision":
            train_vision(model, dataset, optimizer, scheduler, args, checkpoints, metrics)
        else:
            train_gilt(
                model,
                dataset,
                get_dataloader,
                graph_path,
                target,
                optimizer,
                scheduler,
                args,
                checkpoints,
                metrics,
            )
    terminal = args.run_dir / "checkpoint" / f"state_dict_{args.steps}.pt"
    if not terminal.is_file():
        raise RuntimeError(f"terminal checkpoint missing: {terminal}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
