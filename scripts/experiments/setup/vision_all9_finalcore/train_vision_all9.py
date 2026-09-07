#!/usr/bin/env python3
"""Train native VISION on the balanced final-core all-nine graph mixture."""

from __future__ import annotations

import argparse
import gc
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from scripts.experiments.setup.final_core.core_plan import SOURCES
from scripts.experiments.setup.icl_arch_matrix.architecture_adapters import (
    PINS,
    build_adapter,
    build_optimizer,
    vision_contrastive_loss,
)
from scripts.experiments.setup.icl_arch_matrix.common_protocol import (
    TRAIN_BATCH_SIZE,
    build_dataset,
    load_config,
)
from scripts.experiments.setup.icl_arch_matrix.train_native_source_model import (
    VISION_N_QUERY,
    VISION_N_SHOT,
    VISION_N_WAY,
    VISION_POOL_SIZE,
    adaptive_vision_task_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--upstream-root", required=True)
    parser.add_argument("--state-root", required=True, type=Path)
    parser.add_argument("--run-name", default="all9_s0")
    parser.add_argument("--model-id", default="all9")
    parser.add_argument("--sources", default=",".join(SOURCES))
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--checkpoint-steps", default="100,300,900,2500")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def source_node_sets(
    graph, selected_sources: tuple[str, ...] = SOURCES
) -> tuple[list[str], list[torch.Tensor]]:
    names = list(graph.source_graph_names)
    if len(names) != len(SOURCES) or set(names) != set(SOURCES):
        raise ValueError(f"all-nine source registry mismatch: {names} != {list(SOURCES)}")
    if not selected_sources or len(set(selected_sources)) != len(selected_sources):
        raise ValueError(f"sources must be nonempty and unique: {selected_sources}")
    unknown = set(selected_sources) - set(SOURCES)
    if unknown:
        raise ValueError(f"unknown source names: {sorted(unknown)}")
    graph_id = graph.graph_id.detach().cpu().long()
    observed = set(int(value) for value in torch.unique(graph_id).tolist())
    expected = set(range(len(SOURCES)))
    if observed != expected:
        raise ValueError(f"graph ids mismatch: {sorted(observed)} != {sorted(expected)}")
    graph_index = {name: index for index, name in enumerate(names)}
    nodes = [torch.where(graph_id == graph_index[name])[0] for name in selected_sources]
    minimum = VISION_N_WAY * (VISION_N_SHOT + VISION_N_QUERY) + VISION_N_WAY
    if any(part.numel() < minimum for part in nodes):
        raise ValueError(f"source too small for VISION pseudo-tasks: {[part.numel() for part in nodes]}")
    return list(selected_sources), nodes


def sampled_source_nodes(source_nodes: torch.Tensor, count: int) -> torch.Tensor:
    """Sample unique global node IDs without allocating a graph-sized permutation."""
    if count > source_nodes.numel():
        raise ValueError(f"requested {count} nodes from a source of size {source_nodes.numel()}")
    offsets = torch.tensor(random.sample(range(source_nodes.numel()), count), dtype=torch.long)
    return source_nodes[offsets]


def vision_pseudo_task_for_source(
    task_features: torch.Tensor,
    source_nodes: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    per_class = VISION_N_SHOT + VISION_N_QUERY
    pool_size = min(VISION_POOL_SIZE, source_nodes.numel() - VISION_N_WAY)
    selected = sampled_source_nodes(source_nodes, VISION_N_WAY + pool_size)
    anchors = selected[:VISION_N_WAY]
    candidates = selected[VISION_N_WAY:]
    selected_features = task_features[selected].to(device)
    similarities = selected_features[:VISION_N_WAY] @ selected_features[VISION_N_WAY:].t()
    ranked = similarities.argsort(dim=1, descending=True)
    candidates_device = candidates.to(device)
    used = torch.zeros(candidates.numel(), dtype=torch.bool, device=device)
    groups = []
    for class_index in range(VISION_N_WAY):
        available = ranked[class_index][~used[ranked[class_index]]]
        if available.numel() < per_class:
            raise RuntimeError("VISION pseudo-task candidate pool exhausted")
        chosen = available[:per_class]
        used[chosen] = True
        groups.append(candidates_device[chosen])
    samples = torch.stack(groups).cpu()
    return samples[:, :VISION_N_SHOT].reshape(-1), samples[:, VISION_N_SHOT:].reshape(-1)


def first_neighbors(
    rowptr: torch.Tensor,
    col: torch.Tensor,
    node_ids: torch.Tensor,
    max_neighbors: int,
) -> torch.Tensor:
    """Return the first max_neighbors CSR neighbors for each CPU node ID."""
    starts = rowptr[node_ids]
    widths = (rowptr[node_ids + 1] - starts).clamp(max=max_neighbors)
    offsets = torch.arange(max_neighbors, dtype=starts.dtype)
    positions = starts[:, None] + offsets[None, :]
    valid = offsets[None, :] < widths[:, None]
    padded = torch.full((node_ids.numel(), max_neighbors), -1, dtype=torch.long)
    if col.numel():
        padded[valid] = col[positions[valid]]
    return padded


def vision_subgraph_from_csr(
    x: torch.Tensor,
    rowptr: torch.Tensor,
    col: torch.Tensor,
    support: torch.Tensor,
    query: torch.Tensor,
    max_neighbors: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    centers = torch.cat((support, query)).cpu()
    center_adjacency = first_neighbors(rowptr, col, centers, max_neighbors)
    neighbors = center_adjacency.reshape(-1)
    nodes = torch.cat((centers, neighbors[neighbors >= 0])).unique(sorted=True)
    global_adjacency = first_neighbors(rowptr, col, nodes, max_neighbors)
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


def save_checkpoint(path: Path, model, optimizer, scheduler, args, step: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "architecture": "vision",
            "model_id": args.model_id,
            "sources": list(args.selected_sources),
            "seed": args.seed,
            "step": step,
            "upstream": PINS["vision"],
            "pretraining_protocol": "vision_native_feature_similarity_pseudo_episodes",
            "source_sampling": "uniform_independent_per_episode",
            "optimizer_updates": args.steps,
            "episodes_per_update": TRAIN_BATCH_SIZE,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
        },
        path,
    )


def main() -> int:
    args = parse_args()
    args.selected_sources = tuple(value for value in args.sources.split(",") if value)
    checkpoints = {int(value) for value in args.checkpoint_steps.split(",") if value}
    if checkpoints != {100, 300, 900, 2500} or args.steps != 2500:
        raise ValueError("final-core match requires 2,500 updates and checkpoints 100,300,900,2500")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    run_dir = args.state_root / "vision" / args.run_name
    if run_dir.exists():
        raise FileExistsError(f"refusing ambiguous resume into {run_dir}")
    run_dir.mkdir(parents=True)
    metrics_path = run_dir / "train_metrics.jsonl"

    config = load_config(args.config)
    dataset = build_dataset(config)
    graph = dataset.graph
    names, node_sets = source_node_sets(graph, args.selected_sources)
    print(json.dumps({"source_counts": dict(zip(names, [int(x.numel()) for x in node_sets]))}), flush=True)

    print("Precomputing VISION adaptive task features on CPU...", flush=True)
    task_features = adaptive_vision_task_features(graph.x.float(), graph.edge_index.long())
    print("Adaptive task features ready.", flush=True)
    rowptr, col, _ = dataset.neighbor_sampler.whole_adj.csr()
    del graph.edge_index
    gc.collect()

    model = build_adapter("vision", args.upstream_root).to(device)
    optimizer = build_optimizer(model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=model.learning_rate * 0.05
    )
    labels = torch.arange(VISION_N_WAY, device=device).repeat_interleave(VISION_N_QUERY)
    support_labels = torch.arange(VISION_N_WAY, device=device).repeat_interleave(VISION_N_SHOT)
    started = time.time()
    source_episode_counts = [0] * len(names)

    with metrics_path.open("w", encoding="utf-8") as metrics:
        for step in range(1, args.steps + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            losses, accuracies = [], []
            for _ in range(TRAIN_BATCH_SIZE):
                source_index = random.randrange(len(names))
                source_episode_counts[source_index] += 1
                support, query = vision_pseudo_task_for_source(
                    task_features, node_sets[source_index], device
                )
                sub_x, sub_adj, local_support, local_query = vision_subgraph_from_csr(
                    graph.x, rowptr, col, support, query, model.max_neighbors
                )
                sub_x = sub_x.to(device)
                sub_adj = sub_adj.to(device)
                local_support = local_support.to(device)
                local_query = local_query.to(device)
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
                ce = F.cross_entropy(logits, labels, label_smoothing=0.1)
                con = torch.stack([
                    vision_contrastive_loss(q, s, labels, support_labels)
                    for q, s in contrastive
                ]).mean()
                loss = ce + model.contrastive_weight * con
                (loss / TRAIN_BATCH_SIZE).backward()
                losses.append(loss.detach())
                accuracies.append((logits.argmax(1) == labels).float().mean().detach())
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            row = {
                "step": step,
                "loss": float(torch.stack(losses).mean().cpu()),
                "accuracy": float(torch.stack(accuracies).mean().cpu()),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "elapsed_seconds": time.time() - started,
                "source_episode_counts": dict(zip(names, source_episode_counts)),
            }
            metrics.write(json.dumps(row, sort_keys=True) + "\n")
            metrics.flush()
            if step == 1 or step % 20 == 0:
                print(json.dumps(row, sort_keys=True), flush=True)
            if step in checkpoints:
                save_checkpoint(
                    run_dir / "checkpoint" / f"state_dict_{step}.pt",
                    model,
                    optimizer,
                    scheduler,
                    args,
                    step,
                )

    terminal = run_dir / "checkpoint" / "state_dict_2500.pt"
    if not terminal.is_file():
        raise RuntimeError(f"terminal checkpoint missing: {terminal}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
