#!/usr/bin/env python3
"""Evaluate VISION specialists on fixed label-free feature-similarity pseudo-tasks."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from scripts.experiments.setup.icl_arch_matrix.architecture_adapters import (
    build_adapter,
    padded_adjacency,
    vision_contrastive_loss,
)
from scripts.experiments.setup.icl_arch_matrix.common_protocol import (
    build_classification_dataset,
    classification_targets,
)
from scripts.experiments.setup.icl_arch_matrix.train_native_source_model import (
    VISION_N_QUERY,
    VISION_N_SHOT,
    VISION_N_WAY,
    adaptive_vision_task_features,
    vision_pseudo_task,
    vision_subgraph,
)


MODELS = (
    ("ss_covid_political", "covid_political"),
    ("ss_election2020", "election2020"),
    ("ss_ukr_rus_suspended", "ukr_rus_suspended"),
    ("ss_twibot20", "twibot20"),
    ("ss_facebook_page_reference", "facebook_page_reference"),
)
CHECKPOINTS = (20, 60, 100, 300, 900)


def episode_seed(target: str, base_seed: int) -> int:
    return int(base_seed + sum((index + 1) * ord(value) for index, value in enumerate(target)))


def update_fingerprint(digest, support: torch.Tensor, query: torch.Tensor) -> None:
    for values in (support, query):
        array = values.detach().cpu().to(torch.int64).contiguous().numpy()
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())


def build_fixed_episodes(
    task_features: torch.Tensor, *, episodes: int, seed: int
) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], str]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    selected = []
    fingerprint = hashlib.sha256()
    for _ in range(episodes):
        support, query = vision_pseudo_task(task_features)
        update_fingerprint(fingerprint, support, query)
        selected.append((support, query))
    return selected, fingerprint.hexdigest()


@torch.no_grad()
def evaluate_checkpoint(
    model,
    x: torch.Tensor,
    full_adjacency: torch.Tensor,
    *,
    fixed_episodes: list[tuple[torch.Tensor, torch.Tensor]],
    fingerprint: str,
) -> dict[str, float | int | str]:
    model.eval()
    device = x.device
    query_labels = torch.arange(VISION_N_WAY, device=device).repeat_interleave(VISION_N_QUERY)
    support_labels = torch.arange(VISION_N_WAY, device=device).repeat_interleave(VISION_N_SHOT)
    losses, cross_entropies, contrastive_losses, accuracies = [], [], [], []
    for support, query in fixed_episodes:
        sub_x, sub_adj, local_support, local_query = vision_subgraph(
            x, full_adjacency, support, query
        )
        sub_x = sub_x - sub_x.mean(dim=0, keepdim=True)
        logits, contrastive = model.model(
            sub_x,
            sub_adj,
            local_support,
            local_query,
            drop_label_prob=0.0,
            input_noise_std=0.0,
        )
        cross_entropy = F.cross_entropy(logits, query_labels)
        contrastive_loss = torch.stack(
            [
                vision_contrastive_loss(q, s, query_labels, support_labels)
                for q, s in contrastive
            ]
        ).mean()
        loss = cross_entropy + model.contrastive_weight * contrastive_loss
        losses.append(float(loss))
        cross_entropies.append(float(cross_entropy))
        contrastive_losses.append(float(contrastive_loss))
        accuracies.append(float((logits.argmax(1) == query_labels).float().mean()))
    return {
        "episodes": len(fixed_episodes),
        "queries": len(fixed_episodes) * VISION_N_WAY * VISION_N_QUERY,
        "episode_fingerprint": fingerprint,
        "native_ssl_loss": float(np.mean(losses)),
        "pseudo_classification_accuracy": float(np.mean(accuracies)),
        "cross_entropy": float(np.mean(cross_entropies)),
        "contrastive_loss": float(np.mean(contrastive_losses)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--datasets", default="")
    parser.add_argument("--catalog", default="docs/graph_catalog.json")
    parser.add_argument("--data-root", default="/dataMeR1/phil/data")
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--episode-seed", type=int, default=0)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    if args.results.exists():
        raise FileExistsError(f"refusing to overwrite {args.results}")
    if args.episodes != 128:
        raise ValueError("registered cross-SSL protocol requires 128 pseudo-episodes")

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    targets = classification_targets(args.catalog, include_facebook=True)
    selected = tuple(value for value in args.datasets.split(",") if value) or tuple(targets)
    unknown = set(selected) - set(targets)
    if unknown:
        raise ValueError(f"unknown target graphs: {sorted(unknown)}")
    args.results.parent.mkdir(parents=True, exist_ok=True)

    with args.results.open("w", encoding="utf-8") as handle:
        for target_name in selected:
            target = targets[target_name]
            dataset, _, _ = build_classification_dataset(
                dataset_name=target_name,
                data_root=args.data_root,
                target=target,
            )
            graph = dataset.graph
            x = graph.x.float().to(device)
            edge_index = graph.edge_index.long().to(device)
            task_features = adaptive_vision_task_features(x, edge_index)
            seed = episode_seed(target_name, args.episode_seed)
            fixed_episodes, fingerprint = build_fixed_episodes(
                task_features, episodes=args.episodes, seed=seed
            )
            prototype = build_adapter("vision", args.upstream_root)
            full_adjacency = padded_adjacency(
                edge_index, x.size(0), prototype.max_neighbors
            )
            del prototype
            for model_id, source in MODELS:
                for step in CHECKPOINTS:
                    checkpoint_path = (
                        args.state_root
                        / "vision"
                        / f"{model_id}_s0"
                        / "checkpoint"
                        / f"state_dict_{step}.pt"
                    )
                    checkpoint = torch.load(checkpoint_path, map_location="cpu")
                    if checkpoint.get("pretraining_protocol") != "vision_native_feature_similarity_pseudo_episodes":
                        raise ValueError(f"non-native checkpoint: {checkpoint_path}")
                    model = build_adapter("vision", args.upstream_root).to(device)
                    model.load_state_dict(checkpoint["model_state"], strict=True)
                    metrics = evaluate_checkpoint(
                        model,
                        x,
                        full_adjacency,
                        fixed_episodes=fixed_episodes,
                        fingerprint=fingerprint,
                    )
                    row = {
                        "architecture": "vision",
                        "native_pretext": "feature_similarity_pseudo_episodes",
                        "task": "native_feature_similarity_ssl",
                        "model_id": model_id,
                        "source": source,
                        "target": target_name,
                        "training_seed": 0,
                        "checkpoint_step": step,
                        "eval_episode_seed": seed,
                        "compute_regime": "fixed_compute_900_trajectory",
                        **metrics,
                    }
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                    handle.flush()
                    print(json.dumps(row, sort_keys=True), flush=True)
                    del checkpoint, model
                    torch.cuda.empty_cache()
            del dataset, graph, x, edge_index, task_features, fixed_episodes, full_adjacency
            gc.collect()
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
