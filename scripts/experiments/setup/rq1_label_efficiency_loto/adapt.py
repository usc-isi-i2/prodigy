#!/usr/bin/env python3
"""Resumable end-to-end PRODIGY encoder adaptation for one RQ1 cell."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch import nn
from torch_geometric.data import Batch

from models.layer_classes import (
    BackgroundGNNLayer,
    MetagraphLayer,
    SupernodeAggrLayer,
    SupernodeToBgGraphLayer,
)
from scripts.eval.pair_link_ckpt import ENCODER_DEFAULTS, build_encoder, load_frozen_encoder, load_graph_blob
from scripts.experiments.setup.adaptation_efficiency.extract_prodigy import classification_subgraph_dataset
from scripts.experiments.setup.adaptation_efficiency.protocol import (
    fingerprint_indices,
    sampled_labels,
    stratified_node_splits,
)
from scripts.experiments.setup.adaptation_efficiency.targets import TARGETS, load_labels
from scripts.experiments.setup.rq1_label_efficiency_loto.subgraph_cache import CompactCachedSubgraphDataset


class MemoizedSubgraphDataset:
    """Reuse a node's first sampled neighborhood within one adaptation cell.

    Validation already resets the sampler RNG to the same seed on every pass, so
    memoization is exactly equivalent there. Training intentionally changes from
    repeated stochastic sampling to a fixed sampled neighborhood per center node;
    this is recorded in every cell's metadata.
    """

    def __init__(self, dataset):
        self.dataset = dataset
        self.cache = {}
        self.enabled = True
        self.hits = 0
        self.misses = 0

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        index = int(index)
        if not self.enabled:
            return self.dataset[index]
        if index not in self.cache:
            self.cache[index] = self.dataset[index]
            self.misses += 1
        else:
            self.hits += 1
        return self.cache[index]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=sorted(TARGETS), required=True)
    parser.add_argument("--arm", choices=("scratch", "pretrained"), required=True)
    parser.add_argument("--pretrained-checkpoint", type=Path)
    parser.add_argument("--budget", type=int, choices=(1, 10, 100, 1000), required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument(
        "--label-seed",
        type=int,
        help="Override only labeled-example selection; all model/data RNGs remain --seed.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-updates", type=int, default=5000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--first-eval-update", type=int)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--min-updates", type=int, default=300)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--separate-selection-and-stopping", action="store_true")
    parser.add_argument("--protocol-version", default="cached-neighborhoods-patience4-v2")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--subgraph-cache", type=Path)
    parser.add_argument("--val-per-class", type=int, default=1000)
    parser.add_argument("--encoder-lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def atomic_torch_save(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def atomic_json(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def encode_graphs(model, graphs, device):
    graph = Batch.from_data_list(graphs).to(device)
    supernode_idx = graph.supernode + graph.ptr[:-1]
    graph.x = model.initial_input_mlp(graph.x)
    if model.txt_dropout is not None:
        graph.x = model.txt_dropout(graph.x)
    x_orig = graph.x.clone()
    x_input = None
    skip_path = bool(model.params.get("skip_path", False))
    for module in model.layer_list:
        if isinstance(module, MetagraphLayer):
            continue
        if isinstance(module, SupernodeAggrLayer):
            x_input = module.forward(
                graph.x, graph.edge_index_supernode, supernode_idx, graph.batch
            )
            graph.x = graph.x.clone()
            graph.x[supernode_idx] = x_input
        elif isinstance(module, BackgroundGNNLayer):
            new_x = module.forward(
                x_orig,
                graph.x,
                graph.edge_index.long(),
                graph.edge_attr if "edge_attr" in graph else None,
                graph.edge_index_supernode,
                graph.ptr[:-1],
                graph.batch,
            )
            graph.x = graph.x + new_x if skip_path and new_x.shape == graph.x.shape else new_x
        elif isinstance(module, SupernodeToBgGraphLayer):
            if x_input is None:
                raise RuntimeError("supernode-to-background layer precedes pooling")
            new_x = module.forward(
                graph.x, x_input, graph.edge_index_supernode, supernode_idx, graph.batch
            )
            graph.x = graph.x + new_x if skip_path else new_x
        else:
            raise TypeError(f"unknown PRODIGY layer: {type(module)}")
    if x_input is None:
        raise RuntimeError("PRODIGY encoder produced no pooled representation")
    return model.final_input_mlp(x_input)


def balanced_batch(labels, selected, batch_size, seed):
    rng = np.random.default_rng(seed)
    classes = sorted(int(value) for value in np.unique(labels[selected]))
    per_class = max(1, batch_size // len(classes))
    parts = []
    for class_id in classes:
        members = selected[labels[selected] == class_id]
        parts.append(rng.choice(members, size=per_class, replace=members.size < per_class))
    result = np.concatenate(parts).astype(np.int64)
    rng.shuffle(result)
    return result


def fixed_eval_nodes(labels, split_nodes, per_class, seed):
    rng = np.random.default_rng(seed)
    parts = []
    for class_id in sorted(int(value) for value in np.unique(labels[split_nodes])):
        members = split_nodes[labels[split_nodes] == class_id].copy()
        rng.shuffle(members)
        parts.append(members[: min(per_class, members.size)])
    return np.concatenate(parts).astype(np.int64)


@torch.no_grad()
def evaluate(model, head, dataset, labels, nodes, *, device, batch_size, sampling_seed):
    model.eval()
    head.eval()
    truth, scores, predictions = [], [], []
    for batch_index, start in enumerate(range(0, len(nodes), batch_size)):
        random.seed(sampling_seed + batch_index)
        np.random.seed(sampling_seed + batch_index)
        torch.manual_seed(sampling_seed + batch_index)
        chunk = nodes[start : start + batch_size]
        embeddings = encode_graphs(model, [dataset[int(node)] for node in chunk], device)
        logits = head(embeddings)
        probability = torch.softmax(logits, dim=1).cpu().numpy()
        truth.extend(labels[chunk].tolist())
        scores.extend(probability.tolist())
        predictions.extend(probability.argmax(1).tolist())
    truth = np.asarray(truth, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.int64)
    if scores.shape[1] == 2:
        auc = roc_auc_score(truth, scores[:, 1])
    else:
        auc = roc_auc_score(truth, scores, multi_class="ovr", average="macro")
    return {
        "roc_auc": float(auc),
        "accuracy": float(accuracy_score(truth, predictions)),
        "macro_f1": float(f1_score(truth, predictions, average="macro", zero_division=0)),
        "nodes": int(len(nodes)),
    }


def checkpoint_payload(model, head, optimizer, update, best_auc, progress_best_auc, bad_checks, metadata):
    return {
        "model": model.state_dict(),
        "head": head.state_dict(),
        "optimizer": optimizer.state_dict(),
        "update": int(update),
        "best_val_roc_auc": float(best_auc),
        "progress_best_val_roc_auc": float(progress_best_auc),
        "bad_checks": int(bad_checks),
        "metadata": metadata,
    }


def main() -> int:
    args = parse_args()
    if args.arm == "pretrained" and not args.pretrained_checkpoint:
        raise ValueError("pretrained arm requires --pretrained-checkpoint")
    if args.arm == "scratch" and args.pretrained_checkpoint:
        raise ValueError("scratch arm cannot receive --pretrained-checkpoint")
    result_path = args.output / "result.json"
    if result_path.is_file():
        print(f"SKIP completed {result_path}")
        return 0
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    target = TARGETS[args.target]
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    blob, graph = load_graph_blob(str(target.graph))
    labels = load_labels(blob, target.label_key)
    splits = stratified_node_splits(labels, seed=0)
    label_seed = args.seed if args.label_seed is None else args.label_seed
    selected = sampled_labels(labels, splits["train"], budget=args.budget, seed=label_seed)
    val_nodes = fixed_eval_nodes(labels, splits["val"], args.val_per_class, seed=17000 + args.seed)
    test_nodes = splits["test"]
    if args.smoke:
        val_nodes = fixed_eval_nodes(labels, splits["val"], 8, seed=17000 + args.seed)
        test_nodes = fixed_eval_nodes(labels, splits["test"], 8, seed=18000 + args.seed)
        args.max_updates = min(args.max_updates, 3)
        args.eval_every = 1
        args.patience = 2
        args.min_updates = 1
        args.batch_size = min(args.batch_size, 8)
        args.eval_batch_size = min(args.eval_batch_size, 8)
    raw_subgraphs = classification_subgraph_dataset(graph, 2, [9, 9], 101)
    if args.subgraph_cache:
        subgraphs = CompactCachedSubgraphDataset(raw_subgraphs, args.subgraph_cache)
    else:
        subgraphs = MemoizedSubgraphDataset(raw_subgraphs)
    params = dict(ENCODER_DEFAULTS)
    if args.arm == "pretrained":
        model = load_frozen_encoder(str(args.pretrained_checkpoint), params, device=str(device))
        checkpoint_hash = sha256_file(args.pretrained_checkpoint)
    else:
        model = build_encoder(params, device=str(device))
        checkpoint_hash = ""
    classes = len(set(int(value) for value in labels if int(value) >= 0))
    torch.manual_seed(10000 + args.seed)
    head = nn.Linear(int(params["emb_dim"]), classes).to(device)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.parameters(), "lr": args.encoder_lr},
            {"params": head.parameters(), "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )
    metadata = {
        "target": args.target,
        "arm": args.arm,
        "budget_per_class": args.budget,
        "seed": args.seed,
        "label_seed": label_seed,
        "selected_nodes_fingerprint": fingerprint_indices(selected),
        "split_fingerprint": fingerprint_indices(splits["train"], splits["val"], splits["test"]),
        "selected_nodes": selected.tolist(),
        "pretrained_checkpoint": str(args.pretrained_checkpoint or ""),
        "pretrained_checkpoint_sha256": checkpoint_hash,
        "encoder_lr": args.encoder_lr,
        "head_lr": args.head_lr,
        "weight_decay": args.weight_decay,
        "max_updates": args.max_updates,
        "eval_every": args.eval_every,
        "first_eval_update": args.first_eval_update,
        "patience": args.patience,
        "min_delta": args.min_delta,
        "separate_selection_and_stopping": args.separate_selection_and_stopping,
        "validation_nodes": int(len(val_nodes)),
        "test_nodes": int(len(test_nodes)),
        "sampled_neighborhood_cache": "first_sample_per_center_node_in_memory",
        "protocol_version": args.protocol_version,
        "shared_compact_subgraph_cache": str(args.subgraph_cache or ""),
    }
    atomic_json(metadata, args.output / "metadata.json")
    latest_path = args.output / "latest.pt"
    best_path = args.output / "best.pt"
    update = 0
    best_auc = float("-inf")
    progress_best_auc = float("-inf")
    bad_checks = 0
    if latest_path.is_file():
        state = torch.load(latest_path, map_location=device, weights_only=False)
        if state["metadata"] != metadata:
            raise ValueError("resume metadata differs from the registered cell")
        model.load_state_dict(state["model"])
        head.load_state_dict(state["head"])
        optimizer.load_state_dict(state["optimizer"])
        update = int(state["update"])
        best_auc = float(state["best_val_roc_auc"])
        progress_best_auc = float(state.get("progress_best_val_roc_auc", best_auc))
        bad_checks = int(state["bad_checks"])
        print(f"RESUME update={update} best_val={best_auc:.6f}", flush=True)
    trajectory_path = args.output / "trajectory.jsonl"
    started = time.time()
    stop_reason = "max_updates"
    while update < args.max_updates:
        update += 1
        random.seed(1000000 * args.seed + update)
        np.random.seed(1000000 * args.seed + update)
        torch.manual_seed(1000000 * args.seed + update)
        nodes = balanced_batch(labels, selected, args.batch_size, 2000000 * args.seed + update)
        model.train()
        head.train()
        optimizer.zero_grad(set_to_none=True)
        embeddings = encode_graphs(model, [subgraphs[int(node)] for node in nodes], device)
        targets = torch.as_tensor(labels[nodes], dtype=torch.long, device=device)
        loss = F.cross_entropy(head(embeddings), targets)
        loss.backward()
        optimizer.step()
        if args.first_eval_update is None:
            eval_due = update % args.eval_every == 0
        else:
            eval_due = update >= args.first_eval_update and (
                update - args.first_eval_update
            ) % args.eval_every == 0
        if not eval_due and update != args.max_updates:
            continue
        val = evaluate(
            model,
            head,
            subgraphs,
            labels,
            val_nodes,
            device=device,
            batch_size=args.eval_batch_size,
            sampling_seed=3000000 + args.seed,
        )
        raw_improved = val["roc_auc"] > best_auc
        if args.separate_selection_and_stopping:
            meaningful_improved = val["roc_auc"] > progress_best_auc + args.min_delta
        else:
            meaningful_improved = val["roc_auc"] > best_auc + args.min_delta
            raw_improved = meaningful_improved
        if raw_improved:
            best_auc = val["roc_auc"]
        if meaningful_improved:
            progress_best_auc = val["roc_auc"]
            bad_checks = 0
        else:
            bad_checks += 1
        payload = checkpoint_payload(
            model, head, optimizer, update, best_auc, progress_best_auc, bad_checks, metadata
        )
        atomic_torch_save(payload, latest_path)
        if raw_improved:
            atomic_torch_save(payload, best_path)
        row = {
            "update": update,
            "training_loss": float(loss.detach().cpu()),
            "val": val,
            "improved": raw_improved,
            "meaningful_improved": meaningful_improved,
            "best_val_roc_auc": best_auc,
            "bad_checks": bad_checks,
            "elapsed_seconds": time.time() - started,
        }
        with trajectory_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
        print(json.dumps(row, sort_keys=True), flush=True)
        if update >= args.min_updates and bad_checks >= args.patience:
            stop_reason = "validation_patience"
            break
    best = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    head.load_state_dict(best["head"])
    # The test set is evaluated once, so retaining every test neighborhood only
    # increases host RAM without enabling reuse.
    subgraphs.enabled = False
    test = evaluate(
        model,
        head,
        subgraphs,
        labels,
        test_nodes,
        device=device,
        batch_size=args.eval_batch_size,
        sampling_seed=4000000 + args.seed,
    )
    result = {
        **metadata,
        "selected_best_update": int(best["update"]),
        "selected_val_roc_auc": float(best["best_val_roc_auc"]),
        "updates_run": int(update),
        "stop_reason": stop_reason,
        "test": test,
        "elapsed_seconds": time.time() - started,
        "sample_cache_hits": int(subgraphs.hits),
        "sample_cache_misses": int(subgraphs.misses),
    }
    atomic_json(result, result_path)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
