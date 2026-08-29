#!/usr/bin/env python3
"""Evaluate an untrained target head on scratch or LOTO-pretrained encoders."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch import nn

from scripts.eval.pair_link_ckpt import ENCODER_DEFAULTS, build_encoder, load_frozen_encoder, load_graph_blob
from scripts.experiments.setup.adaptation_efficiency.extract_prodigy import classification_subgraph_dataset
from scripts.experiments.setup.adaptation_efficiency.protocol import fingerprint_indices, stratified_node_splits
from scripts.experiments.setup.adaptation_efficiency.targets import TARGETS, load_labels
from scripts.experiments.setup.rq1_label_efficiency_loto.adapt import encode_graphs
from scripts.experiments.setup.rq1_label_efficiency_loto.subgraph_cache import CompactCachedSubgraphDataset


def atomic_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=sorted(TARGETS), required=True)
    parser.add_argument("--arm", choices=("scratch", "pretrained"), required=True)
    parser.add_argument("--pretrained-checkpoint", type=Path)
    parser.add_argument("--training-seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--head-seeds", default="0,1,2,3,4")
    parser.add_argument("--subgraph-cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if (args.arm == "pretrained") != bool(args.pretrained_checkpoint):
        raise ValueError("only pretrained requires --pretrained-checkpoint")
    if args.output.is_file():
        print(f"SKIP {args.output}")
        return 0

    started = time.time()
    device = torch.device(args.device)
    random.seed(args.training_seed)
    np.random.seed(args.training_seed)
    torch.manual_seed(args.training_seed)
    target = TARGETS[args.target]
    blob, graph = load_graph_blob(str(target.graph))
    labels = load_labels(blob, target.label_key)
    splits = stratified_node_splits(labels, seed=0)
    test_nodes = splits["test"]
    raw = classification_subgraph_dataset(graph, 2, [9, 9], 101)
    dataset = CompactCachedSubgraphDataset(raw, args.subgraph_cache) if args.subgraph_cache else raw
    params = dict(ENCODER_DEFAULTS)
    model = (
        load_frozen_encoder(str(args.pretrained_checkpoint), params, device=str(device))
        if args.arm == "pretrained"
        else build_encoder(params, device=str(device))
    )
    model.eval()
    chunks = []
    with torch.no_grad():
        for batch_index, start in enumerate(range(0, len(test_nodes), args.batch_size)):
            random.seed(4_000_000 + args.training_seed + batch_index)
            np.random.seed(4_000_000 + args.training_seed + batch_index)
            torch.manual_seed(4_000_000 + args.training_seed + batch_index)
            nodes = test_nodes[start : start + args.batch_size]
            chunks.append(encode_graphs(model, [dataset[int(node)] for node in nodes], device).cpu())
    embeddings = torch.cat(chunks)
    truth = labels[test_nodes]
    classes = len(set(int(value) for value in labels if int(value) >= 0))
    rows = []
    for head_seed in (int(value) for value in args.head_seeds.split(",") if value):
        torch.manual_seed(10_000 + head_seed)
        head = nn.Linear(int(params["emb_dim"]), classes)
        with torch.no_grad():
            probability = torch.softmax(head(embeddings), dim=1).numpy()
        prediction = probability.argmax(1)
        auc = (
            roc_auc_score(truth, probability[:, 1])
            if classes == 2
            else roc_auc_score(truth, probability, multi_class="ovr", average="macro")
        )
        rows.append({
            "head_seed": head_seed,
            "roc_auc": float(auc),
            "accuracy": float(accuracy_score(truth, prediction)),
            "macro_f1": float(f1_score(truth, prediction, average="macro", zero_division=0)),
        })
    result = {
        "protocol_version": "rq1-zero-label-untrained-random-head-v1",
        "interpretation": "sanity check only; no target-label mapping is learned",
        "target": args.target,
        "arm": args.arm,
        "training_seed": args.training_seed,
        "label_budget_per_class": 0,
        "optimizer_updates": 0,
        "pretrained_checkpoint": str(args.pretrained_checkpoint or ""),
        "split_fingerprint": fingerprint_indices(splits["train"], splits["val"], splits["test"]),
        "test_nodes": int(len(test_nodes)),
        "head_results": rows,
        "mean_test_roc_auc": float(np.mean([row["roc_auc"] for row in rows])),
        "std_test_roc_auc": float(np.std([row["roc_auc"] for row in rows], ddof=1)),
        "elapsed_seconds": time.time() - started,
    }
    atomic_json(result, args.output)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
