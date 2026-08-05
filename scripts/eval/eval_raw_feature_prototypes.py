#!/usr/bin/env python3
"""Evaluate raw node features on PRODIGY's exact classification episodes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from data.dataloader import BatchSampler, MulticlassTask, ParamSampler
from data.midterm import _build_stratified_node_splits, _mask_labels_to_node_split
from data.ukr_rus_twitter import _episode_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--shots", type=int, default=10)
    parser.add_argument("--queries", type=int, default=4)
    parser.add_argument("--seed-offset", type=int, default=0)
    return parser.parse_args()


def load_graph(path: Path) -> tuple[torch.Tensor, np.ndarray, list[str]]:
    raw = torch.load(path, map_location="cpu")
    if not isinstance(raw, dict):
        raise TypeError(f"Expected dict graph artifact, got {type(raw).__name__}")
    features = raw["x"].detach().cpu().float().contiguous()
    labels = raw["y"].detach().cpu().numpy().astype(np.int64, copy=False)
    label_names = [str(value) for value in raw.get("label_names", [])]
    if features.shape[0] != labels.shape[0]:
        raise ValueError("Feature and label node counts differ")
    return features, labels, label_names


def main() -> None:
    args = parse_args()
    features, labels, label_names = load_graph(args.graph)
    classes = sorted(int(value) for value in np.unique(labels) if int(value) >= 0)
    if classes != list(range(len(classes))):
        raise ValueError(f"Expected contiguous non-negative classes, got {classes}")
    if label_names and len(label_names) != len(classes):
        raise ValueError("label_names length does not match observed classes")

    split = _build_stratified_node_splits(labels, seed=0)
    masked = _mask_labels_to_node_split(labels, split["test"])
    task = MulticlassTask(masked, set(classes))
    sampler = BatchSampler(
        args.episodes,
        task,
        ParamSampler(1, len(classes), args.shots, args.queries, 1),
        seed=_episode_seed("test", args.seed_offset),
    )
    normalized = torch.nn.functional.normalize(features, dim=1)

    rows: list[dict[str, float | int]] = []
    manifest: list[dict[str, object]] = []
    all_true: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    for episode_index, (batch, _) in enumerate(sampler):
        sampled = batch[0]
        class_order = [int(value) for value in sampled]
        support_idx: list[int] = []
        support_y: list[int] = []
        query_idx: list[int] = []
        query_y: list[int] = []
        for class_id in class_order:
            chosen = sampled[class_id]
            support_idx.extend(int(value) for value in chosen[: args.shots])
            support_y.extend([class_id] * args.shots)
            query_idx.extend(int(value) for value in chosen[args.shots :])
            query_y.extend([class_id] * args.queries)

        support = normalized[torch.tensor(support_idx)]
        query = normalized[torch.tensor(query_idx)]
        support_y_array = np.asarray(support_y, dtype=np.int64)
        query_y_array = np.asarray(query_y, dtype=np.int64)
        prototypes = torch.stack(
            [support[torch.from_numpy(support_y_array == class_id)].mean(0) for class_id in classes]
        )
        prototypes = torch.nn.functional.normalize(prototypes, dim=1)
        probabilities = torch.softmax(query @ prototypes.T, dim=1).numpy()
        predictions = np.asarray(classes)[probabilities.argmax(1)]
        episode_auc = roc_auc_score(
            query_y_array,
            probabilities,
            labels=classes,
            multi_class="ovr",
            average="macro",
        )
        rows.append(
            {
                "episode": episode_index,
                "accuracy": float(accuracy_score(query_y_array, predictions)),
                "macro_f1": float(f1_score(query_y_array, predictions, average="macro")),
                "roc_auc": float(episode_auc),
            }
        )
        manifest.append(
            {
                "episode": episode_index,
                "class_order": class_order,
                "support_idx": support_idx,
                "support_y": support_y,
                "query_idx": query_idx,
                "query_y": query_y,
            }
        )
        all_true.append(query_y_array)
        all_probs.append(probabilities)
        all_pred.append(predictions)

    y_true = np.concatenate(all_true)
    probabilities = np.concatenate(all_probs)
    predictions = np.concatenate(all_pred)
    summary = {
        "graph": str(args.graph.resolve()),
        "protocol": {
            "split_seed": 0,
            "episode_seed": _episode_seed("test", args.seed_offset),
            "seed_offset": args.seed_offset,
            "episodes": args.episodes,
            "ways": len(classes),
            "shots": args.shots,
            "queries": args.queries,
            "readout": "raw_feature_cosine_prototypes",
        },
        "metrics": {
            "test_accuracy": float(accuracy_score(y_true, predictions)),
            "test_f1": float(f1_score(y_true, predictions, average="macro")),
            "test_roc_auc": float(
                roc_auc_score(
                    y_true,
                    probabilities,
                    labels=classes,
                    multi_class="ovr",
                    average="macro",
                )
            ),
            "episode_accuracy_std": float(np.std([row["accuracy"] for row in rows])),
            "episode_roc_auc_std": float(np.std([row["roc_auc"] for row in rows])),
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "metrics.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    with (args.output_dir / "episode_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "episode_manifest.json").open("w") as handle:
        json.dump(manifest, handle)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
