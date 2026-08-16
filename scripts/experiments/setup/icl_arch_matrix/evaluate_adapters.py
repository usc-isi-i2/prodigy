#!/usr/bin/env python3
"""Evaluate every VISION or GILT matrix model on the shared CLS episodes."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from scripts.experiments.setup.final_core.core_plan import build_models
from scripts.experiments.setup.icl_arch_matrix.architecture_adapters import build_adapter
from scripts.experiments.setup.icl_arch_matrix.common_protocol import (
    EVAL_N_SHOT,
    EVAL_N_WAY,
    TRAIN_STEPS,
    build_classification_dataset,
    build_classification_loader,
    classification_targets,
    iter_episodes,
    new_fingerprint,
    reset_episode_rng,
    update_episode_fingerprint,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", choices=("vision", "gilt"), required=True)
    parser.add_argument("--upstream-root", required=True)
    parser.add_argument("--state-root", default="")
    parser.add_argument("--data-root", default="/dataMeR1/phil/data")
    parser.add_argument("--catalog", default="docs/graph_catalog.json")
    parser.add_argument("--results", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--model-ids", default="")
    parser.add_argument("--datasets", default="")
    parser.add_argument("--checkpoint-step", type=int, choices=(20, 60, 100), default=TRAIN_STEPS)
    parser.add_argument(
        "--random-init",
        action="store_true",
        help="Evaluate one deterministically initialized, untrained model.",
    )
    return parser.parse_args()


def evaluate_model(model, loader, device, *, n_query: int, equal_query_counts: bool):
    labels, scores, predictions = [], [], []
    fingerprint = new_fingerprint()
    episode_count = 0
    model.eval()
    reset_episode_rng()
    with torch.no_grad():
        for batch in loader:
            graphs = batch[0].to(device)
            moved = (graphs,) + tuple(
                value.to(device) if torch.is_tensor(value) else value
                for value in batch[1:]
            )
            for episode in iter_episodes(
                moved,
                n_way=EVAL_N_WAY,
                n_shot=EVAL_N_SHOT,
                n_query=n_query,
                equal_query_counts=equal_query_counts,
            ):
                output = model.episode_logits(episode)
                logits = output[0] if isinstance(output, tuple) else output
                target_local = episode.labels[episode.query_mask]
                probability_local = torch.softmax(logits, dim=1)
                target_global = episode.label_map[target_local]
                global_class1_local_index = torch.where(episode.label_map == 1)[0]
                if global_class1_local_index.numel() != 1:
                    raise ValueError(f"expected one global class 1: {episode.label_map.tolist()}")
                probability_global1 = probability_local[:, global_class1_local_index.item()]
                prediction_global = episode.label_map[logits.argmax(1)]
                labels.extend(target_global.detach().cpu().tolist())
                scores.extend(probability_global1.detach().cpu().tolist())
                predictions.extend(prediction_global.detach().cpu().tolist())
                update_episode_fingerprint(fingerprint, episode)
                episode_count += 1
    y_true = np.asarray(labels, dtype=np.int64)
    y_score = np.asarray(scores, dtype=np.float64)
    y_pred = np.asarray(predictions, dtype=np.int64)
    return {
        "episodes": episode_count,
        "queries": int(y_true.size),
        "episode_fingerprint": fingerprint.hexdigest(),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def main() -> int:
    args = parse_args()
    torch.set_num_threads(16)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    selected = set(filter(None, args.model_ids.split(",")))
    if args.random_init:
        if args.checkpoint_step != TRAIN_STEPS:
            raise ValueError("--checkpoint-step cannot be combined with --random-init")
        if selected:
            raise ValueError("--model-ids cannot be combined with --random-init")
        models = [SimpleNamespace(model_id="random_init", sources=())]
    else:
        if not args.state_root:
            raise ValueError("--state-root is required unless --random-init is used")
        models = [model for model in build_models() if not selected or model.model_id in selected]
        if selected != {model.model_id for model in models} and selected:
            missing = selected - {model.model_id for model in models}
            raise ValueError(f"unknown model ids: {sorted(missing)}")

    result_path = Path(args.results)
    if result_path.exists():
        raise FileExistsError(f"refusing to overwrite results: {result_path}")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    targets = classification_targets(args.catalog)
    selected_datasets = set(filter(None, args.datasets.split(",")))
    if selected_datasets:
        missing = selected_datasets - targets.keys()
        if missing:
            raise ValueError(f"unknown classification datasets: {sorted(missing)}")
        targets = {name: target for name, target in targets.items() if name in selected_datasets}
    expected_fingerprints = {}

    with result_path.open("w", encoding="utf-8") as handle:
        for dataset_name, target in targets.items():
            dataset, get_dataloader, graph_path = build_classification_dataset(
                dataset_name=dataset_name,
                data_root=args.data_root,
                target=target,
            )
            for plan_model in models:
                torch.manual_seed(0)
                torch.cuda.manual_seed_all(0)
                model = build_adapter(args.architecture, args.upstream_root)
                checkpoint = None
                checkpoint_step = 0 if args.random_init else args.checkpoint_step
                if not args.random_init:
                    checkpoint_path = (
                        Path(args.state_root)
                        / args.architecture
                        / plan_model.model_id
                        / "checkpoint"
                        / f"state_dict_{checkpoint_step}.pt"
                    )
                    checkpoint = torch.load(checkpoint_path, map_location="cpu")
                    model.load_state_dict(checkpoint["model_state"], strict=True)
                model.to(device)
                loader = build_classification_loader(
                    dataset_name=dataset_name,
                    data_root=args.data_root,
                    target=target,
                    dataset=dataset,
                    get_dataloader=get_dataloader,
                    graph_path=graph_path,
                    workers=args.workers,
                )
                metrics = evaluate_model(
                    model,
                    loader,
                    device,
                    n_query=int(target["n_query"]),
                    equal_query_counts=not target["eval_random_query"],
                )
                prior = expected_fingerprints.setdefault(
                    dataset_name, metrics["episode_fingerprint"]
                )
                if metrics["episode_fingerprint"] != prior:
                    raise RuntimeError(
                        f"episode drift on {dataset_name}: {metrics['episode_fingerprint']} != {prior}"
                    )
                row = {
                    "architecture": args.architecture,
                    "model_id": plan_model.model_id,
                    "sources": list(plan_model.sources),
                    "seed": 0,
                    "checkpoint_step": checkpoint_step,
                    "baseline": "random_init" if args.random_init else "pretrained",
                    "task": "classification",
                    "dataset": dataset_name,
                    "n_way": EVAL_N_WAY,
                    "n_shot": EVAL_N_SHOT,
                    "n_query": int(target["n_query"]),
                    **metrics,
                }
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                handle.flush()
                print(json.dumps(row, sort_keys=True), flush=True)
                del model, checkpoint, loader
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
