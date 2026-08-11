#!/usr/bin/env python3
"""Train target-supervised MLP/GraphSAGE references and score fixed CLS episodes."""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch import nn
from torch_geometric.nn import SAGEConv

from scripts.experiments.setup.icl_arch_matrix.common_protocol import (
    EVAL_BATCH_SIZE,
    EVAL_EPISODES,
    EVAL_N_SHOT,
    EVAL_N_WAY,
    build_classification_dataset,
    build_classification_loader,
    classification_targets,
    iter_episodes,
    new_fingerprint,
    reset_episode_rng,
    update_episode_fingerprint,
)


BASELINES = ("supervised_mlp", "supervised_graphsage")
DEFAULT_LR_GRID = (1e-3, 3e-4)
TRAIN_UPDATES = 100
TRAIN_BATCH_SIZE = 4
VAL_EPISODES = 32


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=BASELINES, required=True)
    parser.add_argument("--data-root", default="/dataMeR1/phil/data")
    parser.add_argument("--catalog", default="docs/graph_catalog.json")
    parser.add_argument("--results", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--datasets", default="")
    parser.add_argument("--train-updates", type=int, default=TRAIN_UPDATES)
    parser.add_argument("--val-episodes", type=int, default=VAL_EPISODES)
    parser.add_argument("--lr-grid", default=",".join(map(str, DEFAULT_LR_GRID)))
    return parser.parse_args()


class SupervisedMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x, edge_index, centers):
        del edge_index
        return self.net(x[centers])


class SupervisedGraphSAGE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, x, edge_index, centers):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        return self.classifier(x[centers])


def _new_model(model_name: str, input_dim: int) -> nn.Module:
    if model_name == "supervised_mlp":
        return SupervisedMLP(input_dim)
    if model_name == "supervised_graphsage":
        return SupervisedGraphSAGE(input_dim)
    raise ValueError(model_name)


def _build_split_loader(
    *, dataset, get_dataloader, graph_path, target, split: str, batch_count: int, workers: int
):
    return get_dataloader(
        dataset,
        split=split,
        node_split="",
        batch_size=TRAIN_BATCH_SIZE if split == "train" else EVAL_BATCH_SIZE,
        n_way=EVAL_N_WAY,
        n_shot=EVAL_N_SHOT,
        n_query=int(target["n_query"]),
        batch_count=batch_count,
        root=str(graph_path.parent),
        bert=None,
        num_workers=workers,
        aug="",
        aug_test=False,
        split_labels=False,
        train_cap=None,
        linear_probe=False,
        task_name="classification",
        eval_random_query=target["eval_random_query"],
        eval_episode_seed_offset=0,
        seed=0,
    )


def _episode_logits(model, episode, device):
    x = episode.x.to(device)
    edge_index = episode.edge_index.to(device)
    centers = episode.centers.to(device)
    labels = episode.label_map[episode.labels].long().to(device)
    return model(x, edge_index, centers), labels


def train_candidate(
    model_name: str,
    *,
    input_dim: int,
    lr: float,
    dataset,
    get_dataloader,
    graph_path,
    target,
    train_updates: int,
    workers: int,
    device,
):
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    model = _new_model(model_name, input_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loader = _build_split_loader(
        dataset=dataset,
        get_dataloader=get_dataloader,
        graph_path=graph_path,
        target=target,
        split="train",
        batch_count=train_updates,
        workers=workers,
    )
    reset_episode_rng()
    updates = 0
    labeled_centers_seen = 0
    last_loss = float("nan")
    model.train()
    for batch in loader:
        episodes = list(
            iter_episodes(
                batch,
                n_way=EVAL_N_WAY,
                n_shot=EVAL_N_SHOT,
                n_query=int(target["n_query"]),
                equal_query_counts=True,
            )
        )
        optimizer.zero_grad(set_to_none=True)
        batch_loss = 0.0
        for episode in episodes:
            logits, labels = _episode_logits(model, episode, device)
            loss = F.cross_entropy(logits, labels) / len(episodes)
            loss.backward()
            batch_loss += float(loss.detach().cpu())
            labeled_centers_seen += int(labels.numel())
        optimizer.step()
        updates += 1
        last_loss = batch_loss
    if updates != train_updates:
        raise RuntimeError(f"expected {train_updates} updates, got {updates}")
    return model, labeled_centers_seen, last_loss


@torch.no_grad()
def evaluate_model(
    model,
    loader,
    *,
    n_query: int,
    equal_query_counts: bool,
    device,
    fingerprint_episodes: bool,
):
    labels, scores, predictions = [], [], []
    fingerprint = new_fingerprint()
    episode_count = 0
    model.eval()
    reset_episode_rng()
    for batch in loader:
        for episode in iter_episodes(
            batch,
            n_way=EVAL_N_WAY,
            n_shot=EVAL_N_SHOT,
            n_query=n_query,
            equal_query_counts=equal_query_counts,
        ):
            logits, target_global = _episode_logits(model, episode, device)
            query_logits = logits[episode.query_mask.to(device)]
            query_targets = target_global[episode.query_mask.to(device)]
            probabilities = torch.softmax(query_logits, dim=1)[:, 1]
            labels.extend(query_targets.cpu().tolist())
            scores.extend(probabilities.cpu().tolist())
            predictions.extend(query_logits.argmax(1).cpu().tolist())
            if fingerprint_episodes:
                update_episode_fingerprint(fingerprint, episode)
            episode_count += 1
    y_true = np.asarray(labels, dtype=np.int64)
    y_score = np.asarray(scores, dtype=np.float64)
    y_pred = np.asarray(predictions, dtype=np.int64)
    return {
        "episodes": episode_count,
        "queries": int(y_true.size),
        "episode_fingerprint": fingerprint.hexdigest() if fingerprint_episodes else "",
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("supervised target controls require a Tucker GPU")
    device = torch.device(f"cuda:{args.device}")
    result_path = Path(args.results)
    if result_path.exists():
        raise FileExistsError(f"refusing to overwrite results: {result_path}")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    lr_grid = tuple(float(value) for value in args.lr_grid.split(",") if value)
    if not lr_grid or any(value <= 0 for value in lr_grid):
        raise ValueError(f"invalid learning-rate grid: {args.lr_grid}")

    targets = classification_targets(args.catalog)
    selected = set(filter(None, args.datasets.split(",")))
    if selected:
        missing = selected - targets.keys()
        if missing:
            raise ValueError(f"unknown classification datasets: {sorted(missing)}")
        targets = {name: target for name, target in targets.items() if name in selected}

    with result_path.open("w", encoding="utf-8") as handle:
        for dataset_name, target in targets.items():
            dataset, get_dataloader, graph_path = build_classification_dataset(
                dataset_name=dataset_name,
                data_root=args.data_root,
                target=target,
            )
            input_dim = int(dataset.graph.x.shape[1])
            best = None
            tuning_rows = []
            for lr in lr_grid:
                model, labeled_centers_seen, train_loss = train_candidate(
                    args.model,
                    input_dim=input_dim,
                    lr=lr,
                    dataset=dataset,
                    get_dataloader=get_dataloader,
                    graph_path=graph_path,
                    target=target,
                    train_updates=args.train_updates,
                    workers=args.workers,
                    device=device,
                )
                val_loader = _build_split_loader(
                    dataset=dataset,
                    get_dataloader=get_dataloader,
                    graph_path=graph_path,
                    target=target,
                    split="val",
                    batch_count=args.val_episodes // EVAL_BATCH_SIZE,
                    workers=args.workers,
                )
                val_metrics = evaluate_model(
                    model,
                    val_loader,
                    n_query=int(target["n_query"]),
                    equal_query_counts=not target["eval_random_query"],
                    device=device,
                    fingerprint_episodes=False,
                )
                tuning_rows.append({"lr": lr, "val_roc_auc": val_metrics["roc_auc"]})
                candidate = {
                    "val_roc_auc": val_metrics["roc_auc"],
                    "lr": lr,
                    "state_dict": copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()}),
                    "labeled_centers_seen": labeled_centers_seen,
                    "train_loss": train_loss,
                }
                if best is None or (candidate["val_roc_auc"], -candidate["lr"]) > (
                    best["val_roc_auc"], -best["lr"]
                ):
                    best = candidate
                del model
                torch.cuda.empty_cache()

            model = _new_model(args.model, input_dim).to(device)
            model.load_state_dict(best["state_dict"])
            test_loader = build_classification_loader(
                dataset_name=dataset_name,
                data_root=args.data_root,
                target=target,
                dataset=dataset,
                get_dataloader=get_dataloader,
                graph_path=graph_path,
                workers=args.workers,
            )
            test_metrics = evaluate_model(
                model,
                test_loader,
                n_query=int(target["n_query"]),
                equal_query_counts=not target["eval_random_query"],
                device=device,
                fingerprint_episodes=True,
            )
            split_sizes = {
                name: int(len(indices))
                for name, indices in dataset._classification_node_splits.items()
            }
            row = {
                "baseline": args.model,
                "model_id": args.model,
                "sources": [],
                "seed": 0,
                "training_updates": int(args.train_updates),
                "training_label_scope": "target_train_split",
                "validation_selection": "best_of_two_fixed_lrs_at_update100",
                "lr_grid": list(lr_grid),
                "selected_lr": float(best["lr"]),
                "selected_val_roc_auc": float(best["val_roc_auc"]),
                "tuning_results": tuning_rows,
                "train_loss_final": float(best["train_loss"]),
                "labeled_centers_seen": int(best["labeled_centers_seen"]),
                "node_split_sizes": split_sizes,
                "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
                "raw_features_used": True,
                "topology_used": args.model == "supervised_graphsage",
                "test_support_labels_used": False,
                "query_labels_used_for_selection": False,
                "task": "classification",
                "dataset": dataset_name,
                "n_way": EVAL_N_WAY,
                "n_shot": EVAL_N_SHOT,
                "n_query": int(target["n_query"]),
                **test_metrics,
            }
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            print(json.dumps(row, sort_keys=True), flush=True)
            del model, dataset
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
