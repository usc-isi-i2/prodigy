#!/usr/bin/env python3
"""Tune and train a strong, topology-free MLP classification baseline."""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from scripts.experiments.setup.adaptation_efficiency.protocol import (
    sampled_labels,
    standardize_and_pad,
    stratified_node_splits,
)
from scripts.experiments.setup.adaptation_efficiency.targets import (
    graph_field,
    labeled_nodes,
    load_graph,
    load_labels,
    selected_targets,
)


DEFAULT_BUDGETS = (10, 100, 500, 1000, -1)  # -1 means the complete train split.
DEFAULT_SEEDS = (0, 1, 2, 3, 4)


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden: tuple[int, ...], dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_dim
        for width in hidden:
            layers.extend((nn.Linear(previous, width), nn.LayerNorm(width), nn.GELU()))
            if dropout:
                layers.append(nn.Dropout(dropout))
            previous = width
        layers.append(nn.Linear(previous, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values)


def parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--targets", default="covid_political,election2020,ukr_rus_suspended,twibot20")
    parser.add_argument("--budgets", default=",".join(map(str, DEFAULT_BUDGETS)))
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--min-epochs", type=int, default=30)
    parser.add_argument("--wandb-project", default="strong-raw-mlp")
    return parser.parse_args()


def metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    probability = logits.softmax(1)[:, 1].detach().cpu().numpy()
    truth = labels.detach().cpu().numpy()
    prediction = (probability >= 0.5).astype(np.int64)
    return {
        "roc_auc": float(roc_auc_score(truth, probability)),
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, average="macro", zero_division=0)),
        "f1": float(f1_score(truth, prediction, zero_division=0)),
    }


def fingerprint(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        values = np.asarray(array, dtype=np.int64)
        digest.update(values.tobytes())
    return digest.hexdigest()


def log_wandb_run(
    *, project: str, output: Path, name: str, config: dict[str, object],
    curve: list[dict[str, float]], summary: dict[str, object],
) -> None:
    """Replay the canonical curve into a self-contained offline W&B run."""
    import wandb

    run = wandb.init(project=project, name=name, config=config, dir=str(output / "wandb"), reinit=True)
    for values in curve:
        run.log(values, step=int(values["epoch"]))
    for key, value in summary.items():
        if isinstance(value, (str, int, float, bool)):
            run.summary[key] = value
    run.finish()


def hyperparameters() -> list[dict[str, object]]:
    # Compact but meaningful grid: capacity, regularization, and optimizer scale.
    architectures = [((256,), 0.2), ((512,), 0.2), ((512, 256), 0.2), ((512, 256), 0.5)]
    optimizers = [(1e-3, 1e-4), (3e-4, 1e-4), (1e-3, 1e-3)]
    return [
        {"hidden": list(hidden), "dropout": dropout, "learning_rate": lr, "weight_decay": wd}
        for (hidden, dropout), (lr, wd) in itertools.product(architectures, optimizers)
    ]


def train_once(
    x: torch.Tensor,
    y: torch.Tensor,
    train_rows: np.ndarray,
    val_rows: np.ndarray,
    config: dict[str, object],
    *,
    seed: int,
    batch_size: int,
    max_epochs: int,
    patience: int,
    min_epochs: int,
    device: torch.device,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = MLP(x.shape[1], tuple(config["hidden"]), float(config["dropout"])).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"])
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.as_tensor(train_rows), y[train_rows].cpu()),
        batch_size=min(batch_size, len(train_rows)), shuffle=True, generator=generator,
    )
    val_index = torch.as_tensor(val_rows, device=device)
    best = None
    stale = 0
    curve = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        losses = []
        for rows, labels in loader:
            rows, labels = rows.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x[rows]), labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            val = metrics(model(x[val_index]), y[val_index])
        curve.append({"epoch": epoch, "train_loss": float(np.mean(losses)), **{f"val_{k}": v for k, v in val.items()}})
        score = (val["roc_auc"], val["macro_f1"], -epoch)
        if best is None or score > best["score"]:
            best = {
                "score": score,
                "epoch": epoch,
                "metrics": val,
                "model": copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()}),
                "optimizer": copy.deepcopy(optimizer.state_dict()),
            }
            stale = 0
        else:
            stale += 1
        if epoch >= min_epochs and stale >= patience:
            break
    return model, optimizer, best, curve


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {args.output}")
    for subdir in ("curves", "checkpoints", "wandb"):
        (args.output / subdir).mkdir(parents=True, exist_ok=True)
    budgets, seeds = parse_ints(args.budgets), parse_ints(args.seeds)
    protocol = vars(args).copy()
    protocol["output"] = str(protocol["output"])
    protocol["budgets"] = budgets
    protocol["seeds"] = seeds
    protocol["selection_metric"] = "validation_roc_auc_then_macro_f1"
    protocol["grid"] = hyperparameters()
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")

    selection_handle = (args.output / "selection.jsonl").open("w")
    result_handle = (args.output / "results.jsonl").open("w")
    for target in selected_targets(args.targets):
        graph = load_graph(target)
        labels_global = load_labels(graph, target.label_key)
        nodes = labeled_nodes(labels_global)
        labels = labels_global[nodes].astype(np.int64)
        raw = graph_field(graph, "x")[nodes].float().cpu().numpy()
        global_splits = stratified_node_splits(labels_global, seed=0)
        lookup = {int(node): row for row, node in enumerate(nodes)}
        splits = {name: np.asarray([lookup[int(node)] for node in values], dtype=np.int64) for name, values in global_splits.items()}
        features = standardize_and_pad(raw, splits["train"], output_dim=raw.shape[1])
        x = torch.as_tensor(features, device=device)
        y = torch.as_tensor(labels, dtype=torch.long, device=device)
        split_hash = fingerprint(splits["train"], splits["val"], splits["test"])
        for budget in budgets:
            tune_rows = splits["train"] if budget == -1 else sampled_labels(labels, splits["train"], budget=budget, seed=0)
            candidates = []
            for candidate_id, config in enumerate(hyperparameters()):
                _, _, best, curve = train_once(
                    x, y, tune_rows, splits["val"], config, seed=0, batch_size=args.batch_size,
                    max_epochs=args.max_epochs, patience=args.patience, min_epochs=args.min_epochs, device=device,
                )
                row = {"target": target.name, "budget_per_class": budget, "candidate_id": candidate_id,
                       "config": config, "best_epoch": best["epoch"], **{f"val_{k}": v for k, v in best["metrics"].items()}}
                selection_handle.write(json.dumps(row, sort_keys=True) + "\n")
                selection_handle.flush()
                candidates.append((best["score"], candidate_id, config))
            _, candidate_id, config = max(candidates, key=lambda value: value[0])
            for seed in seeds:
                train_rows = splits["train"] if budget == -1 else sampled_labels(labels, splits["train"], budget=budget, seed=seed)
                model, optimizer, best, curve = train_once(
                    x, y, train_rows, splits["val"], config, seed=seed, batch_size=args.batch_size,
                    max_epochs=args.max_epochs, patience=args.patience, min_epochs=args.min_epochs, device=device,
                )
                model.load_state_dict(best["model"])
                model.eval()
                test_index = torch.as_tensor(splits["test"], device=device)
                with torch.no_grad():
                    test = metrics(model(x[test_index]), y[test_index])
                stem = f"{target.name}_b{budget}_s{seed}"
                curve_path = args.output / "curves" / f"{stem}.jsonl"
                curve_path.write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in curve))
                checkpoint_path = args.output / "checkpoints" / f"{stem}.pt"
                torch.save({"model_state_dict": best["model"], "optimizer_state_dict": best["optimizer"],
                            "config": config, "best_epoch": best["epoch"], "target": target.name,
                            "budget_per_class": budget, "seed": seed, "split_fingerprint": split_hash}, checkpoint_path)
                row = {"model": "strong_raw_mlp", "target": target.name, "budget_per_class": budget,
                       "labeled_examples": int(len(train_rows)), "seed": seed, "selected_candidate_id": candidate_id,
                       "config": config, "best_epoch": best["epoch"], "split_fingerprint": split_hash,
                       "train_nodes_fingerprint": fingerprint(train_rows), "checkpoint": str(checkpoint_path),
                       "curve": str(curve_path), **{f"val_{k}": v for k, v in best["metrics"].items()},
                       **{f"test_{k}": v for k, v in test.items()}}
                log_wandb_run(
                    project=args.wandb_project,
                    output=args.output,
                    name=stem,
                    config={**config, "target": target.name, "budget_per_class": budget, "seed": seed},
                    curve=curve,
                    summary=row,
                )
                result_handle.write(json.dumps(row, sort_keys=True) + "\n")
                result_handle.flush()
                print(json.dumps(row, sort_keys=True), flush=True)
        del graph, x, y
        if device.type == "cuda":
            torch.cuda.empty_cache()
    selection_handle.close()
    result_handle.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
