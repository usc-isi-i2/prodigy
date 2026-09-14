#!/usr/bin/env python3
"""Matched raw/linear/nonlinear cosine LP ladder on classic Cora features."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.nn import functional as F


ARMS = ("raw_cosine", "linear_cosine", "nonlinear_mlp_cosine")


def fingerprint(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str((value.shape, value.dtype.str)).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def unordered_pairs(edge_index: torch.Tensor, num_nodes: int) -> np.ndarray:
    edges = edge_index.detach().cpu().numpy().astype(np.int64, copy=False)
    lo, hi = np.minimum(edges[0], edges[1]), np.maximum(edges[0], edges[1])
    keep = lo != hi
    keys = np.unique(lo[keep] * num_nodes + hi[keep])
    return np.column_stack(np.divmod(keys, num_nodes)).astype(np.int64)


def sample_nonedges(
    num_nodes: int, count: int, forbidden: set[int], used: set[int], rng: np.random.Generator
) -> np.ndarray:
    result: list[tuple[int, int]] = []
    while len(result) < count:
        u = int(rng.integers(num_nodes))
        v = int(rng.integers(num_nodes))
        if u == v:
            continue
        lo, hi = min(u, v), max(u, v)
        key = lo * num_nodes + hi
        if key in forbidden or key in used:
            continue
        used.add(key)
        result.append((lo, hi))
    return np.asarray(result, dtype=np.int64)


def make_campaign(edge_index: torch.Tensor, num_nodes: int, seed: int) -> dict[str, np.ndarray]:
    positive = unordered_pairs(edge_index, num_nodes)
    rng = np.random.default_rng(seed)
    positive = positive[rng.permutation(len(positive))]
    n_test = round(0.10 * len(positive))
    n_val = round(0.05 * len(positive))
    splits = {
        "test_pos": positive[:n_test],
        "val_pos": positive[n_test : n_test + n_val],
        "train_pos": positive[n_test + n_val :],
    }
    forbidden = {int(u) * num_nodes + int(v) for u, v in positive}
    used: set[int] = set()
    for split in ("test", "val", "train"):
        splits[f"{split}_neg"] = sample_nonedges(
            num_nodes, len(splits[f"{split}_pos"]), forbidden, used, rng
        )
    # Required leakage and completeness gates.
    positive_sets = [set(map(tuple, splits[f"{s}_pos"])) for s in ("train", "val", "test")]
    assert not (positive_sets[0] & positive_sets[1] or positive_sets[0] & positive_sets[2] or positive_sets[1] & positive_sets[2])
    assert sum(map(len, positive_sets)) == len(positive)
    assert all(len(splits[f"{s}_pos"]) == len(splits[f"{s}_neg"]) for s in ("train", "val", "test"))
    return splits


class Encoder(nn.Module):
    def __init__(self, arm: str, input_dim: int):
        super().__init__()
        if arm == "linear_cosine":
            self.net = nn.Linear(input_dim, 128)
        elif arm == "nonlinear_mlp_cosine":
            self.net = nn.Sequential(nn.Linear(input_dim, 256), nn.ReLU(), nn.Linear(256, 128))
        else:
            raise ValueError(arm)
        self.log_scale = nn.Parameter(torch.tensor(float(np.log(10.0))))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=1)

    def logits(self, z: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
        cosine = (z[pairs[:, 0]] * z[pairs[:, 1]]).sum(dim=1)
        return self.log_scale.exp().clamp(max=100.0) * cosine + self.bias


def labels_and_pairs(pos: np.ndarray, neg: np.ndarray, device: torch.device):
    pairs = torch.as_tensor(np.concatenate([pos, neg]), dtype=torch.long, device=device)
    labels = torch.cat((torch.ones(len(pos)), torch.zeros(len(neg)))).to(device)
    return pairs, labels


def metrics(labels: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    assert np.isfinite(logits).all()
    probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))
    prediction = probability >= 0.5
    return {
        "roc_auc": float(roc_auc_score(labels, logits)),
        "average_precision": float(average_precision_score(labels, logits)),
        "binary_cross_entropy": float(log_loss(labels, probability, labels=[0, 1])),
        "brier_score": float(brier_score_loss(labels, probability)),
        "accuracy_at_0_5": float(accuracy_score(labels, prediction)),
        "balanced_accuracy_at_0_5": float(balanced_accuracy_score(labels, prediction)),
        "precision_at_0_5": float(precision_score(labels, prediction, zero_division=0)),
        "recall_at_0_5": float(recall_score(labels, prediction, zero_division=0)),
        "f1_at_0_5": float(f1_score(labels, prediction, zero_division=0)),
    }


def evaluate(model: Encoder, x: torch.Tensor, pairs: torch.Tensor, labels: torch.Tensor):
    model.eval()
    with torch.no_grad():
        scores = model.logits(model(x), pairs).cpu().numpy()
    return metrics(labels.cpu().numpy(), scores)


def train_arm(arm: str, x: torch.Tensor, campaign: dict[str, np.ndarray], args, out: Path):
    os.environ["WANDB_MODE"] = args.wandb_mode
    os.environ.setdefault("WANDB_SILENT", "true")
    import wandb

    torch.manual_seed(args.seed)
    model = Encoder(arm, x.shape[1]).to(x.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    val_pairs, val_labels = labels_and_pairs(campaign["val_pos"], campaign["val_neg"], x.device)
    num_nodes = len(x)
    all_positive = np.concatenate([campaign[f"{split}_pos"] for split in ("train", "val", "test")])
    forbidden = {int(u) * num_nodes + int(v) for u, v in all_positive}
    reserved = {
        int(u) * num_nodes + int(v)
        for split in ("val", "test")
        for u, v in campaign[f"{split}_neg"]
    }
    negative_rng = np.random.default_rng(args.train_negative_seed)
    best_auc, best_epoch, stale, best_state = -np.inf, -1, 0, None
    history = []
    run = wandb.init(
        project=args.wandb_project,
        name=f"cora_standard_lp_{arm}_s{args.seed}_{args.run_tag}",
        mode=args.wandb_mode,
        dir=str(out),
        reinit=True,
        config={
            "arm": arm,
            "seed": args.seed,
            "input_dim": int(x.shape[1]),
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "epochs": args.epochs,
            "patience": args.patience,
            "pair_fingerprint": args.pair_fingerprint,
            "resample_train_negatives": args.resample_train_negatives,
            "train_negative_seed": args.train_negative_seed,
            "run_tag": args.run_tag,
            "validation_interval": args.val_interval,
        },
    )
    print(json.dumps({"event": "wandb_initialized", "arm": arm, "url": run.url}), flush=True)
    for epoch in range(1, args.epochs + 1):
        if args.resample_train_negatives:
            train_neg = sample_nonedges(
                num_nodes, len(campaign["train_pos"]), forbidden, set(reserved), negative_rng
            )
        else:
            train_neg = campaign["train_neg"]
        train_negative_fingerprint = fingerprint(train_neg)
        train_pairs, train_labels = labels_and_pairs(campaign["train_pos"], train_neg, x.device)
        model.train()
        logits = model.logits(model(x), train_pairs)
        loss = F.binary_cross_entropy_with_logits(logits, train_labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        row = {
            "epoch": epoch,
            "train_loss": float(loss.detach()),
            "logit_scale": float(model.log_scale.detach().exp().clamp(max=100.0)),
            "logit_bias": float(model.bias.detach()),
            "train_negative_fingerprint": train_negative_fingerprint,
        }
        validation_due = epoch % args.val_interval == 0
        if validation_due:
            val = evaluate(model, x, val_pairs, val_labels)
            row.update({f"val_{key}": value for key, value in val.items()})
        history.append(row)
        run.log({k: v for k, v in row.items() if k != "train_negative_fingerprint"}, step=epoch)
        if validation_due:
            if val["roc_auc"] > best_auc:
                best_auc, best_epoch, stale = val["roc_auc"], epoch, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
            if stale >= args.patience:
                break
    assert best_state is not None
    checkpoint = out / f"{arm}.pt"
    torch.save({"arm": arm, "epoch": best_epoch, "state_dict": best_state}, checkpoint)
    model.load_state_dict(best_state)
    run.summary["best_epoch"] = best_epoch
    run.summary["best_validation_roc_auc"] = best_auc
    run.summary["epochs_run"] = len(history)
    run.finish()
    return model, {"arm": arm, "best_epoch": best_epoch, "best_validation_roc_auc": best_auc,
                   "epochs_run": len(history), "history": history, "checkpoint": str(checkpoint)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--val-interval", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--wandb-project", default="cora-standard-mlp-lp")
    parser.add_argument("--wandb-mode", choices=("offline", "online"), default="offline")
    parser.add_argument("--run-tag", default="fixed_negatives")
    parser.add_argument("--resample-train-negatives", action="store_true")
    parser.add_argument("--train-negative-seed", type=int, default=1000)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"output directory already exists: {args.out}")
    args.out.mkdir(parents=True)

    raw = torch.load(args.graph, map_location="cpu", weights_only=False)
    x = raw.x if hasattr(raw, "x") else raw["x"]
    edge_index = raw.edge_index if hasattr(raw, "edge_index") else raw["edge_index"]
    if tuple(x.shape) != (2708, 1433):
        raise ValueError(f"expected classic Cora x shape (2708, 1433), got {tuple(x.shape)}")
    unique = torch.unique(x)
    if not torch.all((unique == 0) | (unique == 1)):
        raise ValueError("classic Cora features must be binary")
    campaign = make_campaign(edge_index, len(x), args.seed)
    np.savez_compressed(args.out / "pairs.npz", **campaign)
    pair_hash = fingerprint(*(campaign[k] for k in sorted(campaign)))
    args.pair_fingerprint = pair_hash
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    protocol = {
        "revision": revision, "graph": str(args.graph), "seed": args.seed, "arms": ARMS,
        "features": "classic Cora 1433-dimensional binary bag-of-words",
        "split": "unique undirected pairs, seeded 85/5/10",
        "negatives": "one fixed unique uniform unordered nonedge per positive, disjoint across splits",
        "pair_fingerprint": pair_hash,
        "counts": {k: len(v) for k, v in campaign.items()},
        "epochs": args.epochs, "patience": args.patience,
        "validation_interval": args.val_interval,
        "patience_unit": "validation checks",
        "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
        "wandb": {"mode": args.wandb_mode, "project": args.wandb_project, "logging": "every epoch, unsmoothed"},
        "run_tag": args.run_tag,
        "training_negatives": (
            f"resampled every epoch from deterministic seed {args.train_negative_seed}; "
            "unique within epoch and disjoint from fixed validation/test negatives"
            if args.resample_train_negatives
            else "fixed campaign train_neg panel"
        ),
        "selection": "maximum validation ROC-AUC on scheduled validation checks; earliest epoch wins ties; test closed until all selections finish",
        "metrics": ["roc_auc", "average_precision", "binary_cross_entropy", "brier_score",
                    "accuracy_at_0_5", "balanced_accuracy_at_0_5", "precision_at_0_5",
                    "recall_at_0_5", "f1_at_0_5"],
    }
    (args.out / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")

    device = torch.device(args.device)
    x = x.float().to(device)
    trained, selections = {}, []
    for arm in ARMS[1:]:
        trained[arm], selection = train_arm(arm, x, campaign, args, args.out)
        selections.append(selection)
    (args.out / "selection.json").write_text(json.dumps(selections, indent=2) + "\n")

    test_pairs, test_labels = labels_and_pairs(campaign["test_pos"], campaign["test_neg"], device)
    raw_z = F.normalize(x, dim=1)
    raw_scores = (raw_z[test_pairs[:, 0]] * raw_z[test_pairs[:, 1]]).sum(dim=1).cpu().numpy()
    results = [{"arm": "raw_cosine", **metrics(test_labels.cpu().numpy(), raw_scores)}]
    for arm in ARMS[1:]:
        results.append({"arm": arm, **evaluate(trained[arm], x, test_pairs, test_labels)})
    payload = {"complete": True, "pair_fingerprint": pair_hash, "results": results}
    (args.out / "results.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
