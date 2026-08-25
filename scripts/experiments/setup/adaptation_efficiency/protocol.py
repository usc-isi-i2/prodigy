"""Deterministic data and head-training contract for adaptation efficiency."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch import nn


LABEL_BUDGETS = (0, 1, 10, 100)
UPDATE_STEPS = (0, 1, 10, 100)
LABEL_SEEDS = (0, 1, 2)
# Preserve the full raw 768-D feature baseline while giving every linear probe
# the exact same parameter shape. Smaller learned representations are zero-padded.
COMMON_DIM = 768
HEAD_LR = 1e-2


def stratified_node_splits(
    labels: np.ndarray,
    *,
    seed: int = 0,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
) -> dict[str, np.ndarray]:
    """Match the final-core SAMGPT 60/20/20 stratified split exactly."""
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    result: dict[str, list[np.ndarray]] = {"train": [], "val": [], "test": []}
    classes = sorted(int(value) for value in np.unique(labels) if int(value) >= 0)
    for class_id in classes:
        indices = np.where(labels == class_id)[0].copy()
        rng.shuffle(indices)
        count = int(indices.size)
        train_count = int(round(count * train_fraction))
        validation_count = int(round(count * validation_fraction))
        if count >= 3:
            train_count = min(max(1, train_count), count - 2)
            validation_count = min(max(1, validation_count), count - train_count - 1)
        elif count == 2:
            train_count, validation_count = 1, 0
        else:
            train_count, validation_count = 1, 0
        result["train"].append(indices[:train_count])
        result["val"].append(indices[train_count : train_count + validation_count])
        result["test"].append(indices[train_count + validation_count :])
    return {
        key: np.concatenate(parts).astype(np.int64) if parts else np.empty(0, dtype=np.int64)
        for key, parts in result.items()
    }


def sampled_labels(
    labels: np.ndarray, train_indices: np.ndarray, *, budget: int, seed: int
) -> np.ndarray:
    """Nested, balanced samples: budget 1 is a subset of 10, which is a subset of 100."""
    if budget == 0:
        return np.empty(0, dtype=np.int64)
    rng = np.random.default_rng(seed)
    picked: list[np.ndarray] = []
    classes = sorted(int(value) for value in np.unique(labels[train_indices]) if int(value) >= 0)
    for class_id in classes:
        members = train_indices[labels[train_indices] == class_id].copy()
        rng.shuffle(members)
        if members.size < budget:
            raise ValueError(
                f"class {class_id} has {members.size} train nodes; budget {budget} is impossible"
            )
        picked.append(members[:budget])
    return np.concatenate(picked).astype(np.int64)


def fingerprint_indices(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        values = np.asarray(array, dtype=np.int64)
        digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
        digest.update(values.tobytes())
    return digest.hexdigest()


def fingerprint_model(model: nn.Module) -> str:
    """Hash an initialized head so cross-model initialization equality is auditable."""
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        array = value.detach().cpu().contiguous().numpy()
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def standardize_and_pad(
    features: np.ndarray, train_rows: np.ndarray, *, output_dim: int = COMMON_DIM
) -> np.ndarray:
    """Train-split, label-free standardization followed by deterministic zero pad/truncate."""
    features = np.asarray(features, dtype=np.float32)
    mean = features[train_rows].mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = features[train_rows].std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    normalized = (features - mean) / scale
    if normalized.shape[1] >= output_dim:
        return normalized[:, :output_dim].astype(np.float32, copy=False)
    result = np.zeros((normalized.shape[0], output_dim), dtype=np.float32)
    result[:, : normalized.shape[1]] = normalized
    return result


class LinearHead(nn.Module):
    def __init__(self, input_dim: int, classes: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, classes)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.linear(values)


class RawMLP(nn.Module):
    def __init__(self, input_dim: int, classes: int, hidden_dim: int = COMMON_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, classes)
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values)


def new_head(kind: str, input_dim: int, classes: int, seed: int) -> nn.Module:
    torch.manual_seed(10_000 + int(seed))
    if kind == "mlp":
        return RawMLP(input_dim, classes)
    if kind == "linear":
        return LinearHead(input_dim, classes)
    raise ValueError(kind)


def classification_metrics(
    model: nn.Module, features: torch.Tensor, labels: np.ndarray, rows: np.ndarray
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        logits = model(features[rows])
        probability = torch.softmax(logits, dim=1).cpu().numpy()
        prediction = probability.argmax(axis=1)
    truth = labels[rows]
    classes = probability.shape[1]
    if classes == 2:
        roc_auc = roc_auc_score(truth, probability[:, 1])
    else:
        roc_auc = roc_auc_score(truth, probability, multi_class="ovr", average="macro")
    return {
        "roc_auc": float(roc_auc),
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, average="macro", zero_division=0)),
    }


def run_curve(
    features: np.ndarray,
    labels: np.ndarray,
    splits: dict[str, np.ndarray],
    *,
    model_id: str,
    target: str,
    label_seed: int,
    budget: int,
    head_kind: str = "linear",
    learning_rate: float = HEAD_LR,
) -> list[dict[str, object]]:
    """Evaluate update 0 then advance one shared full-batch head to 1, 10, and 100."""
    labels = np.asarray(labels, dtype=np.int64)
    x = torch.from_numpy(np.asarray(features, dtype=np.float32))
    classes = len(set(int(value) for value in labels if int(value) >= 0))
    head = new_head(head_kind, x.shape[1], classes, label_seed)
    head_initialization_fingerprint = fingerprint_model(head)
    selected = sampled_labels(labels, splits["train"], budget=budget, seed=label_seed)
    selected_fingerprint = fingerprint_indices(selected)
    split_fingerprint = fingerprint_indices(splits["train"], splits["val"], splits["test"])
    optimizer = (
        None
        if budget == 0
        else torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=0.0)
    )
    rows: list[dict[str, object]] = []
    valid_updates = (0,) if budget == 0 else UPDATE_STEPS
    update = 0
    for milestone in valid_updates:
        while update < milestone:
            assert optimizer is not None
            head.train()
            optimizer.zero_grad(set_to_none=True)
            logits = head(x[selected])
            loss = F.cross_entropy(logits, torch.from_numpy(labels[selected]))
            loss.backward()
            optimizer.step()
            update += 1
        for split_name in ("val", "test"):
            metrics = classification_metrics(head, x, labels, splits[split_name])
            rows.append(
                {
                    "model_id": model_id,
                    "target": target,
                    "head_kind": head_kind,
                    "label_seed": label_seed,
                    "label_budget_per_class": budget,
                    "head_updates": milestone,
                    "split": split_name,
                    "classes": classes,
                    "labeled_examples": int(selected.size),
                    "selected_nodes_fingerprint": selected_fingerprint,
                    "split_fingerprint": split_fingerprint,
                    "head_initialization_fingerprint": head_initialization_fingerprint,
                    "optimizer": "none" if optimizer is None else "AdamW",
                    "learning_rate": 0.0 if optimizer is None else learning_rate,
                    "weight_decay": 0.0,
                    **metrics,
                }
            )
    return rows


@dataclass(frozen=True)
class FeatureCache:
    model_id: str
    target: str
    features: np.ndarray
    labels: np.ndarray
    node_ids: np.ndarray
    metadata: dict[str, object]


def save_feature_cache(path: Path, cache: FeatureCache) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        model_id=np.asarray(cache.model_id),
        target=np.asarray(cache.target),
        features=np.asarray(cache.features, dtype=np.float32),
        labels=np.asarray(cache.labels, dtype=np.int64),
        node_ids=np.asarray(cache.node_ids, dtype=np.int64),
        metadata=np.asarray(json.dumps(cache.metadata, sort_keys=True)),
    )


def load_feature_cache(path: Path) -> FeatureCache:
    with np.load(path, allow_pickle=False) as value:
        return FeatureCache(
            model_id=str(value["model_id"]),
            target=str(value["target"]),
            features=value["features"],
            labels=value["labels"],
            node_ids=value["node_ids"],
            metadata=json.loads(str(value["metadata"])),
        )
