#!/usr/bin/env python3
"""Feature-only raw/linear/nonlinear LP campaign on official ogbl-collab."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.nn import functional as F


ARMS = ("raw_cosine", "linear_cosine", "nonlinear_mlp_cosine")
LEARNED_ARMS = ARMS[1:]
EXPECTED_FEATURE_SHAPE = (235_868, 128)
EXPECTED_EDGE_COUNTS = {"train": 1_179_052, "valid": 60_084, "test": 46_329}
EXPECTED_NEGATIVE_COUNTS = {"valid": 100_000, "test": 100_000}
HITS_K = (10, 50, 100)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_revision() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str((value.shape, value.dtype.str)).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def pair_keys(edges: np.ndarray, num_nodes: int) -> np.ndarray:
    edges = np.asarray(edges, dtype=np.int64)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError(f"expected edge array shaped (N, 2), got {edges.shape}")
    lo = np.minimum(edges[:, 0], edges[:, 1])
    hi = np.maximum(edges[:, 0], edges[:, 1])
    return lo * np.int64(num_nodes) + hi


def test_strata(
    train_edges: np.ndarray,
    valid_edges: np.ndarray,
    test_edges: np.ndarray,
    num_nodes: int,
) -> dict[str, np.ndarray]:
    train_keys = np.unique(pair_keys(train_edges, num_nodes))
    valid_keys = np.unique(pair_keys(valid_edges, num_nodes))
    test_keys = pair_keys(test_edges, num_nodes)
    seen_train = np.isin(test_keys, train_keys)
    seen_pre2019 = np.isin(test_keys, np.union1d(train_keys, valid_keys))
    return {
        "seen_in_train": seen_train,
        "novel_vs_train": ~seen_train,
        "seen_pre2019": seen_pre2019,
        "novel_vs_pre2019": ~seen_pre2019,
    }


def audit_dataset(
    graph: dict[str, Any], split: dict[str, dict[str, np.ndarray]], ogb_version: str
) -> dict[str, Any]:
    x = np.asarray(graph["node_feat"])
    if x.shape != EXPECTED_FEATURE_SHAPE:
        raise ValueError(f"expected feature shape {EXPECTED_FEATURE_SHAPE}, got {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError("node features contain non-finite values")
    num_nodes = int(graph["num_nodes"])
    if num_nodes != EXPECTED_FEATURE_SHAPE[0]:
        raise ValueError(f"expected {EXPECTED_FEATURE_SHAPE[0]} nodes, got {num_nodes}")

    for name, count in EXPECTED_EDGE_COUNTS.items():
        edges = np.asarray(split[name]["edge"])
        if edges.shape != (count, 2):
            raise ValueError(f"unexpected {name} positive shape: {edges.shape}")
        if np.any(edges < 0) or np.any(edges >= num_nodes):
            raise ValueError(f"{name} positive edges contain invalid node IDs")
    for name, count in EXPECTED_NEGATIVE_COUNTS.items():
        edges = np.asarray(split[name]["edge_neg"])
        if edges.shape != (count, 2):
            raise ValueError(f"unexpected {name} negative shape: {edges.shape}")
        if np.any(edges < 0) or np.any(edges >= num_nodes):
            raise ValueError(f"{name} negative edges contain invalid node IDs")

    years = {name: np.asarray(split[name]["year"]) for name in EXPECTED_EDGE_COUNTS}
    if int(years["train"].max()) > 2017:
        raise ValueError("training split contains an edge after 2017")
    if set(np.unique(years["valid"]).tolist()) != {2018}:
        raise ValueError("validation split is not exactly year 2018")
    if set(np.unique(years["test"]).tolist()) != {2019}:
        raise ValueError("test split is not exactly year 2019")

    train_event_keys = pair_keys(split["train"]["edge"], num_nodes)
    train_keys = np.unique(train_event_keys)
    graph_edges = np.asarray(graph["edge_index"], dtype=np.int64).T
    graph_event_keys = pair_keys(graph_edges, num_nodes)
    if not np.array_equal(
        np.sort(graph_event_keys), np.repeat(np.sort(train_event_keys), 2)
    ):
        raise ValueError(
            "loaded graph pair multiplicities do not equal two arcs per training event"
        )

    all_positive_keys = np.union1d(
        np.union1d(train_keys, pair_keys(split["valid"]["edge"], num_nodes)),
        pair_keys(split["test"]["edge"], num_nodes),
    )
    for name in EXPECTED_NEGATIVE_COUNTS:
        overlap = np.intersect1d(
            np.unique(pair_keys(split[name]["edge_neg"], num_nodes)), all_positive_keys
        )
        if len(overlap):
            raise ValueError(f"{name} official negatives overlap {len(overlap)} known positives")

    strata = test_strata(
        split["train"]["edge"], split["valid"]["edge"], split["test"]["edge"], num_nodes
    )
    return {
        "ogb_version": ogb_version,
        "num_nodes": num_nodes,
        "feature_shape": list(x.shape),
        "graph_stored_arcs": int(graph_edges.shape[0]),
        "training_unique_pairs": int(len(train_keys)),
        "training_repeated_events": int(len(train_event_keys) - len(train_keys)),
        "counts": {
            name: {
                "positive": int(len(split[name]["edge"])),
                "negative": int(len(split[name].get("edge_neg", []))),
            }
            for name in ("train", "valid", "test")
        },
        "official_negative_self_pair_counts": {
            name: int(np.sum(split[name]["edge_neg"][:, 0] == split[name]["edge_neg"][:, 1]))
            for name in EXPECTED_NEGATIVE_COUNTS
        },
        "year_ranges": {
            name: [int(years[name].min()), int(years[name].max())]
            for name in ("train", "valid", "test")
        },
        "test_strata_counts": {name: int(mask.sum()) for name, mask in strata.items()},
        "feature_fingerprint": fingerprint(x),
        "split_fingerprint": fingerprint(
            split["train"]["edge"],
            split["train"]["year"],
            split["valid"]["edge"],
            split["valid"]["edge_neg"],
            split["test"]["edge"],
            split["test"]["edge_neg"],
        ),
    }


class Encoder(nn.Module):
    def __init__(self, arm: str, input_dim: int = 128):
        super().__init__()
        if arm == "linear_cosine":
            self.net = nn.Linear(input_dim, 128)
        elif arm == "nonlinear_mlp_cosine":
            self.net = nn.Sequential(
                nn.Linear(input_dim, 256), nn.ReLU(), nn.Linear(256, 128)
            )
        else:
            raise ValueError(f"unknown learned arm: {arm}")
        self.log_scale = nn.Parameter(torch.tensor(float(np.log(10.0))))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(features), dim=-1)

    def logits(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        cosine = (self.encode(left) * self.encode(right)).sum(dim=-1)
        return self.log_scale.exp().clamp(max=100.0) * cosine + self.bias


def raw_cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return (F.normalize(left, dim=-1) * F.normalize(right, dim=-1)).sum(dim=-1)


@torch.no_grad()
def score_edges(
    x: torch.Tensor,
    edges: np.ndarray,
    batch_size: int,
    model: Encoder | None,
) -> np.ndarray:
    if model is not None:
        model.eval()
    scores: list[np.ndarray] = []
    for start in range(0, len(edges), batch_size):
        batch = torch.as_tensor(edges[start : start + batch_size], dtype=torch.long, device=x.device)
        left, right = x[batch[:, 0]], x[batch[:, 1]]
        logits = raw_cosine(left, right) if model is None else model.logits(left, right)
        scores.append(logits.detach().cpu().numpy())
    result = np.concatenate(scores)
    if not np.isfinite(result).all():
        raise ValueError("non-finite evaluation scores")
    return result


def ranking_metrics(evaluator: Any, positive: np.ndarray, negative: np.ndarray) -> dict[str, float]:
    output: dict[str, float] = {}
    for k in HITS_K:
        evaluator.K = k
        value = evaluator.eval({"y_pred_pos": positive, "y_pred_neg": negative})[f"hits@{k}"]
        output[f"hits_at_{k}"] = float(value)
    labels = np.concatenate((np.ones(len(positive)), np.zeros(len(negative))))
    scores = np.concatenate((positive, negative))
    output.update(
        {
            "roc_auc": float(roc_auc_score(labels, scores)),
            "average_precision": float(average_precision_score(labels, scores)),
            "positive_score_mean": float(np.mean(positive)),
            "positive_score_std": float(np.std(positive)),
            "negative_score_mean": float(np.mean(negative)),
            "negative_score_std": float(np.std(negative)),
            "score_mean_gap": float(np.mean(positive) - np.mean(negative)),
        }
    )
    return output


def evaluate_split(
    evaluator: Any,
    x: torch.Tensor,
    split_part: dict[str, np.ndarray],
    model: Encoder | None,
    batch_size: int,
    positive_mask: np.ndarray | None = None,
) -> dict[str, float]:
    positive_edges = split_part["edge"]
    if positive_mask is not None:
        positive_edges = positive_edges[positive_mask]
    positive = score_edges(x, positive_edges, batch_size, model)
    negative = score_edges(x, split_part["edge_neg"], batch_size, model)
    return ranking_metrics(evaluator, positive, negative)


def epoch_training_pairs(
    train_edges: np.ndarray, num_nodes: int, seed: int, epoch: int
) -> tuple[np.ndarray, np.ndarray, str]:
    stream_seed = np.uint64(seed + 1) * np.uint64(1_000_003) + np.uint64(epoch)
    rng = np.random.default_rng(stream_seed)
    order = rng.permutation(len(train_edges))
    negatives = rng.integers(0, num_nodes, size=(len(train_edges), 2), dtype=np.int64)
    return order, negatives, fingerprint(order, negatives)


def parameter_norm(model: nn.Module) -> float:
    values = [parameter.detach().norm(2).square() for parameter in model.parameters()]
    return float(torch.stack(values).sum().sqrt())


def train_arm(
    arm: str,
    x: torch.Tensor,
    split: dict[str, dict[str, np.ndarray]],
    evaluator: Any,
    args: argparse.Namespace,
    out: Path,
) -> tuple[Encoder, dict[str, Any]]:
    import wandb

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model = Encoder(arm, x.shape[1]).to(x.device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    run = wandb.init(
        project=args.wandb_project,
        group=args.run_tag,
        name=f"ogbl_collab_{arm}_s{args.seed}_{args.run_tag}",
        mode=args.wandb_mode,
        dir=str(out),
        reinit=True,
        config={
            "dataset": "ogbl-collab",
            "official_metric": "hits@50",
            "arm": arm,
            "seed": args.seed,
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "negative_ratio": "1:1",
            "negative_policy": "uniform random node pairs; deterministic matched stream",
            "validation_interval_epochs": args.val_interval,
            "test_closed_during_training": True,
            "use_validation_edges": False,
            "revision": args.revision,
            "dataset_fingerprint": args.dataset_fingerprint,
            "run_tag": args.run_tag,
        },
    )
    run_url = getattr(run, "url", None)
    print(json.dumps({"event": "wandb_initialized", "arm": arm, "url": run_url}), flush=True)

    train_edges = split["train"]["edge"]
    best_hits, best_epoch, best_state = -np.inf, -1, None
    stale = 0
    history: list[dict[str, Any]] = []
    updates = 0
    for epoch in range(1, args.epochs + 1):
        order, negatives, stream_fingerprint = epoch_training_pairs(
            train_edges, len(x), args.seed, epoch
        )
        model.train()
        loss_sum = 0.0
        example_count = 0
        grad_norm_sum = 0.0
        for start in range(0, len(train_edges), args.batch_size):
            indices = order[start : start + args.batch_size]
            positive = torch.as_tensor(train_edges[indices], dtype=torch.long, device=x.device)
            negative = torch.as_tensor(negatives[start : start + len(indices)], dtype=torch.long, device=x.device)
            positive_logits = model.logits(x[positive[:, 0]], x[positive[:, 1]])
            negative_logits = model.logits(x[negative[:, 0]], x[negative[:, 1]])
            logits = torch.cat((positive_logits, negative_logits))
            labels = torch.cat((torch.ones_like(positive_logits), torch.zeros_like(negative_logits)))
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optimizer.step()
            count = int(labels.numel())
            loss_sum += float(loss.detach()) * count
            example_count += count
            grad_norm_sum += float(grad_norm)
            updates += 1

        row: dict[str, Any] = {
            "epoch": epoch,
            "optimizer_updates": updates,
            "train_loss": loss_sum / example_count,
            "mean_batch_grad_norm": grad_norm_sum / int(np.ceil(len(train_edges) / args.batch_size)),
            "parameter_norm": parameter_norm(model),
            "logit_scale": float(model.log_scale.detach().exp().clamp(max=100.0)),
            "logit_bias": float(model.bias.detach()),
            "training_stream_fingerprint": stream_fingerprint,
        }
        validation_due = epoch % args.val_interval == 0
        if validation_due:
            validation = evaluate_split(
                evaluator, x, split["valid"], model, args.eval_batch_size
            )
            row.update({f"val_{key}": value for key, value in validation.items()})
            current = validation["hits_at_50"]
            if current > best_hits:
                best_hits = current
                best_epoch = epoch
                stale = 0
                best_state = {
                    key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                }
            else:
                stale += 1
        history.append(row)
        run.log(
            {key: value for key, value in row.items() if key != "training_stream_fingerprint"},
            step=epoch,
        )
        print(
            json.dumps(
                {
                    "event": "epoch",
                    "arm": arm,
                    "epoch": epoch,
                    "train_loss": row["train_loss"],
                    "val_hits_at_50": row.get("val_hits_at_50"),
                    "best_epoch": best_epoch,
                    "best_validation_hits_at_50": best_hits,
                }
            ),
            flush=True,
        )
        if validation_due and stale >= args.patience:
            break

    if best_state is None:
        raise RuntimeError(f"{arm} completed without a validation checkpoint")
    checkpoint = out / f"{arm}.pt"
    torch.save(
        {"arm": arm, "seed": args.seed, "epoch": best_epoch, "state_dict": best_state}, checkpoint
    )
    model.load_state_dict(best_state)
    selection = {
        "arm": arm,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_validation_hits_at_50": best_hits,
        "epochs_run": len(history),
        "optimizer_updates": updates,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "wandb_run_id": getattr(run, "id", None),
        "wandb_url": run_url,
        "history": history,
    }
    run.summary.update(
        {
            "best_epoch": best_epoch,
            "best_validation_hits_at_50": best_hits,
            "epochs_run": len(history),
            "optimizer_updates": updates,
        }
    )
    run.finish()
    return model, selection


def load_official_dataset(root: Path):
    import ogb
    from ogb.linkproppred import Evaluator, LinkPropPredDataset

    dataset = LinkPropPredDataset(name="ogbl-collab", root=str(root))
    graph = dataset[0]
    raw_split = dataset.get_edge_split()
    split = {
        split_name: {key: np.asarray(value) for key, value in values.items()}
        for split_name, values in raw_split.items()
    }
    return graph, split, Evaluator(name="ogbl-collab"), ogb.__version__


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("/dataMeR1/phil/data/ogb"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--val-interval", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=65_536)
    parser.add_argument("--eval-batch-size", type=int, default=262_144)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wandb-project", default="ogbl-collab-mlp-lp")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--run-tag", default="official_v1")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed < 0:
        raise ValueError("seed must be nonnegative")
    if args.epochs < 1 or args.patience < 1 or args.val_interval < 1:
        raise ValueError("epochs, patience, and validation interval must be positive")

    graph, split, evaluator, ogb_version = load_official_dataset(args.dataset_root)
    audit = audit_dataset(graph, split, ogb_version)
    plan = {
        "event": "dry_run" if args.dry_run else "run_start",
        "seed": args.seed,
        "arms": ARMS,
        "dataset_root": str(args.dataset_root),
        "out": str(args.out),
        "device": args.device,
        "epochs": args.epochs,
        "patience": args.patience,
        "validation_interval_epochs": args.val_interval,
        "optimizer_updates_per_epoch": int(
            np.ceil(EXPECTED_EDGE_COUNTS["train"] / args.batch_size)
        ),
        "audit": audit,
    }
    print(json.dumps(plan, indent=2, sort_keys=True), flush=True)
    if args.dry_run:
        return
    if args.out.exists():
        raise FileExistsError(f"output directory already exists: {args.out}")
    args.out.mkdir(parents=True)

    os.environ["WANDB_MODE"] = args.wandb_mode
    os.environ.setdefault("WANDB_SILENT", "true")
    args.revision = git_revision()
    args.dataset_fingerprint = audit["split_fingerprint"]
    protocol_source = Path(__file__).with_name("protocol.yaml")
    resolved_protocol = {
        "started_at": utc_now(),
        "revision": args.revision,
        "protocol_source": str(protocol_source),
        "protocol_source_sha256": sha256_file(protocol_source),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
        },
        "dataset_audit": audit,
        "test_status": "closed",
    }
    (args.out / "protocol.json").write_text(
        json.dumps(resolved_protocol, indent=2, sort_keys=True) + "\n"
    )

    device = torch.device(args.device)
    x = torch.as_tensor(np.asarray(graph["node_feat"]), dtype=torch.float32, device=device)
    raw_validation = evaluate_split(evaluator, x, split["valid"], None, args.eval_batch_size)
    selections: list[dict[str, Any]] = []
    trained: dict[str, Encoder] = {}
    for arm in LEARNED_ARMS:
        trained[arm], selection = train_arm(arm, x, split, evaluator, args, args.out)
        selections.append(selection)

    first_streams = [
        row["training_stream_fingerprint"] for row in selections[0]["history"]
    ]
    second_streams = [
        row["training_stream_fingerprint"] for row in selections[1]["history"]
    ]
    common = min(len(first_streams), len(second_streams))
    if first_streams[:common] != second_streams[:common]:
        raise RuntimeError("learned arms did not receive matching training streams")
    frozen = {
        "frozen_at": utc_now(),
        "selection_metric": "official validation Hits@50",
        "tie_break": "earliest epoch",
        "raw_validation": raw_validation,
        "learned": selections,
        "matched_training_stream_prefix_epochs": common,
        "test_status": "closed",
    }
    (args.out / "selection_frozen.json").write_text(
        json.dumps(frozen, indent=2, sort_keys=True) + "\n"
    )

    strata_masks = test_strata(
        split["train"]["edge"], split["valid"]["edge"], split["test"]["edge"], len(x)
    )
    results: list[dict[str, Any]] = []
    for arm in ARMS:
        model = trained.get(arm)
        official = evaluate_split(evaluator, x, split["test"], model, args.eval_batch_size)
        strata = {
            name: {
                "positive_count": int(mask.sum()),
                **evaluate_split(
                    evaluator,
                    x,
                    split["test"],
                    model,
                    args.eval_batch_size,
                    positive_mask=mask,
                ),
            }
            for name, mask in strata_masks.items()
        }
        results.append({"arm": arm, "seed": None if arm == "raw_cosine" else args.seed,
                        "official_test": official, "test_strata": strata})

    payload = {
        "complete": True,
        "completed_at": utc_now(),
        "revision": args.revision,
        "seed": args.seed,
        "dataset_fingerprint": audit["split_fingerprint"],
        "selection_frozen_sha256": sha256_file(args.out / "selection_frozen.json"),
        "results": results,
    }
    (args.out / "results.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    import wandb

    summary_run = wandb.init(
        project=args.wandb_project,
        group=args.run_tag,
        name=f"ogbl_collab_summary_s{args.seed}_{args.run_tag}",
        mode=args.wandb_mode,
        dir=str(args.out),
        reinit=True,
        config={"dataset": "ogbl-collab", "seed": args.seed, "revision": args.revision,
                "test_opened_after_selection_freeze": True, "run_tag": args.run_tag},
    )
    for result in results:
        arm = result["arm"]
        for key, value in result["official_test"].items():
            summary_run.summary[f"test/{arm}/{key}"] = value
        for stratum, values in result["test_strata"].items():
            for key, value in values.items():
                summary_run.summary[f"test_strata/{arm}/{stratum}/{key}"] = value
    summary_url = getattr(summary_run, "url", None)
    summary_run.finish()
    print(json.dumps({"event": "complete", "summary_url": summary_url, **payload}, indent=2), flush=True)


if __name__ == "__main__":
    main()
