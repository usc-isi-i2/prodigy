#!/usr/bin/env python3
"""Matched GraphSAGE-cosine LP experiment on official ogbl-collab."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


MLP_RUN = Path(__file__).parents[1] / "ogbl_collab_mlp_lp" / "run.py"
SPEC = importlib.util.spec_from_file_location("ogbl_collab_mlp_run", MLP_RUN)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import shared Collab helpers from {MLP_RUN}")
common = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(common)

ARM = "sage_unique_cosine"
EXPECTED_UNIQUE_PAIRS = 967_632


def unique_unweighted_train_graph(
    train_edges: np.ndarray, num_nodes: int
) -> tuple[np.ndarray, str]:
    keys = np.unique(common.pair_keys(train_edges, num_nodes))
    lo = keys // np.int64(num_nodes)
    hi = keys % np.int64(num_nodes)
    forward = np.column_stack((lo, hi))
    non_self = lo != hi
    reverse = np.column_stack((hi[non_self], lo[non_self]))
    arcs = np.concatenate((forward, reverse), axis=0).astype(np.int64, copy=False)
    order = np.lexsort((arcs[:, 1], arcs[:, 0]))
    arcs = arcs[order]
    return arcs.T, common.fingerprint(arcs)


class SAGECosine(nn.Module):
    def __init__(self, input_dim: int = 128, hidden_dim: int = 256, output_dim: int = 128):
        super().__init__()
        from torch_geometric.nn import SAGEConv

        self.conv1 = SAGEConv(input_dim, hidden_dim, aggr="mean")
        self.conv2 = SAGEConv(hidden_dim, output_dim, aggr="mean")
        self.log_scale = nn.Parameter(torch.tensor(float(np.log(10.0))))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x, edge_index))
        return F.normalize(self.conv2(x, edge_index), dim=-1)

    def logits(self, z: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        cosine = (z[edges[:, 0]] * z[edges[:, 1]]).sum(dim=-1)
        return self.log_scale.exp().clamp(max=100.0) * cosine + self.bias


def parameter_norm(model: nn.Module) -> float:
    values = [parameter.detach().norm(2).square() for parameter in model.parameters()]
    return float(torch.stack(values).sum().sqrt())


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


@torch.no_grad()
def score_edges(
    model: SAGECosine,
    z: torch.Tensor,
    edges: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    values = []
    for start in range(0, len(edges), batch_size):
        batch = torch.as_tensor(
            edges[start : start + batch_size], dtype=torch.long, device=z.device
        )
        values.append(model.logits(z, batch).detach().cpu().numpy())
    scores = np.concatenate(values)
    if not np.isfinite(scores).all():
        raise ValueError("non-finite evaluation scores")
    return scores


@torch.no_grad()
def evaluate_split(
    evaluator: Any,
    model: SAGECosine,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    split_part: dict[str, np.ndarray],
    batch_size: int,
) -> dict[str, float]:
    model.eval()
    z = model.encode(x, edge_index)
    positive = score_edges(model, z, split_part["edge"], batch_size)
    negative = score_edges(model, z, split_part["edge_neg"], batch_size)
    return common.ranking_metrics(evaluator, positive, negative)


def train(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    split: dict[str, dict[str, np.ndarray]],
    evaluator: Any,
    args: argparse.Namespace,
) -> tuple[SAGECosine, dict[str, Any]]:
    import wandb

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model = SAGECosine(x.shape[1]).to(x.device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    run = wandb.init(
        project=args.wandb_project,
        group=args.run_tag,
        name=f"ogbl_collab_{ARM}_s{args.seed}_{args.run_tag}",
        mode=args.wandb_mode,
        dir=str(args.out),
        reinit=True,
        config={
            "dataset": "ogbl-collab",
            "arm": ARM,
            "seed": args.seed,
            "encoder": "SAGEConv(128,256,mean)-ReLU-SAGEConv(256,128,mean)",
            "decoder": "scaled_cosine",
            "graph": "unique_unweighted_official_training_pairs",
            "use_validation_edges": False,
            "use_edge_weight": False,
            "use_event_multiplicity": False,
            "test_closed_during_training": True,
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "negative_ratio": "1:1",
            "revision": args.revision,
            "dataset_fingerprint": args.dataset_fingerprint,
            "graph_fingerprint": args.graph_fingerprint,
            "run_tag": args.run_tag,
        },
    )
    run_url = getattr(run, "url", None)
    print(json.dumps({"event": "wandb_initialized", "url": run_url}), flush=True)

    train_edges = split["train"]["edge"]
    checkpoint_dir = args.out / "checkpoints"
    checkpoint_dir.mkdir()
    best_checkpoint = checkpoint_dir / "best.pt"
    last_checkpoint = checkpoint_dir / "last.pt"
    best_hits, best_epoch, best_state = -np.inf, -1, None
    stale = 0
    updates = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        if x.is_cuda:
            torch.cuda.reset_peak_memory_stats(x.device)
        order, negatives, stream_fingerprint = common.epoch_training_pairs(
            train_edges, len(x), args.seed, epoch
        )
        model.train()
        loss_sum = 0.0
        example_count = 0
        grad_norm_sum = 0.0
        for start in range(0, len(train_edges), args.batch_size):
            indices = order[start : start + args.batch_size]
            positive = torch.as_tensor(
                train_edges[indices], dtype=torch.long, device=x.device
            )
            negative = torch.as_tensor(
                negatives[start : start + len(indices)], dtype=torch.long, device=x.device
            )
            optimizer.zero_grad(set_to_none=True)
            z = model.encode(x, edge_index)
            positive_logits = model.logits(z, positive)
            negative_logits = model.logits(z, negative)
            logits = torch.cat((positive_logits, negative_logits))
            labels = torch.cat((torch.ones_like(positive_logits), torch.zeros_like(negative_logits)))
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optimizer.step()
            count = int(labels.numel())
            loss_sum += float(loss.detach()) * count
            example_count += count
            grad_norm_sum += float(grad_norm)
            updates += 1

        validation = evaluate_split(
            evaluator, model, x, edge_index, split["valid"], args.eval_batch_size
        )
        row = {
            "epoch": epoch,
            "optimizer_updates": updates,
            "train_loss": loss_sum / example_count,
            "mean_batch_grad_norm": grad_norm_sum
            / int(np.ceil(len(train_edges) / args.batch_size)),
            "parameter_norm": parameter_norm(model),
            "logit_scale": float(model.log_scale.detach().exp().clamp(max=100.0)),
            "logit_bias": float(model.bias.detach()),
            "training_stream_fingerprint": stream_fingerprint,
            **{f"val_{key}": value for key, value in validation.items()},
        }
        current = validation["hits_at_50"]
        improved = current > best_hits
        if improved:
            best_hits = current
            best_epoch = epoch
            stale = 0
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
        else:
            stale += 1
        epoch_seconds = time.perf_counter() - epoch_started
        row.update(
            {
                "epoch_seconds": epoch_seconds,
                "positive_edges_per_second": len(train_edges) / epoch_seconds,
                "peak_gpu_memory_gb": (
                    float(torch.cuda.max_memory_allocated(x.device) / (1024**3))
                    if x.is_cuda
                    else 0.0
                ),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        checkpoint_payload = {
            "arm": ARM,
            "seed": args.seed,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_epoch": best_epoch,
            "best_validation_hits_at_50": best_hits,
            "stale_validation_checks": stale,
            "training_stream_fingerprint": stream_fingerprint,
            "revision": args.revision,
            "dataset_fingerprint": args.dataset_fingerprint,
            "graph_fingerprint": args.graph_fingerprint,
        }
        save_checkpoint(last_checkpoint, checkpoint_payload)
        if improved:
            save_checkpoint(best_checkpoint, checkpoint_payload)
        history.append(row)
        run.log(
            {key: value for key, value in row.items() if key != "training_stream_fingerprint"},
            step=epoch,
        )
        print(
            json.dumps(
                {
                    "event": "epoch",
                    "epoch": epoch,
                    "train_loss": row["train_loss"],
                    "val_hits_at_50": current,
                    "best_epoch": best_epoch,
                    "best_validation_hits_at_50": best_hits,
                }
            ),
            flush=True,
        )
        if stale >= args.patience:
            break

    if best_state is None:
        raise RuntimeError("training completed without a selected checkpoint")
    model.load_state_dict(best_state)
    selection = {
        "arm": ARM,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_validation_hits_at_50": best_hits,
        "epochs_run": len(history),
        "optimizer_updates": updates,
        "best_checkpoint": str(best_checkpoint),
        "best_checkpoint_sha256": common.sha256_file(best_checkpoint),
        "last_checkpoint": str(last_checkpoint),
        "last_checkpoint_sha256": common.sha256_file(last_checkpoint),
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("/dataMeR1/phil/data/ogb"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=65_536)
    parser.add_argument("--eval-batch-size", type=int, default=262_144)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wandb-project", default="ogbl-collab-sage-lp")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--run-tag", default="matched_v1")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed < 0 or args.epochs < 1 or args.patience < 1:
        raise ValueError("seed must be nonnegative; epochs and patience must be positive")
    graph, split, evaluator, ogb_version = common.load_official_dataset(args.dataset_root)
    audit = common.audit_dataset(graph, split, ogb_version)
    graph_arcs, graph_fingerprint = unique_unweighted_train_graph(
        split["train"]["edge"], int(graph["num_nodes"])
    )
    unique_pairs = int((graph_arcs.shape[1] + np.sum(graph_arcs[0] == graph_arcs[1])) / 2)
    if unique_pairs != EXPECTED_UNIQUE_PAIRS:
        raise ValueError(f"expected {EXPECTED_UNIQUE_PAIRS} unique train pairs, got {unique_pairs}")
    plan = {
        "event": "dry_run" if args.dry_run else "run_start",
        "arm": ARM,
        "seed": args.seed,
        "device": args.device,
        "out": str(args.out),
        "dataset_audit": audit,
        "graph_policy": "unique_unweighted_official_training_pairs",
        "graph_arcs": int(graph_arcs.shape[1]),
        "graph_fingerprint": graph_fingerprint,
        "optimizer_updates_per_epoch": int(
            np.ceil(common.EXPECTED_EDGE_COUNTS["train"] / args.batch_size)
        ),
    }
    print(json.dumps(plan, indent=2, sort_keys=True), flush=True)
    if args.dry_run:
        return
    if args.out.exists():
        raise FileExistsError(f"output directory already exists: {args.out}")
    args.out.mkdir(parents=True)
    os.environ["WANDB_MODE"] = args.wandb_mode
    os.environ.setdefault("WANDB_SILENT", "true")
    args.revision = common.git_revision()
    args.dataset_fingerprint = audit["split_fingerprint"]
    args.graph_fingerprint = graph_fingerprint
    protocol_source = Path(__file__).with_name("protocol.yaml")
    protocol = {
        "started_at": common.utc_now(),
        "revision": args.revision,
        "protocol_source": str(protocol_source),
        "protocol_source_sha256": common.sha256_file(protocol_source),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
        },
        "dataset_audit": audit,
        "graph_arcs": int(graph_arcs.shape[1]),
        "graph_fingerprint": graph_fingerprint,
        "test_status": "closed",
    }
    (args.out / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")

    device = torch.device(args.device)
    x = torch.as_tensor(np.asarray(graph["node_feat"]), dtype=torch.float32, device=device)
    edge_index = torch.as_tensor(graph_arcs, dtype=torch.long, device=device)
    model, selection = train(x, edge_index, split, evaluator, args)
    frozen = {
        "frozen_at": common.utc_now(),
        "selection_metric": "official validation Hits@50",
        "tie_break": "earliest epoch",
        "learned": selection,
        "test_status": "closed",
    }
    selection_path = args.out / "selection_frozen.json"
    selection_path.write_text(json.dumps(frozen, indent=2, sort_keys=True) + "\n")

    model.eval()
    with torch.no_grad():
        z = model.encode(x, edge_index)
    positive = score_edges(model, z, split["test"]["edge"], args.eval_batch_size)
    negative = score_edges(model, z, split["test"]["edge_neg"], args.eval_batch_size)
    official = common.ranking_metrics(evaluator, positive, negative)
    masks = common.test_strata(
        split["train"]["edge"], split["valid"]["edge"], split["test"]["edge"], len(x)
    )
    strata = {
        name: {
            "positive_count": int(mask.sum()),
            **common.ranking_metrics(evaluator, positive[mask], negative),
        }
        for name, mask in masks.items()
    }
    payload = {
        "complete": True,
        "completed_at": common.utc_now(),
        "revision": args.revision,
        "seed": args.seed,
        "arm": ARM,
        "dataset_fingerprint": args.dataset_fingerprint,
        "graph_fingerprint": graph_fingerprint,
        "selection_frozen_sha256": common.sha256_file(selection_path),
        "official_test": official,
        "test_strata": strata,
    }
    (args.out / "results.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    import wandb

    summary = wandb.init(
        project=args.wandb_project,
        group=args.run_tag,
        name=f"ogbl_collab_{ARM}_summary_s{args.seed}_{args.run_tag}",
        mode=args.wandb_mode,
        dir=str(args.out),
        reinit=True,
        config={
            "dataset": "ogbl-collab",
            "arm": ARM,
            "seed": args.seed,
            "revision": args.revision,
            "test_opened_after_selection_freeze": True,
            "run_tag": args.run_tag,
        },
    )
    for key, value in official.items():
        summary.summary[f"test/{key}"] = value
    for name, values in strata.items():
        for key, value in values.items():
            summary.summary[f"test_strata/{name}/{key}"] = value
    summary_url = getattr(summary, "url", None)
    summary.finish()
    print(json.dumps({"event": "complete", "summary_url": summary_url, **payload}, indent=2), flush=True)


if __name__ == "__main__":
    main()
