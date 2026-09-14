#!/usr/bin/env python3
"""Matched scorer-by-negative-policy campaign on official ogbl-collab."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from scripts.experiments.setup.ogbl_collab_mlp_lp import run as common


EXPECTED_FINGERPRINT = "07f7af8e654bda27caad60ed74c479f48780343826543dd613d90cb4e979f9f4"
SCORERS = ("cosine", "interaction")
NEGATIVE_POLICIES = ("uniform", "hard8")


class FeaturePairModel(nn.Module):
    def __init__(self, scorer: str, input_dim: int = 128):
        super().__init__()
        self.scorer = scorer
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Linear(256, 128)
        )
        if scorer == "cosine":
            self.log_scale = nn.Parameter(torch.tensor(float(np.log(10.0))))
            self.bias = nn.Parameter(torch.tensor(0.0))
        elif scorer == "interaction":
            self.predictor = nn.Sequential(
                nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1)
            )
        else:
            raise ValueError(scorer)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.encoder(x), dim=-1)

    def logits(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        left, right = self.encode(left), self.encode(right)
        if self.scorer == "cosine":
            cosine = (left * right).sum(dim=-1)
            return self.log_scale.exp().clamp(max=100.0) * cosine + self.bias
        pair = torch.cat((left * right, torch.abs(left - right)), dim=-1)
        return self.predictor(pair).squeeze(-1)


def hard_candidate_pool(
    train_edges: np.ndarray, num_nodes: int, seed: int, epoch: int, factor: int = 8
) -> tuple[np.ndarray, np.ndarray, str]:
    order, base, _ = common.epoch_training_pairs(train_edges, num_nodes, seed, epoch)
    rng = np.random.default_rng((seed + 1) * 10_000_019 + epoch)
    extra = rng.integers(0, num_nodes, size=(len(train_edges), factor - 1, 2), dtype=np.int64)
    pool = np.concatenate((base[:, None, :], extra), axis=1)
    return order, pool, common.fingerprint(order, pool)


@torch.no_grad()
def choose_hardest(
    model: FeaturePairModel, x: torch.Tensor, candidates: np.ndarray
) -> np.ndarray:
    shape = candidates.shape
    flat = torch.as_tensor(candidates.reshape(-1, 2), dtype=torch.long, device=x.device)
    scores = model.logits(x[flat[:, 0]], x[flat[:, 1]]).reshape(shape[0], shape[1])
    winner = scores.argmax(dim=1).cpu().numpy()
    return candidates[np.arange(shape[0]), winner]


@torch.no_grad()
def score_edges(model: FeaturePairModel, x: torch.Tensor, edges: np.ndarray, batch: int) -> np.ndarray:
    model.eval()
    values = []
    for start in range(0, len(edges), batch):
        pair = torch.as_tensor(edges[start:start + batch], dtype=torch.long, device=x.device)
        values.append(model.logits(x[pair[:, 0]], x[pair[:, 1]]).cpu().numpy())
    result = np.concatenate(values)
    if not np.isfinite(result).all():
        raise ValueError("non-finite scores")
    return result


def evaluate(evaluator: Any, model: FeaturePairModel, x: torch.Tensor,
             part: dict[str, np.ndarray], batch: int) -> dict[str, float]:
    return common.ranking_metrics(
        evaluator, score_edges(model, x, part["edge"], batch),
        score_edges(model, x, part["edge_neg"], batch)
    )


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def train(model: FeaturePairModel, x: torch.Tensor, split: dict, evaluator: Any,
          args: argparse.Namespace) -> tuple[FeaturePairModel, dict[str, Any]]:
    import wandb

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model.to(x.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate,
                                 weight_decay=args.weight_decay)
    arm = f"{args.scorer}_{args.negative_policy}"
    run = wandb.init(project=args.wandb_project, group=args.run_tag,
                     name=f"ogbl_collab_{arm}_s{args.seed}", mode=args.wandb_mode,
                     dir=str(args.out), reinit=True,
                     config={"dataset": "ogbl-collab", "arm": arm, "seed": args.seed,
                             "scorer": args.scorer, "negative_policy": args.negative_policy,
                             "hard_candidate_factor": 8 if args.negative_policy == "hard8" else 1,
                             "loss": "balanced_bce", "parameters": sum(p.numel() for p in model.parameters()),
                             "revision": args.revision, "dataset_fingerprint": args.dataset_fingerprint,
                             "test_closed_during_training": True})
    best_hits, best_epoch, stale, updates = -np.inf, -1, 0, 0
    best_path = args.out / "best.pt"
    history = []
    train_edges = split["train"]["edge"]
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        if args.negative_policy == "hard8":
            order, negative_pool, stream_hash = hard_candidate_pool(
                train_edges, len(x), args.seed, epoch
            )
        else:
            order, negatives, stream_hash = common.epoch_training_pairs(
                train_edges, len(x), args.seed, epoch
            )
        model.train()
        loss_sum = grad_sum = 0.0
        example_count = 0
        for start in range(0, len(train_edges), args.batch_size):
            ids = order[start:start + args.batch_size]
            positive = torch.as_tensor(train_edges[ids], dtype=torch.long, device=x.device)
            if args.negative_policy == "hard8":
                model.eval()
                negative_np = choose_hardest(model, x, negative_pool[start:start + len(ids)])
                model.train()
            else:
                negative_np = negatives[start:start + len(ids)]
            negative = torch.as_tensor(negative_np, dtype=torch.long, device=x.device)
            pos_logits = model.logits(x[positive[:, 0]], x[positive[:, 1]])
            neg_logits = model.logits(x[negative[:, 0]], x[negative[:, 1]])
            loss = F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits))
            loss += F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_sum += float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm))
            optimizer.step()
            loss_sum += float(loss.detach()) * len(ids)
            example_count += len(ids)
            updates += 1
        validation = evaluate(evaluator, model, x, split["valid"], args.eval_batch_size)
        improved = validation["hits_at_50"] > best_hits
        if improved:
            best_hits, best_epoch, stale = validation["hits_at_50"], epoch, 0
            save_checkpoint(best_path, {"arm": arm, "seed": args.seed, "epoch": epoch,
                                        "state_dict": model.state_dict(), "revision": args.revision,
                                        "dataset_fingerprint": args.dataset_fingerprint})
        else:
            stale += 1
        row = {"epoch": epoch, "optimizer_updates": updates,
               "train_loss": loss_sum / example_count,
               "mean_batch_grad_norm": grad_sum / int(np.ceil(len(train_edges) / args.batch_size)),
               "training_stream_fingerprint": stream_hash,
               "epoch_seconds": time.perf_counter() - started,
               **{f"val_{k}": v for k, v in validation.items()}}
        history.append(row)
        run.log({k: v for k, v in row.items() if k != "training_stream_fingerprint"}, step=epoch)
        print(json.dumps({"event": "epoch", "arm": arm, "seed": args.seed,
                          "epoch": epoch, "val_hits_at_50": validation["hits_at_50"],
                          "best_epoch": best_epoch, "stale": stale,
                          "epoch_seconds": row["epoch_seconds"]}), flush=True)
        if stale >= args.patience:
            break
    checkpoint = torch.load(best_path, map_location=x.device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    selection = {"arm": arm, "seed": args.seed, "best_epoch": best_epoch,
                 "best_validation_hits_at_50": best_hits, "epochs_run": len(history),
                 "optimizer_updates": updates, "checkpoint": str(best_path),
                 "checkpoint_sha256": common.sha256_file(best_path), "history": history,
                 "wandb_run_id": getattr(run, "id", None), "wandb_url": getattr(run, "url", None)}
    run.finish()
    return model, selection


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=Path, default=Path("/dataMeR1/phil/data/ogb"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--scorer", choices=SCORERS, required=True)
    p.add_argument("--negative-policy", choices=NEGATIVE_POLICIES, required=True)
    p.add_argument("--seed", type=int, choices=range(3), required=True)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=65_536)
    p.add_argument("--eval-batch-size", type=int, default=262_144)
    p.add_argument("--learning-rate", type=float, default=0.001)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--wandb-project", default="ogbl-collab-mlp-hardneg")
    p.add_argument("--wandb-mode", choices=("offline", "online", "disabled"), default="offline")
    p.add_argument("--run-tag", default="hardneg_v1")
    p.add_argument("--smoke", action="store_true", help="train and validate only; never score test")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    graph, split, evaluator, ogb_version = common.load_official_dataset(args.dataset_root)
    audit = common.audit_dataset(graph, split, ogb_version)
    if audit["split_fingerprint"] != EXPECTED_FINGERPRINT:
        raise ValueError("dataset fingerprint mismatch")
    plan = {"event": "dry_run" if args.dry_run else "run_start",
            "cell": f"{args.scorer}_{args.negative_policy}_seed{args.seed}",
            "device": args.device, "out": str(args.out), "dataset_audit": audit,
            "optimizer_updates_per_epoch": int(np.ceil(len(split["train"]["edge"]) / args.batch_size)),
            "hard_candidates_per_positive": 8 if args.negative_policy == "hard8" else 1}
    print(json.dumps(plan, indent=2, sort_keys=True), flush=True)
    if args.dry_run:
        return
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    args.revision, args.dataset_fingerprint = common.git_revision(), audit["split_fingerprint"]
    protocol_path = Path(__file__).with_name("protocol.yaml")
    (args.out / "protocol.json").write_text(json.dumps({"started_at": common.utc_now(),
        "revision": args.revision, "protocol_source": str(protocol_path),
        "protocol_source_sha256": common.sha256_file(protocol_path), "arguments": vars(args),
        "dataset_audit": audit, "test_status": "closed"}, indent=2, sort_keys=True, default=str) + "\n")
    os.environ["WANDB_MODE"] = args.wandb_mode
    x = torch.as_tensor(np.asarray(graph["node_feat"]), dtype=torch.float32, device=args.device)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model, selection = train(FeaturePairModel(args.scorer, x.shape[1]), x, split, evaluator, args)
    selection_path = args.out / "selection_frozen.json"
    selection_path.write_text(json.dumps({"frozen_at": common.utc_now(), "selection_metric": "official validation Hits@50",
        "tie_break": "earliest epoch", "learned": selection, "test_status": "closed"}, indent=2, sort_keys=True) + "\n")
    if args.smoke:
        (args.out / "smoke_complete.json").write_text(json.dumps({
            "complete": True, "classification": "smoke_validation_only",
            "test_scored": False, "cell": f"{args.scorer}_{args.negative_policy}_seed{args.seed}",
            "epochs": selection["epochs_run"], "revision": args.revision,
            "dataset_fingerprint": args.dataset_fingerprint,
        }, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"event": "smoke_complete", "test_scored": False,
                          "selection": selection}, indent=2), flush=True)
        return
    positive = score_edges(model, x, split["test"]["edge"], args.eval_batch_size)
    negative = score_edges(model, x, split["test"]["edge_neg"], args.eval_batch_size)
    masks = common.test_strata(split["train"]["edge"], split["valid"]["edge"], split["test"]["edge"], len(x))
    result = {"complete": True, "revision": args.revision, "seed": args.seed,
              "arm": f"{args.scorer}_{args.negative_policy}", "dataset_fingerprint": args.dataset_fingerprint,
              "selection_frozen_sha256": common.sha256_file(selection_path),
              "official_test": common.ranking_metrics(evaluator, positive, negative),
              "test_strata": {name: {"positive_count": int(mask.sum()),
                  **common.ranking_metrics(evaluator, positive[mask], negative)} for name, mask in masks.items()}}
    (args.out / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "complete", **result}, indent=2), flush=True)


if __name__ == "__main__":
    main()
