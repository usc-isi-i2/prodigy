#!/usr/bin/env python3
"""OGB-style GraphSAGE link prediction on official ogbl-collab."""
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


HERE = Path(__file__).parent
MATCHED_RUN = HERE / "run.py"
SPEC = importlib.util.spec_from_file_location("ogbl_collab_sage_matched", MATCHED_RUN)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import helpers from {MATCHED_RUN}")
matched = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(matched)
common = matched.common

ARM = "sage_ogb_style"
EXPECTED_GRAPH_ARCS = 2_358_104
EXPECTED_PARAMETERS = 460_289


class SAGE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, layers: int = 3):
        super().__init__()
        from torch_geometric.nn import SAGEConv

        self.convs = nn.ModuleList([SAGEConv(input_dim, hidden_dim)])
        self.convs.extend(SAGEConv(hidden_dim, hidden_dim) for _ in range(layers - 1))

    def forward(self, x: torch.Tensor, adj_t: Any) -> torch.Tensor:
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, adj_t))
        return self.convs[-1](x, adj_t)


class LinkPredictor(nn.Module):
    def __init__(self, dim: int = 256, layers: int = 3):
        super().__init__()
        self.lins = nn.ModuleList([nn.Linear(dim, dim)])
        self.lins.extend(nn.Linear(dim, dim) for _ in range(layers - 2))
        self.lins.append(nn.Linear(dim, 1))

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        x = left * right
        for lin in self.lins[:-1]:
            x = F.relu(lin(x))
        return self.lins[-1](x).squeeze(-1)


def parameter_norm(*modules: nn.Module) -> float:
    squares = [p.detach().norm(2).square() for m in modules for p in m.parameters()]
    return float(torch.stack(squares).sum().sqrt())


@torch.no_grad()
def score_edges(predictor: LinkPredictor, z: torch.Tensor, edges: np.ndarray, batch: int) -> np.ndarray:
    predictor.eval()
    values = []
    for start in range(0, len(edges), batch):
        pair = torch.as_tensor(edges[start:start + batch], dtype=torch.long, device=z.device)
        values.append(predictor(z[pair[:, 0]], z[pair[:, 1]]).cpu().numpy())
    scores = np.concatenate(values)
    if not np.isfinite(scores).all():
        raise ValueError("non-finite evaluation scores")
    return scores


@torch.no_grad()
def evaluate(evaluator: Any, encoder: SAGE, predictor: LinkPredictor, x: torch.Tensor,
             adj_t: Any, part: dict[str, np.ndarray], batch: int) -> dict[str, float]:
    encoder.eval()
    z = encoder(x, adj_t)
    return common.ranking_metrics(
        evaluator,
        score_edges(predictor, z, part["edge"], batch),
        score_edges(predictor, z, part["edge_neg"], batch),
    )


def train(x: torch.Tensor, adj_t: Any, split: dict[str, dict[str, np.ndarray]], evaluator: Any,
          args: argparse.Namespace) -> tuple[SAGE, LinkPredictor, dict[str, Any]]:
    import wandb

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    encoder, predictor = SAGE(x.shape[1]).to(x.device), LinkPredictor().to(x.device)
    count = sum(p.numel() for p in encoder.parameters()) + sum(p.numel() for p in predictor.parameters())
    if count != EXPECTED_PARAMETERS:
        raise ValueError(f"expected {EXPECTED_PARAMETERS} parameters, got {count}")
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(predictor.parameters()), lr=args.learning_rate
    )
    run = wandb.init(
        project=args.wandb_project, group=args.run_tag,
        name=f"ogbl_collab_{ARM}_s{args.seed}_{args.run_tag}", mode=args.wandb_mode,
        dir=str(args.out), reinit=True,
        config={
            "dataset": "ogbl-collab", "arm": ARM, "seed": args.seed,
            "encoder": "SAGEConv(128,256)-ReLU-SAGEConv(256,256)-ReLU-SAGEConv(256,256)",
            "decoder": "MLP(256,256,256,1) on endpoint product", "parameters": count,
            "graph": "original_weighted_training_adjacency", "use_validation_edges": False,
            "test_closed_during_training": True, "epochs": args.epochs,
            "batch_size": args.batch_size, "learning_rate": args.learning_rate,
            "weight_decay": 0.0, "negative_ratio": "1:1", "revision": args.revision,
            "dataset_fingerprint": args.dataset_fingerprint, "graph_fingerprint": args.graph_fingerprint,
            "run_tag": args.run_tag,
        },
    )
    run_url = getattr(run, "url", None)
    print(json.dumps({"event": "wandb_initialized", "url": run_url}), flush=True)
    train_edges = split["train"]["edge"]
    ckpt_dir = args.out / "checkpoints"
    ckpt_dir.mkdir()
    best_path, last_path = ckpt_dir / "best.pt", ckpt_dir / "last.pt"
    best_hits, best_epoch = -np.inf, -1
    history: list[dict[str, Any]] = []
    updates = 0
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats(x.device)
        order, negatives, stream_hash = common.epoch_training_pairs(
            train_edges, len(x), args.seed, epoch
        )
        encoder.train(); predictor.train()
        loss_sum = grad_encoder_sum = grad_predictor_sum = 0.0
        examples = 0
        for start in range(0, len(train_edges), args.batch_size):
            ids = order[start:start + args.batch_size]
            pos = torch.as_tensor(train_edges[ids], dtype=torch.long, device=x.device)
            neg = torch.as_tensor(negatives[start:start + len(ids)], dtype=torch.long, device=x.device)
            optimizer.zero_grad(set_to_none=True)
            z = encoder(x, adj_t)
            pos_logits = predictor(z[pos[:, 0]], z[pos[:, 1]])
            neg_logits = predictor(z[neg[:, 0]], z[neg[:, 1]])
            loss = F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits))
            loss = loss + F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits))
            loss.backward()
            grad_encoder_sum += float(torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0))
            grad_predictor_sum += float(torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0))
            optimizer.step()
            loss_sum += float(loss.detach()) * len(ids)
            examples += len(ids)
            updates += 1
        validation = evaluate(evaluator, encoder, predictor, x, adj_t, split["valid"], args.eval_batch_size)
        improved = validation["hits_at_50"] > best_hits
        if improved:
            best_hits, best_epoch = validation["hits_at_50"], epoch
        seconds = time.perf_counter() - started
        batches = int(np.ceil(len(train_edges) / args.batch_size))
        row = {
            "epoch": epoch, "optimizer_updates": updates, "train_loss": loss_sum / examples,
            "encoder_grad_norm": grad_encoder_sum / batches,
            "predictor_grad_norm": grad_predictor_sum / batches,
            "parameter_norm": parameter_norm(encoder, predictor), "epoch_seconds": seconds,
            "positive_edges_per_second": len(train_edges) / seconds,
            "peak_gpu_memory_gb": float(torch.cuda.max_memory_allocated(x.device) / 1024**3),
            "learning_rate": optimizer.param_groups[0]["lr"], "training_stream_fingerprint": stream_hash,
            **{f"val_{k}": v for k, v in validation.items()},
        }
        payload = {
            "arm": ARM, "seed": args.seed, "epoch": epoch,
            "encoder_state_dict": encoder.state_dict(), "predictor_state_dict": predictor.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(), "best_epoch": best_epoch,
            "best_validation_hits_at_50": best_hits, "revision": args.revision,
            "dataset_fingerprint": args.dataset_fingerprint, "graph_fingerprint": args.graph_fingerprint,
            "training_stream_fingerprint": stream_hash,
        }
        matched.save_checkpoint(last_path, payload)
        if improved:
            matched.save_checkpoint(best_path, payload)
        history.append(row)
        run.log({k: v for k, v in row.items() if k != "training_stream_fingerprint"}, step=epoch)
        print(json.dumps({"event": "epoch", "epoch": epoch, "train_loss": row["train_loss"],
                          "val_hits_at_50": validation["hits_at_50"], "best_epoch": best_epoch,
                          "best_validation_hits_at_50": best_hits}), flush=True)
    checkpoint = torch.load(best_path, map_location=x.device)
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    predictor.load_state_dict(checkpoint["predictor_state_dict"])
    selection = {
        "arm": ARM, "seed": args.seed, "best_epoch": best_epoch,
        "best_validation_hits_at_50": best_hits, "epochs_run": len(history),
        "optimizer_updates": updates, "best_checkpoint": str(best_path),
        "best_checkpoint_sha256": common.sha256_file(best_path), "last_checkpoint": str(last_path),
        "last_checkpoint_sha256": common.sha256_file(last_path), "wandb_run_id": getattr(run, "id", None),
        "wandb_url": run_url, "history": history,
    }
    run.summary.update({"best_epoch": best_epoch, "best_validation_hits_at_50": best_hits,
                        "epochs_run": len(history), "optimizer_updates": updates})
    run.finish()
    return encoder, predictor, selection


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=Path, default=Path("/dataMeR1/phil/data/ogb"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=65536)
    p.add_argument("--eval-batch-size", type=int, default=262144)
    p.add_argument("--learning-rate", type=float, default=0.001)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--wandb-project", default="ogbl-collab-sage-lp")
    p.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    p.add_argument("--run-tag", default="ogb_style_v1")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    graph, split, evaluator, ogb_version = common.load_official_dataset(args.dataset_root)
    audit = common.audit_dataset(graph, split, ogb_version)
    arcs = np.asarray(graph["edge_index"], dtype=np.int64)
    weights = np.asarray(graph["edge_weight"], dtype=np.float32).reshape(-1)
    if arcs.shape[1] != EXPECTED_GRAPH_ARCS or len(weights) != EXPECTED_GRAPH_ARCS:
        raise ValueError(f"unexpected supplied graph shapes {arcs.shape}, {weights.shape}")
    graph_hash = common.fingerprint(arcs, weights)
    print(json.dumps({"event": "dry_run" if args.dry_run else "run_start", "arm": ARM,
                      "seed": args.seed, "device": args.device, "out": str(args.out),
                      "dataset_audit": audit, "graph_arcs": arcs.shape[1],
                      "edge_weight_min": float(weights.min()), "edge_weight_max": float(weights.max()),
                      "graph_fingerprint": graph_hash,
                      "optimizer_updates_per_epoch": int(np.ceil(len(split["train"]["edge"]) / args.batch_size))},
                     indent=2, sort_keys=True), flush=True)
    if args.dry_run:
        return
    if args.out.exists():
        raise FileExistsError(f"output directory already exists: {args.out}")
    args.out.mkdir(parents=True)
    os.environ["WANDB_MODE"] = args.wandb_mode
    os.environ.setdefault("WANDB_SILENT", "true")
    args.revision, args.dataset_fingerprint, args.graph_fingerprint = common.git_revision(), audit["split_fingerprint"], graph_hash
    source = HERE / "protocol_ogb_style.yaml"
    protocol = {"started_at": common.utc_now(), "revision": args.revision,
                "protocol_source": str(source), "protocol_source_sha256": common.sha256_file(source),
                "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "dataset_audit": audit, "graph_arcs": int(arcs.shape[1]),
                "edge_weight_min": float(weights.min()), "edge_weight_max": float(weights.max()),
                "graph_fingerprint": graph_hash, "test_status": "closed"}
    (args.out / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    from torch_sparse import SparseTensor
    device = torch.device(args.device)
    x = torch.as_tensor(np.asarray(graph["node_feat"]), dtype=torch.float32, device=device)
    adj_t = SparseTensor(row=torch.as_tensor(arcs[1]), col=torch.as_tensor(arcs[0]),
                         value=torch.as_tensor(weights).reshape(-1, 1),
                         sparse_sizes=(len(x), len(x))).to(device)
    encoder, predictor, selection = train(x, adj_t, split, evaluator, args)
    frozen = {"frozen_at": common.utc_now(), "selection_metric": "official validation Hits@50",
              "tie_break": "earliest epoch", "learned": selection, "test_status": "closed"}
    selection_path = args.out / "selection_frozen.json"
    selection_path.write_text(json.dumps(frozen, indent=2, sort_keys=True) + "\n")
    encoder.eval(); predictor.eval()
    with torch.no_grad():
        z = encoder(x, adj_t)
    positive = score_edges(predictor, z, split["test"]["edge"], args.eval_batch_size)
    negative = score_edges(predictor, z, split["test"]["edge_neg"], args.eval_batch_size)
    official = common.ranking_metrics(evaluator, positive, negative)
    masks = common.test_strata(split["train"]["edge"], split["valid"]["edge"], split["test"]["edge"], len(x))
    strata = {name: {"positive_count": int(mask.sum()),
                     **common.ranking_metrics(evaluator, positive[mask], negative)} for name, mask in masks.items()}
    result = {"complete": True, "completed_at": common.utc_now(), "revision": args.revision,
              "seed": args.seed, "arm": ARM, "dataset_fingerprint": args.dataset_fingerprint,
              "graph_fingerprint": graph_hash, "selection_frozen_sha256": common.sha256_file(selection_path),
              "official_test": official, "test_strata": strata}
    (args.out / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    import wandb
    summary = wandb.init(project=args.wandb_project, group=args.run_tag,
                         name=f"ogbl_collab_{ARM}_summary_s{args.seed}_{args.run_tag}",
                         mode=args.wandb_mode, dir=str(args.out), reinit=True,
                         config={"dataset": "ogbl-collab", "arm": ARM, "seed": args.seed,
                                 "revision": args.revision, "test_opened_after_selection_freeze": True,
                                 "run_tag": args.run_tag})
    for key, value in official.items(): summary.summary[f"test/{key}"] = value
    for name, values in strata.items():
        for key, value in values.items(): summary.summary[f"test_strata/{name}/{key}"] = value
    summary_url = getattr(summary, "url", None)
    summary.finish()
    print(json.dumps({"event": "complete", "summary_url": summary_url, **result}, indent=2), flush=True)


if __name__ == "__main__":
    main()
