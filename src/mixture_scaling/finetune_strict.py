from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .config import load_config
from .model import GraphSAGE
from .strict_data import induced_partition, load_raw, load_split, split_hash
from .strict_metrics import classification_metrics


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def label_mapping(y: torch.Tensor) -> tuple[list[int], dict[int, int]]:
    classes = [int(value) for value in torch.unique(y[y >= 0], sorted=True)]
    return classes, {label: index for index, label in enumerate(classes)}


def targets(y: torch.Tensor, mapping: dict[int, int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    mask = y >= 0
    encoded = torch.tensor([mapping[int(value)] for value in y[mask]], dtype=torch.long, device=device)
    return mask.to(device), encoded


def forward_labeled(
    encoder: GraphSAGE,
    head: nn.Linear,
    graph,
    mapping: dict[int, int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    mask, y = targets(graph.data.y, mapping, device)
    embedding = encoder(graph.data.x.to(device), graph.data.edge_index.to(device))
    return head(embedding[mask]), y


def metrics_from_logits(logits: torch.Tensor, y: torch.Tensor, classes: list[int]) -> dict:
    probability = torch.softmax(logits, dim=-1).detach().cpu().numpy()
    encoded_y = y.detach().cpu().numpy()
    prediction = probability.argmax(axis=1)
    return classification_metrics(encoded_y, prediction, probability, np.arange(len(classes)))


def save(path: Path, encoder: GraphSAGE, head: nn.Linear, optimizer, epoch: int, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "encoder": encoder.state_dict(),
            "head": head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "metadata": metadata,
        },
        path,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split-root", required=True)
    parser.add_argument("--checkpoint", required=True, help="best.pt or the literal scratch")
    parser.add_argument("--target", default="twibot20")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    if args.device not in (2, 3):
        raise ValueError("strict pilot may use only GPUs 2 and 3")
    run_dir = Path(args.output_root) / args.run_id
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite fine-tune run: {run_dir}")
    (run_dir / "checkpoints").mkdir(parents=True)
    config = load_config(args.config)
    protocol = config["protocol"]
    raw = load_raw(config["graphs"][args.target]["path"])
    split = load_split(Path(args.split_root) / f"{args.target}.pt", args.target, int(raw["x"].shape[0]))
    train_graph = induced_partition(args.target, config["graphs"][args.target]["path"], split, "train")
    validation_graph = induced_partition(args.target, config["graphs"][args.target]["path"], split, "validation")
    classes, mapping = label_mapping(raw["y"].reshape(-1).long())
    seed_everything(args.seed)
    device = torch.device(f"cuda:{args.device}")
    encoder = GraphSAGE(
        int(protocol["input_dim"]), int(protocol["hidden_dim"]), int(protocol["output_dim"]),
        int(protocol["layers"]), float(protocol["dropout"]),
    )
    if args.checkpoint == "scratch":
        sources, pretrain_seed, pretrain_step = [], args.seed, 0
    else:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        encoder.load_state_dict(checkpoint["model"])
        sources = checkpoint["metadata"]["sources"]
        pretrain_seed = int(checkpoint["metadata"]["seed"])
        pretrain_step = int(checkpoint["step"])
    encoder.to(device)
    head = nn.Linear(int(protocol["output_dim"]), len(classes)).to(device)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(head.parameters()),
        lr=float(protocol["finetune_learning_rate"]),
        weight_decay=float(protocol["finetune_weight_decay"]),
    )
    metadata = {
        "run_id": args.run_id,
        "target": args.target,
        "target_split_hash": split_hash(split),
        "initial_checkpoint": args.checkpoint,
        "initial_sources": sources,
        "pretrain_seed": pretrain_seed,
        "pretrain_step": pretrain_step,
        "finetune_seed": args.seed,
        "classes": classes,
        "test_evaluations": 0,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    best_loss, best_epoch, patience = float("inf"), 0, 0
    start = time.monotonic()
    for epoch in range(1, int(protocol["finetune_max_epochs"]) + 1):
        encoder.train(); head.train(); optimizer.zero_grad(set_to_none=True)
        train_logits, train_y = forward_labeled(encoder, head, train_graph, mapping, device)
        train_loss = F.cross_entropy(train_logits, train_y)
        train_loss.backward(); optimizer.step()
        encoder.eval(); head.eval()
        with torch.no_grad():
            validation_logits, validation_y = forward_labeled(encoder, head, validation_graph, mapping, device)
            validation_loss = F.cross_entropy(validation_logits, validation_y)
            validation_metrics = metrics_from_logits(validation_logits, validation_y, classes)
        row = {
            "epoch": epoch,
            "train_loss": float(train_loss.detach()),
            "validation_loss": float(validation_loss.detach()),
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
            "elapsed_seconds": time.monotonic() - start,
        }
        with (run_dir / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        if epoch % int(protocol["finetune_checkpoint_interval"]) == 0:
            save(run_dir / "checkpoints" / f"epoch_{epoch}.pt", encoder, head, optimizer, epoch, metadata)
        current_validation_loss = float(validation_loss.detach())
        if current_validation_loss < best_loss - 1e-5:
            best_loss, best_epoch, patience = current_validation_loss, epoch, 0
            save(run_dir / "best.pt", encoder, head, optimizer, epoch, metadata)
        else:
            patience += 1
        if epoch % 10 == 0:
            print(json.dumps(row), flush=True)
        if epoch >= int(protocol["finetune_min_epochs"]) and patience >= int(protocol["finetune_patience"]):
            break
    final_epoch = epoch
    save(run_dir / "checkpoints" / f"epoch_{final_epoch}.pt", encoder, head, optimizer, final_epoch, metadata)
    # Model selection is complete. Load the selected validation checkpoint, then
    # construct and evaluate the test partition exactly once.
    selected = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    encoder.load_state_dict(selected["encoder"]); head.load_state_dict(selected["head"])
    test_graph = induced_partition(args.target, config["graphs"][args.target]["path"], split, "test")
    encoder.eval(); head.eval()
    with torch.no_grad():
        test_logits, test_y = forward_labeled(encoder, head, test_graph, mapping, device)
        test_metrics = metrics_from_logits(test_logits, test_y, classes)
    summary = {
        **metadata,
        "status": "complete",
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "final_epoch": final_epoch,
        "elapsed_seconds": time.monotonic() - start,
        "train_nodes": int((train_graph.data.y >= 0).sum()),
        "validation_nodes": int((validation_graph.data.y >= 0).sum()),
        "test_nodes": int((test_graph.data.y >= 0).sum()),
        "test_evaluations": 1,
        **test_metrics,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
