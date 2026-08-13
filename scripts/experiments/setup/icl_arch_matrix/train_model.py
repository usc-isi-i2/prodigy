#!/usr/bin/env python3
"""Train one VISION or GILT source-set model on the common NM protocol."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from scripts.experiments.setup.icl_arch_matrix.architecture_adapters import (
    PINS,
    build_adapter,
    build_optimizer,
)
from scripts.experiments.setup.icl_arch_matrix.common_protocol import (
    CHECKPOINT_STEPS,
    TRAIN_BATCH_SIZE,
    TRAIN_STEPS,
    build_dataset,
    build_loader,
    iter_episodes,
    load_config,
    reset_episode_rng,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", choices=("vision", "gilt"), required=True)
    parser.add_argument("--upstream-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--sources", required=True)
    parser.add_argument("--state-root", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--steps", type=int, default=TRAIN_STEPS)
    parser.add_argument("--workers", type=int, default=0)
    return parser.parse_args()


def save_checkpoint(path: Path, model, optimizer, scheduler, args, step: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "architecture": args.architecture,
            "model_id": args.model_id,
            "sources": args.sources,
            "seed": 0,
            "step": step,
            "upstream": PINS[args.architecture],
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
        },
        path,
    )


def main() -> int:
    args = parse_args()
    if args.steps != TRAIN_STEPS:
        raise ValueError(f"registered comparison budget is exactly {TRAIN_STEPS} updates")
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    config = load_config(args.config)
    dataset = build_dataset(config)
    loader = build_loader(
        dataset,
        config,
        split="train",
        sources=args.sources,
        batch_count=args.steps,
        batch_size=TRAIN_BATCH_SIZE,
        workers=args.workers,
    )
    model = build_adapter(args.architecture, args.upstream_root).to(device)
    optimizer = build_optimizer(model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=model.learning_rate * 0.05
    )

    run_dir = Path(args.state_root) / args.architecture / args.model_id
    metrics_path = run_dir / "train_metrics.jsonl"
    if run_dir.exists():
        raise FileExistsError(f"refusing ambiguous resume into existing run: {run_dir}")
    run_dir.mkdir(parents=True)
    started = time.time()
    reset_episode_rng()
    with metrics_path.open("w", encoding="utf-8") as metrics:
        for step, batch in enumerate(loader, start=1):
            model.train()
            graphs = batch[0].to(device)
            moved = (graphs,) + tuple(x.to(device) if torch.is_tensor(x) else x for x in batch[1:])
            optimizer.zero_grad(set_to_none=True)
            losses, accuracies = [], []
            for episode in iter_episodes(moved):
                episode_loss, accuracy = model.episode_loss_and_accuracy(episode)
                # Backpropagate each episode immediately. This is mathematically the
                # same mean loss over the registered four-episode update, while
                # avoiding retention of four large VISION autograd graphs at once.
                (episode_loss / TRAIN_BATCH_SIZE).backward()
                losses.append(episode_loss.detach())
                accuracies.append(accuracy.detach())
            if len(losses) != TRAIN_BATCH_SIZE:
                raise RuntimeError(
                    f"expected {TRAIN_BATCH_SIZE} episodes in an update, got {len(losses)}"
                )
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            loss = torch.stack(losses).mean()

            row = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "accuracy": float(torch.stack(accuracies).mean().cpu()),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "elapsed_seconds": time.time() - started,
            }
            metrics.write(json.dumps(row, sort_keys=True) + "\n")
            metrics.flush()
            if step == 1 or step % 20 == 0:
                print(json.dumps(row, sort_keys=True), flush=True)
            if step in CHECKPOINT_STEPS:
                save_checkpoint(
                    run_dir / "checkpoint" / f"state_dict_{step}.pt",
                    model,
                    optimizer,
                    scheduler,
                    args,
                    step,
                )
    terminal = run_dir / "checkpoint" / f"state_dict_{TRAIN_STEPS}.pt"
    if not terminal.is_file():
        raise RuntimeError(f"terminal checkpoint was not written: {terminal}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
