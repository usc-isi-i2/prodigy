from __future__ import annotations

import argparse
import json
import random
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch

from .config import load_config
from .model import GraphSAGE
from .strict_data import load_raw, load_split, split_hash, ssl_train_edge_partition
from .train import batch_loss, configure_sampling_backend, make_loader, next_batch, save_checkpoint, validation_loss


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def update_selection(
    value: float,
    step: int,
    best_loss: float,
    best_step: int,
    patience_reference: float,
    patience: int,
    min_improvement: float,
) -> tuple[float, int, float, int, bool]:
    """Track the absolute best checkpoint separately from patience resets."""
    save_best = value < best_loss
    if save_best:
        best_loss, best_step = value, step
    relative = (
        (patience_reference - value) / patience_reference
        if np.isfinite(patience_reference) else float("inf")
    )
    if relative >= min_improvement:
        patience_reference, patience = value, 0
    else:
        patience += 1
    return best_loss, best_step, patience_reference, patience, save_best


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split-root", required=True)
    parser.add_argument("--sources", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    if args.device not in (2, 3):
        raise ValueError("strict pilot may use only GPUs 2 and 3")
    config = load_config(args.config)
    protocol = dict(config["protocol"])
    configure_sampling_backend(protocol)
    sources = [source for source in args.sources.split(",") if source]
    if not sources or len(sources) != len(set(sources)):
        raise ValueError("sources must be nonempty and unique")
    seed_everything(args.seed)

    ssl_graphs, split_hashes = [], {}
    for source in sources:
        graph_config = config["graphs"][source]
        raw = load_raw(graph_config["path"])
        split = load_split(Path(args.split_root) / f"{source}.pt", source, int(raw["x"].shape[0]))
        ssl_graphs.append(ssl_train_edge_partition(
            source, graph_config["path"], split,
            validation_fraction=float(protocol["ssl_edge_validation_fraction"]),
            seed=int(protocol["ssl_edge_split_seed"]),
        ))
        split_hashes[source] = split_hash(split)

    train_loaders = [make_loader(graph, protocol, validation=False) for graph in ssl_graphs]
    validation_loaders = [make_loader(graph, protocol, validation=True) for graph in ssl_graphs]
    device = torch.device(f"cuda:{args.device}")
    model = GraphSAGE(
        int(protocol["input_dim"]), int(protocol["hidden_dim"]), int(protocol["output_dim"]),
        int(protocol["layers"]), float(protocol["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(protocol["learning_rate"]), weight_decay=float(protocol["weight_decay"])
    )
    run_dir = Path(args.output_root) / args.run_id
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing run: {run_dir}")
    (run_dir / "checkpoints").mkdir(parents=True)
    metadata = {
        "run_id": args.run_id,
        "sources": sources,
        "seed": args.seed,
        "protocol": protocol,
        "split_hashes": split_hashes,
        "ssl_train_partition": "train_nodes/train_edges",
        "ssl_validation_partition": "train_nodes/heldout_edges",
        "ssl_edge_validation_fraction": float(protocol["ssl_edge_validation_fraction"]),
        "ssl_edge_split_seed": int(protocol["ssl_edge_split_seed"]),
        "source_confined": True,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    checkpoints = set(map(int, protocol["checkpoint_steps"]))
    max_steps = int(args.max_steps or protocol["max_steps"])
    interval = int(protocol["validation_interval"])
    iterators: list[Iterator | None] = [None] * len(train_loaders)
    best_loss, best_step, patience_reference, patience = float("inf"), 0, float("inf"), 0
    start = time.monotonic()
    model.train()
    for step in range(1, max_steps + 1):
        # Uniform deterministic source rotation; an entire batch is confined to
        # the selected graph, including negative samples.
        source_index = (step - 1) % len(train_loaders)
        batch, iterators[source_index] = next_batch(train_loaders[source_index], iterators[source_index])
        optimizer.zero_grad(set_to_none=True)
        loss = batch_loss(model, batch, device)
        loss.backward()
        optimizer.step()
        if step in checkpoints:
            save_checkpoint(run_dir / "checkpoints" / f"step_{step}.pt", model, optimizer, step, metadata)
        if step % interval:
            continue
        value, per_source = validation_loss(
            model, validation_loaders, device, int(protocol["validation_batches"]), args.seed
        )
        row = {
            "step": step,
            "train_loss": float(loss),
            "validation_loss": value,
            "per_source_validation_loss": dict(zip(sources, per_source.values())),
            "elapsed_seconds": time.monotonic() - start,
        }
        with (run_dir / "validation.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        best_loss, best_step, patience_reference, patience, save_best = update_selection(
            value, step, best_loss, best_step, patience_reference, patience,
            float(protocol["min_relative_improvement"]),
        )
        if save_best:
            save_checkpoint(run_dir / "best.pt", model, optimizer, step, metadata)
        print(json.dumps(row), flush=True)
        if step >= int(protocol["minimum_steps"]) and patience >= int(protocol["patience_evaluations"]):
            break
    final_step = step
    save_checkpoint(run_dir / "checkpoints" / f"step_{final_step}.pt", model, optimizer, final_step, metadata)
    summary = {
        **metadata,
        "status": "complete",
        "best_step": best_step,
        "best_validation_loss": best_loss,
        "final_step": final_step,
        "elapsed_seconds": time.monotonic() - start,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
