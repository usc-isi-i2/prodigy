from __future__ import annotations

import argparse
import json
import random
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
import torch_geometric.typing
from torch_geometric.loader import LinkNeighborLoader

from .config import load_config
from .data import GraphArtifact, load_graph
from .model import GraphSAGE


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_sampling_backend(protocol: dict) -> None:
    backend = protocol.get("sampling_backend", "auto")
    if backend == "torch_sparse":
        if not torch_geometric.typing.WITH_TORCH_SPARSE:
            raise RuntimeError("sampling_backend=torch_sparse but torch-sparse is unavailable")
        # Tucker's PyG 2.3.1 and newer pyg-lib expose incompatible
        # neighbor_sample signatures. torch-sparse matches this PyG release.
        torch_geometric.typing.WITH_PYG_LIB = False
    elif backend != "auto":
        raise ValueError(f"unknown sampling backend: {backend}")


def make_loader(graph: GraphArtifact, protocol: dict, *, validation: bool) -> LinkNeighborLoader:
    edges = graph.validation_edges if validation else graph.train_edges
    return LinkNeighborLoader(
        graph.data,
        num_neighbors=[int(protocol["fanout"])] * int(protocol["layers"]),
        edge_label_index=edges,
        edge_label=torch.ones(edges.shape[1]),
        neg_sampling_ratio=float(protocol["negatives_per_positive"]),
        batch_size=int(protocol["ssl_batch_size"]),
        shuffle=not validation,
        num_workers=0,
    )


def next_batch(loader: LinkNeighborLoader, iterator: Iterator | None) -> tuple[object, Iterator]:
    if iterator is None:
        iterator = iter(loader)
    try:
        batch = next(iterator)
    except StopIteration:
        iterator = iter(loader)
        batch = next(iterator)
    return batch, iterator


def batch_loss(model: GraphSAGE, batch, device: torch.device) -> torch.Tensor:
    batch = batch.to(device)
    embedding = model(batch.x, batch.edge_index)
    src, dst = batch.edge_label_index
    score = (embedding[src] * embedding[dst]).sum(-1)
    return torch.nn.functional.binary_cross_entropy_with_logits(score, batch.edge_label.float())


@torch.no_grad()
def validation_loss(
    model: GraphSAGE,
    loaders: list[LinkNeighborLoader],
    device: torch.device,
    batches: int,
    seed: int,
) -> tuple[float, dict[str, float]]:
    model.eval()
    cpu_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    torch.manual_seed(seed + 9173)
    per_source: dict[str, float] = {}
    for index, loader in enumerate(loaders):
        values = []
        iterator = iter(loader)
        for _ in range(batches):
            try:
                batch = next(iterator)
            except StopIteration:
                break
            values.append(float(batch_loss(model, batch, device)))
        if not values:
            raise RuntimeError(f"validation loader {index} yielded no batches")
        per_source[str(index)] = float(np.mean(values))
    torch.random.set_rng_state(cpu_state)
    if cuda_state:
        torch.cuda.set_rng_state_all(cuda_state)
    model.train()
    return float(np.mean(list(per_source.values()))), per_source


def save_checkpoint(path: Path, model: GraphSAGE, optimizer, step: int, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "metadata": metadata,
        },
        path,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/graphs.yaml")
    parser.add_argument("--sources", required=True, help="comma-separated graph names")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-root", default="state")
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()

    config = load_config(args.config)
    protocol = dict(config["protocol"])
    configure_sampling_backend(protocol)
    seed = int(protocol["seed"] if args.seed is None else args.seed)
    max_steps = int(args.max_steps or protocol["max_steps"])
    sources = [item for item in args.sources.split(",") if item]
    if not sources or len(sources) != len(set(sources)):
        raise ValueError("sources must be a non-empty unique list")
    unknown = set(sources) - set(config["graphs"])
    if unknown:
        raise ValueError(f"unknown sources: {sorted(unknown)}")

    seed_everything(seed)
    graphs = [load_graph(name, config["graphs"][name]["path"], seed=seed) for name in sources]
    train_loaders = [make_loader(graph, protocol, validation=False) for graph in graphs]
    validation_loaders = [make_loader(graph, protocol, validation=True) for graph in graphs]
    device = torch.device(f"cuda:{args.device}")
    model = GraphSAGE(
        int(protocol["input_dim"]), int(protocol["hidden_dim"]), int(protocol["output_dim"]),
        int(protocol["layers"]), float(protocol["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(protocol["learning_rate"]),
        weight_decay=float(protocol["weight_decay"]),
    )
    run_dir = Path(args.output_root) / args.run_id
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing run: {run_dir}")
    (run_dir / "checkpoints").mkdir(parents=True)
    metadata = {"run_id": args.run_id, "sources": sources, "seed": seed, "protocol": protocol}
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    checkpoints = {int(value) for value in protocol["checkpoint_steps"] if int(value) <= max_steps}
    interval = int(protocol["validation_interval"])
    minimum_steps = int(protocol["minimum_steps"])
    patience_limit = int(protocol["patience_evaluations"])
    min_improvement = float(protocol["min_relative_improvement"])
    iterators: list[Iterator | None] = [None] * len(graphs)
    history: list[dict] = []
    best_loss, best_step, patience = float("inf"), 0, 0
    start = time.monotonic()
    model.train()
    for step in range(1, max_steps + 1):
        source_index = (step - 1) % len(graphs)
        batch, iterators[source_index] = next_batch(train_loaders[source_index], iterators[source_index])
        optimizer.zero_grad(set_to_none=True)
        loss = batch_loss(model, batch, device)
        loss.backward()
        optimizer.step()
        if step in checkpoints:
            save_checkpoint(run_dir / "checkpoints" / f"step_{step}.pt", model, optimizer, step, metadata)
        if step % interval == 0:
            value, per_source = validation_loss(
                model, validation_loaders, device, int(protocol["validation_batches"]), seed
            )
            row = {
                "step": step, "train_loss": float(loss), "validation_loss": value,
                "per_source_validation_loss": dict(zip(sources, per_source.values())),
                "elapsed_seconds": time.monotonic() - start,
            }
            history.append(row)
            with (run_dir / "validation.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            relative = (best_loss - value) / best_loss if np.isfinite(best_loss) else float("inf")
            if relative >= min_improvement:
                best_loss, best_step, patience = value, step, 0
                save_checkpoint(run_dir / "best.pt", model, optimizer, step, metadata)
            else:
                patience += 1
            print(json.dumps(row), flush=True)
            if step >= minimum_steps and patience >= patience_limit:
                break
    final_step = step
    save_checkpoint(run_dir / "checkpoints" / f"step_{final_step}.pt", model, optimizer, final_step, metadata)
    summary = {
        **metadata, "status": "complete", "best_step": best_step, "best_validation_loss": best_loss,
        "final_step": final_step, "elapsed_seconds": time.monotonic() - start,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
