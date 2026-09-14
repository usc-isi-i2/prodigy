from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch_geometric.loader import NeighborLoader

from .config import load_config
from .model import GraphSAGE
from .strict_data import load_raw, load_split, split_hash, ssl_train_edge_partition
from .structural_logistic_baseline import structural_features
from .train import batch_loss, configure_sampling_backend, make_loader, next_batch, save_checkpoint, validation_loss
from .train_strict import seed_everything, update_selection


def set_features(graph, mode: str) -> dict:
    if mode == "existing":
        return {"mode": mode, "dimension": int(graph.data.x.shape[1])}
    if mode != "structural":
        raise ValueError(f"unknown feature mode: {mode}")
    values, names = structural_features(graph.data.edge_index, int(graph.data.num_nodes))
    mean = values.mean(axis=0, dtype=np.float64)
    std = values.std(axis=0, dtype=np.float64)
    std[std < 1e-8] = 1.0
    graph.data.x = torch.from_numpy(((values - mean) / std).astype(np.float32))
    return {"mode": mode, "dimension": int(graph.data.x.shape[1]), "names": names,
            "normalization": "per_source_train_partition_zscore"}


def make_node_loader(graph, protocol: dict, *, shuffle: bool) -> NeighborLoader:
    return NeighborLoader(
        graph.data,
        input_nodes=torch.arange(graph.data.num_nodes),
        num_neighbors=[int(protocol["fanout"])] * int(protocol["layers"]),
        batch_size=int(protocol["ssl_batch_size"]),
        shuffle=shuffle,
        num_workers=0,
    )


def graphmae_loss(encoder, decoder, batch, device, mask_rate: float, alpha: float) -> torch.Tensor:
    batch = batch.to(device)
    seeds = int(batch.batch_size)
    count = max(1, int(round(seeds * mask_rate)))
    masked = torch.randperm(seeds, device=device)[:count]
    target = batch.x[masked].clone()
    corrupted = batch.x.clone()
    corrupted[masked] = 0.0
    prediction = decoder(encoder(corrupted, batch.edge_index)[masked])
    cosine = torch.nn.functional.cosine_similarity(prediction, target, dim=-1, eps=1e-8)
    return torch.pow(1.0 - cosine, alpha).mean()


@torch.no_grad()
def graphmae_validation(encoder, decoder, loaders, device, batches, seed, mask_rate, alpha):
    encoder.eval(); decoder.eval()
    cpu_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    torch.manual_seed(seed + 19073)
    values = []
    for loader in loaders:
        local = []
        for index, batch in enumerate(loader):
            if index >= batches:
                break
            local.append(float(graphmae_loss(encoder, decoder, batch, device, mask_rate, alpha)))
        if not local:
            raise RuntimeError("GraphMAE validation loader yielded no batches")
        values.append(float(np.mean(local)))
    torch.random.set_rng_state(cpu_state)
    if cuda_state:
        torch.cuda.set_rng_state_all(cuda_state)
    encoder.train(); decoder.train()
    return float(np.mean(values)), values


def save(path, encoder, decoder, optimizer, step, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": encoder.state_dict(), "decoder": decoder.state_dict(),
                "optimizer": optimizer.state_dict(), "step": step, "metadata": metadata}, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split-root", required=True)
    parser.add_argument("--sources", required=True)
    parser.add_argument("--objective", choices=("link", "graphmae"), required=True)
    parser.add_argument("--feature-mode", choices=("existing", "structural"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    if args.device not in (2, 3):
        raise ValueError("transfer pilot may use only GPUs 2 and 3")
    config = load_config(args.config)
    protocol = dict(config["protocol"])
    configure_sampling_backend(protocol)
    sources = [item for item in args.sources.split(",") if item]
    if not sources or len(sources) != len(set(sources)):
        raise ValueError("sources must be nonempty and unique")
    seed_everything(args.seed)
    graphs, hashes, feature_metadata = [], {}, {}
    for source in sources:
        raw = load_raw(config["graphs"][source]["path"])
        split = load_split(Path(args.split_root) / f"{source}.pt", source, int(raw["x"].shape[0]))
        graph = ssl_train_edge_partition(
            source, config["graphs"][source]["path"], split,
            validation_fraction=float(protocol["ssl_edge_validation_fraction"]),
            seed=int(protocol["ssl_edge_split_seed"]),
        )
        feature_metadata[source] = set_features(graph, args.feature_mode)
        graphs.append(graph); hashes[source] = split_hash(split)
    input_dim = int(graphs[0].data.x.shape[1])
    if any(int(graph.data.x.shape[1]) != input_dim for graph in graphs):
        raise ValueError("source input dimensions differ")
    protocol["input_dim"] = input_dim
    device = torch.device(f"cuda:{args.device}")
    encoder = GraphSAGE(input_dim, int(protocol["hidden_dim"]), int(protocol["output_dim"]),
                        int(protocol["layers"]), float(protocol["dropout"])).to(device)
    decoder = nn.Linear(int(protocol["output_dim"]), input_dim).to(device)
    parameters = list(encoder.parameters()) + (list(decoder.parameters()) if args.objective == "graphmae" else [])
    optimizer = torch.optim.AdamW(parameters, lr=float(protocol["learning_rate"]),
                                  weight_decay=float(protocol["weight_decay"]))
    metadata = {
        "run_id": args.run_id, "sources": sources, "seed": args.seed, "protocol": protocol,
        "objective": args.objective, "feature_mode": args.feature_mode,
        "feature_metadata": feature_metadata, "split_hashes": hashes,
        "ssl_train_partition": "train_nodes", "source_confined": True,
        "graphmae_mask_rate": 0.70, "graphmae_loss": "scaled_cosine_alpha2",
    }
    run_dir = Path(args.output_root) / args.run_id
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite {run_dir}")
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if args.objective == "link":
        train_loaders = [make_loader(graph, protocol, validation=False) for graph in graphs]
        validation_loaders = [make_loader(graph, protocol, validation=True) for graph in graphs]
    else:
        train_loaders = [make_node_loader(graph, protocol, shuffle=True) for graph in graphs]
        validation_loaders = [make_node_loader(graph, protocol, shuffle=False) for graph in graphs]
    iterators: list[Iterator | None] = [None] * len(train_loaders)
    best_loss, best_step, reference, patience = float("inf"), 0, float("inf"), 0
    checkpoints = set(map(int, protocol["checkpoint_steps"]))
    start = time.monotonic()
    for step in range(1, int(protocol["max_steps"]) + 1):
        source_index = (step - 1) % len(train_loaders)
        batch, iterators[source_index] = next_batch(train_loaders[source_index], iterators[source_index])
        optimizer.zero_grad(set_to_none=True)
        loss = (batch_loss(encoder, batch, device) if args.objective == "link" else
                graphmae_loss(encoder, decoder, batch, device, 0.70, 2.0))
        loss.backward(); optimizer.step()
        if step in checkpoints:
            save(run_dir / "checkpoints" / f"step_{step}.pt", encoder, decoder, optimizer, step, metadata)
        if step % int(protocol["validation_interval"]):
            continue
        if args.objective == "link":
            value, per_source_map = validation_loss(
                encoder, validation_loaders, device, int(protocol["validation_batches"]), args.seed
            )
            per_source = list(per_source_map.values())
        else:
            value, per_source = graphmae_validation(
                encoder, decoder, validation_loaders, device, int(protocol["validation_batches"]),
                args.seed, 0.70, 2.0,
            )
        row = {"step": step, "train_loss": float(loss), "validation_loss": value,
               "per_source_validation_loss": dict(zip(sources, per_source)),
               "elapsed_seconds": time.monotonic() - start}
        with (run_dir / "validation.jsonl").open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        best_loss, best_step, reference, patience, save_best = update_selection(
            value, step, best_loss, best_step, reference, patience,
            float(protocol["min_relative_improvement"]),
        )
        if save_best:
            save(run_dir / "best.pt", encoder, decoder, optimizer, step, metadata)
        print(json.dumps(row), flush=True)
        if step >= int(protocol["minimum_steps"]) and patience >= int(protocol["patience_evaluations"]):
            break
    save(run_dir / "checkpoints" / f"step_{step}.pt", encoder, decoder, optimizer, step, metadata)
    summary = {**metadata, "status": "complete", "best_step": best_step,
               "best_validation_loss": best_loss, "final_step": step,
               "elapsed_seconds": time.monotonic() - start}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
