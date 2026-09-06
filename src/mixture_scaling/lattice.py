from __future__ import annotations

import argparse
import fcntl
import itertools
import json
import math
import random
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.typing
from torch_geometric.loader import LinkNeighborLoader, NeighborLoader
from torch_geometric.data import Data

from .config import load_config
from .data import GraphArtifact, load_graph
from .model import GraphMAE, GraphSAGE
from .train import configure_sampling_backend, next_batch


SOURCE_ORDER = (
    "ukr_rus_twitter",
    "covid19_twitter",
    "midterm",
    "covid_political",
    "election2020",
    "ukr_rus_suspended",
    "twibot20",
    "cp_hk_twitter",
    "facebook_page_reference",
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def lattice_rows() -> list[tuple[str, tuple[str, ...]]]:
    rows: list[tuple[str, tuple[str, ...]]] = []
    rows.extend((f"ss_{source}", (source,)) for source in SOURCE_ORDER)
    rows.extend(
        (f"pair_{left}__{right}", (left, right))
        for left, right in itertools.combinations(SOURCE_ORDER, 2)
    )
    rows.extend(
        (f"loo_{heldout}", tuple(source for source in SOURCE_ORDER if source != heldout))
        for heldout in SOURCE_ORDER
    )
    assert len(rows) == 54
    return rows


def selected_rows(selection: str) -> list[tuple[str, tuple[str, ...]]]:
    rows = lattice_rows()
    if selection == "full":
        return rows
    if selection == "gate":
        wanted = {
            "ss_covid19_twitter",
            "pair_ukr_rus_suspended__facebook_page_reference",
            "loo_covid19_twitter",
        }
        return [row for row in rows if row[0] in wanted]
    names = {name for name in selection.split(",") if name}
    chosen = [row for row in rows if row[0] in names]
    if len(chosen) != len(names):
        raise ValueError(f"unknown run ids: {sorted(names - {name for name, _ in chosen})}")
    return chosen


def make_lp_loader(graph: GraphArtifact, protocol: dict, validation: bool):
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


def _raw_data(name: str, path: str | Path) -> Data:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(raw, dict):
        raw = raw.to_dict()
    x, edge_index, y = raw.get("x"), raw.get("edge_index"), raw.get("y")
    if x is None or edge_index is None:
        raise ValueError(f"{name}: artifact requires x and edge_index")
    if x.ndim != 2 or x.shape[1] != 768:
        raise ValueError(f"{name}: expected [N,768] features, got {tuple(x.shape)}")
    return Data(x=x.float(), edge_index=edge_index.long(), y=y)


def load_shared_graph(name, path, objective, protocol, seed, cache_root: Path) -> GraphArtifact:
    """Load raw topology for MAE or an exactly cached LP edge partition."""
    path = Path(path)
    if objective == "graphmae":
        data = _raw_data(name, path)
        empty = torch.empty((2, 0), dtype=torch.long)
        return GraphArtifact(name, path, data, empty, empty)

    cache_root.mkdir(parents=True, exist_ok=True)
    cache_path = cache_root / f"{name}_edge_split_s{seed}.pt"
    lock_path = cache_root / f"{name}_edge_split_s{seed}.lock"
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not cache_path.is_file():
            graph = load_graph(
                name, path,
                validation_fraction=float(protocol["edge_validation_fraction"]),
                seed=seed,
            )
            temporary = cache_path.with_suffix(".tmp")
            torch.save(
                {"train_edges": graph.train_edges, "validation_edges": graph.validation_edges},
                temporary,
            )
            temporary.replace(cache_path)
            return graph
        split = torch.load(cache_path, map_location="cpu", weights_only=False)
    data = _raw_data(name, path)
    train_edges = split["train_edges"].long()
    data.edge_index = torch.cat((train_edges, train_edges.flip(0)), dim=1)
    return GraphArtifact(name, path, data, train_edges, split["validation_edges"].long())


def node_partition(graph: GraphArtifact, validation: bool, seed: int, fraction: float) -> torch.Tensor:
    count = int(graph.data.num_nodes)
    order = torch.randperm(count, generator=torch.Generator().manual_seed(seed))
    n_validation = max(1, round(count * fraction))
    return order[:n_validation] if validation else order[n_validation:]


def make_mae_loader(graph: GraphArtifact, protocol: dict, validation: bool, seed: int):
    nodes = node_partition(
        graph, validation, seed + 104729, float(protocol["node_validation_fraction"])
    )
    return NeighborLoader(
        graph.data,
        input_nodes=nodes,
        num_neighbors=[int(protocol["fanout"])] * int(protocol["layers"]),
        batch_size=int(protocol["ssl_batch_size"]),
        shuffle=not validation,
        num_workers=0,
    )


def lp_loss(model: GraphSAGE, batch, device: torch.device) -> torch.Tensor:
    batch = batch.to(device)
    embedding = model(batch.x, batch.edge_index)
    src, dst = batch.edge_label_index
    logits = (embedding[src] * embedding[dst]).sum(-1)
    return F.binary_cross_entropy_with_logits(logits, batch.edge_label.float())


def mae_loss(
    model: GraphMAE,
    batch,
    device: torch.device,
    mask_rate: float,
    alpha: float,
    generator: torch.Generator,
) -> torch.Tensor:
    batch = batch.to(device)
    roots = int(batch.batch_size)
    mask_count = max(1, round(roots * mask_rate))
    mask = torch.randperm(roots, generator=generator)[:mask_count].to(device)
    target = batch.x[mask].float()
    corrupted = batch.x.clone()
    corrupted[mask] = model.mask_token
    prediction = model(corrupted, batch.edge_index)[mask]
    cosine_error = 1.0 - F.cosine_similarity(prediction, target, dim=-1)
    return cosine_error.clamp_min(0).pow(alpha).mean()


def save_checkpoint(path: Path, model, optimizer, step: int, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoder = model.encoder if isinstance(model, GraphMAE) else model
    torch.save(
        {
            "model": encoder.state_dict(),
            "pretrain_model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "metadata": metadata,
        },
        path,
    )


def update_selection(value, step, best_loss, best_step, reference, patience, threshold):
    save_best = value < best_loss
    if save_best:
        best_loss, best_step = value, step
    relative = (reference - value) / reference if math.isfinite(reference) else math.inf
    if relative >= threshold:
        reference, patience = value, 0
    else:
        patience += 1
    return best_loss, best_step, reference, patience, save_best


@torch.no_grad()
def validate(model, loaders, objective, device, protocol, seed):
    model.eval()
    values = {}
    generator = torch.Generator().manual_seed(seed + 32452843)
    for source, loader in loaders.items():
        losses = []
        iterator = iter(loader)
        for _ in range(int(protocol["validation_batches"])):
            try:
                batch = next(iterator)
            except StopIteration:
                break
            if objective == "lp":
                loss = lp_loss(model, batch, device)
            else:
                loss = mae_loss(
                    model, batch, device, float(protocol["mask_rate"]),
                    float(protocol["sce_alpha"]), generator,
                )
            losses.append(float(loss))
        if not losses:
            raise RuntimeError(f"validation loader for {source} yielded no batches")
        values[source] = float(np.mean(losses))
    model.train()
    return float(np.mean(list(values.values()))), values


def train_one(run_id, sources, objective, graphs, config, device, output_root, seed):
    protocol = dict(config["protocol"])
    run_dir = output_root / objective / run_id
    if (run_dir / "summary.json").is_file():
        return json.loads((run_dir / "summary.json").read_text())
    if run_dir.exists():
        raise FileExistsError(f"partial run exists: {run_dir}")
    (run_dir / "checkpoints").mkdir(parents=True)
    seed_everything(seed)
    selected = {source: graphs[source] for source in sources}
    if objective == "lp":
        train_loaders = {s: make_lp_loader(g, protocol, False) for s, g in selected.items()}
        val_loaders = {s: make_lp_loader(g, protocol, True) for s, g in selected.items()}
        model = GraphSAGE(
            int(protocol["input_dim"]), int(protocol["hidden_dim"]),
            int(protocol["output_dim"]), int(protocol["layers"]), float(protocol["dropout"]),
        ).to(device)
    else:
        train_loaders = {s: make_mae_loader(g, protocol, False, seed) for s, g in selected.items()}
        val_loaders = {s: make_mae_loader(g, protocol, True, seed) for s, g in selected.items()}
        model = GraphMAE(
            int(protocol["input_dim"]), int(protocol["hidden_dim"]),
            int(protocol["output_dim"]), int(protocol["layers"]), float(protocol["dropout"]),
        ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(protocol["learning_rate"]),
        weight_decay=float(protocol["weight_decay"]),
    )
    metadata = {
        "run_id": run_id, "sources": list(sources), "seed": seed,
        "objective": objective, "protocol": protocol, "source_confined": True,
        "source_schedule": "uniform_round_robin", "shared_graph_worker": True,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    wandb_run = None
    try:
        import wandb

        wandb_run = wandb.init(
            project="sage-graphmae-source-lattice",
            name=f"{objective}_{run_id}_s{seed}",
            dir=str(run_dir),
            mode="offline",
            config=metadata,
            reinit=True,
        )
    except ImportError:
        (run_dir / "WANDB_UNAVAILABLE").write_text("wandb is not installed\n")
    iterators: dict[str, Iterator | None] = {source: None for source in sources}
    best_loss = reference = math.inf
    best_step = patience = 0
    start = time.monotonic()
    mask_generator = torch.Generator().manual_seed(seed + 49979687)
    train_window: list[float] = []
    model.train()
    checkpoints = set(map(int, protocol["checkpoint_steps"]))
    for step in range(1, int(protocol["max_steps"]) + 1):
        source = sources[(step - 1) % len(sources)]
        batch, iterators[source] = next_batch(train_loaders[source], iterators[source])
        optimizer.zero_grad(set_to_none=True)
        if objective == "lp":
            loss = lp_loss(model, batch, device)
        else:
            loss = mae_loss(
                model, batch, device, float(protocol["mask_rate"]),
                float(protocol["sce_alpha"]), mask_generator,
            )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}: {loss}")
        loss.backward()
        optimizer.step()
        train_window.append(float(loss))
        if step in checkpoints:
            save_checkpoint(run_dir / "checkpoints" / f"step_{step}.pt", model, optimizer, step, metadata)
        if step % int(protocol["validation_interval"]):
            continue
        value, per_source = validate(model, val_loaders, objective, device, protocol, seed)
        elapsed = time.monotonic() - start
        row = {
            "step": step, "train_loss": float(np.mean(train_window)), "validation_loss": value,
            "per_source_validation_loss": per_source, "elapsed_seconds": elapsed,
            "updates_per_second": step / elapsed,
        }
        with (run_dir / "validation.jsonl").open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        if wandb_run is not None:
            wandb_run.log(row, step=step)
        print(json.dumps({"run_id": run_id, **row}), flush=True)
        train_window.clear()
        best_loss, best_step, reference, patience, save_best = update_selection(
            value, step, best_loss, best_step, reference, patience,
            float(protocol["min_relative_improvement"]),
        )
        if save_best:
            save_checkpoint(run_dir / "best.pt", model, optimizer, step, metadata)
        if step >= int(protocol["minimum_steps"]) and patience >= int(protocol["patience_evaluations"]):
            break
    save_checkpoint(run_dir / "checkpoints" / f"step_{step}.pt", model, optimizer, step, metadata)
    summary = {
        **metadata, "status": "complete", "best_step": best_step,
        "best_validation_loss": best_loss, "final_step": step,
        "elapsed_seconds": time.monotonic() - start,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if wandb_run is not None:
        wandb_run.summary.update(summary)
        wandb_run.finish()
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--objective", choices=("lp", "graphmae"), required=True)
    parser.add_argument("--selection", default="full")
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--cache-root")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.device not in (0, 1, 2, 3):
        raise ValueError("only owned Tucker GPUs 0-3 are allowed")
    if not 0 <= args.worker_index < args.workers:
        raise ValueError("invalid worker index")
    config = load_config(args.config)
    configure_sampling_backend(config["protocol"])
    rows = selected_rows(args.selection)[args.worker_index::args.workers]
    required = {source for _, sources in rows for source in sources}
    # Each worker loads every graph it needs exactly once and reuses it across models.
    graphs = {
        source: load_shared_graph(
            source, config["graphs"][source]["path"], args.objective,
            config["protocol"], args.seed,
            Path(args.cache_root or (Path(args.output_root) / "_cache")),
        )
        for source in SOURCE_ORDER if source in required
    }
    device = torch.device(f"cuda:{args.device}")
    output_root = Path(args.output_root)
    for run_id, sources in rows:
        summary = train_one(
            run_id, sources, args.objective, graphs, config, device, output_root, args.seed
        )
        print(json.dumps({"completed": run_id, "final_step": summary["final_step"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
