"""Fixed-budget LP ladder using the existing endpoint-only MLP fast path."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch

from .config import load_config
from .evaluate_lattice import LP_TARGETS, load_pair_module
from .evaluate_node_only import evaluate_lp
from .lattice import SOURCE_ORDER, load_shared_graph, next_batch
from .node_only_transfer import build_model, lp_loss, make_direct_lp_loader, save_checkpoint, seed_everything, validate


def ladder_rows():
    return [(f"ladder_r{k}", SOURCE_ORDER[:k]) for k in range(1, len(SOURCE_ORDER) + 1)]


def source_schedule(sources, steps):
    if not sources or steps < 1:
        raise ValueError("sources and positive steps are required")
    return [sources[i % len(sources)] for i in range(steps)]


def resident_sources(sizes, budget):
    """Keep the greatest number of sources resident; stream the remainder."""
    chosen = set()
    for source, size in sorted(sizes.items(), key=lambda item: (item[1], item[0])):
        if size <= budget:
            chosen.add(source)
            budget -= size
    return chosen


def train_rung(run_id, sources, graphs, config, args, device):
    run_dir = Path(args.state_root) / "lp" / run_id
    if (run_dir / "summary.json").is_file():
        summary = json.loads((run_dir / "summary.json").read_text())
        if (summary["sources"] != list(sources) or summary["final_step"] != args.steps
                or summary["seed"] != args.seed or not (run_dir / "best.pt").is_file()):
            raise ValueError(f"completed run does not match requested protocol: {run_dir}")
        return
    if run_dir.exists():
        raise FileExistsError(f"refusing partial run: {run_dir}")
    run_dir.mkdir(parents=True)
    protocol = dict(config["protocol"], max_steps=args.steps)
    seed_everything(args.seed)
    model = build_model("lp", protocol, device)
    optimizer = torch.optim.AdamW(model.parameters(),
        lr=float(protocol["node_mlp_lp_learning_rate"]), weight_decay=float(protocol["weight_decay"]))
    free, _ = torch.cuda.mem_get_info(device)
    budget = max(0, min(int(args.feature_budget_gib * 2**30), free - 6 * 2**30))
    sizes = {s: graphs[s].data.x.numel() * graphs[s].data.x.element_size() for s in sources}
    resident = resident_sources(sizes, budget)
    features = {s: graphs[s].data.x.to(device) if s in resident else graphs[s].data.x for s in sources}
    train = {s: make_direct_lp_loader(graphs[s], protocol, False, args.seed, features[s]) for s in sources}
    val = {s: make_direct_lp_loader(graphs[s], protocol, True, args.seed, features[s]) for s in sources}
    iterators = {s: None for s in sources}
    counts = dict.fromkeys(sources, 0)
    metadata = {
        "run_id": run_id, "sources": list(sources), "seed": args.seed, "objective": "lp",
        "architecture": "node_mlp", "input_view": "center_node_features_only",
        "uses_topology_in_encoder": False, "protocol": protocol,
        "checkpoint_selection": "fixed_terminal_budget", "source_sampling": "uniform_round_robin",
        "training_negative_sampling": "source_confined_uniform_approximate_excluding_self_loops",
        "feature_residency": {s: "gpu" if s in resident else "cpu_direct_pinned" for s in sources},
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"starting": run_id, "feature_residency": metadata["feature_residency"]}), flush=True)
    started = time.monotonic()
    for step, source in enumerate(source_schedule(sources, args.steps), 1):
        batch, iterators[source] = next_batch(train[source], iterators[source])
        optimizer.zero_grad(set_to_none=True)
        loss = lp_loss(model, batch, device)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss: {run_id} step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(protocol["gradient_clip_norm"]))
        optimizer.step()
        counts[source] += 1
        if step % 250 == 0 or step == args.steps:
            print(json.dumps({"run_id": run_id, "step": step, "loss": float(loss),
                "elapsed_seconds": time.monotonic() - started}), flush=True)
    training_seconds = time.monotonic() - started
    validation = {s: validate(model, graphs[s], val[s], "lp", device, protocol, args.seed) for s in sources}
    metadata["source_update_counts"] = counts
    save_checkpoint(run_dir / "checkpoints" / f"step_{args.steps}.pt", model, optimizer, args.steps, metadata)
    # The existing evaluator expects best.pt. It is explicitly the terminal model,
    # not a model chosen on target scores or an unequal early-stopping budget.
    (run_dir / "best.pt").symlink_to(f"checkpoints/step_{args.steps}.pt")
    summary = {**metadata, "status": "complete", "final_step": args.steps, "best_step": args.steps,
        "training_seconds": training_seconds, "elapsed_seconds": time.monotonic() - started,
        "source_validation_losses": validation}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"completed": run_id, "training_seconds": training_seconds}), flush=True)


def aggregate(args):
    rows = []
    for run_id, sources in ladder_rows():
        for target in LP_TARGETS:
            path = Path(args.output_root) / "lp" / f"{run_id}__to__{target}.json"
            payload = json.loads(path.read_text())
            if payload["sources"] != list(sources) or payload["checkpoint_step"] != args.steps:
                raise ValueError(f"unexpected evaluation metadata: {path}")
            if payload["gates"]["holdout_leakage_edges"] != 0 or payload["gates"]["endpoint_sensitivity"] <= 0:
                raise ValueError(f"failed LP gate: {path}")
            rows.append({"run_id": run_id, "n_sources": len(sources), "sources": ",".join(sources),
                "target": target, "target_in_pretraining": target in sources,
                "seed": payload["seed"], "checkpoint_step": payload["checkpoint_step"],
                "roc_auc": payload["report"]["auc"]})
    path = Path(args.output_root) / "ladder_results.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (path.parent / "COMPLETE.json").write_text(json.dumps({"models": 9, "cells": len(rows),
        "status": "complete", "seed": args.seed, "steps": args.steps}, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=("train", "eval", "aggregate", "plan"))
    p.add_argument("--config", default="configs/node_only_transfer.yaml")
    p.add_argument("--state-root", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--cache-root", required=True)
    p.add_argument("--prodigy-root", default="/dataMeR1/phil/gfm/prodigy-nm-pairs")
    p.add_argument("--steps", type=int, default=2500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=int, choices=(2, 3), default=2)
    p.add_argument("--worker-index", type=int, default=0)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--feature-budget-gib", type=float, default=70)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    if args.steps < 1 or not 0 <= args.worker_index < args.workers:
        p.error("invalid budget or worker assignment")
    if args.phase == "plan":
        print(json.dumps({"rows": ladder_rows(), "targets": LP_TARGETS, "steps_per_rung": args.steps,
            "total_updates": 9 * args.steps, "cells": 54}, indent=2))
        return
    if args.phase == "aggregate":
        aggregate(args)
        return
    torch.set_num_threads(args.threads)
    config = load_config(args.config)
    device = torch.device(f"cuda:{args.device}")
    if args.phase == "train":
        graphs = {}
        for run_id, sources in ladder_rows()[args.worker_index::args.workers]:
            for source in sources:
                if source not in graphs:
                    started = time.monotonic()
                    graphs[source] = load_shared_graph(source, config["graphs"][source]["path"],
                        "lp", config["protocol"], args.seed, Path(args.cache_root))
                    # The node-only encoder never uses message-passing adjacency.
                    graphs[source].data.edge_index = torch.empty((2, 0), dtype=torch.long)
                    print(json.dumps({"loaded": source, "seconds": time.monotonic()-started}), flush=True)
            train_rung(run_id, sources, graphs, config, args, device)
            torch.cuda.empty_cache()
    else:
        pair = load_pair_module(Path(args.prodigy_root))
        for target in LP_TARGETS[args.worker_index::args.workers]:
            evaluate_lp(target, args, config, ladder_rows(), device, Path(args.output_root), pair)


if __name__ == "__main__":
    main()
