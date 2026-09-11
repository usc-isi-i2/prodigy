"""Fixed-budget LP ladder using the existing endpoint-only MLP fast path."""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import torch

from .ladder_queue import claimed_rows
from .prefetch_links import PrefetchLinks
from .convergence import SourcePlateau
from .ladder_tracking import LossWindow, tracked_run
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
    limit = args.max_steps_per_source * len(sources) if args.convergence else args.steps
    run_dir = Path(args.state_root) / "lp" / run_id
    if (run_dir / "summary.json").is_file():
        summary = json.loads((run_dir / "summary.json").read_text())
        if (summary["sources"] != list(sources) or summary["protocol"]["max_steps"] != limit
                or summary.get("convergence", False) != args.convergence
                or summary["seed"] != args.seed or not (run_dir / "best.pt").is_file()):
            raise ValueError(f"completed run does not match requested protocol: {run_dir}")
        return
    if run_dir.exists():
        raise FileExistsError(f"refusing partial run: {run_dir}")
    run_dir.mkdir(parents=True)
    protocol = dict(config["protocol"], max_steps=limit)
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
        "checkpoint_selection": "source_validation_macro_loss" if args.convergence else "fixed_terminal_budget",
        "convergence": args.convergence,
        "stopping": {"minimum_steps_per_source": args.minimum_steps_per_source,
                     "validation_every_per_source": args.validation_every_per_source,
                     "patience": args.patience, "min_delta": args.min_delta,
                     "max_steps_per_source": args.max_steps_per_source}, "source_sampling": "uniform_round_robin",
        "training_negative_sampling": "source_confined_uniform_approximate_excluding_self_loops",
        "prefetch_depth": args.prefetch_depth, "prefetch_workers": args.prefetch_workers,
        "feature_residency": {s: "gpu" if s in resident else "cpu_direct_pinned" for s in sources},
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"starting": run_id, "feature_residency": metadata["feature_residency"]}), flush=True)
    with tracked_run(run_dir, metadata, args) as (run, log), PrefetchLinks(
            train, source_schedule(sources, limit), device, depth=args.prefetch_depth,
            workers=args.prefetch_workers) as batches:
        window = LossWindow()
        started = time.monotonic()
        stopper = SourcePlateau(sources, args.patience, args.min_delta)
        best_value, best_step = float("inf"), 0
        stop_reason = "safety_cap" if args.convergence else "fixed_budget"
        validation = None
        for step, source in enumerate(source_schedule(sources, limit), 1):
            batch_source, batch = next(batches)
            if batch_source != source:
                raise RuntimeError("prefetch changed source order")
            optimizer.zero_grad(set_to_none=True)
            loss = lp_loss(model, batch, device)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss: {run_id} step {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(protocol["gradient_clip_norm"]))
            optimizer.step()
            counts[source] += 1
            window.add(source, float(loss.detach()), batch.edge_label.numel())
            if step % args.log_interval == 0 or step == limit:
                log(step, {**window.flush(), "train/elapsed_seconds": time.monotonic() - started})
            if step % 250 == 0 or step == limit:
                print(json.dumps({"run_id": run_id, "step": step, "loss": float(loss),
                    "elapsed_seconds": time.monotonic() - started}), flush=True)
            if args.convergence and (step % (args.validation_every_per_source * len(sources)) == 0 or step == limit):
                validation = {s: validate(model, graphs[s], val[s], "lp", device, protocol, args.seed) for s in sources}
                value = sum(validation.values()) / len(validation)
                plateau = stopper.update(validation, step >= args.minimum_steps_per_source * len(sources))
                log(step, {"validation/loss": value,
                    **{f"validation/source/{s}/loss": v for s, v in validation.items()},
                    **{f"validation/source/{s}/stale_checks": n for s, n in stopper.stale.items()}})
                print(json.dumps({"run_id": run_id, "step": step, "validation_loss": value,
                                  "stale_checks": stopper.stale}), flush=True)
                if value < best_value:
                    best_value, best_step = value, step
                    save_checkpoint(run_dir / "best.pt", model, optimizer, step,
                                    {**metadata, "source_update_counts": dict(counts)})
                # Retain the latest full state without accumulating large checkpoint histories.
                save_checkpoint(run_dir / "latest.pt", model, optimizer, step,
                                {**metadata, "source_update_counts": dict(counts)})
                if plateau:
                    stop_reason = "validation_plateau"
                    break
        training_seconds = time.monotonic() - started
        if window.weights:
            log(step, window.flush())
        if not args.convergence:
            validation = {s: validate(model, graphs[s], val[s], "lp", device, protocol, args.seed) for s in sources}
            log(step, {"validation/loss": sum(validation.values()) / len(validation),
                **{f"validation/source/{s}/loss": value for s, value in validation.items()}})
            best_step = step
        metadata["source_update_counts"] = counts
        save_checkpoint(run_dir / "checkpoints" / f"step_{step}.pt", model, optimizer, step, metadata)
        if not args.convergence:
            (run_dir / "best.pt").symlink_to(f"checkpoints/step_{step}.pt")
        summary = {**metadata, "status": "complete", "final_step": step, "best_step": best_step,
            "stop_reason": stop_reason, "converged": stop_reason == "validation_plateau",
            "training_seconds": training_seconds, "elapsed_seconds": time.monotonic() - started,
            "source_validation_losses": validation}
        run.summary.update({"final_step": step, "best_step": best_step,
                            "stop_reason": stop_reason, "converged": summary["converged"],
                            "training_seconds": training_seconds, "status": "complete"})
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"completed": run_id, "final_step": step, "stop_reason": stop_reason}), flush=True)



def aggregate(args):
    rows = []
    for run_id, sources in ladder_rows():
        summary = json.loads((Path(args.state_root) / "lp" / run_id / "summary.json").read_text())
        for target in LP_TARGETS:
            path = Path(args.output_root) / "lp" / f"{run_id}__to__{target}.json"
            payload = json.loads(path.read_text())
            if payload["sources"] != list(sources) or payload["checkpoint_step"] != summary["best_step"]:
                raise ValueError(f"unexpected evaluation metadata: {path}")
            if payload["gates"]["holdout_leakage_edges"] != 0 or payload["gates"]["endpoint_sensitivity"] <= 0:
                raise ValueError(f"failed LP gate: {path}")
            rows.append({"run_id": run_id, "n_sources": len(sources), "sources": ",".join(sources),
                "target": target, "target_in_pretraining": target in sources,
                "seed": payload["seed"], "checkpoint_step": payload["checkpoint_step"],
                "stop_reason": summary.get("stop_reason", "fixed_budget"),
                "converged": summary.get("converged", False),
                "final_step": summary["final_step"], "roc_auc": payload["report"]["auc"]})
    path = Path(args.output_root) / "ladder_results.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (path.parent / "COMPLETE.json").write_text(json.dumps({"models": 9, "cells": len(rows),
        "status": "complete", "seed": args.seed, "convergence": args.convergence,
        "all_converged": all(r["converged"] for r in rows),
        "steps": None if args.convergence else args.steps}, indent=2) + "\n")


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
    p.add_argument("--device", type=int, choices=(0, 1, 2, 3), default=0)
    p.add_argument("--worker-index", type=int, default=0)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--feature-budget-gib", type=float, default=70)
    p.add_argument("--queue", action="store_true")
    p.add_argument("--rungs", help="Comma-separated rung numbers; default all")
    p.add_argument("--prefetch-depth", type=int, default=8)
    p.add_argument("--prefetch-workers", type=int, default=4)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--wandb-mode", choices=("offline", "online", "disabled"),
                   default=os.environ.get("WANDB_MODE", "offline"))
    p.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "node-mlp-ladder"))
    p.add_argument("--wandb-group", default=os.environ.get("WANDB_RUN_GROUP"))
    p.add_argument("--log-interval", type=int, default=25)
    p.add_argument("--convergence", action="store_true")
    p.add_argument("--max-steps-per-source", type=int, default=100000)
    p.add_argument("--minimum-steps-per-source", type=int, default=2500)
    p.add_argument("--validation-every-per-source", type=int, default=500)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--min-delta", type=float, default=1e-4)
    args = p.parse_args()
    if (min(args.max_steps_per_source, args.minimum_steps_per_source,
            args.validation_every_per_source, args.patience) < 1
            or args.min_delta < 0 or args.minimum_steps_per_source > args.max_steps_per_source):
        p.error("invalid convergence settings")
    if args.log_interval < 1 or args.wandb_mode not in ("offline", "online", "disabled"):
        p.error("invalid tracking mode or logging interval")
    if args.steps < 1 or not 0 <= args.worker_index < args.workers:
        p.error("invalid budget or worker assignment")
    if args.phase == "plan":
        print(json.dumps({"rows": ladder_rows(), "targets": LP_TARGETS, "steps_per_rung": args.steps,
            "total_updates": None if args.convergence else 9 * args.steps, "cells": 54,
            "convergence": args.convergence, "max_steps_per_source": args.max_steps_per_source,
            "minimum_steps_per_source": args.minimum_steps_per_source,
            "validation_every_per_source": args.validation_every_per_source,
            "patience": args.patience, "min_delta": args.min_delta}, indent=2))
        return
    if args.phase == "aggregate":
        aggregate(args)
        return
    torch.set_num_threads(args.threads)
    config = load_config(args.config)
    torch.cuda.set_device(args.device)
    device = torch.device(f"cuda:{args.device}")
    if args.phase == "train":
        graphs = {}
        selected = ladder_rows()
        if args.rungs:
            wanted = {int(k) for k in args.rungs.split(",")}
            if not wanted or not wanted <= set(range(1, 10)):
                p.error("rungs must be in 1..9")
            selected = [row for row in selected if len(row[1]) in wanted]
        assigned = claimed_rows(args.state_root, selected) if args.queue else selected[args.worker_index::args.workers]
        for run_id, sources in assigned:
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
        selected = ladder_rows()
        if args.rungs:
            wanted = {int(k) for k in args.rungs.split(",")}
            if not wanted or not wanted <= set(range(1, 10)):
                p.error("rungs must be in 1..9")
            selected = [row for row in selected if len(row[1]) in wanted]
        for run_id, _ in selected:
            if not (Path(args.state_root) / "lp" / run_id / "summary.json").is_file():
                raise ValueError(f"evaluation requires a completed rung: {run_id}")
        pair = load_pair_module(Path(args.prodigy_root))
        for target in LP_TARGETS[args.worker_index::args.workers]:
            evaluate_lp(target, args, config, selected, device, Path(args.output_root), pair)


if __name__ == "__main__":
    main()
