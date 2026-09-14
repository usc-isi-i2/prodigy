#!/usr/bin/env python3
"""Save aligned per-node FP errors for source-specialist overlap analysis."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from mixture_scaling.config import load_config
from mixture_scaling.evaluate_lattice import unwrap
from mixture_scaling.evaluate_node_only import load_model
from mixture_scaling.lattice import SOURCE_ORDER
from mixture_scaling.node_only_transfer import feature_split


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--state-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--worker-index", required=True, type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", required=True, type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample-nodes", type=int, default=10000)
    parser.add_argument("--replicates", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2048)
    args = parser.parse_args()
    if args.device not in (0, 1, 2, 3):
        raise ValueError("only owned Tucker GPUs 0-3 are allowed")
    if not 0 <= args.worker_index < args.workers:
        raise ValueError("invalid worker index")

    config = load_config(args.config)
    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    args.output_root.mkdir(parents=True, exist_ok=True)
    targets = list(SOURCE_ORDER)[args.worker_index::args.workers]
    run_ids = [f"ss_{source}" for source in SOURCE_ORDER]

    for target in targets:
        output = args.output_root / f"{target}.npz"
        if output.is_file():
            continue
        graph = unwrap(config["graphs"][target]["path"])
        indices = feature_split(
            int(graph.num_nodes), True, args.seed,
            float(config["protocol"]["node_validation_fraction"]),
        )[: args.sample_nodes]
        x = graph.x[indices].float()
        del graph
        errors = np.empty((len(run_ids), len(indices)), dtype=np.float32)

        for model_index, run_id in enumerate(run_ids):
            checkpoint = torch.load(
                args.state_root / "fp" / run_id / "best.pt",
                map_location="cpu", weights_only=False,
            )
            model, protocol = load_model(config, checkpoint, "fp", device)
            accumulated = torch.zeros(len(indices), dtype=torch.float64)
            for replicate in range(args.replicates):
                generator = torch.Generator().manual_seed(args.seed + 80021 + replicate)
                mask = torch.rand(x.shape, generator=generator) < float(protocol["mask_rate"])
                empty = ~mask.any(dim=1)
                if empty.any():
                    mask[empty, 0] = True
                for start in range(0, len(indices), args.batch_size):
                    stop = min(start + args.batch_size, len(indices))
                    batch = x[start:stop].to(device)
                    batch_mask = mask[start:stop].to(device)
                    corrupted = torch.where(batch_mask, model.mask_token.expand_as(batch), batch)
                    prediction = model(corrupted).masked_fill(~batch_mask, 0.0)
                    target_values = batch.masked_fill(~batch_mask, 0.0)
                    error = (1.0 - F.cosine_similarity(prediction, target_values, dim=-1))
                    error = error.clamp_min(0).pow(float(protocol["sce_alpha"]))
                    accumulated[start:stop] += error.cpu().double()
            errors[model_index] = (accumulated / args.replicates).float().numpy()
            del model, checkpoint
            torch.cuda.empty_cache()

        np.savez_compressed(
            output, node_ids=indices.numpy(), errors=errors,
            model_ids=np.asarray(run_ids), target=np.asarray(target),
            replicates=np.asarray(args.replicates), seed=np.asarray(args.seed),
        )
        print(json.dumps({"target": target, "nodes": len(indices), "models": len(run_ids)}), flush=True)


if __name__ == "__main__":
    main()
