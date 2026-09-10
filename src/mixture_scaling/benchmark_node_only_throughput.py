from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .config import load_config
from .lattice import load_shared_graph
from .model import NodeMLP


def endpoint_ids(edges, n_nodes, batch_size, generator):
    positive_index = torch.randint(edges.shape[1], (batch_size,), generator=generator)
    positive = edges[:, positive_index]
    n_negative = batch_size * 5
    negative = torch.randint(n_nodes, (2, n_negative), generator=generator)
    self_loops = negative[0] == negative[1]
    while self_loops.any():
        negative[1, self_loops] = torch.randint(
            n_nodes, (int(self_loops.sum()),), generator=generator
        )
        self_loops = negative[0] == negative[1]
    pairs = torch.cat((positive, negative), dim=1)
    return torch.cat((pairs[0], pairs[1])), pairs.shape[1]


def benchmark_mode(x_cpu, edges, protocol, device, mode, batch_size, total_positives, seed):
    torch.manual_seed(seed)
    model = NodeMLP(
        int(protocol["input_dim"]), int(protocol["hidden_dim"]),
        int(protocol["output_dim"]), float(protocol["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(protocol["node_mlp_lp_learning_rate"]),
        weight_decay=float(protocol["weight_decay"]),
    )
    x_device = x_cpu.to(device) if mode == "gpu_resident" else None
    steps = total_positives // batch_size
    if steps < 10 or steps * batch_size != total_positives:
        raise ValueError("total positives must be divisible by every batch size and give >=10 steps")
    generator = torch.Generator().manual_seed(seed + 982451653)
    warmup = min(10, max(2, steps // 10))
    elapsed = None
    for step in range(steps + warmup):
        if step == warmup:
            torch.cuda.synchronize(device)
            started = time.perf_counter()
        ids, n_pairs = endpoint_ids(edges, len(x_cpu), batch_size, generator)
        if mode == "gpu_resident":
            features = x_device[ids.to(device, non_blocking=True)]
        else:
            features = x_cpu[ids].pin_memory().to(device, non_blocking=True)
        embedding = model(features)
        src, dst = embedding[:n_pairs], embedding[n_pairs:]
        labels = torch.cat((
            torch.ones(batch_size, device=device),
            torch.zeros(n_pairs - batch_size, device=device),
        ))
        loss = F.binary_cross_entropy_with_logits((src * dst).sum(-1), labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    total_pairs = total_positives * 6
    result = {
        "mode": mode, "batch_positives": batch_size, "steps": steps,
        "total_positives": total_positives, "total_pairs": total_pairs,
        "elapsed_seconds": elapsed, "updates_per_second": steps / elapsed,
        "positives_per_second": total_positives / elapsed,
        "pairs_per_second": total_pairs / elapsed,
        "feature_matrix_gib": x_cpu.numel() * x_cpu.element_size() / 2**30,
        "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "final_loss": float(loss),
    }
    del model, optimizer, x_device, features, embedding, loss
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--source", default="covid19_twitter")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch-sizes", default="1024,4096,8192,16384")
    parser.add_argument("--total-positives", type=int, default=1048576)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.device not in (0, 1, 2, 3):
        raise ValueError("only owned Tucker GPUs 0-3 are allowed")
    config = load_config(args.config)
    protocol = config["protocol"]
    graph = load_shared_graph(
        args.source, config["graphs"][args.source]["path"], "lp", protocol,
        args.seed, Path(args.cache_root),
    )
    x = graph.data.x.float().contiguous()
    edges = graph.train_edges.long().contiguous()
    device = torch.device(f"cuda:{args.device}")
    rows = []
    for batch_size in map(int, args.batch_sizes.split(",")):
        for mode in ("cpu_direct", "gpu_resident"):
            row = benchmark_mode(
                x, edges, protocol, device, mode, batch_size,
                args.total_positives, args.seed,
            )
            rows.append(row)
            print(json.dumps(row), flush=True)
    payload = {
        "status": "complete", "source": args.source, "seed": args.seed,
        "negative_ratio": 5, "fixed_total_examples": True, "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
