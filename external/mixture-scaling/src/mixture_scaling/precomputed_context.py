from __future__ import annotations

import fcntl
import hashlib
import json
import time
from pathlib import Path

import torch
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from .context_views import neighbor_mean


ALGORITHM = "pyg_neighborloader_incoming_mean_v1"


def tensor_sha256(tensor: torch.Tensor, chunk_bytes: int = 64 << 20) -> str:
    value = tensor.detach().cpu().contiguous().view(torch.uint8).numpy()
    digest = hashlib.sha256()
    flat = value.reshape(-1)
    for start in range(0, flat.size, chunk_bytes):
        digest.update(flat[start:start + chunk_bytes])
    return digest.hexdigest()


def cache_key(name: str, edge_index: torch.Tensor, fanout: int, seed: int,
              batch_size: int, topology_scope: str) -> tuple[str, dict]:
    receipt = {
        "algorithm": ALGORITHM,
        "graph": name,
        "topology_scope": topology_scope,
        "fanout": int(fanout),
        "seed": int(seed),
        "batch_size": int(batch_size),
        "edge_count": int(edge_index.shape[1]),
        "topology_sha256": tensor_sha256(edge_index),
    }
    encoded = json.dumps(receipt, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()[:20], receipt


def sampled_neighbor_means(x: torch.Tensor, edge_index: torch.Tensor, fanout: int,
                           seed: int, batch_size: int, *, limit_nodes: int | None = None,
                           progress=None) -> torch.Tensor:
    """Materialize one deterministic fanout sample and incoming mean per root node."""
    count = len(x) if limit_nodes is None else min(len(x), int(limit_nodes))
    roots = torch.arange(count)
    graph = Data(x=x.float(), edge_index=edge_index.long())
    output = torch.empty((count, x.shape[1]), dtype=torch.float32)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        loader = NeighborLoader(
            graph, input_nodes=roots, num_neighbors=[int(fanout)],
            batch_size=int(batch_size), shuffle=False, num_workers=0,
        )
        for index, batch in enumerate(loader):
            root_ids = batch.n_id[:batch.batch_size]
            values = neighbor_mean(batch.x.float(), batch.edge_index)[:batch.batch_size]
            output[root_ids] = values
            if progress is not None:
                progress(index, int(batch.batch_size))
    return output


def load_or_materialize(cache_root: Path, name: str, x: torch.Tensor,
                        edge_index: torch.Tensor, fanout: int, seed: int,
                        batch_size: int, topology_scope: str) -> tuple[torch.Tensor, dict]:
    cache_root.mkdir(parents=True, exist_ok=True)
    key, expected = cache_key(name, edge_index, fanout, seed, batch_size, topology_scope)
    path = cache_root / f"{name}_{topology_scope}_f{fanout}_s{seed}_{key}.pt"
    lock_path = path.with_suffix(".lock")
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.is_file():
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if any(payload["receipt"].get(key) != value for key, value in expected.items()):
                raise ValueError(f"context cache receipt mismatch: {path}")
            return payload["neighbor_mean"], payload["receipt"]
        started = time.monotonic()
        values = sampled_neighbor_means(x, edge_index, fanout, seed, batch_size)
        receipt = {
            **expected,
            "node_count": int(len(x)),
            "feature_dim": int(x.shape[1]),
            "dtype": str(values.dtype),
            "materialization_seconds": time.monotonic() - started,
            "neighbor_mean_sha256": tensor_sha256(values),
        }
        temporary = path.with_suffix(".tmp")
        torch.save({"neighbor_mean": values, "receipt": receipt}, temporary)
        temporary.replace(path)
    return values, receipt
