"""Build a CP-HK PyG retweet graph from staged parquet and bio embeddings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import sys
import time

import numpy as np
import pandas as pd
import torch

try:
    from torch_geometric.data import Data as PyGData
except ImportError:  # pragma: no cover
    PyGData = None


EDGE_ATTR_FEATURE_NAMES = ["n_retweets"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet-dir", required=True)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--history-fraction", type=float, default=0.7)
    parser.add_argument("--max-event-files", type=int, default=0)
    return parser.parse_args()


def _make_data(x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, y: torch.Tensor):
    if PyGData is not None:
        return PyGData(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    return {"x": x, "edge_index": edge_index, "edge_attr": edge_attr, "y": y}


def _load_embeddings(path: str) -> torch.Tensor:
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    emb = obj.get("meanpool")
    if emb is None:
        raise KeyError(f"{path} must contain meanpool embeddings")
    return emb.float()


def _pairs_to_tensor(pairs: pd.DataFrame) -> torch.Tensor:
    if pairs.empty:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.from_numpy(pairs[["source_node_id", "target_node_id"]].to_numpy(dtype=np.int64).T).long()


def main() -> None:
    args = parse_args()
    started = time.time()
    command = " ".join(shlex.quote(x) for x in [sys.executable, *sys.argv])
    parquet_dir = Path(args.parquet_dir)
    users = pd.read_parquet(parquet_dir / "user_bios.parquet", columns=["node_id", "user_id", "profile"])
    users = users.sort_values("node_id").reset_index(drop=True)
    if not (users["node_id"].to_numpy() == np.arange(len(users))).all():
        raise ValueError("node_id must be contiguous and 0-indexed in users parquet.")
    user_to_node = dict(zip(users["user_id"].astype(str), users["node_id"].astype(np.int64)))

    event_files = sorted((parquet_dir / "retweet_events").glob("part-*.parquet"))
    if args.max_event_files > 0:
        event_files = event_files[: args.max_event_files]
    if not event_files:
        raise FileNotFoundError(f"No event parquet files found under {parquet_dir / 'retweet_events'}")

    print(f"Loading {len(event_files):,} event parquet files", flush=True)
    edges_parts = []
    for idx, path in enumerate(event_files):
        df = pd.read_parquet(path, columns=["source_user_id", "target_user_id", "timestamp_ms"])
        df["source_node_id"] = df["source_user_id"].astype(str).map(user_to_node)
        df["target_node_id"] = df["target_user_id"].astype(str).map(user_to_node)
        df = df.dropna(subset=["source_node_id", "target_node_id"])
        df["source_node_id"] = df["source_node_id"].astype(np.int64)
        df["target_node_id"] = df["target_node_id"].astype(np.int64)
        df = df[df["source_node_id"] != df["target_node_id"]]
        edges_parts.append(df[["source_node_id", "target_node_id", "timestamp_ms"]])
        if (idx + 1) % 10 == 0:
            print(f"  loaded {idx + 1:,}/{len(event_files):,} files", flush=True)
    events = pd.concat(edges_parts, ignore_index=True)
    del edges_parts

    if events.empty:
        raise ValueError("No retweet events after mapping users.")

    events["timestamp_ms"] = pd.to_numeric(events["timestamp_ms"], errors="coerce").astype("Int64")
    valid_time = events["timestamp_ms"].dropna().astype(np.int64)
    if len(valid_time) > 1:
        cutoff = int(valid_time.quantile(float(args.history_fraction)))
        hist_events = events[events["timestamp_ms"].fillna(cutoff) <= cutoff]
        future_events = events[events["timestamp_ms"].fillna(cutoff) > cutoff]
    else:
        cutoff = None
        hist_events = events
        future_events = events.iloc[0:0]

    grouped = (
        events.groupby(["source_node_id", "target_node_id"], sort=False)
        .size()
        .rename("n_retweets")
        .reset_index()
    )
    hist_grouped = (
        hist_events.groupby(["source_node_id", "target_node_id"], sort=False)
        .size()
        .rename("n_retweets")
        .reset_index()
    )
    future_pairs = future_events[["source_node_id", "target_node_id"]].drop_duplicates()
    if not hist_grouped.empty and not future_pairs.empty:
        hist_pairs = set(map(tuple, hist_grouped[["source_node_id", "target_node_id"]].to_numpy(dtype=np.int64)))
        future_pairs = future_pairs[
            [tuple(row) not in hist_pairs for row in future_pairs.to_numpy(dtype=np.int64)]
        ]

    edge_index = _pairs_to_tensor(grouped)
    edge_attr = torch.from_numpy(np.log1p(grouped[["n_retweets"]].to_numpy(dtype=np.float32))).float()
    hist_edge_index = _pairs_to_tensor(hist_grouped)
    hist_edge_attr = torch.from_numpy(np.log1p(hist_grouped[["n_retweets"]].to_numpy(dtype=np.float32))).float()
    future_edge_index = _pairs_to_tensor(future_pairs)

    x = _load_embeddings(args.embeddings)
    if x.shape[0] != len(users):
        raise ValueError(f"Embedding rows {x.shape[0]} do not match users {len(users)}")
    y = torch.full((len(users),), -1, dtype=torch.long)
    data = _make_data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    graph_obj = {
        "data": data,
        "x": x,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "edge_attr_feature_names": EDGE_ATTR_FEATURE_NAMES,
        "edge_index_views": {
            "retweet_all": edge_index,
            "temporal_history": hist_edge_index,
        },
        "edge_attr_views": {
            "retweet_all": edge_attr,
            "temporal_history": hist_edge_attr,
        },
        "edge_attr_feature_names_views": {
            "retweet_all": EDGE_ATTR_FEATURE_NAMES,
            "temporal_history": EDGE_ATTR_FEATURE_NAMES,
        },
        "target_edge_index_views": {
            "temporal_new": future_edge_index,
            "future": future_edge_index,
        },
        "future_edge_index": future_edge_index,
        "y": y,
        "user_ids": users["user_id"].astype(str).tolist(),
        "feature_names": [f"bio_emb_{i}" for i in range(int(x.shape[1]))],
        "label_names": [],
        "label_type": "classification",
        "dataset_name": "cp_hk_twitter",
        "metadata": {
            "parquet_dir": str(parquet_dir),
            "embeddings": args.embeddings,
            "users": int(len(users)),
            "users_with_bio": int((users["profile"].fillna("").str.len() > 0).sum()),
            "events": int(len(events)),
            "edges": int(edge_index.shape[1]),
            "history_edges": int(hist_edge_index.shape[1]),
            "future_target_edges": int(future_edge_index.shape[1]),
            "history_fraction": float(args.history_fraction),
            "history_cutoff_timestamp_ms": cutoff,
            "command": command,
            "wall_min": round((time.time() - started) / 60, 2),
        },
    }

    # Benchmark targets: static-LP edge views only (cp_hk has no profile metrics,
    # so node regression is skipped automatically). Opt out with SKIP_BENCHMARK_TARGETS=1.
    try:
        import os

        if os.environ.get("SKIP_BENCHMARK_TARGETS", "") not in {"1", "true", "True"}:
            from enrich_graph_targets import enrich_graph_obj

            bt_stats = enrich_graph_obj(graph_obj, "cp_hk_twitter")
            print(f"benchmark targets attached: static={bt_stats['static_split']}", flush=True)
    except Exception as exc:  # noqa: BLE001 - enrichment must not break a build
        print(f"[warn] benchmark-target enrichment failed: {exc}", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(graph_obj, out_path)
    with out_path.with_suffix(".meta.json").open("w", encoding="utf-8") as handle:
        json.dump(graph_obj["metadata"], handle, indent=2)
    print(json.dumps(graph_obj["metadata"], indent=2), flush=True)
    print(f"Saved graph: {out_path}", flush=True)


if __name__ == "__main__":
    main()
