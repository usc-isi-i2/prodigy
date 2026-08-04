#!/usr/bin/env python3
"""Assemble a benchmark-ready Facebook page-reference graph artifact."""

from __future__ import annotations

import argparse
import collections
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from scripts.graph_construction.benchmark_targets import build_static_edge_split
from scripts.graph_construction.generate_twibot20_retweet_graph import resolve_bio_features

try:
    from torch_geometric.data import Data as PyGData
except ImportError:  # pragma: no cover
    PyGData = None


EDGE_FEATURE_NAMES = ["n_reference_posts", "n_content_reference_posts"]


def log(message: str) -> None:
    print(f"[progress] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables-root", required=True, type=Path)
    parser.add_argument("--bio-embeddings-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--category-top-k", type=int, default=30)
    parser.add_argument("--country-top-k", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--static-holdout-frac", type=float, default=0.15)
    parser.add_argument("--history-fraction", type=float, default=0.70)
    return parser.parse_args()


def make_data(x, edge_index, edge_attr, y):
    if PyGData is not None:
        return PyGData(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    return SimpleNamespace(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


def normalize_label(value: object) -> str:
    text = str(value or "").strip()
    return "" if text.lower() in {"", "none", "null", "nan", "<na>"} else text


def topk_target(values: list[object], top_k: int):
    normalized = [normalize_label(value) for value in values]
    counts = collections.Counter(value for value in normalized if value)
    names = [
        value for value, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:top_k]
    ]
    mapping = {value: index for index, value in enumerate(names)}
    y = torch.tensor([mapping.get(value, -1) for value in normalized], dtype=torch.long)
    return y, names, {value: counts[value] for value in names}


def verified_target(values: list[object]):
    y = torch.full((len(values),), -1, dtype=torch.long)
    for index, value in enumerate(values):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue
        y[index] = int(bool(value))
    return y, ["not_verified", "verified"]


def stratified_masks(y: torch.Tensor, seed: int):
    rng = np.random.default_rng(seed)
    masks = {name: torch.zeros(y.shape[0], dtype=torch.bool) for name in ("train", "val", "test")}
    labels = sorted(int(value) for value in torch.unique(y[y >= 0]).tolist())
    y_numpy = y.numpy()
    for label in labels:
        indices = np.flatnonzero(y_numpy == label)
        rng.shuffle(indices)
        n_train = max(1, int(0.70 * len(indices)))
        n_val = max(1, int(0.15 * len(indices)))
        if n_train + n_val >= len(indices):
            n_val = max(0, len(indices) - n_train - 1)
        masks["train"][indices[:n_train]] = True
        masks["val"][indices[n_train:n_train + n_val]] = True
        masks["test"][indices[n_train + n_val:]] = True
    return masks


def build_regression_targets(profiles, user_ids, reference_date: datetime):
    subscribers = np.full(len(user_ids), np.nan, dtype=np.float32)
    ages = np.full(len(user_ids), np.nan, dtype=np.float32)
    for index, user_id in enumerate(user_ids):
        record = profiles[user_id]
        try:
            subscriber = float(record.get("subscriber_count"))
        except (TypeError, ValueError):
            subscriber = float("nan")
        if np.isfinite(subscriber) and subscriber >= 0:
            subscribers[index] = subscriber
        created = pd.to_datetime(record.get("page_created_date"), utc=True, errors="coerce")
        if not pd.isna(created):
            age = (reference_date - created.to_pydatetime()).total_seconds() / 86400.0
            if age >= 0:
                ages[index] = age
    targets = {
        "subscriber_count": torch.from_numpy(subscribers),
        "account_age_days": torch.from_numpy(ages),
    }
    coverage = {name: int(torch.isfinite(value).sum()) for name, value in targets.items()}
    return targets, coverage


def pairs_to_tensors(pairs, u2i, weights=None):
    if not pairs:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 2), dtype=torch.float32) if weights is not None else None
        return edge_index, edge_attr
    source = np.fromiter((u2i[left] for left, _ in pairs), dtype=np.int64, count=len(pairs))
    target = np.fromiter((u2i[right] for _, right in pairs), dtype=np.int64, count=len(pairs))
    edge_index = torch.from_numpy(np.vstack([source, target])).long()
    edge_attr = torch.tensor(weights, dtype=torch.float32) if weights is not None else None
    return edge_index, edge_attr


def temporal_views(events_path: Path, u2i, history_fraction: float):
    if not 0.0 < history_fraction < 1.0:
        raise ValueError("history_fraction must be in (0, 1)")
    events = pq.read_table(
        events_path,
        columns=["source_account_id", "target_account_id", "event_date", "is_content_reference"],
    ).to_pandas()
    timestamps = pd.to_datetime(events["event_date"], utc=True, errors="coerce")
    missing_time = int(timestamps.isna().sum())
    valid = events.loc[~timestamps.isna()].copy()
    valid["timestamp"] = timestamps.loc[~timestamps.isna()]
    valid.sort_values(
        ["timestamp", "source_account_id", "target_account_id"], inplace=True, kind="mergesort"
    )
    cutoff = min(max(1, int(len(valid) * history_fraction)), max(1, len(valid) - 1))
    history, future = valid.iloc[:cutoff], valid.iloc[cutoff:]
    grouped = history.groupby(["source_account_id", "target_account_id"], sort=True).agg(
        n_reference_posts=("source_account_id", "size"),
        n_content_reference_posts=("is_content_reference", "sum"),
    ).reset_index()
    history_pairs = list(zip(grouped["source_account_id"], grouped["target_account_id"]))
    history_weights = list(zip(grouped["n_reference_posts"], grouped["n_content_reference_posts"]))
    history_index, history_attr = pairs_to_tensors(history_pairs, u2i, history_weights)
    history_set = set(history_pairs)
    future_pairs = sorted(set(zip(future["source_account_id"], future["target_account_id"])) - history_set)
    future_index, _ = pairs_to_tensors(future_pairs, u2i)
    cutoff_time = valid.iloc[cutoff - 1]["timestamp"] if len(valid) else None
    stats = {
        "history_fraction": history_fraction,
        "valid_events": len(valid),
        "missing_timestamp_events": missing_time,
        "history_events": len(history),
        "future_events": len(future),
        "history_edges": int(history_index.shape[1]),
        "future_new_edges": int(future_index.shape[1]),
        "history_cutoff": cutoff_time.isoformat() if cutoff_time is not None else "",
        "future_target_mode": "new_only",
    }
    return history_index, history_attr, future_index, stats


def validate_graph(graph):
    x, edge_index, edge_attr = graph["x"], graph["edge_index"], graph["edge_attr"]
    n = x.shape[0]
    if x.shape != (n, 768) or not torch.isfinite(x).all():
        raise ValueError(f"invalid node features: {tuple(x.shape)}")
    if edge_index.shape[0] != 2 or edge_attr.shape != (edge_index.shape[1], 2):
        raise ValueError("edge tensors are misaligned")
    if len(graph["user_ids"]) != n or len(set(graph["user_ids"])) != n:
        raise ValueError("user_ids are not unique and aligned")
    if edge_index.numel() and (int(edge_index.min()) < 0 or int(edge_index.max()) >= n):
        raise ValueError("edge_index contains an out-of-range node")
    for target in [*graph["node_targets"].values(), *graph["node_classification_targets"].values()]:
        if target.shape != (n,):
            raise ValueError("node target is misaligned")
    static_bg = {tuple(sorted(pair)) for pair in graph["edge_index_views"]["static_background"].t().tolist()}
    static_ho = {tuple(sorted(pair)) for pair in graph["target_edge_index_views"]["static_holdout"].t().tolist()}
    if static_bg & static_ho:
        raise ValueError("static holdout leaks through a reverse background edge")
    temporal_bg = set(map(tuple, graph["edge_index_views"]["temporal_history"].t().tolist()))
    if any(tuple(pair) in temporal_bg for pair in graph["target_edge_index_views"]["temporal_new"].t().tolist()):
        raise ValueError("temporal target contains a history edge")


def main() -> int:
    args = parse_args()
    started = time.monotonic()
    tables_root = args.tables_root.expanduser().resolve()
    bio_root = args.bio_embeddings_root.expanduser().resolve()
    out_path = args.out.expanduser().resolve()
    meta_path = out_path.with_suffix(".meta.json")
    if out_path.exists() or meta_path.exists():
        raise FileExistsError(f"Refusing to overwrite graph output: {out_path}")
    edges_path = tables_root / "page_reference_edges.parquet"
    events_path = tables_root / "page_reference_events.parquet"
    profiles_path = tables_root / "page_profiles.parquet"
    for path in (edges_path, events_path, profiles_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    log("reading full page-reference edges")
    edges = pq.read_table(
        edges_path,
        columns=[
            "source_account_id", "target_account_id", "n_reference_posts",
            "n_content_reference_posts", "last_event_date",
        ],
    ).to_pandas()
    source_ids = edges["source_account_id"].astype(str).tolist()
    target_ids = edges["target_account_id"].astype(str).tolist()
    user_ids = sorted(set(source_ids) | set(target_ids))
    u2i = {user_id: index for index, user_id in enumerate(user_ids)}
    pairs = list(zip(source_ids, target_ids))
    weights = list(zip(edges["n_reference_posts"], edges["n_content_reference_posts"]))
    edge_index, edge_attr = pairs_to_tensors(pairs, u2i, weights)
    assert edge_attr is not None
    log(f"nodes={len(user_ids):,} directed_edges={edge_index.shape[1]:,}")

    log("reading and aligning page profiles")
    profile_table = pq.read_table(profiles_path)
    graph_nodes = set(user_ids)
    profiles = {
        str(row["account_id"]): row
        for row in profile_table.to_pylist()
        if str(row["account_id"]) in graph_nodes
    }
    missing = sorted(graph_nodes - set(profiles))
    if missing:
        raise ValueError(f"Missing {len(missing)} graph profiles: {missing[:5]}")

    log("resolving GTE page-description features")
    x, feature_names, bio_stats = resolve_bio_features(user_ids, bio_root)
    category_y, category_names, category_counts = topk_target(
        [profiles[user_id].get("page_category") for user_id in user_ids], args.category_top_k
    )
    country_y, country_names, country_counts = topk_target(
        [profiles[user_id].get("page_admin_top_country") for user_id in user_ids],
        args.country_top_k,
    )
    verified_y, verified_names = verified_target(
        [profiles[user_id].get("verified") for user_id in user_ids]
    )
    classification_targets = {
        "page_category_top30": category_y,
        "admin_country_top30": country_y,
        "verified": verified_y,
    }
    classification_names = {
        "page_category_top30": category_names,
        "admin_country_top30": country_names,
        "verified": verified_names,
    }
    split_masks = stratified_masks(category_y, args.seed)

    observed_end = pd.to_datetime(edges["last_event_date"], utc=True, errors="coerce").max()
    reference_date = (
        observed_end.to_pydatetime() if not pd.isna(observed_end) else datetime.now(timezone.utc)
    )
    node_targets, target_coverage = build_regression_targets(profiles, user_ids, reference_date)
    source_post_count = torch.tensor(
        [float(profiles[user_id].get("source_post_count") or 0) for user_id in user_ids],
        dtype=torch.float32,
    )

    log("constructing link-prediction views")
    static = build_static_edge_split(
        edge_index, holdout_frac=args.static_holdout_frac, seed=args.seed
    )
    history_index, history_attr, future_index, temporal_stats = temporal_views(
        events_path, u2i, args.history_fraction
    )
    assert history_attr is not None
    content_mask = edge_attr[:, 1] > 0
    edge_index_views = {
        "page_reference_all": edge_index,
        "content_only": edge_index[:, content_mask],
        "temporal_history": history_index,
        "static_background": static.background_edge_index,
    }
    edge_attr_views = {
        "page_reference_all": edge_attr,
        "content_only": edge_attr[content_mask],
        "temporal_history": history_attr,
        "static_background": edge_attr[static.background_mask],
    }
    target_views = {
        "temporal_new": future_index,
        "static_holdout": static.holdout_edge_index,
    }
    data = make_data(x, edge_index, edge_attr, category_y)
    data.feature_names = feature_names
    data.edge_attr_feature_names = EDGE_FEATURE_NAMES
    data.user_ids = user_ids
    data.label_names = category_names
    for split, mask in split_masks.items():
        setattr(data, f"{split}_mask", mask)

    graph = {
        "x": x,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "edge_attr_feature_names": EDGE_FEATURE_NAMES,
        "edge_index_views": edge_index_views,
        "edge_attr_views": edge_attr_views,
        "edge_attr_feature_names_views": {
            name: list(EDGE_FEATURE_NAMES) for name in edge_attr_views
        },
        "target_edge_index_views": target_views,
        "future_edge_index": future_index,
        "user_ids": user_ids,
        "u2i": u2i,
        "feature_names": feature_names,
        "y": category_y,
        "label_names": category_names,
        "label_type": "classification",
        "primary_classification_target": "page_category_top30",
        "node_classification_targets": classification_targets,
        "node_classification_label_names": classification_names,
        "node_split_masks": split_masks,
        "node_targets": node_targets,
        "node_target_names": list(node_targets),
        "node_attributes": {"source_post_count": source_post_count},
        "bio_embedding_policy": bio_stats["policy"],
        "static_split_stats": static.stats,
        "temporal_split_stats": temporal_stats,
        "data": data,
    }
    validate_graph(graph)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(out_path.name + f".tmp.{os.getpid()}")
    if tmp_path.exists():
        raise FileExistsError(tmp_path)
    log(f"writing graph artifact to {out_path}")
    torch.save(graph, tmp_path)
    tmp_path.rename(out_path)

    meta = {
        "canonical_name": "facebook-page-reference",
        "dataset_key": "facebook_page_reference",
        "nodes": len(user_ids),
        "edges": int(edge_index.shape[1]),
        "reference_events": int(edge_attr[:, 0].sum()),
        "content_only_edges": int(content_mask.sum()),
        "node_feature_dim": int(x.shape[1]),
        "edge_feature_names": EDGE_FEATURE_NAMES,
        "tables_root": str(tables_root),
        "bio_embeddings_root": str(bio_root),
        "bio_embedding_policy": bio_stats["policy"],
        "bio_embedding_matched_users": int(bio_stats["matched_users"]),
        "bio_embedding_missing_users": int(bio_stats["missing_users"]),
        "primary_classification_target": "page_category_top30",
        "label_names": category_names,
        "label_counts": {
            "page_category_top30": category_counts,
            "admin_country_top30": country_counts,
            "verified": {
                "not_verified": int((verified_y == 0).sum()),
                "verified": int((verified_y == 1).sum()),
                "missing": int((verified_y < 0).sum()),
            },
        },
        "node_split_counts": {split: int(mask.sum()) for split, mask in split_masks.items()},
        "node_regression_targets": list(node_targets),
        "node_regression_coverage": target_coverage,
        "regression_reference_date": reference_date.isoformat(),
        "static_split": static.stats,
        "temporal_split": temporal_stats,
        "source_profile_columns": profile_table.column_names,
        "construction_script": str(Path(__file__).relative_to(REPO_ROOT)),
        "duration_seconds": time.monotonic() - started,
    }
    tmp_meta = meta_path.with_name(meta_path.name + f".tmp.{os.getpid()}")
    with tmp_meta.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp_meta.rename(meta_path)
    log(f"done in {meta['duration_seconds']:.1f}s")
    print(json.dumps(meta, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
