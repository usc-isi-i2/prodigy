"""Build the TwiBot-20 retweet graph with zero-filled bio-embedding features.

Consumes the reconstructed retweet edge list
(``data/data/twibot20/scripts/extract_retweet_edges.py``) plus TwiBot-20's
``label.csv`` / ``split.csv`` and the bio-embedding store, and emits a graph
artifact structurally compatible with the other retweet graphs in this repo
(see ``generate_covid19_twitter_retweet_graph_from_parquet.py``).

Differences from the covid/ukr_rus builders:
- userids are strings (``u17461978``), not integers;
- edges come from a precomputed ``retweet_edges.parquet`` (no rt_* columns,
  no timestamps), so there are **no temporal / link-prediction views**;
- nodes carry real bot/human labels (``y``) and train/val/test splits;
- missing bios are **zero-filled**, matching the established convention.

Node set = retweet-edge participants ∪ all labeled users (so every labeled node
is present even if it has no retweet edge). Edge direction: retweeter -> retweeted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pyarrow.parquet as pq
import torch

try:
    from torch_geometric.data import Data as PyGData
except ImportError:  # pragma: no cover - depends on environment.
    PyGData = None


DEFAULT_EDGES = "/dataMeR1/phil/data/twibot20/graph_build/retweet_edges.parquet"
DEFAULT_LABELS = "/dataMeR1/phil/data/twibot20/raw/Twibot-20/label.csv"
DEFAULT_SPLITS = "/dataMeR1/phil/data/twibot20/raw/Twibot-20/split.csv"
DEFAULT_BIO_ROOT = (
    "/dataMeR1/phil/data/twibot20/bio_embeddings/gte-multilingual-base/version=v001"
)
DEFAULT_OUT = "/dataMeR1/phil/data/twibot20/graphs/retweet_graph.pt"

EDGE_ATTR_FEATURE_NAMES = ["n_retweets"]
LABEL_NAMES = ["human", "bot"]
LABEL_TO_Y = {"human": 0, "bot": 1}


def _log(message: str) -> None:
    print(f"[progress] {message}", flush=True)


def _make_data_object(x, edge_index, edge_attr, y):
    if PyGData is not None:
        return PyGData(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    return SimpleNamespace(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


def read_csv_map(path: Path) -> dict[str, str]:
    """Read a two-column ``id,value`` CSV (with header) into a dict."""
    import csv

    out: dict[str, str] = {}
    with open(path, newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row in reader:
            if len(row) >= 2 and row[0]:
                out[row[0]] = row[1]
    return out


def resolve_bio_features(
    user_ids: list[str],
    bio_root: Path,
    embedding_dim_fallback: int = 768,
) -> tuple[torch.Tensor, list[str], dict[str, Any]]:
    """Zero-fill features, then fill matched bios from the embedding shards.

    Mirrors the covid generator: ``np.zeros((N, dim))`` then scatter shard rows
    for users that have a resolvable bio. TwiBot-20 has exactly one bio per user,
    so no latest/cutoff policy is needed.
    """
    started = time.monotonic()
    user_bio_path = bio_root / "user_bio_observations.parquet"
    bio_index_path = bio_root / "bio_embedding_index.parquet"
    if not user_bio_path.exists():
        raise FileNotFoundError(f"Missing bio observations parquet: {user_bio_path}")
    if not bio_index_path.exists():
        raise FileNotFoundError(f"Missing bio embedding index parquet: {bio_index_path}")

    node_set = set(user_ids)

    # userid -> bio_hash (restricted to graph nodes; one bio per user in TwiBot-20)
    obs = pq.read_table(user_bio_path, columns=["userid", "bio_hash"])
    userid_to_hash: dict[str, str] = {}
    for uid, bh in zip(
        obs.column("userid").to_pylist(), obs.column("bio_hash").to_pylist()
    ):
        if uid in node_set and bh:
            userid_to_hash[str(uid)] = str(bh)

    # bio_hash -> (shard, row, dim)
    idx = pq.read_table(
        bio_index_path,
        columns=["bio_hash", "embedding_shard", "embedding_row", "embedding_dim"],
    )
    hash_to_loc: dict[str, tuple[str, int, int]] = {}
    for bh, shard, row, dim in zip(
        idx.column("bio_hash").to_pylist(),
        idx.column("embedding_shard").to_pylist(),
        idx.column("embedding_row").to_pylist(),
        idx.column("embedding_dim").to_pylist(),
    ):
        if bh is not None and shard is not None and row is not None:
            hash_to_loc[str(bh)] = (str(shard), int(row), int(dim))

    embedding_dim = (
        max((loc[2] for loc in hash_to_loc.values()), default=embedding_dim_fallback)
        if hash_to_loc
        else embedding_dim_fallback
    )

    features = np.zeros((len(user_ids), embedding_dim), dtype=np.float32)
    u2i = {uid: i for i, uid in enumerate(user_ids)}

    # Group scatter targets by shard for efficient mmap reads.
    shard_to_nodes: dict[str, list[tuple[int, int]]] = {}
    for uid, bh in userid_to_hash.items():
        loc = hash_to_loc.get(bh)
        if loc is None:
            continue
        shard_path, row_idx, _ = loc
        shard_to_nodes.setdefault(shard_path, []).append((u2i[uid], row_idx))

    matched_users = 0
    for shard_path, entries in sorted(shard_to_nodes.items()):
        shard_abs = Path(shard_path)
        if not shard_abs.is_absolute():
            shard_abs = bio_root / shard_abs
        vectors = np.load(shard_abs, mmap_mode="r")
        node_rows = np.fromiter((n for n, _ in entries), dtype=np.int64, count=len(entries))
        shard_rows = np.fromiter((r for _, r in entries), dtype=np.int64, count=len(entries))
        features[node_rows] = np.asarray(vectors[shard_rows], dtype=np.float32)
        matched_users += len(entries)

    feature_names = [f"bio_emb_{i}" for i in range(embedding_dim)]
    stats = {
        "policy": "single_bio_per_user_zero_fill",
        "embedding_dim": embedding_dim,
        "matched_users": matched_users,
        "missing_users": len(user_ids) - matched_users,
    }
    _log(
        f"bio features resolved in {time.monotonic() - started:.1f}s "
        f"(matched={matched_users:,}, missing={len(user_ids) - matched_users:,})"
    )
    return torch.from_numpy(features).float(), feature_names, stats


def validate_graph_artifact(graph_obj: dict[str, Any]) -> None:
    data = graph_obj["data"]
    user_ids = graph_obj["user_ids"]
    feature_names = graph_obj["feature_names"]
    edge_attr_feature_names = graph_obj["edge_attr_feature_names"]

    if data.x.dim() != 2:
        raise ValueError("data.x must be 2D")
    if data.edge_index.dim() != 2 or data.edge_index.shape[0] != 2:
        raise ValueError("data.edge_index must have shape [2, E]")
    if len(user_ids) != data.x.shape[0]:
        raise ValueError("len(user_ids) must equal data.x.shape[0]")
    if len(feature_names) != data.x.shape[1]:
        raise ValueError("len(feature_names) must equal data.x.shape[1]")
    if data.edge_attr is not None:
        if data.edge_attr.dim() != 2:
            raise ValueError("data.edge_attr must be 2D when present")
        if data.edge_attr.shape[0] != data.edge_index.shape[1]:
            raise ValueError("data.edge_attr rows must equal edge count")
        if len(edge_attr_feature_names) != data.edge_attr.shape[1]:
            raise ValueError("edge_attr_feature_names must align with data.edge_attr")
    if data.y.shape[0] != data.x.shape[0]:
        raise ValueError("data.y must have one entry per node")
    if torch.isnan(data.x).any():
        raise ValueError("data.x contains NaN")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edges", default=DEFAULT_EDGES)
    parser.add_argument("--labels", default=DEFAULT_LABELS)
    parser.add_argument("--splits", default=DEFAULT_SPLITS)
    parser.add_argument("--bio-embeddings-root", default=DEFAULT_BIO_ROOT)
    parser.add_argument("--out", default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.monotonic()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --- edges ---
    _log(f"reading retweet edges from {args.edges}")
    edge_table = pq.read_table(args.edges, columns=["userid", "rt_userid", "n_retweets"])
    src = edge_table.column("userid").to_pylist()
    dst = edge_table.column("rt_userid").to_pylist()
    weight = np.asarray(edge_table.column("n_retweets").to_pylist(), dtype=np.float32)
    _log(f"  {len(src):,} directed edges")

    # --- labels & splits ---
    labels = read_csv_map(Path(args.labels))
    splits = read_csv_map(Path(args.splits))
    _log(f"  {len(labels):,} labeled users, {len(splits):,} split assignments")

    # --- node set: edge participants ∪ all labeled users (deterministic order) ---
    node_set = set(src) | set(dst) | set(labels.keys())
    user_ids = sorted(node_set)
    u2i = {uid: i for i, uid in enumerate(user_ids)}
    n = len(user_ids)
    _log(f"  node set = {n:,} users (edge participants ∪ labeled)")

    # --- edge tensors ---
    src_idx = np.fromiter((u2i[u] for u in src), dtype=np.int64, count=len(src))
    dst_idx = np.fromiter((u2i[u] for u in dst), dtype=np.int64, count=len(dst))
    edge_index = torch.from_numpy(np.vstack([src_idx, dst_idx])).long()
    edge_attr = torch.from_numpy(weight.reshape(-1, 1)).float()

    # --- labels y and split masks ---
    y = torch.full((n,), -1, dtype=torch.long)
    n_labeled = 0
    for uid, lab in labels.items():
        if uid in u2i and lab in LABEL_TO_Y:
            y[u2i[uid]] = LABEL_TO_Y[lab]
            n_labeled += 1

    split_masks = {
        name: torch.zeros(n, dtype=torch.bool)
        for name in ("train", "val", "test", "support")
    }
    for uid, split in splits.items():
        if uid in u2i and split in split_masks:
            split_masks[split][u2i[uid]] = True
    split_counts = {name: int(mask.sum()) for name, mask in split_masks.items()}
    _log(f"  labeled nodes={n_labeled:,}  split node counts={split_counts}")

    # --- node features (zero-fill bio embeddings) ---
    x, feature_names, bio_stats = resolve_bio_features(
        user_ids=user_ids, bio_root=Path(args.bio_embeddings_root)
    )

    # --- assemble & validate ---
    data = _make_data_object(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data.feature_names = feature_names
    data.edge_attr_feature_names = EDGE_ATTR_FEATURE_NAMES
    data.user_ids = list(user_ids)
    for name, mask in split_masks.items():
        setattr(data, f"{name}_mask", mask)

    graph_obj: dict[str, Any] = {
        "x": x,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "edge_attr_feature_names": EDGE_ATTR_FEATURE_NAMES,
        "user_ids": user_ids,
        "u2i": u2i,
        "feature_names": feature_names,
        "y": y,
        "label_names": LABEL_NAMES,
        "split_masks": {name: mask for name, mask in split_masks.items()},
        "bio_embedding_policy": bio_stats["policy"],
        "data": data,
    }

    _log("validating graph artifact in memory")
    validate_graph_artifact(graph_obj)

    _log(f"writing graph artifact to {out_path}")
    torch.save(graph_obj, out_path)

    isolated = int((torch.bincount(edge_index.reshape(-1), minlength=n) == 0).sum())
    meta = {
        "nodes": n,
        "edges": int(edge_index.shape[1]),
        "retweet_events": int(weight.sum()),
        "isolated_nodes": isolated,
        "labeled_nodes": n_labeled,
        "label_names": LABEL_NAMES,
        "label_counts": {
            "human": int((y == 0).sum()),
            "bot": int((y == 1).sum()),
            "unlabeled": int((y == -1).sum()),
        },
        "split_counts": split_counts,
        "node_feature_dim": int(x.shape[1]),
        "edge_feature_names": EDGE_ATTR_FEATURE_NAMES,
        "bio_embeddings_root": args.bio_embeddings_root,
        "bio_embedding_policy": bio_stats["policy"],
        "bio_embedding_matched_users": int(bio_stats["matched_users"]),
        "bio_embedding_missing_users": int(bio_stats["missing_users"]),
    }
    meta_path = out_path.with_suffix(".meta.json")
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)

    _log(f"done in {time.monotonic() - started:.1f}s")
    print(f"Saved graph: {out_path}")
    print(f"Saved meta:  {meta_path}")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
