"""Build the covid19_twitter retweet graph with optional embeddings and temporal views.

Dataset-specific logic: JSON loading and external parquet label attachment.
All shared graph-building primitives come from rapids.graph.build.
"""
import argparse
import glob
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from rapids.graph.build import (
    EDGE_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
    aggregate_edge_features,
    build_node_features,
    build_temporal_views,
    build_user_index,
    build_user_metadata,
    drop_isolates_from_graph,
    maybe_attach_embeddings,
    prepare_retweet_rows,
    save_graph,
    to_edge_tensors,
    trim_rt_to_max_nodes,
)
from rapids.loaders.json_loader import load_json_items
from rapids.utils import normalize_handle, normalize_user_id

DEFAULT_JSON_GLOB = "/scratch1/eibl/data/covid19_twitter/raw/*/*.json"
DEFAULT_OUT = "data/data/covid19_twitter/graphs/retweet_graph.pt"
DEFAULT_LABELS_PARQUET_GLOB = "/scratch1/eibl/data/covid_masking/masking_2020-*.parquet"
DEFAULT_LABEL_HANDLE_COL = "screen_name"
DEFAULT_LABEL_VALUE_COL = "political_gen"


# ---------------------------------------------------------------------------
# Raw data loading
# ---------------------------------------------------------------------------

def load_raw_rows(json_glob: str, max_files: int) -> pd.DataFrame:
    files = sorted(glob.glob(json_glob))
    if max_files > 0:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"No files matched: {json_glob}")

    rows = []
    print(f"Found {len(files)} files")
    for i, path in enumerate(files, start=1):
        print(f"[{i}/{len(files)}] Loading {os.path.basename(path)}", flush=True)
        try:
            items = load_json_items(path)
        except Exception as exc:
            print(f"  [WARN] failed {os.path.basename(path)}: {exc}", flush=True)
            continue

        for tweet in items:
            user = tweet.get("user") or {}
            rt = tweet.get("retweeted_status") or {}
            rt_user = rt.get("user") or {} if rt else {}
            rows.append({
                "screen_name": normalize_handle(user.get("screen_name")),
                "userid": normalize_user_id(user.get("id")),
                "rt_screen": normalize_handle(rt_user.get("screen_name")) if rt else None,
                "rt_userid": normalize_user_id(rt_user.get("id")) if rt else None,
                "date": tweet.get("created_at"),
                "followers_count": user.get("followers_count"),
                "verified": user.get("verified"),
                "statuses_count": user.get("statuses_count"),
                "rt_fav_count": rt.get("favorite_count") if rt else None,
                "rt_reply_count": rt.get("reply_count") if rt else None,
                "sent_vader": None,
                "hashtag": tweet.get("entities", {}).get("hashtags", []),
                "mentionsn": tweet.get("entities", {}).get("user_mentions", []),
                "media_urls": (tweet.get("extended_entities") or {}).get("media", []),
            })

        if i % 10 == 0 or i == len(files):
            print(f"  processed {i}/{len(files)} files", flush=True)

    if not rows:
        raise RuntimeError("No valid rows parsed")
    df = pd.DataFrame(rows)
    print(f"Loaded rows: {len(df):,}")
    return df


# ---------------------------------------------------------------------------
# External label loading
# ---------------------------------------------------------------------------

def load_external_labels(labels_parquet_glob: str) -> Optional[Dict]:
    if not labels_parquet_glob:
        return None
    files = sorted(glob.glob(labels_parquet_glob))
    if not files:
        raise FileNotFoundError(f"No label parquet files matched: {labels_parquet_glob}")

    chunks = [
        pd.read_parquet(p, columns=[DEFAULT_LABEL_HANDLE_COL, DEFAULT_LABEL_VALUE_COL])
        for p in files
    ]
    labels_df = pd.concat(chunks, ignore_index=True)
    labels_df = labels_df.dropna(subset=[DEFAULT_LABEL_HANDLE_COL, DEFAULT_LABEL_VALUE_COL]).copy()
    labels_df["handle"] = labels_df[DEFAULT_LABEL_HANDLE_COL].map(normalize_handle)
    labels_df = labels_df.dropna(subset=["handle"]).drop_duplicates(subset=["handle"], keep="first")

    if labels_df.empty:
        raise RuntimeError("No usable external labels remained after normalisation.")

    unique_raw = sorted(labels_df[DEFAULT_LABEL_VALUE_COL].drop_duplicates().tolist(), key=str)
    raw_to_idx = {v: i for i, v in enumerate(unique_raw)}
    handle_to_label = labels_df.set_index("handle")[DEFAULT_LABEL_VALUE_COL].map(raw_to_idx).to_dict()
    return {
        "handle_to_label": handle_to_label,
        "label_names": [str(v) for v in unique_raw],
        "n_rows": len(labels_df),
        "n_files": len(files),
    }


def build_node_labels(
    handles: List[Optional[str]],
    label_info: Optional[Dict],
) -> Tuple[torch.Tensor, List[str], Dict]:
    if not label_info:
        return torch.full((len(handles),), -1, dtype=torch.long), [], {"labeled_nodes": 0, "label_count": 0}
    handle_to_label = label_info["handle_to_label"]
    label_names = list(label_info["label_names"])
    mapped = pd.Series(handles).map(handle_to_label)
    y_np = mapped.fillna(-1).to_numpy(dtype=np.int64)
    labeled = int((y_np >= 0).sum())
    return torch.from_numpy(y_np), label_names, {"labeled_nodes": labeled, "label_count": len(label_names)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Build covid19_twitter retweet graph with optional embeddings and temporal views."
    )
    p.add_argument("--json_glob", default=DEFAULT_JSON_GLOB)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--embeddings", default="")
    p.add_argument("--embedding_pool", choices=["meanpool", "maxpool"], default="meanpool")
    p.add_argument("--max_files", type=int, default=0)
    p.add_argument("--max_nodes", type=int, default=0)
    p.add_argument("--strict_dates", action="store_true")
    p.add_argument("--history_fraction", type=float, default=0.3)
    p.add_argument("--future_target_mode", choices=["new_only", "all_future"], default="new_only")
    p.add_argument("--no_temporal_views", action="store_true")
    p.add_argument("--labels_parquet_glob", default=DEFAULT_LABELS_PARQUET_GLOB)
    p.add_argument("--keep-isolates", dest="keep_isolates",
                   action=argparse.BooleanOptionalAction, default=False)
    return p.parse_args()


def main():
    args = parse_args()

    raw = load_raw_rows(args.json_glob, args.max_files)
    rt = prepare_retweet_rows(raw, args.strict_dates, timestamp_format="%a %b %d %H:%M:%S %z %Y")
    rt = trim_rt_to_max_nodes(rt, args.max_nodes)

    user_ids, u2i = build_user_index(rt)
    handles = build_user_metadata(rt, user_ids)
    print(f"Nodes: {len(user_ids):,}")

    edge_all_df = aggregate_edge_features(rt)
    edge_index, edge_attr = to_edge_tensors(edge_all_df, u2i)
    print(f"Directed edges: {edge_index.shape[1]:,}")

    x, feature_names = build_node_features(rt, u2i, edge_all_df)
    x, feature_names, emb_stats = maybe_attach_embeddings(
        x, feature_names, user_ids, handles,
        embeddings_path=args.embeddings, embedding_pool=args.embedding_pool,
    )

    isolated_before = int(
        ((torch.bincount(edge_index[0], minlength=len(user_ids)) == 0) &
         (torch.bincount(edge_index[1], minlength=len(user_ids)) == 0)).sum()
    )
    isolated_dropped = 0
    if not args.keep_isolates:
        x, edge_index, edge_attr, user_ids, _, handles, u2i, isolated_dropped = drop_isolates_from_graph(
            x, edge_index, edge_attr, user_ids, handles=handles
        )
        print(f"Dropped isolated nodes: {isolated_dropped:,}")

    label_info = load_external_labels(args.labels_parquet_glob)
    y, label_names, label_stats = build_node_labels(handles, label_info)
    if label_info:
        print(f"Attached labels: {label_stats['labeled_nodes']:,} labeled, "
              f"classes={label_names}", flush=True)

    graph_obj = {
        "x": x, "edge_index": edge_index, "edge_attr": edge_attr,
        "edge_attr_feature_names": EDGE_FEATURE_NAMES,
        "user_ids": user_ids, "u2i": u2i,
        "feature_names": feature_names, "y": y, "label_names": label_names,
        "handles": handles,
    }

    temporal_stats = {}
    if not args.no_temporal_views:
        temporal_entries, temporal_stats = build_temporal_views(
            rt, u2i, args.history_fraction, args.future_target_mode,
            edge_index, edge_attr,
        )
        graph_obj.update(temporal_entries)

    meta = {
        "json_glob": args.json_glob,
        "max_files": args.max_files,
        "nodes": int(len(user_ids)),
        "edges": int(edge_index.shape[1]),
        "node_feature_dim": int(x.shape[1]),
        "edge_feature_names": EDGE_FEATURE_NAMES,
        **label_stats,
        "embeddings": args.embeddings,
        "embedding_pool": args.embedding_pool,
        **emb_stats,
        "labels_parquet_glob": args.labels_parquet_glob,
        "keep_isolates": args.keep_isolates,
        "isolated_nodes_before_drop": isolated_before,
        "isolated_nodes_dropped": isolated_dropped,
        "temporal": temporal_stats,
    }
    save_graph(args.out, graph_obj, meta)


if __name__ == "__main__":
    main()
