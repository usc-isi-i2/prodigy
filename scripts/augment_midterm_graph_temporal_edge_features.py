"""
Augment a single midterm graph file with temporal link-prediction views and edge features.

Adds:
- edge_index_views[<edge_view_name>]      : history co-retweet edges
- edge_attr_views[<edge_view_name>]       : [first_retweet_time, n_retweets]
- edge_attr_feature_names_views[...]      : feature names for the edge view
- target_edge_index_views[<target_view>]  : new future co-retweet edges (labels for LP)

Optional:
- set the selected edge/target views as defaults via --set-default-edge-view / --set-default-target-view.
"""

import argparse
import glob
import os
from itertools import combinations

import numpy as np
import pandas as pd
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Augment midterm graph with temporal edge features.")
    parser.add_argument("--source-graph", required=True, help="Path to existing graph_data.pt")
    parser.add_argument("--output-graph", default=None, help="Output graph path (default: overwrite source)")
    parser.add_argument("--csv-glob", default="/project2/ll_774_951/midterm/*/*.csv")
    parser.add_argument("--history-fraction", type=float, default=0.8)
    parser.add_argument("--max-group-size", type=int, default=500)
    parser.add_argument("--strict-dates", action="store_true")
    parser.add_argument("--edge-view-name", default="temporal_history")
    parser.add_argument("--target-view-name", default="temporal_new")
    parser.add_argument("--set-default-edge-view", action="store_true")
    parser.add_argument("--set-default-target-view", action="store_true")
    return parser.parse_args()


def to_int_user_id(uid):
    if isinstance(uid, (int, np.integer)):
        return int(uid)
    if isinstance(uid, (float, np.floating)):
        if np.isfinite(uid):
            return int(uid)
        return None
    s = str(uid).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return None
    v = pd.to_numeric(s, errors="coerce")
    if pd.isna(v):
        return None
    return int(v)


def load_and_split_rows(csv_glob, history_fraction, strict_dates):
    files = sorted(glob.glob(csv_glob))
    if not files:
        raise FileNotFoundError(f"No CSV files matched: {csv_glob}")
    print(f"Found {len(files)} CSV files")

    chunks = []
    for fpath in files:
        chunks.append(pd.read_csv(fpath, usecols=["userid", "rt_userid", "date"]))
    df = pd.concat(chunks, ignore_index=True)
    print(f"Loaded rows: {len(df):,}")

    df["userid"] = pd.to_numeric(df["userid"], errors="coerce")
    df["rt_userid"] = pd.to_numeric(df["rt_userid"], errors="coerce")
    df = df.dropna(subset=["userid", "rt_userid"]).copy()
    df["userid"] = df["userid"].astype(np.int64)
    df["rt_userid"] = df["rt_userid"].astype(np.int64)

    df["date"] = df["date"].astype(str).str.strip()
    df["timestamp"] = pd.to_datetime(
        df["date"],
        format="%a %b %d %H:%M:%S +0000 %Y",
        utc=True,
        errors="coerce",
    )
    invalid_ts = df["timestamp"].isna()
    if invalid_ts.any():
        n_bad = int(invalid_ts.sum())
        bad_examples = df.loc[invalid_ts, "date"].drop_duplicates().head(5).tolist()
        if strict_dates:
            raise ValueError(
                f"Invalid timestamp rows: {n_bad:,} / {len(df):,}. Examples: {bad_examples}"
            )
        print(f"Dropping invalid timestamp rows: {n_bad:,} / {len(df):,}")
        print(f"Example invalid date values: {bad_examples}")
        df = df.loc[~invalid_ts].copy()

    if len(df) == 0:
        raise ValueError("No valid rows after cleanup.")

    df = df.sort_values("timestamp").reset_index(drop=True)
    cutoff_idx = int(len(df) * history_fraction)
    cutoff_idx = max(1, min(len(df) - 1, cutoff_idx))
    cutoff_time = df["timestamp"].iloc[cutoff_idx]
    print(f"History cutoff idx={cutoff_idx:,} ({history_fraction:.2f}), time={cutoff_time}")

    history_df = df.iloc[:cutoff_idx].copy()
    future_df = df.iloc[cutoff_idx:].copy()
    t0 = history_df["timestamp"].min()
    return history_df, future_df, t0


def build_pair_stats(df, max_group_size):
    """
    For each undirected user pair (u, v):
    - first_retweet_time: first timestamp where both users had co-retweeted any account
    - n_retweets: number of distinct retweeted accounts co-retweeted by the pair
    """
    stats = {}
    grouped = df.groupby("rt_userid")[["userid", "timestamp"]]
    for _, grp in grouped:
        grp = grp.sort_values("timestamp").drop_duplicates("userid", keep="first")
        users = grp["userid"].astype(np.int64).tolist()
        if len(users) < 2:
            continue
        if max_group_size is not None and len(users) > max_group_size:
            continue
        first_ts = grp["timestamp"].view("int64").tolist()  # ns since epoch
        for i, j in combinations(range(len(users)), 2):
            u, v = users[i], users[j]
            if u > v:
                u, v = v, u
                i, j = j, i
            first_pair_ts = max(first_ts[i], first_ts[j])
            key = (u, v)
            prev = stats.get(key)
            if prev is None:
                stats[key] = [first_pair_ts, 1]
            else:
                if first_pair_ts < prev[0]:
                    prev[0] = first_pair_ts
                prev[1] += 1
    return stats


def stats_to_edge_tensors(pair_stats, id_to_idx, t0):
    src, dst = [], []
    attrs = []
    kept = 0
    t0_ns = int(pd.Timestamp(t0).value)
    for (u, v), (first_ts_ns, n_retweets) in pair_stats.items():
        if u not in id_to_idx or v not in id_to_idx:
            continue
        ui, vi = id_to_idx[u], id_to_idx[v]
        first_hours = max(0.0, float(first_ts_ns - t0_ns) / 3_600_000_000_000.0)
        n_retweets_log = float(np.log1p(n_retweets))
        feat = [first_hours, n_retweets_log]

        src.extend([ui, vi])
        dst.extend([vi, ui])
        attrs.extend([feat, feat])
        kept += 1

    if not src:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 2), dtype=torch.float)
    else:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.tensor(attrs, dtype=torch.float)
    return edge_index, edge_attr, kept


def pairs_to_edge_index(pair_keys, id_to_idx):
    src, dst = [], []
    kept = 0
    for u, v in pair_keys:
        if u not in id_to_idx or v not in id_to_idx:
            continue
        ui, vi = id_to_idx[u], id_to_idx[v]
        src.extend([ui, vi])
        dst.extend([vi, ui])
        kept += 1
    if not src:
        return torch.zeros((2, 0), dtype=torch.long), kept
    return torch.tensor([src, dst], dtype=torch.long), kept


def main():
    args = parse_args()
    out_graph = args.output_graph or args.source_graph

    print(f"Loading source graph: {args.source_graph}")
    raw = torch.load(args.source_graph, map_location="cpu")
    if "user_ids" not in raw:
        raise KeyError("source graph_data.pt must contain 'user_ids'")
    user_ids = raw["user_ids"]

    id_to_idx = {}
    invalid_uid = 0
    duplicate_uid = 0
    for i, uid in enumerate(user_ids):
        u = to_int_user_id(uid)
        if u is None:
            invalid_uid += 1
            continue
        if u in id_to_idx:
            duplicate_uid += 1
            continue
        id_to_idx[u] = i
    print(
        f"Usable user ids: {len(id_to_idx):,} / {len(user_ids):,} "
        f"(invalid={invalid_uid:,}, duplicates={duplicate_uid:,})"
    )

    history_df, future_df, t0 = load_and_split_rows(
        args.csv_glob, args.history_fraction, args.strict_dates
    )
    print("Building history pair stats...")
    history_stats = build_pair_stats(history_df, args.max_group_size)
    print(f"History undirected pairs: {len(history_stats):,}")

    print("Building future pair stats...")
    future_stats = build_pair_stats(future_df, args.max_group_size)
    print(f"Future undirected pairs: {len(future_stats):,}")

    history_pairs = set(history_stats.keys())
    future_pairs = set(future_stats.keys())
    future_new_pairs = future_pairs - history_pairs
    print(f"Future-new undirected pairs: {len(future_new_pairs):,}")

    history_edge_index, history_edge_attr, kept_history_pairs = stats_to_edge_tensors(
        history_stats, id_to_idx, t0
    )
    future_new_edge_index, kept_future_pairs = pairs_to_edge_index(future_new_pairs, id_to_idx)
    print(
        f"Kept history pairs in node set: {kept_history_pairs:,} "
        f"(directed edges={history_edge_index.shape[1]:,})"
    )
    print(
        f"Kept future-new pairs in node set: {kept_future_pairs:,} "
        f"(directed edges={future_new_edge_index.shape[1]:,})"
    )

    edge_index_views = dict(raw.get("edge_index_views", {}))
    edge_attr_views = dict(raw.get("edge_attr_views", {}))
    edge_attr_feature_names_views = dict(raw.get("edge_attr_feature_names_views", {}))
    target_edge_index_views = dict(raw.get("target_edge_index_views", {}))

    edge_index_views[args.edge_view_name] = history_edge_index
    edge_attr_views[args.edge_view_name] = history_edge_attr
    edge_attr_feature_names_views[args.edge_view_name] = ["first_retweet_time", "n_retweets"]
    target_edge_index_views[args.target_view_name] = future_new_edge_index

    raw["edge_index_views"] = edge_index_views
    raw["edge_attr_views"] = edge_attr_views
    raw["edge_attr_feature_names_views"] = edge_attr_feature_names_views
    raw["target_edge_index_views"] = target_edge_index_views

    # Backward-compatible aliases for existing temporal LP code paths.
    raw["future_edge_index"] = future_new_edge_index

    if args.set_default_edge_view:
        raw["edge_index"] = history_edge_index
        raw["edge_attr"] = history_edge_attr
        raw["edge_attr_feature_names"] = ["first_retweet_time", "n_retweets"]
    if args.set_default_target_view:
        raw["future_edge_index"] = future_new_edge_index

    out_dir = os.path.dirname(out_graph)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(raw, out_graph)
    print(f"Saved augmented graph to: {out_graph}")
    print(f"Edge view name: {args.edge_view_name}")
    print(f"Target edge view name: {args.target_view_name}")


if __name__ == "__main__":
    main()
