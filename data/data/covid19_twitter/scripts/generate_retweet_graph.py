import argparse
import glob
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data


DEFAULT_JSON_GLOB = "/scratch1/eibl/data/covid19_twitter/raw/*/*.json"
DEFAULT_OUT = "data/data/covid19_twitter/graphs/retweet_graph.pt"
DEFAULT_HISTORY_FRACTION = 0.8
DEFAULT_LABELS_PARQUET_GLOB = "/scratch1/eibl/data/covid_masking/masking_2020-*.parquet"
DEFAULT_LABEL_HANDLE_COL = "screen_name"
DEFAULT_LABEL_VALUE_COL = "political_gen"

NODE_FEATURE_NAMES = [
    "subscriber_count",
    "verified",
    "avg_favorites",
    "avg_comments",
    "avg_score",
    "avg_n_hashtags",
    "avg_n_mentions",
    "avg_has_media",
    "post_count",
    "in_degree",
    "out_degree",
]

EDGE_FEATURE_NAMES = [
    "first_retweet_time",
    "n_retweets",
    "avg_rt_fav",
    "avg_rt_reply",
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Build unified covid19_twitter retweet graph with optional embeddings and temporal views."
    )
    p.add_argument("--json_glob", default=DEFAULT_JSON_GLOB)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--embeddings", default="", help="Optional user_embeddings_*.pt with user_ids + meanpool/maxpool (handles fallback supported)")
    p.add_argument("--embedding_pool", choices=["meanpool", "maxpool"], default="meanpool")
    p.add_argument("--max_files", type=int, default=0)
    p.add_argument("--max_nodes", type=int, default=0, help="Trim the final graph to exactly this many nodes when possible (0 = no limit)")
    p.add_argument("--strict_dates", action="store_true")
    p.add_argument("--history_fraction", type=float, default=DEFAULT_HISTORY_FRACTION)
    p.add_argument("--future_target_mode", choices=["new_only", "all_future"], default="new_only")
    p.add_argument("--no_temporal_views", action="store_true")
    p.add_argument(
        "--labels_parquet_glob",
        default=DEFAULT_LABELS_PARQUET_GLOB,
        help="Optional parquet glob containing external node labels keyed by handle.",
    )
    p.add_argument(
        "--keep-isolates",
        dest="keep_isolates",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return p.parse_args()


def normalize_handle(handle):
    if handle is None:
        return None
    s = str(handle).strip().lower()
    return s if s and s not in {"nan", "none", "<na>"} else None


def normalize_user_id(user_id):
    if user_id is None:
        return None
    try:
        return int(user_id)
    except Exception:
        return None


def load_json_items(path: str):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read().strip()
    if not text:
        return []
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            if isinstance(obj.get("statuses"), list):
                return obj["statuses"]
            if isinstance(obj.get("data"), list):
                return obj["data"]
            return [obj]
        return []
    except json.JSONDecodeError:
        items = []
        bad_lines = 0
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                bad_lines += 1
        if bad_lines:
            print(f"  [WARN] skipped {bad_lines:,} malformed JSON lines in {os.path.basename(path)}", flush=True)
        return items


def count_list_like(items) -> int:
    if items is None:
        return 0
    if isinstance(items, list):
        return len(items)
    return 0


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
            rt_user = rt.get("user") or {}

            uid = normalize_user_id(user.get("id"))
            rt_uid = normalize_user_id(rt_user.get("id")) if rt else None

            rows.append(
                {
                    "screen_name": normalize_handle(user.get("screen_name")),
                    "userid": uid,
                    "rt_screen": normalize_handle(rt_user.get("screen_name")) if rt else None,
                    "rt_userid": rt_uid,
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
                }
            )

        if i % 10 == 0 or i == len(files):
            print(f"  processed {i}/{len(files)} files", flush=True)

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No valid rows parsed")
    print(f"Loaded rows: {len(df):,}")
    return df


def prepare_retweet_rows(df: pd.DataFrame, strict_dates: bool) -> pd.DataFrame:
    df = df.copy()
    for col in ["userid", "rt_userid", "followers_count", "statuses_count", "rt_fav_count", "rt_reply_count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "verified" in df.columns:
        df["verified"] = df["verified"].map({True: 1, False: 0, "True": 1, "False": 0}).fillna(0)

    df["date"] = df.get("date", "").astype(str).str.strip()
    df["timestamp"] = pd.to_datetime(
        df["date"],
        format="%a %b %d %H:%M:%S %z %Y",
        utc=True,
        errors="coerce",
    )

    bad_ts = df["timestamp"].isna()
    if bad_ts.any():
        n_bad = int(bad_ts.sum())
        examples = df.loc[bad_ts, "date"].drop_duplicates().head(5).tolist()
        if strict_dates:
            raise ValueError(f"Invalid timestamp rows: {n_bad:,}, examples={examples}")
        print(f"Dropping invalid timestamp rows: {n_bad:,}")
        df = df.loc[~bad_ts].copy()

    rt = df.dropna(subset=["userid", "rt_userid"]).copy()
    rt = rt[rt["userid"] != rt["rt_userid"]].copy()
    rt["userid"] = rt["userid"].astype(np.int64)
    rt["rt_userid"] = rt["rt_userid"].astype(np.int64)
    if rt.empty:
        raise RuntimeError("No valid retweet rows after cleaning")
    return rt


def trim_rt_to_max_nodes(rt: pd.DataFrame, max_nodes: int) -> pd.DataFrame:
    if max_nodes <= 0:
        return rt

    total_unique_nodes = len(set(rt["userid"].tolist()) | set(rt["rt_userid"].tolist()))
    if total_unique_nodes < max_nodes:
        raise ValueError(
            f"Requested max_nodes={max_nodes:,}, but only {total_unique_nodes:,} unique retweet nodes exist "
            "after cleaning."
        )

    src = rt["userid"].to_numpy(dtype=np.int64, copy=False)
    dst = rt["rt_userid"].to_numpy(dtype=np.int64, copy=False)
    keep_rows = np.zeros(len(rt), dtype=bool)
    keep_nodes = set()

    for i, (u, v) in enumerate(zip(src, dst)):
        add = 0
        if u not in keep_nodes:
            add += 1
        if v not in keep_nodes:
            add += 1
        if len(keep_nodes) + add <= max_nodes:
            keep_rows[i] = True
            keep_nodes.add(int(u))
            keep_nodes.add(int(v))
        elif len(keep_nodes) >= max_nodes and u in keep_nodes and v in keep_nodes:
            keep_rows[i] = True
        if len(keep_nodes) == max_nodes:
            continue

    rt2 = rt.loc[keep_rows].copy()
    actual_nodes = sorted(set(rt2["userid"].tolist()) | set(rt2["rt_userid"].tolist()))
    print(
        f"Applied exact max_nodes={max_nodes:,}: kept rows={len(rt2):,} "
        f"nodes={len(actual_nodes):,}",
        flush=True,
    )
    if len(actual_nodes) != max_nodes:
        raise RuntimeError(
            f"Exact max_nodes trim failed: requested {max_nodes:,}, got {len(actual_nodes):,} nodes"
        )
    return rt2


def build_user_index(rt: pd.DataFrame) -> Tuple[List[int], Dict[int, int]]:
    user_ids = sorted(set(rt["userid"].tolist()) | set(rt["rt_userid"].tolist()))
    return user_ids, {int(user_id): i for i, user_id in enumerate(user_ids)}


def build_user_metadata(raw: pd.DataFrame, user_ids: List[int]) -> List[str]:
    frames = []
    left = raw[["userid", "screen_name", "timestamp"]].rename(columns={"userid": "user_id", "screen_name": "handle"})
    frames.append(left)
    right = raw[["rt_userid", "rt_screen", "timestamp"]].rename(columns={"rt_userid": "user_id", "rt_screen": "handle"})
    frames.append(right)
    meta = pd.concat(frames, ignore_index=True)
    meta = meta.dropna(subset=["user_id"]).copy()
    meta["user_id"] = pd.to_numeric(meta["user_id"], errors="coerce")
    meta = meta.dropna(subset=["user_id"]).copy()
    meta["user_id"] = meta["user_id"].astype(np.int64)
    meta["handle"] = meta["handle"].map(normalize_handle)
    meta = meta.dropna(subset=["handle"]).copy()
    meta = meta.sort_values("timestamp").drop_duplicates(subset=["user_id"], keep="last")
    handle_map = meta.set_index("user_id")["handle"].to_dict()
    return [handle_map.get(int(user_id)) for user_id in user_ids]


def aggregate_edge_features(rt: pd.DataFrame) -> pd.DataFrame:
    edge_grp = rt.groupby(["userid", "rt_userid"], as_index=False).agg(
        first_ts=("timestamp", "min"),
        n_retweets=("timestamp", "size"),
        avg_rt_fav=("rt_fav_count", "mean"),
        avg_rt_reply=("rt_reply_count", "mean"),
    )
    min_ns = int(edge_grp["first_ts"].min().value)
    edge_grp["first_retweet_time"] = (edge_grp["first_ts"].astype("int64") - min_ns) / 3_600_000_000_000.0
    edge_grp["n_retweets"] = np.log1p(edge_grp["n_retweets"].astype(float))
    edge_grp["avg_rt_fav"] = np.log1p(pd.to_numeric(edge_grp["avg_rt_fav"], errors="coerce").fillna(0).clip(lower=0))
    edge_grp["avg_rt_reply"] = np.log1p(pd.to_numeric(edge_grp["avg_rt_reply"], errors="coerce").fillna(0).clip(lower=0))
    return edge_grp


def to_edge_tensors(edge_df: pd.DataFrame, u2i: Dict[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
    tmp = edge_df.copy()
    tmp["src"] = tmp["userid"].map(u2i)
    tmp["dst"] = tmp["rt_userid"].map(u2i)
    tmp = tmp.dropna(subset=["src", "dst"]).copy()
    tmp[["src", "dst"]] = tmp[["src", "dst"]].astype(int)
    edge_index = torch.tensor(tmp[["src", "dst"]].values.T, dtype=torch.long)
    edge_attr = torch.tensor(tmp[EDGE_FEATURE_NAMES].fillna(0).values.astype(np.float32), dtype=torch.float)
    return edge_index, edge_attr


def build_node_features(rt: pd.DataFrame, u2i: Dict[int, int], edge_df: pd.DataFrame):
    base = rt.copy()
    base["n_hashtags"] = base.get("hashtag", []).apply(count_list_like)
    base["n_mentions"] = base.get("mentionsn", []).apply(count_list_like)
    base["has_media"] = base.get("media_urls", []).apply(lambda x: int(count_list_like(x) > 0))
    if "sent_vader" not in base.columns:
        base["sent_vader"] = 0.0

    node_agg = base.groupby("userid", as_index=False).agg(
        subscriber_count=("followers_count", "max"),
        verified=("verified", "max"),
        avg_favorites=("rt_fav_count", "mean"),
        avg_comments=("rt_reply_count", "mean"),
        avg_score=("sent_vader", "mean"),
        avg_n_hashtags=("n_hashtags", "mean"),
        avg_n_mentions=("n_mentions", "mean"),
        avg_has_media=("has_media", "mean"),
        post_count=("statuses_count", "max"),
    )

    for col in ["subscriber_count", "avg_favorites", "avg_comments", "post_count"]:
        node_agg[col] = np.log1p(pd.to_numeric(node_agg[col], errors="coerce").fillna(0).clip(lower=0))

    in_deg = edge_df.groupby("rt_userid").size().rename("in_degree")
    out_deg = edge_df.groupby("userid").size().rename("out_degree")
    node_agg = node_agg.merge(out_deg, left_on="userid", right_index=True, how="left")
    node_agg = node_agg.merge(in_deg, left_on="userid", right_index=True, how="left")
    node_agg["in_degree"] = np.log1p(node_agg["in_degree"].fillna(0))
    node_agg["out_degree"] = np.log1p(node_agg["out_degree"].fillna(0))

    n_nodes = len(u2i)
    x_np = np.zeros((n_nodes, len(NODE_FEATURE_NAMES)), dtype=np.float32)
    node_agg["node_idx"] = node_agg["userid"].map(u2i)
    node_agg = node_agg.dropna(subset=["node_idx"])
    rows = node_agg["node_idx"].astype(int).values
    x_np[rows] = node_agg[NODE_FEATURE_NAMES].fillna(0).values.astype(np.float32)

    all_out = np.zeros(n_nodes, dtype=np.float32)
    all_in = np.zeros(n_nodes, dtype=np.float32)
    for user_id, c in edge_df.groupby("userid").size().items():
        all_out[u2i[int(user_id)]] = np.log1p(float(c))
    for user_id, c in edge_df.groupby("rt_userid").size().items():
        all_in[u2i[int(user_id)]] = np.log1p(float(c))
    x_np[:, NODE_FEATURE_NAMES.index("out_degree")] = np.maximum(x_np[:, NODE_FEATURE_NAMES.index("out_degree")], all_out)
    x_np[:, NODE_FEATURE_NAMES.index("in_degree")] = np.maximum(x_np[:, NODE_FEATURE_NAMES.index("in_degree")], all_in)

    nonzero = np.any(x_np != 0, axis=1)
    if nonzero.any():
        scaler = StandardScaler()
        x_np[nonzero] = scaler.fit_transform(x_np[nonzero]).astype(np.float32)
    return torch.tensor(x_np, dtype=torch.float), list(NODE_FEATURE_NAMES)


def maybe_attach_embeddings(x, feature_names, user_ids, handles, embeddings_path, embedding_pool):
    if not embeddings_path:
        return x, feature_names, {"matched_users": 0, "embedding_dim": 0}
    emb = torch.load(embeddings_path, map_location="cpu")
    emb_mat = emb.get(embedding_pool)
    if emb_mat is None:
        raise KeyError(f"Embeddings file must contain '{embedding_pool}'")
    emb_dim = int(emb_mat.shape[1])
    extra = torch.zeros((len(handles), emb_dim), dtype=torch.float)
    matched = 0
    emb_user_ids = emb.get("user_ids")
    if emb_user_ids is not None:
        emb_map = {int(uid): i for i, uid in enumerate(np.asarray(emb_user_ids))}
        for i, user_id in enumerate(user_ids):
            j = emb_map.get(int(user_id))
            if j is None:
                continue
            extra[i] = emb_mat[j].float()
            matched += 1
    else:
        emb_handles = emb.get("handles")
        if emb_handles is None:
            raise KeyError(f"Embeddings file must contain either 'user_ids' or 'handles' plus '{embedding_pool}'")
        emb_map = {str(handle).strip().lower(): i for i, handle in enumerate(emb_handles)}
        for i, handle in enumerate(handles):
            if not handle:
                continue
            j = emb_map.get(handle)
            if j is None:
                continue
            extra[i] = emb_mat[j].float()
            matched += 1
    return torch.cat([x, extra], dim=1), feature_names + [f"emb_{k}" for k in range(emb_dim)], {"matched_users": matched, "embedding_dim": emb_dim}


def load_external_labels(labels_parquet_glob: str):
    if not labels_parquet_glob:
        return None
    files = sorted(glob.glob(labels_parquet_glob))
    if not files:
        raise FileNotFoundError(f"No label parquet files matched: {labels_parquet_glob}")

    chunks = []
    for path in files:
        df = pd.read_parquet(path, columns=[DEFAULT_LABEL_HANDLE_COL, DEFAULT_LABEL_VALUE_COL])
        chunks.append(df)
    labels_df = pd.concat(chunks, ignore_index=True)
    if labels_df.empty:
        raise RuntimeError("External labels dataframe is empty.")

    labels_df = labels_df.dropna(subset=[DEFAULT_LABEL_HANDLE_COL, DEFAULT_LABEL_VALUE_COL]).copy()
    labels_df["handle"] = labels_df[DEFAULT_LABEL_HANDLE_COL].map(normalize_handle)
    labels_df = labels_df.dropna(subset=["handle"]).copy()
    labels_df = labels_df.drop_duplicates(subset=["handle"], keep="first")
    if labels_df.empty:
        raise RuntimeError("No usable external labels remained after normalization.")

    raw_values = labels_df[DEFAULT_LABEL_VALUE_COL].tolist()
    unique_raw_values = sorted(pd.Series(raw_values).drop_duplicates().tolist(), key=lambda v: str(v))
    label_names = [str(v) for v in unique_raw_values]

    raw_to_idx = {raw_value: idx for idx, raw_value in enumerate(unique_raw_values)}
    handle_to_label = labels_df.set_index("handle")[DEFAULT_LABEL_VALUE_COL].map(raw_to_idx).to_dict()
    return {
        "handle_to_label": handle_to_label,
        "label_names": label_names,
        "raw_values": unique_raw_values,
        "n_rows": int(len(labels_df)),
        "n_files": int(len(files)),
    }


def build_node_labels(handles, label_info):
    y = torch.full((len(handles),), -1, dtype=torch.long)
    label_names = []
    if not label_info:
        return y, label_names, {"labeled_nodes": 0, "label_count": 0}

    handle_to_label = label_info["handle_to_label"]
    label_names = list(label_info["label_names"])
    labeled = 0
    for i, handle in enumerate(handles):
        label = handle_to_label.get(handle)
        if label is None:
            continue
        y[i] = int(label)
        labeled += 1
    return y, label_names, {"labeled_nodes": labeled, "label_count": len(label_names)}


def drop_isolates_from_graph(x, edge_index, edge_attr, user_ids, handles):
    n = len(user_ids)
    degrees_out = torch.bincount(edge_index[0], minlength=n)
    degrees_in = torch.bincount(edge_index[1], minlength=n)
    keep_mask = (degrees_out > 0) | (degrees_in > 0)
    isolated = int((~keep_mask).sum().item())
    if isolated == 0:
        return x, edge_index, edge_attr, user_ids, handles, {int(uid): i for i, uid in enumerate(user_ids)}, 0
    kept_nodes = int(keep_mask.sum().item())
    remap = torch.full((n,), -1, dtype=torch.long)
    remap[keep_mask] = torch.arange(kept_nodes, dtype=torch.long)
    x = x[keep_mask]
    edge_index = remap[edge_index]
    user_ids = [int(user_id) for user_id, keep in zip(user_ids, keep_mask.tolist()) if keep]
    handles = [handle for handle, keep in zip(handles, keep_mask.tolist()) if keep]
    return x, edge_index, edge_attr, user_ids, handles, {int(uid): i for i, uid in enumerate(user_ids)}, isolated


def main():
    args = parse_args()
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    raw = load_raw_rows(args.json_glob, args.max_files)
    rt = prepare_retweet_rows(raw, args.strict_dates)
    rt = trim_rt_to_max_nodes(rt, args.max_nodes)
    user_ids, u2i = build_user_index(rt)
    handles = build_user_metadata(rt, user_ids)
    print(f"Nodes: {len(user_ids):,}")

    edge_all_df = aggregate_edge_features(rt)
    edge_index, edge_attr = to_edge_tensors(edge_all_df, u2i)
    print(f"Directed edges: {edge_index.shape[1]:,}")

    x, feature_names = build_node_features(rt, u2i, edge_all_df)
    x, feature_names, emb_stats = maybe_attach_embeddings(x, feature_names, user_ids, handles, args.embeddings, args.embedding_pool)

    isolated_before_drop = int(((torch.bincount(edge_index[0], minlength=len(user_ids)) == 0) & (torch.bincount(edge_index[1], minlength=len(user_ids)) == 0)).sum().item())
    if not args.keep_isolates:
        x, edge_index, edge_attr, user_ids, handles, u2i, isolated_dropped = drop_isolates_from_graph(
            x, edge_index, edge_attr, user_ids, handles
        )
        print(f"Dropped isolated nodes: {isolated_dropped:,}")
    else:
        isolated_dropped = 0

    label_info = load_external_labels(args.labels_parquet_glob)
    y, label_names, label_stats = build_node_labels(handles, label_info)
    if label_info:
        print(
            f"Attached labels: labeled_nodes={label_stats['labeled_nodes']:,} "
            f"label_count={label_stats['label_count']} labels={label_names}",
            flush=True,
        )

    graph_obj = {
        "x": x,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "edge_attr_feature_names": EDGE_FEATURE_NAMES,
        "user_ids": user_ids,
        "u2i": u2i,
        "handles": handles,
        "feature_names": feature_names,
        "y": y,
        "label_names": label_names,
    }

    temporal_stats = {}
    if not args.no_temporal_views:
        rt_sorted = rt.sort_values("timestamp").reset_index(drop=True)
        cutoff_idx = int(len(rt_sorted) * args.history_fraction)
        cutoff_idx = max(1, min(len(rt_sorted) - 1, cutoff_idx))
        hist_rt = rt_sorted.iloc[:cutoff_idx].copy()
        fut_rt = rt_sorted.iloc[cutoff_idx:].copy()
        hist_edges_df = aggregate_edge_features(hist_rt)
        fut_edges_df = aggregate_edge_features(fut_rt)
        hist_edge_index, hist_edge_attr = to_edge_tensors(hist_edges_df, u2i)
        hist_pairs = set(zip(hist_edges_df["userid"], hist_edges_df["rt_userid"]))
        fut_pairs = set(zip(fut_edges_df["userid"], fut_edges_df["rt_userid"]))
        target_pairs = fut_pairs - hist_pairs if args.future_target_mode == "new_only" else fut_pairs
        if target_pairs:
            target_df = pd.DataFrame(sorted(list(target_pairs)), columns=["userid", "rt_userid"])
            target_df["src"] = target_df["userid"].map(u2i)
            target_df["dst"] = target_df["rt_userid"].map(u2i)
            target_df = target_df.dropna(subset=["src", "dst"])
            target_new_edge_index = torch.tensor(target_df[["src", "dst"]].astype(int).values.T, dtype=torch.long)
        else:
            target_new_edge_index = torch.zeros((2, 0), dtype=torch.long)

        graph_obj["edge_index_views"] = {"retweet_all": edge_index, "temporal_history": hist_edge_index}
        graph_obj["edge_attr_views"] = {"retweet_all": edge_attr, "temporal_history": hist_edge_attr}
        graph_obj["edge_attr_feature_names_views"] = {"retweet_all": EDGE_FEATURE_NAMES, "temporal_history": EDGE_FEATURE_NAMES}
        graph_obj["target_edge_index_views"] = {"temporal_new": target_new_edge_index}
        graph_obj["future_edge_index"] = target_new_edge_index
        temporal_stats = {
            "history_fraction": args.history_fraction,
            "future_target_mode": args.future_target_mode,
            "history_rows": int(len(hist_rt)),
            "future_rows": int(len(fut_rt)),
            "history_edges": int(hist_edge_index.shape[1]),
            "future_edges": int(len(fut_pairs)),
            "future_overlap_edges": int(len(hist_pairs & fut_pairs)),
            "future_target_edges": int(target_new_edge_index.shape[1]),
        }

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data.feature_names = feature_names
    data.edge_attr_feature_names = EDGE_FEATURE_NAMES
    data.label_names = label_names
    data.user_ids = list(user_ids)
    data.handles = list(handles)
    graph_obj["data"] = data
    torch.save(graph_obj, args.out)

    meta = {
        "json_glob": args.json_glob,
        "max_files": args.max_files,
        "nodes": int(len(user_ids)),
        "edges": int(edge_index.shape[1]),
        "node_feature_dim": int(x.shape[1]),
        "edge_feature_names": EDGE_FEATURE_NAMES,
        "label_count": int(label_stats["label_count"]),
        "labeled_nodes": int(label_stats["labeled_nodes"]),
        "embeddings": args.embeddings,
        "embedding_pool": args.embedding_pool,
        "embedding_dim": emb_stats["embedding_dim"],
        "embedding_matched_users": emb_stats["matched_users"],
        "labels_parquet_glob": args.labels_parquet_glob,
        "labels_handle_col": DEFAULT_LABEL_HANDLE_COL,
        "labels_value_col": DEFAULT_LABEL_VALUE_COL,
        "keep_isolates": args.keep_isolates,
        "isolated_nodes_before_drop": isolated_before_drop,
        "isolated_nodes_dropped": isolated_dropped,
        "temporal": temporal_stats,
    }
    meta_path = args.out.replace(".pt", ".meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved graph: {args.out}")
    print(f"Saved meta:  {meta_path}")


if __name__ == "__main__":
    main()
