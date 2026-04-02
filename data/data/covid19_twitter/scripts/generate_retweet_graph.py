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


DEFAULT_JSON_GLOB = "/scratch1/eibl/data/covid19_twitter/*/*.json"
DEFAULT_OUT = "data/data/covid19_twitter/graphs/retweet_graph.pt"
DEFAULT_HISTORY_FRACTION = 0.8

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
    p.add_argument("--embeddings", default="", help="Optional user_embeddings_*.pt with handles + meanpool/maxpool")
    p.add_argument("--embedding_pool", choices=["meanpool", "maxpool"], default="meanpool")
    p.add_argument("--max_files", type=int, default=0)
    p.add_argument("--strict_dates", action="store_true")
    p.add_argument("--history_fraction", type=float, default=DEFAULT_HISTORY_FRACTION)
    p.add_argument("--future_target_mode", choices=["new_only", "all_future"], default="new_only")
    p.add_argument("--no_temporal_views", action="store_true")
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
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
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

            rows.append(
                {
                    "screen_name": normalize_handle(user.get("screen_name")),
                    "userid": user.get("id"),
                    "rt_screen": normalize_handle(rt_user.get("screen_name")) if rt else None,
                    "rt_userid": rt_user.get("id") if rt else None,
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
    df = df[df["screen_name"].notna()].copy()
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

    rt = df.dropna(subset=["screen_name", "rt_screen"]).copy()
    rt = rt[rt["screen_name"] != rt["rt_screen"]].copy()
    if rt.empty:
        raise RuntimeError("No valid retweet rows after cleaning")
    return rt


def build_handle_index(rt: pd.DataFrame) -> Tuple[List[str], Dict[str, int]]:
    handles = sorted(set(rt["screen_name"].tolist()) | set(rt["rt_screen"].tolist()))
    return handles, {handle: i for i, handle in enumerate(handles)}


def aggregate_edge_features(rt: pd.DataFrame) -> pd.DataFrame:
    edge_grp = rt.groupby(["screen_name", "rt_screen"], as_index=False).agg(
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


def to_edge_tensors(edge_df: pd.DataFrame, h2i: Dict[str, int]) -> Tuple[torch.Tensor, torch.Tensor]:
    tmp = edge_df.copy()
    tmp["src"] = tmp["screen_name"].map(h2i)
    tmp["dst"] = tmp["rt_screen"].map(h2i)
    tmp = tmp.dropna(subset=["src", "dst"]).copy()
    tmp[["src", "dst"]] = tmp[["src", "dst"]].astype(int)
    edge_index = torch.tensor(tmp[["src", "dst"]].values.T, dtype=torch.long)
    edge_attr = torch.tensor(tmp[EDGE_FEATURE_NAMES].fillna(0).values.astype(np.float32), dtype=torch.float)
    return edge_index, edge_attr


def build_node_features(rt: pd.DataFrame, h2i: Dict[str, int], edge_df: pd.DataFrame):
    base = rt.copy()
    base["n_hashtags"] = base.get("hashtag", []).apply(count_list_like)
    base["n_mentions"] = base.get("mentionsn", []).apply(count_list_like)
    base["has_media"] = base.get("media_urls", []).apply(lambda x: int(count_list_like(x) > 0))
    if "sent_vader" not in base.columns:
        base["sent_vader"] = 0.0

    node_agg = base.groupby("screen_name", as_index=False).agg(
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

    in_deg = edge_df.groupby("rt_screen").size().rename("in_degree")
    out_deg = edge_df.groupby("screen_name").size().rename("out_degree")
    node_agg = node_agg.merge(out_deg, left_on="screen_name", right_index=True, how="left")
    node_agg = node_agg.merge(in_deg, left_on="screen_name", right_index=True, how="left")
    node_agg["in_degree"] = np.log1p(node_agg["in_degree"].fillna(0))
    node_agg["out_degree"] = np.log1p(node_agg["out_degree"].fillna(0))

    n_nodes = len(h2i)
    x_np = np.zeros((n_nodes, len(NODE_FEATURE_NAMES)), dtype=np.float32)
    node_agg["node_idx"] = node_agg["screen_name"].map(h2i)
    node_agg = node_agg.dropna(subset=["node_idx"])
    rows = node_agg["node_idx"].astype(int).values
    x_np[rows] = node_agg[NODE_FEATURE_NAMES].fillna(0).values.astype(np.float32)

    all_out = np.zeros(n_nodes, dtype=np.float32)
    all_in = np.zeros(n_nodes, dtype=np.float32)
    for handle, c in edge_df.groupby("screen_name").size().items():
        all_out[h2i[handle]] = np.log1p(float(c))
    for handle, c in edge_df.groupby("rt_screen").size().items():
        all_in[h2i[handle]] = np.log1p(float(c))
    x_np[:, NODE_FEATURE_NAMES.index("out_degree")] = np.maximum(x_np[:, NODE_FEATURE_NAMES.index("out_degree")], all_out)
    x_np[:, NODE_FEATURE_NAMES.index("in_degree")] = np.maximum(x_np[:, NODE_FEATURE_NAMES.index("in_degree")], all_in)

    nonzero = np.any(x_np != 0, axis=1)
    if nonzero.any():
        scaler = StandardScaler()
        x_np[nonzero] = scaler.fit_transform(x_np[nonzero]).astype(np.float32)
    return torch.tensor(x_np, dtype=torch.float), list(NODE_FEATURE_NAMES)


def maybe_attach_embeddings(x, feature_names, handles, embeddings_path, embedding_pool):
    if not embeddings_path:
        return x, feature_names, {"matched_users": 0, "embedding_dim": 0}
    emb = torch.load(embeddings_path, map_location="cpu")
    emb_handles = emb.get("handles")
    emb_mat = emb.get(embedding_pool)
    if emb_handles is None or emb_mat is None:
        raise KeyError(f"Embeddings file must contain 'handles' and '{embedding_pool}'")
    emb_dim = int(emb_mat.shape[1])
    emb_map = {str(handle).strip().lower(): i for i, handle in enumerate(emb_handles)}
    extra = torch.zeros((len(handles), emb_dim), dtype=torch.float)
    matched = 0
    for i, handle in enumerate(handles):
        j = emb_map.get(handle)
        if j is None:
            continue
        extra[i] = emb_mat[j].float()
        matched += 1
    return torch.cat([x, extra], dim=1), feature_names + [f"emb_{k}" for k in range(emb_dim)], {"matched_users": matched, "embedding_dim": emb_dim}


def drop_isolates_from_graph(x, edge_index, edge_attr, handles):
    n = len(handles)
    degrees_out = torch.bincount(edge_index[0], minlength=n)
    degrees_in = torch.bincount(edge_index[1], minlength=n)
    keep_mask = (degrees_out > 0) | (degrees_in > 0)
    isolated = int((~keep_mask).sum().item())
    if isolated == 0:
        return x, edge_index, edge_attr, handles, {h: i for i, h in enumerate(handles)}, 0
    kept_nodes = int(keep_mask.sum().item())
    remap = torch.full((n,), -1, dtype=torch.long)
    remap[keep_mask] = torch.arange(kept_nodes, dtype=torch.long)
    x = x[keep_mask]
    edge_index = remap[edge_index]
    handles = [handle for handle, keep in zip(handles, keep_mask.tolist()) if keep]
    return x, edge_index, edge_attr, handles, {h: i for i, h in enumerate(handles)}, isolated


def main():
    args = parse_args()
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    raw = load_raw_rows(args.json_glob, args.max_files)
    rt = prepare_retweet_rows(raw, args.strict_dates)
    handles, h2i = build_handle_index(rt)
    print(f"Nodes: {len(handles):,}")

    edge_all_df = aggregate_edge_features(rt)
    edge_index, edge_attr = to_edge_tensors(edge_all_df, h2i)
    print(f"Directed edges: {edge_index.shape[1]:,}")

    x, feature_names = build_node_features(rt, h2i, edge_all_df)
    x, feature_names, emb_stats = maybe_attach_embeddings(x, feature_names, handles, args.embeddings, args.embedding_pool)

    isolated_before_drop = int(((torch.bincount(edge_index[0], minlength=len(handles)) == 0) & (torch.bincount(edge_index[1], minlength=len(handles)) == 0)).sum().item())
    if not args.keep_isolates:
        x, edge_index, edge_attr, handles, h2i, isolated_dropped = drop_isolates_from_graph(x, edge_index, edge_attr, handles)
        print(f"Dropped isolated nodes: {isolated_dropped:,}")
    else:
        isolated_dropped = 0

    y = torch.full((len(handles),), -1, dtype=torch.long)
    label_names = []

    graph_obj = {
        "x": x,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "edge_attr_feature_names": EDGE_FEATURE_NAMES,
        "handles": handles,
        "h2i": h2i,
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
        hist_edge_index, hist_edge_attr = to_edge_tensors(hist_edges_df, h2i)
        hist_pairs = set(zip(hist_edges_df["screen_name"], hist_edges_df["rt_screen"]))
        fut_pairs = set(zip(fut_edges_df["screen_name"], fut_edges_df["rt_screen"]))
        target_pairs = fut_pairs - hist_pairs if args.future_target_mode == "new_only" else fut_pairs
        if target_pairs:
            target_df = pd.DataFrame(sorted(list(target_pairs)), columns=["screen_name", "rt_screen"])
            target_df["src"] = target_df["screen_name"].map(h2i)
            target_df["dst"] = target_df["rt_screen"].map(h2i)
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
    graph_obj["data"] = data
    torch.save(graph_obj, args.out)

    meta = {
        "json_glob": args.json_glob,
        "max_files": args.max_files,
        "nodes": int(len(handles)),
        "edges": int(edge_index.shape[1]),
        "node_feature_dim": int(x.shape[1]),
        "edge_feature_names": EDGE_FEATURE_NAMES,
        "label_count": 0,
        "labeled_nodes": 0,
        "embeddings": args.embeddings,
        "embedding_pool": args.embedding_pool,
        "embedding_dim": emb_stats["embedding_dim"],
        "embedding_matched_users": emb_stats["matched_users"],
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
