import argparse
import glob
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler


def parse_args():
    p = argparse.ArgumentParser(
        description="Build unified retweet graph (nodes, edge features, temporal views, optional embeddings)."
    )
    p.add_argument("--csv_glob", default="/project2/ll_774_951/midterm/*/*.csv")
    p.add_argument("--out", default="data/data/midterm/graphs/retweet_graph.pt")
    p.add_argument("--embeddings", default="", help="Optional embeddings_<model>.pt with user_ids + meanpool/maxpool")
    p.add_argument("--embedding_pool", choices=["meanpool", "maxpool"], default="meanpool")
    p.add_argument("--max_files", type=int, default=0)
    p.add_argument("--strict_dates", action="store_true")
    p.add_argument("--history_fraction", type=float, default=0.8)
    p.add_argument("--no_temporal_views", action="store_true")
    return p.parse_args()


def count_list_like(val) -> int:
    if pd.isna(val):
        return 0
    s = str(val).strip()
    if s in {"", "[]", "nan", "None"}:
        return 0
    s = s.strip("[]")
    if not s:
        return 0
    return len([x for x in s.split(",") if x.strip()])


def load_raw_rows(csv_glob: str, max_files: int) -> pd.DataFrame:
    files = sorted(glob.glob(csv_glob))
    if max_files > 0:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"No files matched: {csv_glob}")

    usecols = {
        "userid", "rt_userid", "date", "state",
        "followers_count", "verified", "statuses_count", "sent_vader",
        "hashtag", "mentionsn", "media_urls",
        "rt_fav_count", "rt_reply_count",
    }

    chunks: List[pd.DataFrame] = []
    print(f"Found {len(files)} files")
    for i, fpath in enumerate(files, start=1):
        try:
            dfi = pd.read_csv(fpath, low_memory=False, on_bad_lines="skip")
            if dfi.empty:
                continue
            cols = [c for c in dfi.columns if c in usecols]
            if not cols:
                continue
            chunks.append(dfi[cols].copy())
        except Exception as exc:
            print(f"[WARN] failed {os.path.basename(fpath)}: {exc}")

        if i % 10 == 0 or i == len(files):
            print(f"  processed {i}/{len(files)}")

    if not chunks:
        raise RuntimeError("No valid rows parsed")

    df = pd.concat(chunks, ignore_index=True)
    print(f"Loaded rows: {len(df):,}")
    return df


def normalize_ids_and_timestamps(df: pd.DataFrame, strict_dates: bool) -> pd.DataFrame:
    df = df.copy()
    df["userid"] = pd.to_numeric(df.get("userid"), errors="coerce")
    df["rt_userid"] = pd.to_numeric(df.get("rt_userid"), errors="coerce")

    df["date"] = df.get("date", "").astype(str).str.strip()
    df["timestamp"] = pd.to_datetime(
        df["date"],
        format="%a %b %d %H:%M:%S +0000 %Y",
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
    rt["userid"] = rt["userid"].astype(np.int64)
    rt["rt_userid"] = rt["rt_userid"].astype(np.int64)
    return rt


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


def to_edge_tensors(edge_df: pd.DataFrame, id_to_idx: Dict[int, int], edge_feature_names: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    tmp = edge_df.copy()
    tmp["src"] = tmp["userid"].map(id_to_idx)
    tmp["dst"] = tmp["rt_userid"].map(id_to_idx)
    tmp = tmp.dropna(subset=["src", "dst"]).copy()
    tmp[["src", "dst"]] = tmp[["src", "dst"]].astype(int)

    edge_index = torch.tensor(tmp[["src", "dst"]].values.T, dtype=torch.long)
    edge_attr = torch.tensor(tmp[edge_feature_names].fillna(0).values.astype(np.float32), dtype=torch.float)
    return edge_index, edge_attr


def build_node_features(rt: pd.DataFrame, id_to_idx: Dict[int, int], edge_df: pd.DataFrame) -> Tuple[torch.Tensor, List[str], torch.Tensor, List[str]]:
    base = rt.dropna(subset=["userid"]).copy()

    for col in ["followers_count", "statuses_count", "sent_vader", "rt_fav_count", "rt_reply_count"]:
        if col in base.columns:
            base[col] = pd.to_numeric(base[col], errors="coerce")

    base["verified"] = base.get("verified", 0).map({True: 1, False: 0, "True": 1, "False": 0}).fillna(0).astype(float)
    base["n_hashtags"] = base.get("hashtag", "").apply(count_list_like)
    base["n_mentions"] = base.get("mentionsn", "").apply(count_list_like)
    base["has_media"] = base.get("media_urls", "").apply(lambda x: 0 if str(x).strip() in {"", "[]", "nan", "None"} else 1)

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

    feature_names = [
        "subscriber_count", "verified", "avg_favorites", "avg_comments", "avg_score",
        "avg_n_hashtags", "avg_n_mentions", "avg_has_media", "post_count", "in_degree", "out_degree",
    ]

    n_nodes = len(id_to_idx)
    X = np.zeros((n_nodes, len(feature_names)), dtype=np.float32)
    node_agg["node_idx"] = node_agg["userid"].map(id_to_idx)
    node_agg = node_agg.dropna(subset=["node_idx"])
    rows = node_agg["node_idx"].astype(int).values
    X[rows] = node_agg[feature_names].fillna(0).values.astype(np.float32)

    # ensure targets-only nodes get degree features
    all_out = np.zeros(n_nodes, dtype=np.float32)
    all_in = np.zeros(n_nodes, dtype=np.float32)
    for uid, c in edge_df.groupby("userid").size().items():
        all_out[id_to_idx[int(uid)]] = np.log1p(float(c))
    for uid, c in edge_df.groupby("rt_userid").size().items():
        all_in[id_to_idx[int(uid)]] = np.log1p(float(c))
    X[:, feature_names.index("out_degree")] = np.maximum(X[:, feature_names.index("out_degree")], all_out)
    X[:, feature_names.index("in_degree")] = np.maximum(X[:, feature_names.index("in_degree")], all_in)

    nonzero = np.any(X != 0, axis=1)
    if nonzero.any():
        scaler = StandardScaler()
        X[nonzero] = scaler.fit_transform(X[nonzero]).astype(np.float32)

    x = torch.tensor(X, dtype=torch.float)

    y = torch.full((n_nodes,), -1, dtype=torch.long)
    label_names: List[str] = []
    if "state" in base.columns:
        st = base[base["state"].notna() & base["state"].astype(str).ne("")][["userid", "state"]].drop_duplicates("userid")
        if len(st):
            label_names = sorted(st["state"].astype(str).unique().tolist())
            st2i = {s: i for i, s in enumerate(label_names)}
            for _, r in st.iterrows():
                uid = int(r["userid"])
                if uid in id_to_idx:
                    y[id_to_idx[uid]] = st2i[str(r["state"])]

    return x, feature_names, y, label_names


def maybe_attach_embeddings(
    x: torch.Tensor,
    feature_names: List[str],
    node_ids: np.ndarray,
    embeddings_path: str,
    embedding_pool: str,
) -> Tuple[torch.Tensor, List[str], Dict[str, int]]:
    if not embeddings_path:
        return x, feature_names, {"matched_users": 0, "embedding_dim": 0}

    emb = torch.load(embeddings_path, map_location="cpu")
    emb_user_ids = emb.get("user_ids")
    emb_mat = emb.get(embedding_pool)
    if emb_user_ids is None or emb_mat is None:
        raise KeyError(f"Embeddings file must contain 'user_ids' and '{embedding_pool}'")

    emb_user_ids = np.asarray(emb_user_ids)
    emb_dim = int(emb_mat.shape[1])
    emb_map = {int(uid): i for i, uid in enumerate(emb_user_ids)}

    extra = torch.zeros((len(node_ids), emb_dim), dtype=torch.float)
    matched = 0
    for i, uid in enumerate(node_ids):
        j = emb_map.get(int(uid))
        if j is None:
            continue
        extra[i] = emb_mat[j].float()
        matched += 1

    x2 = torch.cat([x, extra], dim=1)
    names2 = feature_names + [f"emb_{k}" for k in range(emb_dim)]
    return x2, names2, {"matched_users": matched, "embedding_dim": emb_dim}


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    raw = load_raw_rows(args.csv_glob, args.max_files)
    rt = normalize_ids_and_timestamps(raw, args.strict_dates)

    node_ids = np.array(sorted(set(rt["userid"].tolist()) | set(rt["rt_userid"].tolist())), dtype=np.int64)
    id_to_idx = {int(uid): i for i, uid in enumerate(node_ids)}
    print(f"Nodes: {len(node_ids):,}")

    edge_feature_names = ["first_retweet_time", "n_retweets", "avg_rt_fav", "avg_rt_reply"]
    edge_all_df = aggregate_edge_features(rt)
    edge_index, edge_attr = to_edge_tensors(edge_all_df, id_to_idx, edge_feature_names)
    print(f"Directed edges: {edge_index.shape[1]:,}")

    x, feature_names, y, label_names = build_node_features(rt, id_to_idx, edge_all_df)
    x, feature_names, emb_stats = maybe_attach_embeddings(
        x, feature_names, node_ids, args.embeddings, args.embedding_pool
    )

    graph_obj = {
        "x": x,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "edge_attr_feature_names": edge_feature_names,
        "user_ids": node_ids,
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

        hist_edge_index, hist_edge_attr = to_edge_tensors(hist_edges_df, id_to_idx, edge_feature_names)

        hist_pairs = set(zip(hist_edges_df["userid"].astype(int), hist_edges_df["rt_userid"].astype(int)))
        fut_pairs = set(zip(fut_edges_df["userid"].astype(int), fut_edges_df["rt_userid"].astype(int)))
        new_pairs = fut_pairs - hist_pairs

        if new_pairs:
            new_df = pd.DataFrame(sorted(list(new_pairs)), columns=["userid", "rt_userid"])
            new_df["src"] = new_df["userid"].map(id_to_idx)
            new_df["dst"] = new_df["rt_userid"].map(id_to_idx)
            new_df = new_df.dropna(subset=["src", "dst"])
            target_new_edge_index = torch.tensor(new_df[["src", "dst"]].astype(int).values.T, dtype=torch.long)
        else:
            target_new_edge_index = torch.zeros((2, 0), dtype=torch.long)

        graph_obj["edge_index_views"] = {
            "retweet_all": edge_index,
            "temporal_history": hist_edge_index,
        }
        graph_obj["edge_attr_views"] = {
            "retweet_all": edge_attr,
            "temporal_history": hist_edge_attr,
        }
        graph_obj["edge_attr_feature_names_views"] = {
            "retweet_all": edge_feature_names,
            "temporal_history": edge_feature_names,
        }
        graph_obj["target_edge_index_views"] = {
            "temporal_new": target_new_edge_index,
        }
        graph_obj["future_edge_index"] = target_new_edge_index

        temporal_stats = {
            "history_fraction": args.history_fraction,
            "history_rows": int(len(hist_rt)),
            "future_rows": int(len(fut_rt)),
            "history_edges": int(hist_edge_index.shape[1]),
            "future_new_edges": int(target_new_edge_index.shape[1]),
        }

    torch.save(graph_obj, args.out)

    meta = {
        "csv_glob": args.csv_glob,
        "max_files": args.max_files,
        "nodes": int(len(node_ids)),
        "edges": int(edge_index.shape[1]),
        "node_feature_dim": int(x.shape[1]),
        "edge_feature_names": edge_feature_names,
        "label_count": int(len(label_names)),
        "labeled_nodes": int((y >= 0).sum().item()),
        "embeddings": args.embeddings,
        "embedding_pool": args.embedding_pool,
        "embedding_dim": emb_stats["embedding_dim"],
        "embedding_matched_users": emb_stats["matched_users"],
        "temporal": temporal_stats,
    }
    meta_path = args.out.replace(".pt", ".meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved graph: {args.out}")
    print(f"Saved meta:  {meta_path}")


if __name__ == "__main__":
    main()
