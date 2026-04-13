import argparse
import ast
import glob
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler


REP_HASHTAGS = [
    "voteredtosaveamerica",
    "votered",
    "redwavecoming",
    "democratsaretheproblem",
    "2a",
    "1a",
    "fjb",
    "americafirst",
    "kag",
]
DEM_HASHTAGS = [
    "voteblue",
    "voteblue2022",
    "votebluetosavedemocracy",
    "votebluetosaveamerica",
    "votebluein2022",
    "votebluenomatterwho",
    "votebluefordemocracy",
    "votebluetoprotectwomen",
    "voteblueforwomensrights",
    "votebluetoprotectyourrights",
    "voteblueforsomanyreasons",
    "votebluetoendtheinsanity",
    "votebluenotq",
    "votebluedownballot",
    "votebluedownballotlocalstatefederal",
    "votebluetosavesocialsecurity",
    "votebluetosavesocialsecurityandmedicare",
    "votebluetosaveourkids",
    "bluewave",
    "bluewave2022",
    "bluecrew",
    "bluevoters",
    "ourbluevoice",
    "bluein22",
    "proudblue22",
    "demvoice1",
    "wtpblue",
    "democratsdeliver",
    "demsact",
    "voteouteveryrepublican",
    "stopvotingforrepublicans",
    "neverrepublicanagain",
    "republicansaretheproblem",
    "republicanwaronwomen",
    "goptraitorstodemocracy",
    "gopliesabouteverything",
    "magaidiots",
    "blm",
    "blacklivesmatter",
    "resist",
    "fbr",
]
DEM_MEDIA_OUTLETS = [
    "abcnews", "bbc", "buzzfeednews", "huffpost", "msnbc", "cnn",
    "nytimes", "washingtonpost", "latimes", "guardian",
]
REP_MEDIA_OUTLETS = [
    "breitbartnews", "dailycaller", "dailymail", "foxnews", "infowars", "oann", "breitbart",
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Build unified retweet graph (nodes, edge features, temporal views, optional embeddings)."
    )
    p.add_argument("--csv_glob", default="/project2/ll_774_951/midterm/*/*.csv")
    p.add_argument("--out", default="data/data/midterm/graphs/retweet_graph.pt")
    p.add_argument("--embeddings", default="", help="Optional embeddings_<model>.pt with user_ids + meanpool/maxpool")
    p.add_argument("--embedding_pool", choices=["meanpool", "maxpool"], default="meanpool")
    p.add_argument("--max_files", type=int, default=0)
    p.add_argument("--max_nodes", type=int, default=0, help="Trim the final graph to exactly this many nodes when possible (0 = no limit)")
    p.add_argument("--strict_dates", action="store_true")
    p.add_argument("--history_fraction", type=float, default=0.8)
    p.add_argument(
        "--label_source",
        choices=["political_leaning", "state"],
        default="political_leaning",
        help=(
            "Node label source. "
            "'political_leaning' uses hashtag-based political leaning labels; "
            "'state' uses the original state column."
        ),
    )
    p.add_argument(
        "--future_target_mode",
        choices=["new_only", "all_future"],
        default="new_only",
        help=(
            "How to define temporal LP targets: "
            "'new_only' keeps only future pairs not seen in history; "
            "'all_future' keeps every pair that appears after the split."
        ),
    )
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
        "hashtag", "rt_hashtag", "mentionsn", "media_urls",
        "rt_fav_count", "rt_reply_count", "description", "urls", "urls_list",
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
    for col in ("rt_fav_count", "rt_reply_count", "followers_count", "statuses_count", "sent_vader"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

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
            # Continue scanning to retain later internal edges among the chosen nodes.
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


def _parse_url_entries(val) -> List[str]:
    if isinstance(val, list):
        raw = val
    elif pd.isna(val):
        return []
    else:
        s = str(val).strip()
        if s in {"", "[]", "nan", "None"}:
            return []
        try:
            raw = ast.literal_eval(s)
        except Exception:
            raw = val

    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        raw = [raw]

    urls: List[str] = []
    for item in raw:
        if isinstance(item, dict):
            url = item.get("expanded_url") or item.get("url")
            if url:
                urls.append(str(url).lower())
        elif isinstance(item, str):
            urls.append(item.lower())
    return urls


def build_political_leaning_labels(base: pd.DataFrame, id_to_idx: Dict[int, int]) -> Tuple[torch.Tensor, List[str]]:
    label_names = ["rep", "dem"]
    y = torch.full((len(id_to_idx),), -1, dtype=torch.long)

    label_df = base[["userid"]].copy()
    label_df["description"] = base.get("description", "").fillna("").astype(str).str.lower()
    if "urls_list" in base.columns:
        url_source = base["urls_list"]
    elif "urls" in base.columns:
        url_source = base["urls"]
    else:
        url_source = pd.Series(index=base.index, dtype=object)
    label_df["urls"] = url_source.apply(_parse_url_entries)

    rep_pat = r"\#(?:" + "|".join(REP_HASHTAGS) + r")"
    dem_pat = r"\#(?:" + "|".join(DEM_HASHTAGS) + r")"
    rep_media_pat = "|".join(REP_MEDIA_OUTLETS)
    dem_media_pat = "|".join(DEM_MEDIA_OUTLETS)

    label_df["rep_domains"] = (
        label_df["urls"].explode().str.findall(rep_media_pat).explode().dropna().groupby(level=0).agg(list)
    )
    label_df["dem_domains"] = (
        label_df["urls"].explode().str.findall(dem_media_pat).explode().dropna().groupby(level=0).agg(list)
    )
    label_df["has_rep_dom"] = label_df["rep_domains"].str.len()
    label_df["has_dem_dom"] = label_df["dem_domains"].str.len()

    label_df["rep_hashs"] = label_df["description"].str.findall(rep_pat)
    label_df["rep_hashs_len"] = label_df["rep_hashs"].str.len()
    user2rep_hash = (
        label_df.sort_values("rep_hashs_len")
        .drop_duplicates(["userid", "description"], keep="first")
        .groupby("userid")
        .agg({"rep_hashs": "sum"})
    )
    user2rep_hash["rep_hashs"] = user2rep_hash["rep_hashs"].apply(set)
    user2rep_hash["rep_hashs_len"] = user2rep_hash["rep_hashs"].str.len()

    label_df["dem_hashs"] = label_df["description"].str.findall(dem_pat)
    label_df["dem_hashs_len"] = label_df["dem_hashs"].str.len()
    user2dem_hash = (
        label_df.sort_values("dem_hashs_len")
        .drop_duplicates(["userid", "description"], keep="first")
        .groupby("userid")
        .agg({"dem_hashs": "sum"})
    )
    user2dem_hash["dem_hashs"] = user2dem_hash["dem_hashs"].apply(set)
    user2dem_hash["dem_hashs_len"] = user2dem_hash["dem_hashs"].str.len()

    user2pol = user2rep_hash.join(user2dem_hash)
    user2pol["user2rep_dom_tweet_count"] = label_df.groupby("userid")["has_rep_dom"].sum()
    user2pol["user2dem_dom_tweet_count"] = label_df.groupby("userid")["has_dem_dom"].sum()
    user2pol["is_rep"] = (
        user2pol["user2rep_dom_tweet_count"].gt(0) | user2pol["rep_hashs_len"].gt(0)
    )
    user2pol["is_dem"] = (
        user2pol["user2dem_dom_tweet_count"].gt(0) | user2pol["dem_hashs_len"].gt(0)
    )
    user2pol = user2pol[~(user2pol["is_rep"] & user2pol["is_dem"])].copy()

    labeled_count = 0
    for uid in user2pol.index[user2pol["is_rep"]]:
        node_idx = id_to_idx.get(int(uid))
        if node_idx is None:
            continue
        y[node_idx] = 0
        labeled_count += 1
    for uid in user2pol.index[user2pol["is_dem"]]:
        node_idx = id_to_idx.get(int(uid))
        if node_idx is None:
            continue
        y[node_idx] = 1
        labeled_count += 1

    print(
        f"Political-leaning labels: {labeled_count:,} / {len(id_to_idx):,} "
        f"({100 * labeled_count / max(1, len(id_to_idx)):.1f}%) labeled"
    )
    return y, label_names


def build_state_labels(base: pd.DataFrame, id_to_idx: Dict[int, int]) -> Tuple[torch.Tensor, List[str]]:
    y = torch.full((len(id_to_idx),), -1, dtype=torch.long)
    label_names: List[str] = []
    if "state" not in base.columns:
        return y, label_names

    st = base[base["state"].notna() & base["state"].astype(str).ne("")][["userid", "state"]].drop_duplicates("userid")
    if len(st):
        label_names = sorted(st["state"].astype(str).unique().tolist())
        st2i = {s: i for i, s in enumerate(label_names)}
        for _, r in st.iterrows():
            uid = int(r["userid"])
            node_idx = id_to_idx.get(uid)
            if node_idx is not None:
                y[node_idx] = st2i[str(r["state"])]
    return y, label_names


def build_node_features(
    rt: pd.DataFrame,
    id_to_idx: Dict[int, int],
    edge_df: pd.DataFrame,
    label_source: str,
) -> Tuple[torch.Tensor, List[str], torch.Tensor, List[str]]:
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

    if label_source == "political_leaning":
        y, label_names = build_political_leaning_labels(base, id_to_idx)
    elif label_source == "state":
        y, label_names = build_state_labels(base, id_to_idx)
    else:
        raise ValueError(f"Unknown label_source='{label_source}'")

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
    rt = trim_rt_to_max_nodes(rt, args.max_nodes)

    node_ids = np.array(sorted(set(rt["userid"].tolist()) | set(rt["rt_userid"].tolist())), dtype=np.int64)
    id_to_idx = {int(uid): i for i, uid in enumerate(node_ids)}
    print(f"Nodes: {len(node_ids):,}")

    edge_feature_names = ["first_retweet_time", "n_retweets", "avg_rt_fav", "avg_rt_reply"]
    edge_all_df = aggregate_edge_features(rt)
    edge_index, edge_attr = to_edge_tensors(edge_all_df, id_to_idx, edge_feature_names)
    print(f"Directed edges: {edge_index.shape[1]:,}")

    x, feature_names, y, label_names = build_node_features(
        rt,
        id_to_idx,
        edge_all_df,
        label_source=args.label_source,
    )
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
        if args.future_target_mode == "new_only":
            target_pairs = fut_pairs - hist_pairs
        else:
            target_pairs = fut_pairs

        if target_pairs:
            target_df = pd.DataFrame(sorted(list(target_pairs)), columns=["userid", "rt_userid"])
            target_df["src"] = target_df["userid"].map(id_to_idx)
            target_df["dst"] = target_df["rt_userid"].map(id_to_idx)
            target_df = target_df.dropna(subset=["src", "dst"])
            target_new_edge_index = torch.tensor(target_df[["src", "dst"]].astype(int).values.T, dtype=torch.long)
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
            "future_target_mode": args.future_target_mode,
            "history_rows": int(len(hist_rt)),
            "future_rows": int(len(fut_rt)),
            "history_edges": int(hist_edge_index.shape[1]),
            "future_edges": int(len(fut_pairs)),
            "future_overlap_edges": int(len(hist_pairs & fut_pairs)),
            "future_target_edges": int(target_new_edge_index.shape[1]),
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
        "label_source": args.label_source,
        "temporal": temporal_stats,
    }
    meta_path = args.out.replace(".pt", ".meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved graph: {args.out}")
    print(f"Saved meta:  {meta_path}")


if __name__ == "__main__":
    main()
