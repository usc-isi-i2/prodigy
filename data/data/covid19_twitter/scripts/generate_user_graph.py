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
DEFAULT_HISTORY_FRACTION = 0.3
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

RETWEET_EDGE_FEATURE_NAMES = [
    "first_retweet_time",
    "n_retweets",
    "avg_rt_fav",
    "avg_rt_reply",
]

INTERACTION_EDGE_FEATURE_NAMES = [
    "first_interaction_time",
    "n_retweets",
    "n_quotes",
    "n_replies",
    "n_mentions",
]

INTERACTION_COUNT_COLUMNS = {
    "retweet": "n_retweets",
    "quote": "n_quotes",
    "reply": "n_replies",
    "mention": "n_mentions",
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Build unified covid19_twitter graph with retweet or interaction edges plus optional embeddings and temporal views."
    )
    p.add_argument("--json_glob", default=DEFAULT_JSON_GLOB)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--embeddings", default="", help="Optional user_embeddings_*.pt with user_ids + meanpool/maxpool (handles fallback supported)")
    p.add_argument("--embedding_pool", choices=["meanpool", "maxpool"], default="meanpool")
    p.add_argument("--graph_mode", choices=["retweet", "interaction"], default="interaction")
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


def _fast_list_lens(series: pd.Series) -> np.ndarray:
    # Much faster than .apply on millions of rows.
    return np.fromiter(
        (len(x) if isinstance(x, list) else 0 for x in series),
        dtype=np.int32,
        count=len(series),
    )


def extract_mentions(tweet: dict) -> Tuple[List[int], List[str]]:
    mention_userids: List[int] = []
    mention_screens: List[str] = []
    seen = set()
    for mention in tweet.get("entities", {}).get("user_mentions", []) or []:
        if not isinstance(mention, dict):
            continue
        mention_id = normalize_user_id(mention.get("id"))
        mention_handle = normalize_handle(mention.get("screen_name"))
        if mention_id is None:
            continue
        key = (mention_id, mention_handle)
        if key in seen:
            continue
        seen.add(key)
        mention_userids.append(mention_id)
        mention_screens.append(mention_handle)
    return mention_userids, mention_screens


def load_raw_rows(
        json_glob: str,
        max_files: int,
        strict_dates: bool = False,
        max_nodes: int = 0,
        graph_mode: str = "interaction",
) -> pd.DataFrame:
    files = sorted(glob.glob(json_glob))
    if max_files > 0:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"No files matched: {json_glob}")

    rows = []
    seen_nodes = set()
    stopped_early = False
    print(f"Found {len(files)} files")
    for i, path in enumerate(files, start=1):
        print(f"[{i}/{len(files)}] Loading {os.path.basename(path)}", flush=True)
        try:
            items = load_json_items(path)
        except Exception as exc:
            print(f"  [WARN] failed {os.path.basename(path)}: {exc}", flush=True)
            continue

        file_rows = []
        for tweet in items:
            user = tweet.get("user") or {}
            rt = tweet.get("retweeted_status") or {}
            rt_user = rt.get("user") or {}
            quoted = tweet.get("quoted_status") or {}
            quoted_user = quoted.get("user") or {}
            mention_userids, mention_screens = extract_mentions(tweet)

            uid = normalize_user_id(user.get("id"))
            rt_uid = normalize_user_id(rt_user.get("id")) if rt else None
            reply_uid = normalize_user_id(tweet.get("in_reply_to_user_id"))
            quote_uid = normalize_user_id(quoted_user.get("id")) if quoted else None

            file_rows.append(
                {
                    "screen_name": normalize_handle(user.get("screen_name")),
                    "userid": uid,
                    "rt_screen": normalize_handle(rt_user.get("screen_name")) if rt else None,
                    "rt_userid": rt_uid,
                    "reply_screen": normalize_handle(tweet.get("in_reply_to_screen_name")),
                    "reply_userid": reply_uid,
                    "qtd_screen": normalize_handle(quoted_user.get("screen_name")) if quoted else None,
                    "qtd_userid": quote_uid,
                    "date": tweet.get("created_at"),
                    "followers_count": user.get("followers_count"),
                    "verified": user.get("verified"),
                    "statuses_count": user.get("statuses_count"),
                    "rt_fav_count": rt.get("favorite_count") if rt else None,
                    "rt_reply_count": rt.get("reply_count") if rt else None,
                    "sent_vader": None,
                    "hashtag": tweet.get("entities", {}).get("hashtags", []),
                    "mentionsn": tweet.get("entities", {}).get("user_mentions", []),
                    "mention_userids": mention_userids,
                    "mention_screens": mention_screens,
                    "media_urls": (tweet.get("extended_entities") or {}).get("media", []),
                }
            )

        if file_rows:
            rows.extend(file_rows)
            if max_nodes > 0:
                file_df = pd.DataFrame(file_rows)
                try:
                    if graph_mode == "retweet":
                        file_events = prepare_retweet_rows(file_df, strict_dates)
                    else:
                        file_events = prepare_interaction_rows(file_df, strict_dates)
                except RuntimeError:
                    file_events = pd.DataFrame(columns=["userid", "target_userid"])
                if not file_events.empty:
                    seen_nodes.update(file_events["userid"].tolist())
                    seen_nodes.update(file_events["target_userid"].tolist())
                    print(
                        f"  cleaned participants seen so far: {len(seen_nodes):,}",
                        flush=True,
                    )
                    if len(seen_nodes) >= max_nodes:
                        print(
                            f"  reached max_nodes target during ingestion after file {i}/{len(files)}; "
                            "stopping raw file loading before full corpus read",
                            flush=True,
                        )
                        stopped_early = True
                        break

        if i % 10 == 0 or i == len(files):
            print(f"  processed {i}/{len(files)} files", flush=True)

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No valid rows parsed")
    print(f"Loaded rows: {len(df):,}")
    if max_nodes > 0:
        print(
            f"Ingestion summary: cleaned participants seen={len(seen_nodes):,} "
            f"(target={max_nodes:,}, early_stop={stopped_early})",
            flush=True,
        )
    return df


def _normalize_base_rows(df: pd.DataFrame, strict_dates: bool) -> pd.DataFrame:
    df = df.copy()
    for col in [
        "userid",
        "rt_userid",
        "reply_userid",
        "qtd_userid",
        "followers_count",
        "statuses_count",
        "rt_fav_count",
        "rt_reply_count",
    ]:
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

    return df


def prepare_retweet_rows(df: pd.DataFrame, strict_dates: bool) -> pd.DataFrame:
    df = _normalize_base_rows(df, strict_dates)
    common_cols = [
        "userid",
        "screen_name",
        "timestamp",
        "followers_count",
        "verified",
        "statuses_count",
        "rt_fav_count",
        "rt_reply_count",
        "sent_vader",
        "hashtag",
        "mentionsn",
        "media_urls",
    ]
    retweet = df.dropna(subset=["userid", "rt_userid"]).copy()
    retweet = retweet[retweet["userid"] != retweet["rt_userid"]].copy()
    if retweet.empty:
        raise RuntimeError("No valid retweet rows after cleaning")
    retweet["userid"] = retweet["userid"].astype(np.int64)
    retweet["target_userid"] = retweet["rt_userid"].astype(np.int64)
    retweet["target_screen"] = retweet["rt_screen"]
    retweet["interaction_type"] = "retweet"
    return retweet[common_cols + ["target_userid", "target_screen", "interaction_type"]]


def prepare_interaction_rows(df: pd.DataFrame, strict_dates: bool) -> pd.DataFrame:
    df = _normalize_base_rows(df, strict_dates)
    common_cols = [
        "userid",
        "screen_name",
        "timestamp",
        "followers_count",
        "verified",
        "statuses_count",
        "rt_fav_count",
        "rt_reply_count",
        "sent_vader",
        "hashtag",
        "mentionsn",
        "media_urls",
    ]
    frames = []

    try:
        retweet = prepare_retweet_rows(df, strict_dates=False)
        frames.append(retweet)
    except RuntimeError:
        pass

    reply = df.dropna(subset=["userid", "reply_userid"]).copy()
    if not reply.empty:
        reply = reply[reply["userid"] != reply["reply_userid"]].copy()
        reply["userid"] = reply["userid"].astype(np.int64)
        reply["target_userid"] = reply["reply_userid"].astype(np.int64)
        reply["target_screen"] = reply["reply_screen"]
        reply["interaction_type"] = "reply"
        frames.append(reply[common_cols + ["target_userid", "target_screen", "interaction_type"]])

    quote = df.dropna(subset=["userid", "qtd_userid"]).copy()
    if not quote.empty:
        quote = quote[quote["userid"] != quote["qtd_userid"]].copy()
        quote["userid"] = quote["userid"].astype(np.int64)
        quote["target_userid"] = quote["qtd_userid"].astype(np.int64)
        quote["target_screen"] = quote["qtd_screen"]
        quote["interaction_type"] = "quote"
        frames.append(quote[common_cols + ["target_userid", "target_screen", "interaction_type"]])

    mention = df[common_cols + ["mention_userids", "mention_screens"]].copy()
    if not mention.empty:
        mention["mention_userids"] = mention["mention_userids"].apply(lambda xs: xs if isinstance(xs, list) else [])
        mention["mention_screens"] = mention["mention_screens"].apply(lambda xs: xs if isinstance(xs, list) else [])
        mention = mention.explode(["mention_userids", "mention_screens"], ignore_index=True)
        mention = mention.dropna(subset=["userid", "mention_userids"]).copy()
        if not mention.empty:
            mention["userid"] = mention["userid"].astype(np.int64)
            mention["target_userid"] = pd.to_numeric(mention["mention_userids"], errors="coerce")
            mention = mention.dropna(subset=["target_userid"]).copy()
            mention["target_userid"] = mention["target_userid"].astype(np.int64)
            mention = mention[mention["userid"] != mention["target_userid"]].copy()
            mention["target_screen"] = mention["mention_screens"]
            mention["interaction_type"] = "mention"
            frames.append(mention[common_cols + ["target_userid", "target_screen", "interaction_type"]])

    if not frames:
        raise RuntimeError("No valid interaction rows after cleaning")

    interactions = pd.concat(frames, ignore_index=True, copy=False)
    if interactions.empty:
        raise RuntimeError("No valid interaction rows after cleaning")
    return interactions


def trim_events_to_max_nodes(events: pd.DataFrame, max_nodes: int, graph_mode: str) -> pd.DataFrame:
    if max_nodes <= 0:
        print(f"No max_nodes trim requested; keeping all cleaned {graph_mode} participants.", flush=True)
        return events

    total_unique_nodes = len(set(events["userid"].tolist()) | set(events["target_userid"].tolist()))
    if total_unique_nodes < max_nodes:
        print(
            f"Warning: requested max_nodes={max_nodes:,}, but only {total_unique_nodes:,} unique {graph_mode} nodes exist "
            "after cleaning. Proceeding with all available cleaned nodes.",
            flush=True,
        )
        return events

    src = events["userid"].to_numpy(dtype=np.int64, copy=False)
    dst = events["target_userid"].to_numpy(dtype=np.int64, copy=False)
    keep_rows = np.zeros(len(events), dtype=bool)
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

    trimmed = events.loc[keep_rows].copy()
    actual_nodes = sorted(set(trimmed["userid"].tolist()) | set(trimmed["target_userid"].tolist()))
    print(
        f"Applied exact max_nodes={max_nodes:,}: kept rows={len(trimmed):,} "
        f"nodes={len(actual_nodes):,}",
        flush=True,
    )
    if len(actual_nodes) != max_nodes:
        print(
            f"Warning: exact max_nodes trim requested {max_nodes:,}, got {len(actual_nodes):,} nodes. "
            "Proceeding with the largest feasible trimmed graph.",
            flush=True,
        )
    return trimmed


def build_user_index(interactions: pd.DataFrame) -> Tuple[List[int], Dict[int, int]]:
    user_ids = sorted(set(interactions["userid"].tolist()) | set(interactions["target_userid"].tolist()))
    return user_ids, {int(user_id): i for i, user_id in enumerate(user_ids)}


def build_user_metadata(interactions: pd.DataFrame, user_ids: List[int]) -> List[str]:
    left = interactions[["userid", "screen_name", "timestamp"]].rename(
        columns={"userid": "user_id", "screen_name": "handle"}
    )
    right = interactions[["target_userid", "target_screen", "timestamp"]].rename(
        columns={"target_userid": "user_id", "target_screen": "handle"}
    )
    meta = pd.concat([left, right], ignore_index=True, copy=False)
    meta = meta.dropna(subset=["user_id", "handle"])
    if meta.empty:
        return [None] * len(user_ids)
    meta["user_id"] = meta["user_id"].astype(np.int64)
    latest_idx = meta.groupby("user_id")["timestamp"].idxmax()
    latest = meta.loc[latest_idx, ["user_id", "handle"]]
    handle_map = dict(zip(latest["user_id"].to_numpy(), latest["handle"].to_numpy()))
    return [handle_map.get(int(user_id)) for user_id in user_ids]


def aggregate_retweet_edge_features(events: pd.DataFrame) -> pd.DataFrame:
    edge_grp = events.groupby(["userid", "target_userid"], as_index=False).agg(
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


def aggregate_interaction_edge_features(interactions: pd.DataFrame) -> pd.DataFrame:
    edge_grp = interactions.groupby(["userid", "target_userid"], as_index=False).agg(
        first_ts=("timestamp", "min"),
    )
    type_counts = (
        interactions.groupby(["userid", "target_userid", "interaction_type"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    edge_grp = edge_grp.merge(type_counts, on=["userid", "target_userid"], how="left")
    min_ns = int(edge_grp["first_ts"].min().value)
    edge_grp["first_interaction_time"] = (edge_grp["first_ts"].astype("int64") - min_ns) / 3_600_000_000_000.0
    for interaction_type, feature_name in INTERACTION_COUNT_COLUMNS.items():
        counts = (
            edge_grp[interaction_type]
            if interaction_type in edge_grp.columns
            else pd.Series(0.0, index=edge_grp.index)
        )
        edge_grp[feature_name] = np.log1p(pd.to_numeric(counts, errors="coerce").fillna(0).astype(float))
    return edge_grp


def to_edge_tensors(
    edge_df: pd.DataFrame,
    u2i: Dict[int, int],
    edge_feature_names: List[str],
) -> Tuple[torch.Tensor, torch.Tensor]:
    tmp = edge_df.copy()
    tmp["src"] = tmp["userid"].map(u2i)
    tmp["dst"] = tmp["target_userid"].map(u2i)
    tmp = tmp.dropna(subset=["src", "dst"]).copy()
    tmp[["src", "dst"]] = tmp[["src", "dst"]].astype(int)
    edge_index = torch.tensor(tmp[["src", "dst"]].values.T, dtype=torch.long)
    edge_attr = torch.tensor(
        tmp[edge_feature_names].fillna(0).values.astype(np.float32),
        dtype=torch.float,
    )
    return edge_index, edge_attr


def build_node_features(raw: pd.DataFrame, u2i: Dict[int, int], edge_df: pd.DataFrame):
    base = raw.copy()
    # Fast list-length extraction (avoids .apply on millions of rows).
    base["n_hashtags"] = _fast_list_lens(base["hashtag"]) if "hashtag" in base.columns else 0
    base["n_mentions"] = _fast_list_lens(base["mentionsn"]) if "mentionsn" in base.columns else 0
    if "media_urls" in base.columns:
        base["has_media"] = (_fast_list_lens(base["media_urls"]) > 0).astype(np.int8)
    else:
        base["has_media"] = 0
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

    # Degrees from the aggregated edge frame. Merge once; no Python loops.
    in_deg = edge_df.groupby("target_userid").size().rename("in_degree")
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

    emb_user_ids = emb.get("user_ids")
    if emb_user_ids is not None:
        # Vectorized join via searchsorted on a sorted copy of the embedding ids.
        emb_ids = np.asarray(emb_user_ids).astype(np.int64)
        order = np.argsort(emb_ids)
        sorted_ids = emb_ids[order]
        query = np.asarray(user_ids, dtype=np.int64)
        pos = np.searchsorted(sorted_ids, query)
        pos_clipped = np.clip(pos, 0, len(sorted_ids) - 1)
        hit = (pos < len(sorted_ids)) & (sorted_ids[pos_clipped] == query)
        tgt_rows = np.where(hit)[0]
        src_rows = order[pos[hit]]
        if len(tgt_rows):
            extra[torch.from_numpy(tgt_rows)] = emb_mat[torch.from_numpy(src_rows)].float()
        matched = int(hit.sum())
    else:
        emb_handles = emb.get("handles")
        if emb_handles is None:
            raise KeyError(f"Embeddings file must contain either 'user_ids' or 'handles' plus '{embedding_pool}'")
        # Vectorized handle lookup via pandas.
        emb_handle_series = pd.Series(
            np.arange(len(emb_handles), dtype=np.int64),
            index=[str(h).strip().lower() for h in emb_handles],
        )
        # Dedup in case of repeated handles in the embedding file.
        emb_handle_series = emb_handle_series[~emb_handle_series.index.duplicated(keep="first")]
        mapped = pd.Series(handles).map(emb_handle_series)
        hit_mask = mapped.notna().to_numpy()
        tgt_rows = np.where(hit_mask)[0]
        src_rows = mapped.dropna().to_numpy(dtype=np.int64)
        if len(tgt_rows):
            extra[torch.from_numpy(tgt_rows)] = emb_mat[torch.from_numpy(src_rows)].float()
        matched = int(hit_mask.sum())

    return (
        torch.cat([x, extra], dim=1),
        feature_names + [f"emb_{k}" for k in range(emb_dim)],
        {"matched_users": matched, "embedding_dim": emb_dim},
    )


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
    label_names: List[str] = []
    if not label_info:
        y = torch.full((len(handles),), -1, dtype=torch.long)
        return y, label_names, {"labeled_nodes": 0, "label_count": 0}

    handle_to_label = label_info["handle_to_label"]
    label_names = list(label_info["label_names"])
    # Vectorized lookup via pandas.
    mapped = pd.Series(handles).map(handle_to_label)
    y_np = mapped.fillna(-1).to_numpy(dtype=np.int64)
    y = torch.from_numpy(y_np)
    labeled = int((y_np >= 0).sum())
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
    print("Configuration")
    print(f"  json_glob: {args.json_glob}")
    print(f"  out: {args.out}")
    print(f"  embeddings: {args.embeddings or '<none>'}")
    print(f"  embedding_pool: {args.embedding_pool}")
    print(f"  max_files: {args.max_files if args.max_files > 0 else 'all'}")
    print(f"  max_nodes: {args.max_nodes if args.max_nodes > 0 else 'all'}")
    print(f"  history_fraction: {args.history_fraction}")
    print(f"  future_target_mode: {args.future_target_mode}")
    print(f"  labels_parquet_glob: {args.labels_parquet_glob or '<none>'}")
    print(f"  keep_isolates: {args.keep_isolates}")
    print()

    print(f"  graph_mode: {args.graph_mode}")
    raw = load_raw_rows(
        args.json_glob,
        args.max_files,
        strict_dates=args.strict_dates,
        max_nodes=args.max_nodes,
        graph_mode=args.graph_mode,
    )
    print(f"Raw frame: rows={len(raw):,} cols={len(raw.columns):,}", flush=True)
    if args.graph_mode == "retweet":
        events = prepare_retweet_rows(raw, args.strict_dates)
        edge_feature_names = RETWEET_EDGE_FEATURE_NAMES
        default_view_name = "retweet_all"
        edge_all_df = aggregate_retweet_edge_features
    else:
        events = prepare_interaction_rows(raw, args.strict_dates)
        edge_feature_names = INTERACTION_EDGE_FEATURE_NAMES
        default_view_name = "interaction_all"
        edge_all_df = aggregate_interaction_edge_features
    pretrim_nodes = len(set(events["userid"].tolist()) | set(events["target_userid"].tolist()))
    print(f"Cleaned {args.graph_mode} events: rows={len(events):,} unique_nodes={pretrim_nodes:,}", flush=True)
    events = trim_events_to_max_nodes(events, args.max_nodes, args.graph_mode)
    posttrim_nodes = len(set(events["userid"].tolist()) | set(events["target_userid"].tolist()))
    print(f"Post-trim {args.graph_mode} events: rows={len(events):,} unique_nodes={posttrim_nodes:,}", flush=True)
    user_ids, u2i = build_user_index(events)
    handles = build_user_metadata(events, user_ids)
    print(f"Nodes: {len(user_ids):,}")

    edges_df = edge_all_df(events)
    edge_index, edge_attr = to_edge_tensors(edges_df, u2i, edge_feature_names)
    print(f"Directed edges: {edge_index.shape[1]:,}")

    x, feature_names = build_node_features(raw, u2i, edges_df)
    x, feature_names, emb_stats = maybe_attach_embeddings(
        x, feature_names, user_ids, handles, args.embeddings, args.embedding_pool
    )
    print(
        f"After attaching embeddings: {x.shape[1]} dims total, "
        f"matched_users={emb_stats['matched_users']:,}, embedding_dim={emb_stats['embedding_dim']}",
        flush=True,
    )

    # Cache bincounts once; reuse for the "before drop" stat and the drop itself.
    n_nodes = len(user_ids)
    out_deg_t = torch.bincount(edge_index[0], minlength=n_nodes)
    in_deg_t = torch.bincount(edge_index[1], minlength=n_nodes)
    isolated_before_drop = int(((out_deg_t == 0) & (in_deg_t == 0)).sum().item())

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
        "edge_attr_feature_names": edge_feature_names,
        "user_ids": user_ids,
        "u2i": u2i,
        "feature_names": feature_names,
        "y": y,
        "label_names": label_names,
    }

    temporal_stats = {}
    if not args.no_temporal_views:
        events_sorted = events.sort_values("timestamp").reset_index(drop=True)
        cutoff_idx = int(len(events_sorted) * args.history_fraction)
        cutoff_idx = max(1, min(len(events_sorted) - 1, cutoff_idx))
        hist_events = events_sorted.iloc[:cutoff_idx].copy()
        fut_events = events_sorted.iloc[cutoff_idx:].copy()
        hist_edges_df = edge_all_df(hist_events)
        fut_edges_df = edge_all_df(fut_events)
        hist_edge_index, hist_edge_attr = to_edge_tensors(
            hist_edges_df,
            u2i,
            edge_feature_names,
        )
        hist_pairs = set(zip(hist_edges_df["userid"], hist_edges_df["target_userid"]))
        fut_pairs = set(zip(fut_edges_df["userid"], fut_edges_df["target_userid"]))
        target_pairs = fut_pairs - hist_pairs if args.future_target_mode == "new_only" else fut_pairs
        if target_pairs:
            target_df = pd.DataFrame(sorted(list(target_pairs)), columns=["userid", "target_userid"])
            target_df["src"] = target_df["userid"].map(u2i)
            target_df["dst"] = target_df["target_userid"].map(u2i)
            target_df = target_df.dropna(subset=["src", "dst"])
            target_new_edge_index = torch.tensor(target_df[["src", "dst"]].astype(int).values.T, dtype=torch.long)
        else:
            target_new_edge_index = torch.zeros((2, 0), dtype=torch.long)

        graph_obj["edge_index_views"] = {default_view_name: edge_index, "temporal_history": hist_edge_index}
        graph_obj["edge_attr_views"] = {default_view_name: edge_attr, "temporal_history": hist_edge_attr}
        graph_obj["edge_attr_feature_names_views"] = {default_view_name: edge_feature_names, "temporal_history": edge_feature_names}
        if args.graph_mode == "interaction":
            graph_obj["edge_index_views"]["retweet_all"] = edge_index
            graph_obj["edge_attr_views"]["retweet_all"] = edge_attr
            graph_obj["edge_attr_feature_names_views"]["retweet_all"] = edge_feature_names
        graph_obj["target_edge_index_views"] = {"temporal_new": target_new_edge_index}
        graph_obj["future_edge_index"] = target_new_edge_index
        temporal_stats = {
            "history_fraction": args.history_fraction,
            "future_target_mode": args.future_target_mode,
            "history_rows": int(len(hist_events)),
            "future_rows": int(len(fut_events)),
            "history_edges": int(hist_edge_index.shape[1]),
            "future_edges": int(len(fut_pairs)),
            "future_overlap_edges": int(len(hist_pairs & fut_pairs)),
            "future_target_edges": int(target_new_edge_index.shape[1]),
        }
        print(
            "Temporal views: "
            f"history_rows={temporal_stats['history_rows']:,} "
            f"future_rows={temporal_stats['future_rows']:,} "
            f"history_edges={temporal_stats['history_edges']:,} "
            f"future_target_edges={temporal_stats['future_target_edges']:,}",
            flush=True,
        )

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data.feature_names = feature_names
    data.edge_attr_feature_names = edge_feature_names
    data.label_names = label_names
    data.user_ids = list(user_ids)
    graph_obj["data"] = data
    torch.save(graph_obj, args.out)

    meta = {
        "json_glob": args.json_glob,
        "max_files": args.max_files,
        "nodes": int(len(user_ids)),
        "edges": int(edge_index.shape[1]),
        "node_feature_dim": int(x.shape[1]),
        "edge_feature_names": edge_feature_names,
        "graph_mode": args.graph_mode,
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
