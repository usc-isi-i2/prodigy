"""Shared graph-building primitives used by all dataset preprocessing scripts.

All functions operate on a pandas DataFrame of retweet rows with at minimum
the columns ``userid``, ``rt_userid``, and ``timestamp``.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data


NODE_FEATURE_NAMES: List[str] = [
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

EDGE_FEATURE_NAMES: List[str] = [
    "first_retweet_time",
    "n_retweets",
    "avg_rt_fav",
    "avg_rt_reply",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _count_list_like(val) -> int:
    """Count elements in a value that may be a Python list or a serialised list string."""
    if val is None:
        return 0
    if isinstance(val, float) and np.isnan(val):
        return 0
    if isinstance(val, list):
        return len(val)
    s = str(val).strip()
    if s in {"", "[]", "nan", "None"}:
        return 0
    return len([x for x in s.strip("[]").split(",") if x.strip()])


def _fast_list_lens(series: pd.Series) -> np.ndarray:
    return np.fromiter(
        (_count_list_like(x) for x in series),
        dtype=np.int32,
        count=len(series),
    )


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_retweet_rows(
    df: pd.DataFrame,
    strict_dates: bool = False,
    timestamp_format: str = "%a %b %d %H:%M:%S +0000 %Y",
    src_col: str = "userid",
    dst_col: str = "rt_userid",
) -> pd.DataFrame:
    """Normalise types, parse timestamps, and retain only valid retweet rows.

    Drops rows where either endpoint is missing or where a user retweets
    themselves. Raises ``RuntimeError`` if no rows remain.

    Parameters
    ----------
    src_col, dst_col:
        Column names for the source and destination user ids in ``df``.
        They are renamed to the canonical ``userid`` / ``rt_userid`` so all
        downstream functions work without change. Defaults match Twitter CSVs.
    """
    df = df.copy()

    # Rename dataset-specific endpoint columns to the canonical names expected
    # by all downstream graph-building functions.
    if src_col != "userid" or dst_col != "rt_userid":
        df = df.rename(columns={src_col: "userid", dst_col: "rt_userid"})

    for col in ("userid", "rt_userid", "followers_count", "statuses_count",
                "rt_fav_count", "rt_reply_count", "sent_vader"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "verified" in df.columns:
        df["verified"] = (
            df["verified"]
            .map({True: 1, False: 0, "True": 1, "False": 0, 1: 1, 0: 0})
            .fillna(0)
        )

    df["date"] = df.get("date", "").astype(str).str.strip()
    df["timestamp"] = pd.to_datetime(
        df["date"], format=timestamp_format, utc=True, errors="coerce"
    )

    bad_ts = df["timestamp"].isna()
    if bad_ts.any():
        n_bad = int(bad_ts.sum())
        if strict_dates:
            examples = df.loc[bad_ts, "date"].drop_duplicates().head(5).tolist()
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
    """Greedily trim retweet rows so the node count does not exceed ``max_nodes``.

    After the cap is reached the scan continues to collect any later edges
    whose both endpoints are already in the kept set.
    """
    if max_nodes <= 0:
        return rt

    total_unique = len(set(rt["userid"].tolist()) | set(rt["rt_userid"].tolist()))
    if total_unique < max_nodes:
        raise ValueError(
            f"Requested max_nodes={max_nodes:,}, but only {total_unique:,} unique nodes exist."
        )

    src = rt["userid"].to_numpy(dtype=np.int64, copy=False)
    dst = rt["rt_userid"].to_numpy(dtype=np.int64, copy=False)
    keep_rows = np.zeros(len(rt), dtype=bool)
    keep_nodes: set = set()

    for i, (u, v) in enumerate(zip(src, dst)):
        add = (u not in keep_nodes) + (v not in keep_nodes)
        if len(keep_nodes) + add <= max_nodes:
            keep_rows[i] = True
            keep_nodes.add(int(u))
            keep_nodes.add(int(v))
        elif len(keep_nodes) >= max_nodes and u in keep_nodes and v in keep_nodes:
            keep_rows[i] = True

    rt2 = rt.loc[keep_rows].copy()
    actual = sorted(set(rt2["userid"].tolist()) | set(rt2["rt_userid"].tolist()))
    print(f"Applied exact max_nodes={max_nodes:,}: kept rows={len(rt2):,} nodes={len(actual):,}", flush=True)
    if len(actual) != max_nodes:
        raise RuntimeError(
            f"max_nodes trim failed: requested {max_nodes:,}, got {len(actual):,} nodes"
        )
    return rt2


# ---------------------------------------------------------------------------
# Node / edge index construction
# ---------------------------------------------------------------------------

def build_user_index(rt: pd.DataFrame) -> Tuple[np.ndarray, Dict[int, int]]:
    """Return a sorted node array and a userid→index mapping."""
    user_ids = np.array(
        sorted(set(rt["userid"].tolist()) | set(rt["rt_userid"].tolist())),
        dtype=np.int64,
    )
    u2i = {int(uid): i for i, uid in enumerate(user_ids)}
    return user_ids, u2i


def build_user_metadata(
    rt: pd.DataFrame,
    user_ids: np.ndarray,
) -> List[Optional[str]]:
    """Return a list of screen-name handles aligned with ``user_ids`` (latest per user)."""
    frames = []
    if "screen_name" in rt.columns:
        frames.append(
            rt[["userid", "screen_name", "timestamp"]].rename(
                columns={"userid": "user_id", "screen_name": "handle"}
            )
        )
    if "rt_screen" in rt.columns:
        frames.append(
            rt[["rt_userid", "rt_screen", "timestamp"]].rename(
                columns={"rt_userid": "user_id", "rt_screen": "handle"}
            )
        )
    if not frames:
        return [None] * len(user_ids)

    meta = pd.concat(frames, ignore_index=True, copy=False)
    meta = meta.dropna(subset=["user_id", "handle"])
    if meta.empty:
        return [None] * len(user_ids)

    meta["user_id"] = meta["user_id"].astype(np.int64)
    latest_idx = meta.groupby("user_id")["timestamp"].idxmax()
    latest = meta.loc[latest_idx, ["user_id", "handle"]]
    handle_map = dict(zip(latest["user_id"].to_numpy(), latest["handle"].to_numpy()))
    return [handle_map.get(int(uid)) for uid in user_ids]


# ---------------------------------------------------------------------------
# Edge features
# ---------------------------------------------------------------------------

def aggregate_edge_features(rt: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-(src, dst) edge features from raw retweet rows."""
    edge_grp = rt.groupby(["userid", "rt_userid"], as_index=False).agg(
        first_ts=("timestamp", "min"),
        n_retweets=("timestamp", "size"),
        avg_rt_fav=("rt_fav_count", "mean"),
        avg_rt_reply=("rt_reply_count", "mean"),
    )
    min_ns = int(edge_grp["first_ts"].min().value)
    edge_grp["first_retweet_time"] = (
        (edge_grp["first_ts"].astype("int64") - min_ns) / 3_600_000_000_000.0
    )
    edge_grp["n_retweets"] = np.log1p(edge_grp["n_retweets"].astype(float))
    edge_grp["avg_rt_fav"] = np.log1p(
        pd.to_numeric(edge_grp["avg_rt_fav"], errors="coerce").fillna(0).clip(lower=0)
    )
    edge_grp["avg_rt_reply"] = np.log1p(
        pd.to_numeric(edge_grp["avg_rt_reply"], errors="coerce").fillna(0).clip(lower=0)
    )
    return edge_grp


def to_edge_tensors(
    edge_df: pd.DataFrame,
    u2i: Dict[int, int],
    edge_feature_names: List[str] = EDGE_FEATURE_NAMES,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert an aggregated edge DataFrame to ``edge_index`` and ``edge_attr`` tensors."""
    tmp = edge_df.copy()
    tmp["src"] = tmp["userid"].map(u2i)
    tmp["dst"] = tmp["rt_userid"].map(u2i)
    tmp = tmp.dropna(subset=["src", "dst"]).copy()
    tmp[["src", "dst"]] = tmp[["src", "dst"]].astype(int)
    edge_index = torch.tensor(tmp[["src", "dst"]].values.T, dtype=torch.long)
    edge_attr = torch.tensor(
        tmp[edge_feature_names].fillna(0).values.astype(np.float32), dtype=torch.float
    )
    return edge_index, edge_attr


# ---------------------------------------------------------------------------
# Node features
# ---------------------------------------------------------------------------

def build_node_features(
    rt: pd.DataFrame,
    u2i: Dict[int, int],
    edge_df: pd.DataFrame,
    feature_spec: Optional[List[str]] = None,
) -> Tuple[torch.Tensor, List[str]]:
    """Build per-node account/graph features, StandardScaled across non-zero rows.

    Returns ``(x, feature_names)`` where ``x`` has shape ``[N, F]``.

    Parameters
    ----------
    feature_spec:
        Ordered list of feature names to include in the output matrix.
        Each name must either be a column that already exists in ``rt`` after
        aggregation, or one of the built-in computed features:

        Built-in aggregations (always computed when source columns are present):
          ``subscriber_count`` ← ``followers_count`` max (log1p)
          ``verified``         ← ``verified`` max
          ``avg_favorites``    ← ``rt_fav_count`` mean (log1p)
          ``avg_comments``     ← ``rt_reply_count`` mean (log1p)
          ``avg_score``        ← ``sent_vader`` mean
          ``avg_n_hashtags``   ← ``hashtag`` list-length mean
          ``avg_n_mentions``   ← ``mentionsn`` list-length mean
          ``avg_has_media``    ← ``media_urls`` presence mean
          ``post_count``       ← ``statuses_count`` max (log1p)
          ``in_degree``        ← from edge_df (log1p)
          ``out_degree``       ← from edge_df (log1p)

        Any name not in the built-in set is looked up directly as a column in
        ``rt`` (aggregated with mean) so new datasets can extend the feature set
        without modifying this function.

        Defaults to ``NODE_FEATURE_NAMES`` (the standard 11-feature Twitter set).
    """
    if feature_spec is None:
        feature_spec = NODE_FEATURE_NAMES

    base = rt.copy()

    # --- computed helper columns (always safe to build; ignored if not selected) ---
    if "verified" not in base.columns:
        base["verified"] = 0.0
    base["verified"] = (
        base["verified"]
        .map({True: 1, False: 0, "True": 1, "False": 0, 1: 1, 0: 0})
        .fillna(0)
        .astype(float)
    )
    if "sent_vader" not in base.columns:
        base["sent_vader"] = 0.0

    base["n_hashtags"] = _fast_list_lens(base["hashtag"]) if "hashtag" in base.columns else 0
    base["n_mentions"] = _fast_list_lens(base["mentionsn"]) if "mentionsn" in base.columns else 0
    base["has_media"] = (
        (_fast_list_lens(base["media_urls"]) > 0).astype(np.int8)
        if "media_urls" in base.columns
        else 0
    )

    # --- standard aggregations ---
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

    # --- extra columns requested by caller but not in the standard set ---
    _standard = set(NODE_FEATURE_NAMES) | {"userid"}
    extra_cols = [f for f in feature_spec if f not in _standard and f not in ("in_degree", "out_degree")]
    for col in extra_cols:
        if col in base.columns:
            extra_agg = base.groupby("userid")[col].mean().rename(col)
            node_agg = node_agg.merge(extra_agg, left_on="userid", right_index=True, how="left")
        else:
            node_agg[col] = 0.0

    for col in ("subscriber_count", "avg_favorites", "avg_comments", "post_count"):
        if col in node_agg.columns:
            node_agg[col] = np.log1p(
                pd.to_numeric(node_agg[col], errors="coerce").fillna(0).clip(lower=0)
            )

    # --- degree features ---
    in_deg = edge_df.groupby("rt_userid").size().rename("in_degree")
    out_deg = edge_df.groupby("userid").size().rename("out_degree")
    node_agg = node_agg.merge(out_deg, left_on="userid", right_index=True, how="left")
    node_agg = node_agg.merge(in_deg, left_on="userid", right_index=True, how="left")
    node_agg["in_degree"] = np.log1p(node_agg["in_degree"].fillna(0))
    node_agg["out_degree"] = np.log1p(node_agg["out_degree"].fillna(0))

    # --- assemble final matrix using only the requested features ---
    n_nodes = len(u2i)
    X = np.zeros((n_nodes, len(feature_spec)), dtype=np.float32)
    node_agg["node_idx"] = node_agg["userid"].map(u2i)
    node_agg = node_agg.dropna(subset=["node_idx"])
    rows = node_agg["node_idx"].astype(int).values
    available = [f for f in feature_spec if f in node_agg.columns]
    if available:
        X[np.ix_(rows, [feature_spec.index(f) for f in available])] = (
            node_agg[available].fillna(0).values.astype(np.float32)
        )

    nonzero = np.any(X != 0, axis=1)
    if nonzero.any():
        scaler = StandardScaler()
        X[nonzero] = scaler.fit_transform(X[nonzero]).astype(np.float32)

    return torch.tensor(X, dtype=torch.float), list(feature_spec)


# ---------------------------------------------------------------------------
# Embedding attachment
# ---------------------------------------------------------------------------

def maybe_attach_embeddings(
    x: torch.Tensor,
    feature_names: List[str],
    user_ids: np.ndarray,
    handles: Optional[List[Optional[str]]],
    embeddings_path: str,
    embedding_pool: str = "meanpool",
) -> Tuple[torch.Tensor, List[str], Dict]:
    """Optionally concatenate pre-built user embeddings onto ``x``.

    Matches by ``user_ids`` first; falls back to handle matching if the
    embedding artifact does not contain numeric ids.

    Returns ``(x, feature_names, stats)``.
    """
    if not embeddings_path:
        return x, feature_names, {"matched_users": 0, "embedding_dim": 0}

    emb = torch.load(embeddings_path, map_location="cpu")
    emb_mat = emb.get(embedding_pool)
    if emb_mat is None:
        raise KeyError(f"Embeddings file must contain '{embedding_pool}'")

    emb_dim = int(emb_mat.shape[1])
    extra = torch.zeros((len(user_ids), emb_dim), dtype=torch.float)

    emb_user_ids = emb.get("user_ids")
    if emb_user_ids is not None:
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
        if emb_handles is None or handles is None:
            raise KeyError(
                f"Embeddings file must contain either 'user_ids' or 'handles' "
                f"plus '{embedding_pool}'"
            )
        emb_handle_series = pd.Series(
            np.arange(len(emb_handles), dtype=np.int64),
            index=[str(h).strip().lower() for h in emb_handles],
        )
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


# ---------------------------------------------------------------------------
# Isolate removal
# ---------------------------------------------------------------------------

def drop_isolates_from_graph(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    user_ids: np.ndarray,
    y: Optional[torch.Tensor] = None,
    handles: Optional[List] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray,
           Optional[torch.Tensor], Optional[List], Dict[int, int], int]:
    """Remove nodes with zero in-degree and zero out-degree and remap indices."""
    n = len(user_ids)
    degrees_out = torch.bincount(edge_index[0], minlength=n)
    degrees_in = torch.bincount(edge_index[1], minlength=n)
    keep_mask = (degrees_out > 0) | (degrees_in > 0)
    isolated = int((~keep_mask).sum().item())

    if isolated == 0:
        u2i = {int(uid): i for i, uid in enumerate(user_ids.tolist())}
        return x, edge_index, edge_attr, user_ids, y, handles, u2i, 0

    kept_nodes = int(keep_mask.sum().item())
    remap = torch.full((n,), -1, dtype=torch.long)
    remap[keep_mask] = torch.arange(kept_nodes, dtype=torch.long)

    x = x[keep_mask]
    edge_index = remap[edge_index]
    user_ids = user_ids[keep_mask.cpu().numpy()]
    if y is not None:
        y = y[keep_mask]
    if handles is not None:
        mask_list = keep_mask.tolist()
        handles = [h for h, k in zip(handles, mask_list) if k]

    u2i = {int(uid): i for i, uid in enumerate(user_ids.tolist())}
    return x, edge_index, edge_attr, user_ids, y, handles, u2i, isolated


# ---------------------------------------------------------------------------
# Temporal views
# ---------------------------------------------------------------------------

def build_temporal_views(
    rt: pd.DataFrame,
    u2i: Dict[int, int],
    history_fraction: float,
    future_target_mode: str,
    full_edge_index: torch.Tensor,
    full_edge_attr: torch.Tensor,
    edge_feature_names: List[str] = EDGE_FEATURE_NAMES,
) -> Tuple[Dict, Dict]:
    """Build temporal edge views for link-prediction tasks.

    Returns ``(graph_entries, stats)`` where ``graph_entries`` is a dict
    ready to be merged into the top-level graph object.
    """
    rt_sorted = rt.sort_values("timestamp").reset_index(drop=True)
    cutoff_idx = int(len(rt_sorted) * history_fraction)
    cutoff_idx = max(1, min(len(rt_sorted) - 1, cutoff_idx))

    hist_rt = rt_sorted.iloc[:cutoff_idx].copy()
    fut_rt = rt_sorted.iloc[cutoff_idx:].copy()

    hist_edges_df = aggregate_edge_features(hist_rt)
    fut_edges_df = aggregate_edge_features(fut_rt)

    hist_edge_index, hist_edge_attr = to_edge_tensors(hist_edges_df, u2i, edge_feature_names)

    hist_pairs = set(zip(hist_edges_df["userid"].astype(int), hist_edges_df["rt_userid"].astype(int)))
    fut_pairs = set(zip(fut_edges_df["userid"].astype(int), fut_edges_df["rt_userid"].astype(int)))
    target_pairs = (fut_pairs - hist_pairs) if future_target_mode == "new_only" else fut_pairs

    if target_pairs:
        target_df = pd.DataFrame(sorted(target_pairs), columns=["userid", "rt_userid"])
        target_df["src"] = target_df["userid"].map(u2i)
        target_df["dst"] = target_df["rt_userid"].map(u2i)
        target_df = target_df.dropna(subset=["src", "dst"])
        target_new_ei = torch.tensor(
            target_df[["src", "dst"]].astype(int).values.T, dtype=torch.long
        )
    else:
        target_new_ei = torch.zeros((2, 0), dtype=torch.long)

    graph_entries = {
        "edge_index_views": {
            "retweet_all": full_edge_index,
            "temporal_history": hist_edge_index,
        },
        "edge_attr_views": {
            "retweet_all": full_edge_attr,
            "temporal_history": hist_edge_attr,
        },
        "edge_attr_feature_names_views": {
            "retweet_all": edge_feature_names,
            "temporal_history": edge_feature_names,
        },
        "target_edge_index_views": {"temporal_new": target_new_ei},
        "future_edge_index": target_new_ei,
    }

    stats = {
        "history_fraction": history_fraction,
        "future_target_mode": future_target_mode,
        "history_rows": int(len(hist_rt)),
        "future_rows": int(len(fut_rt)),
        "history_edges": int(hist_edge_index.shape[1]),
        "future_edges": int(len(fut_pairs)),
        "future_overlap_edges": int(len(hist_pairs & fut_pairs)),
        "future_target_edges": int(target_new_ei.shape[1]),
    }

    return graph_entries, stats


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_graph(out: str, graph_obj: Dict, meta: Optional[Dict] = None) -> None:
    """Save a graph dict to ``out`` and optionally write a ``.meta.json`` sidecar.

    Adds a ``data`` key containing a ``torch_geometric.data.Data`` compatibility
    object before saving.
    """
    out_dir = os.path.dirname(out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    x = graph_obj["x"]
    edge_index = graph_obj["edge_index"]
    edge_attr = graph_obj.get("edge_attr")
    y = graph_obj.get("y")
    feature_names = graph_obj.get("feature_names", [])
    label_names = graph_obj.get("label_names", [])
    user_ids = graph_obj.get("user_ids", [])
    uid_list = user_ids.tolist() if isinstance(user_ids, np.ndarray) else list(user_ids)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data.feature_names = feature_names
    data.edge_attr_feature_names = graph_obj.get("edge_attr_feature_names", EDGE_FEATURE_NAMES)
    data.label_names = label_names
    data.user_ids = uid_list
    graph_obj["data"] = data

    torch.save(graph_obj, out)
    print(f"Saved graph: {out}")

    if meta is not None:
        meta_path = out.replace(".pt", ".meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"Saved meta:  {meta_path}")
