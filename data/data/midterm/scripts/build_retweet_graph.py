import argparse
import ast
import glob
import json
import os
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data


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

LABEL_NAMES = ["rep", "dem"]
LABEL_TO_ID = {name: i for i, name in enumerate(LABEL_NAMES)}
HASHTAG_TO_LABEL = {tag: "rep" for tag in REP_HASHTAGS} | {tag: "dem" for tag in DEM_HASHTAGS}
URL_TO_LABEL = {domain: "rep" for domain in REP_MEDIA_OUTLETS} | {domain: "dem" for domain in DEM_MEDIA_OUTLETS}


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
    p.add_argument("--history_fraction", type=float, default=0.3)
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
    p.add_argument(
        "--keep-isolates",
        dest="keep_isolates",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep nodes with zero in-degree and zero out-degree in the saved graph.",
    )
    p.add_argument(
        "--pseudo-label-margin",
        type=int,
        default=2,
        help="Minimum one-sided evidence count required to assign a pseudo label.",
    )
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


def _fast_list_lens(series: pd.Series) -> np.ndarray:
    return np.fromiter(
        (count_list_like(x) for x in series),
        dtype=np.int32,
        count=len(series),
    )


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


def extract_domain(url: str) -> str:
    s = str(url or "").strip().lower()
    if not s:
        return ""
    s = s.replace("https://", "").replace("http://", "")
    s = s.split("/", 1)[0]
    if s.startswith("www."):
        s = s[4:]
    return s


def score_url_iter(urls: List[str]) -> Dict[str, int]:
    counts = {label: 0 for label in LABEL_NAMES}
    for url in urls:
        domain = extract_domain(url)
        if not domain:
            continue
        matched = None
        for outlet, label in URL_TO_LABEL.items():
            if domain == outlet or domain.endswith(f".{outlet}"):
                matched = label
                break
        if matched is not None:
            counts[matched] += 1
    return counts


def map_url_scores(series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    keys = series.astype("string").fillna("")
    unique_vals = pd.unique(keys)

    rep_map: Dict[str, int] = {}
    dem_map: Dict[str, int] = {}
    for val in unique_vals:
        counts = score_url_iter(_parse_url_entries(val))
        rep_map[val] = counts["rep"]
        dem_map[val] = counts["dem"]

    rep_series = keys.map(rep_map).fillna(0).astype(np.int32)
    dem_series = keys.map(dem_map).fillna(0).astype(np.int32)
    return rep_series, dem_series


def extract_hashtags_from_text(text) -> List[str]:
    s = str(text or "").lower()
    return [match[1:] for match in re.findall(r"#\w+", s)]


def extract_hashtags_from_field(val) -> List[str]:
    tags: List[str] = []
    s = str(val).strip()
    if s in {"", "[]", "nan", "None", "<NA>"}:
        return tags
    s = s.strip("[]").replace("'", "").replace('"', "")
    for item in s.split(","):
        tag = item.strip().lower()
        if not tag:
            continue
        if tag.startswith("#"):
            tag = tag[1:]
        if tag:
            tags.append(tag)
    return tags


def score_hashtag_iter(tags: List[str]) -> Dict[str, int]:
    counts = {label: 0 for label in LABEL_NAMES}
    for tag in tags:
        label = HASHTAG_TO_LABEL.get(tag)
        if label is not None:
            counts[label] += 1
    return counts


def map_hashtag_field_scores(series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    keys = series.astype("string").fillna("")
    unique_vals = pd.unique(keys)

    rep_map: Dict[str, int] = {}
    dem_map: Dict[str, int] = {}
    for val in unique_vals:
        counts = score_hashtag_iter(extract_hashtags_from_field(val))
        rep_map[val] = counts["rep"]
        dem_map[val] = counts["dem"]

    rep_series = keys.map(rep_map).fillna(0).astype(np.int32)
    dem_series = keys.map(dem_map).fillna(0).astype(np.int32)
    return rep_series, dem_series


def build_political_leaning_labels(
    base: pd.DataFrame,
    id_to_idx: Dict[int, int],
    min_margin: int = 2,
) -> Tuple[torch.Tensor, List[str]]:
    y = torch.full((len(id_to_idx),), -1, dtype=torch.long)

    label_df = base[["userid"]].copy()
    if "urls_list" in base.columns:
        url_source = base["urls_list"]
    elif "urls" in base.columns:
        url_source = base["urls"]
    else:
        url_source = pd.Series(index=base.index, dtype=object)
    label_df["url_source"] = url_source
    label_df["description"] = base.get("description", "").fillna("").astype(str).str.lower()
    label_df["hashtag"] = base.get("hashtag", "").astype("string").fillna("")
    label_df["rt_hashtag"] = base.get("rt_hashtag", "").astype("string").fillna("")

    rep_url, dem_url = map_url_scores(label_df["url_source"])
    rep_hashtag, dem_hashtag = map_hashtag_field_scores(label_df["hashtag"])
    rep_rt_hashtag, dem_rt_hashtag = map_hashtag_field_scores(label_df["rt_hashtag"])

    url_agg = (
        pd.DataFrame(
            {
                "userid": label_df["userid"],
                "rep_url": rep_url,
                "dem_url": dem_url,
                "rep_hashtag": rep_hashtag + rep_rt_hashtag,
                "dem_hashtag": dem_hashtag + dem_rt_hashtag,
            }
        )
        .groupby("userid", sort=False)[["rep_url", "dem_url", "rep_hashtag", "dem_hashtag"]]
        .sum()
    )

    desc_df = label_df[["userid", "description"]].copy()
    desc_df["description"] = desc_df["description"].astype("string").fillna("").str.strip().str.lower()
    desc_df = desc_df[desc_df["description"] != ""].drop_duplicates(subset=["userid", "description"])
    if not desc_df.empty:
        desc_scores = desc_df["description"].apply(lambda text: score_hashtag_iter(extract_hashtags_from_text(text)))
        desc_df["rep_desc"] = desc_scores.map(lambda d: d["rep"]).astype(np.int32)
        desc_df["dem_desc"] = desc_scores.map(lambda d: d["dem"]).astype(np.int32)
        desc_agg = desc_df.groupby("userid", sort=False)[["rep_desc", "dem_desc"]].sum()
    else:
        desc_agg = pd.DataFrame(columns=["rep_desc", "dem_desc"])

    label_agg = url_agg.join(desc_agg, how="outer").fillna(0)
    rep_totals = label_agg.get("rep_url", 0) + label_agg.get("rep_hashtag", 0) + label_agg.get("rep_desc", 0)
    dem_totals = label_agg.get("dem_url", 0) + label_agg.get("dem_hashtag", 0) + label_agg.get("dem_desc", 0)

    node_ids = np.asarray(sorted(id_to_idx, key=id_to_idx.get), dtype=np.int64)
    user_index = pd.Index(node_ids)
    rep_votes = user_index.map(rep_totals).fillna(0).astype(int)
    dem_votes = user_index.map(dem_totals).fillna(0).astype(int)

    has_enough = (rep_votes >= min_margin) | (dem_votes >= min_margin)
    not_mixed = ~((rep_votes > 0) & (dem_votes > 0))
    valid = has_enough & not_mixed

    y_np = np.full(len(node_ids), -1, dtype=np.int64)
    y_np[valid & (rep_votes > 0)] = LABEL_TO_ID["rep"]
    y_np[valid & (rep_votes == 0)] = LABEL_TO_ID["dem"]
    y = torch.from_numpy(y_np)

    labeled_count = int(valid.sum())
    print(
        f"Political-leaning labels: {labeled_count:,} / {len(id_to_idx):,} "
        f"({100 * labeled_count / max(1, len(id_to_idx)):.1f}%) labeled"
    )
    return y, list(LABEL_NAMES)


def build_node_features(
    rt: pd.DataFrame,
    id_to_idx: Dict[int, int],
    edge_df: pd.DataFrame,
    pseudo_label_margin: int,
) -> Tuple[torch.Tensor, List[str], torch.Tensor, List[str]]:
    base = rt.dropna(subset=["userid"]).copy()

    for col in ["followers_count", "statuses_count", "sent_vader", "rt_fav_count", "rt_reply_count"]:
        if col in base.columns:
            base[col] = pd.to_numeric(base[col], errors="coerce")

    base["verified"] = base.get("verified", 0).map({True: 1, False: 0, "True": 1, "False": 0}).fillna(0).astype(float)
    base["n_hashtags"] = _fast_list_lens(base["hashtag"]) if "hashtag" in base.columns else 0
    base["n_mentions"] = _fast_list_lens(base["mentionsn"]) if "mentionsn" in base.columns else 0
    if "media_urls" in base.columns:
        base["has_media"] = (_fast_list_lens(base["media_urls"]) > 0).astype(np.int8)
    else:
        base["has_media"] = 0

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

    nonzero = np.any(X != 0, axis=1)
    if nonzero.any():
        scaler = StandardScaler()
        X[nonzero] = scaler.fit_transform(X[nonzero]).astype(np.float32)

    x = torch.tensor(X, dtype=torch.float)

    y, label_names = build_political_leaning_labels(base, id_to_idx, min_margin=pseudo_label_margin)

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

    emb_user_ids = np.asarray(emb_user_ids).astype(np.int64)
    emb_dim = int(emb_mat.shape[1])
    extra = torch.zeros((len(node_ids), emb_dim), dtype=torch.float)
    order = np.argsort(emb_user_ids)
    sorted_ids = emb_user_ids[order]
    query = np.asarray(node_ids, dtype=np.int64)
    pos = np.searchsorted(sorted_ids, query)
    pos_clipped = np.clip(pos, 0, len(sorted_ids) - 1)
    hit = (pos < len(sorted_ids)) & (sorted_ids[pos_clipped] == query)
    tgt_rows = np.where(hit)[0]
    src_rows = order[pos[hit]]
    if len(tgt_rows):
        extra[torch.from_numpy(tgt_rows)] = emb_mat[torch.from_numpy(src_rows)].float()
    matched = int(hit.sum())

    x2 = torch.cat([x, extra], dim=1)
    names2 = feature_names + [f"emb_{k}" for k in range(emb_dim)]
    return x2, names2, {"matched_users": matched, "embedding_dim": emb_dim}


def drop_isolates_from_graph(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    user_ids: np.ndarray,
    y: torch.Tensor,
):
    n = len(user_ids)
    degrees_out = torch.bincount(edge_index[0], minlength=n)
    degrees_in = torch.bincount(edge_index[1], minlength=n)
    keep_mask = (degrees_out > 0) | (degrees_in > 0)
    isolated = int((~keep_mask).sum().item())
    if isolated == 0:
        return x, edge_index, edge_attr, user_ids, y, {int(uid): i for i, uid in enumerate(user_ids.tolist())}, 0

    kept_nodes = int(keep_mask.sum().item())
    remap = torch.full((n,), -1, dtype=torch.long)
    remap[keep_mask] = torch.arange(kept_nodes, dtype=torch.long)
    x = x[keep_mask]
    y = y[keep_mask]
    edge_index = remap[edge_index]
    user_ids = user_ids[keep_mask.cpu().numpy()]
    u2i = {int(uid): i for i, uid in enumerate(user_ids.tolist())}
    return x, edge_index, edge_attr, user_ids, y, u2i, isolated


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    raw = load_raw_rows(args.csv_glob, args.max_files)
    rt = normalize_ids_and_timestamps(raw, args.strict_dates)
    rt = trim_rt_to_max_nodes(rt, args.max_nodes)

    user_ids = np.array(sorted(set(rt["userid"].tolist()) | set(rt["rt_userid"].tolist())), dtype=np.int64)
    u2i = {int(uid): i for i, uid in enumerate(user_ids)}
    print(f"Nodes: {len(user_ids):,}")

    edge_feature_names = ["first_retweet_time", "n_retweets", "avg_rt_fav", "avg_rt_reply"]
    edge_all_df = aggregate_edge_features(rt)
    edge_index, edge_attr = to_edge_tensors(edge_all_df, u2i, edge_feature_names)
    print(f"Directed edges: {edge_index.shape[1]:,}")

    x, feature_names, y, label_names = build_node_features(
        rt,
        u2i,
        edge_all_df,
        pseudo_label_margin=args.pseudo_label_margin,
    )
    x, feature_names, emb_stats = maybe_attach_embeddings(
        x, feature_names, user_ids, args.embeddings, args.embedding_pool
    )

    isolated_before_drop = int(((torch.bincount(edge_index[0], minlength=len(user_ids)) == 0) & (torch.bincount(edge_index[1], minlength=len(user_ids)) == 0)).sum().item())
    if not args.keep_isolates:
        x, edge_index, edge_attr, user_ids, y, u2i, isolated_dropped = drop_isolates_from_graph(
            x, edge_index, edge_attr, user_ids, y
        )
        print(f"Dropped isolated nodes: {isolated_dropped:,}")
    else:
        isolated_dropped = 0

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
        rt_sorted = rt.sort_values("timestamp").reset_index(drop=True)
        cutoff_idx = int(len(rt_sorted) * args.history_fraction)
        cutoff_idx = max(1, min(len(rt_sorted) - 1, cutoff_idx))

        hist_rt = rt_sorted.iloc[:cutoff_idx].copy()
        fut_rt = rt_sorted.iloc[cutoff_idx:].copy()

        hist_edges_df = aggregate_edge_features(hist_rt)
        fut_edges_df = aggregate_edge_features(fut_rt)

        hist_edge_index, hist_edge_attr = to_edge_tensors(hist_edges_df, u2i, edge_feature_names)

        hist_pairs = set(zip(hist_edges_df["userid"].astype(int), hist_edges_df["rt_userid"].astype(int)))
        fut_pairs = set(zip(fut_edges_df["userid"].astype(int), fut_edges_df["rt_userid"].astype(int)))
        if args.future_target_mode == "new_only":
            target_pairs = fut_pairs - hist_pairs
        else:
            target_pairs = fut_pairs

        if target_pairs:
            target_df = pd.DataFrame(sorted(list(target_pairs)), columns=["userid", "rt_userid"])
            target_df["src"] = target_df["userid"].map(u2i)
            target_df["dst"] = target_df["rt_userid"].map(u2i)
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

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data.feature_names = feature_names
    data.edge_attr_feature_names = edge_feature_names
    data.label_names = label_names
    data.user_ids = list(user_ids.tolist())
    graph_obj["data"] = data
    torch.save(graph_obj, args.out)

    meta = {
        "csv_glob": args.csv_glob,
        "max_files": args.max_files,
        "nodes": int(len(user_ids)),
        "edges": int(edge_index.shape[1]),
        "node_feature_dim": int(x.shape[1]),
        "edge_feature_names": edge_feature_names,
        "label_count": int(len(label_names)),
        "labeled_nodes": int((y >= 0).sum().item()),
        "embeddings": args.embeddings,
        "embedding_pool": args.embedding_pool,
        "embedding_dim": emb_stats["embedding_dim"],
        "embedding_matched_users": emb_stats["matched_users"],
        "pseudo_label_margin": int(args.pseudo_label_margin),
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
