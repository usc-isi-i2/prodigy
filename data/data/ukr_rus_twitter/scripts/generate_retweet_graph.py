"""Build the ukr_rus_twitter retweet graph with optional embeddings and temporal views.

Dataset-specific logic: interleaved CSV loading and URL/hashtag/handle-based
weak left/right political labels.
All shared graph-building primitives come from rapids.graph.build.
"""
import argparse
import ast
import glob
import os
import re
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import torch

from rapids.graph.build import (
    EDGE_FEATURE_NAMES,
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
from rapids.loaders.csv_loader import load_ukr_rus_file

DEFAULT_CSV = "/project2/ll_774_951/uk_ru/twitter/data/*/*.csv"
DEFAULT_OUT = "data/data/ukr_rus_twitter/graphs/retweet_graph.pt"
DEFAULT_HISTORY_FRACTION = 0.3

LABEL_NAMES = ["left", "right"]
LABEL_TO_ID = {n: i for i, n in enumerate(LABEL_NAMES)}

# Outlet bias scores: < 3 → left, > 3 → right, == 3 → neutral
OUTLET_SCORES = {
    "abcnews.go.com": 2.0, "bbc.com": 3.0, "breitbart.com": 5.0,
    "bostonglobe.com": 2.0, "businessinsider.com": 3.0, "buzzfeednews.com": 1.0,
    "cbsnews.com": 2.0, "chicagotribune.com": 3.0, "cnbc.com": 3.0, "cnn.com": 2.0,
    "dailycaller.com": 5.0, "dailymail.co.uk": 5.0, "foxnews.com": 4.0,
    "huffpost.com": 1.0, "infowars.com": 5.0, "latimes.com": 2.0, "msnbc.om": 1.0,
    "nbcnews.com": 2.0, "nytimes.com": 2.0, "npr.org": 3.0, "oann.com": 4.0,
    "pbs.org": 3.0, "reuters.com": 3.0, "theguardian.com": 2.0, "usatoday.com": 3.0,
    "yahoo.com": 2.0, "vice.com": 1.0, "washingtonpost.com": 2.0, "wsj.com": 3.0,
    "thehill.com": 2.7, "rt.com": 3.7, "rawstory.com": 1.7,
    "news.sky.com": 2.3, "independent.co.uk": 2.3, "dailykos.com": 1.7,
}
URL_TO_LABEL = {d: ("left" if s < 3 else "right") for d, s in OUTLET_SCORES.items() if s != 3}

HANDLE_SCORES = {
    "abc": 2, "bbcworld": 3, "breitbartnews": 5, "bostonglobe": 2,
    "businessinsider": 3, "buzzfeednews": 1, "cbsnews": 2, "chicagotribune": 3,
    "cnbc": 3, "cnn": 2, "dailycaller": 5, "dailymail": 5, "foxnews": 4,
    "huffpost": 1, "infowars": 5, "latimes": 2, "msnbc": 1, "nbcnews": 2,
    "nytimes": 2, "npr": 3, "oann": 4, "pbs": 3, "reuters": 3, "guardian": 2,
    "usatoday": 3, "yahoonews": 2, "vice": 1, "washingtonpost": 2, "wsj": 3,
    "thehill": 2.7, "rt_com": 3.7, "rawstory": 1.7, "skynews": 2.3,
    "independent": 2.3, "dailykos": 1.7,
}

HASHTAG_TO_LABEL = {
    "blm": 0, "resist": 0, "fbpe": 0, "blacklivesmatter": 0, "fbr": 0, "maga": 1,
    "theresistance": 0, "voteblue": 0, "resistance": 0, "bidenharris": 0,
    "johnsonout": 0, "lgbtq": 0, "bidenharris2020": 0, "2a": 1, "fbpa": 0,
    "resister": 0, "fbppr": 0, "bluewave": 0, "gtto": 0, "freepalestine": 0,
    "rejoineu": 0, "voteblue2022": 0, "kag": 1, "wearamask": 0, "getvaccinated": 0,
    "humanrights": 0, "votingrights": 0, "science": 0, "goodtrouble": 0,
    "strongertogether": 0, "stillwithher": 0, "climatecrisis": 0, "metoo": 0,
    "demvoice1": 0, "biden": 0, "climatechange": 0, "justicematters": 0,
    "americafirst": 1, "nevertrump": 0, "khive": 0, "democrat": 0, "vaccinated": 0,
    "buildbackbetter": 0, "stopasianhate": 0, "prochoice": 0, "drcole": 1, "bds": 0,
    "votebluenomatterwho": 0, "teampelosi": 0, "handmarkedpaperballots": 1,
    "reconquête": 1, "biden2020": 0, "patriot": 1, "equality": 0, "prolife": 1,
    "antifa": 0, "fjb": 1, "lgbt": 0, "nra": 1, "climateaction": 0,
    "unionpopulaire": 0, "zemmour2022": 1, "trump": 1, "demcast": 0, "m4a": 0,
    "vaxxed": 0, "democracy": 0, "indivisible": 0, "teamjustice": 0, "noafd": 0,
    "alllivesmatter": 1,
}

USECOLS = {
    "screen_name", "userid", "rt_userid", "rt_screen", "date",
    "followers_count", "verified", "statuses_count", "rt_fav_count", "rt_reply_count",
    "sent_vader", "hashtag", "mentionsn", "media_urls", "description",
    "urls_list", "rt_urls_list", "rt_hashtag",
}


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------

def _parse_list(val) -> list:
    if pd.isna(val):
        return []
    if isinstance(val, list):
        return val
    s = str(val).strip()
    if s in {"", "[]", "nan", "None", "<NA>"}:
        return []
    try:
        parsed = ast.literal_eval(s)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


def _normalize_domain(domain: str) -> str:
    d = (domain or "").strip().lower()
    return d[4:] if d.startswith("www.") else d


def _extract_domains(val) -> List[str]:
    domains = []
    for item in _parse_list(val):
        raw = str(item.get("expanded_url") or item.get("url") or "") if isinstance(item, dict) else str(item)
        raw = raw.strip()
        if not raw:
            continue
        try:
            netloc = urlparse(raw).netloc
        except Exception:
            netloc = ""
        d = _normalize_domain(netloc)
        if d:
            domains.append(d)
    return domains


def _score_url_val(val) -> Dict[str, int]:
    counts = {"left": 0, "right": 0}
    for domain in _extract_domains(val):
        label = URL_TO_LABEL.get(domain)
        if label:
            counts[label] += 1
    return counts


def _extract_hashtags_field(val) -> List[str]:
    tags = []
    for item in _parse_list(val):
        tag = str(item.get("text") or item.get("tag") or "") if isinstance(item, dict) else str(item)
        tag = tag.strip().lower().lstrip("#")
        if tag:
            tags.append(tag)
    return tags


def _extract_hashtags_text(text) -> List[str]:
    return [m[1:] for m in re.findall(r"#\w+", str(text or "").lower())]


def _score_hashtags(tags: List[str]) -> Dict[str, int]:
    counts = {"left": 0, "right": 0}
    for t in tags:
        val = HASHTAG_TO_LABEL.get(t)
        if val is not None:
            counts[LABEL_NAMES[val]] += 1
    return counts


def _map_scores(series: pd.Series, score_fn) -> Tuple[pd.Series, pd.Series]:
    keys = series.astype("string").fillna("")
    unique_vals = pd.unique(keys)
    left_map, right_map = {}, {}
    for val in unique_vals:
        c = score_fn(val)
        left_map[val] = c.get("left", 0)
        right_map[val] = c.get("right", 0)
    return keys.map(left_map).fillna(0).astype(np.int32), keys.map(right_map).fillna(0).astype(np.int32)


def _normalize_handle_series(series: pd.Series) -> pd.Series:
    normalized = series.astype("string").str.strip().str.lower()
    return normalized.mask(normalized.isin(["", "nan", "none", "<na>"]))


def build_pseudo_political_labels(
    raw: pd.DataFrame,
    handles: List[Optional[str]],
    min_margin: int = 2,
) -> Tuple[torch.Tensor, List[str], int]:
    needed = [c for c in ("screen_name", "rt_screen", "description", "urls_list") if c in raw.columns]
    user_df = raw[needed].copy()
    user_df["screen_name"] = _normalize_handle_series(user_df["screen_name"])
    user_df = user_df[user_df["screen_name"].notna()].copy()
    if user_df.empty:
        return torch.full((len(handles),), -1, dtype=torch.long), list(LABEL_NAMES), 0

    # Signal 1: URLs
    url_left = pd.Series(0, index=user_df.index, dtype=np.int32)
    url_right = pd.Series(0, index=user_df.index, dtype=np.int32)
    if "urls_list" in user_df.columns:
        url_left, url_right = _map_scores(user_df["urls_list"], _score_url_val)

    url_agg = (
        pd.DataFrame({"screen_name": user_df["screen_name"],
                      "left_url": url_left, "right_url": url_right})
        .groupby("screen_name", sort=False)[["left_url", "right_url"]].sum()
    )

    # Signal 2: Retweet-target handles
    rt_agg = pd.DataFrame(index=url_agg.index, data={"left_rt": 0, "right_rt": 0})
    if "rt_screen" in user_df.columns:
        rt_score = user_df["rt_screen"].str.lower().map(HANDLE_SCORES)
        rt_df = pd.DataFrame({
            "screen_name": user_df["screen_name"],
            "left_rt": (rt_score.lt(3) & rt_score.notna()).astype(np.int32).values,
            "right_rt": (rt_score.gt(3) & rt_score.notna()).astype(np.int32).values,
        })
        rt_agg = rt_df.groupby("screen_name", sort=False)[["left_rt", "right_rt"]].sum()

    # Signal 3: Profile description hashtags
    desc_agg = pd.DataFrame(index=url_agg.index, data={"left_ht": 0, "right_ht": 0})
    if "description" in user_df.columns:
        desc_df = user_df[["screen_name", "description"]].copy()
        desc_df["description"] = desc_df["description"].astype("string").fillna("").str.strip().str.lower()
        desc_df = desc_df[desc_df["description"] != ""].drop_duplicates(subset=["screen_name", "description"])
        if not desc_df.empty:
            scores = desc_df["description"].apply(lambda t: _score_hashtags(_extract_hashtags_text(t)))
            desc_df = desc_df.copy()
            desc_df["left_ht"] = scores.map(lambda d: d["left"]).astype(np.int32)
            desc_df["right_ht"] = scores.map(lambda d: d["right"]).astype(np.int32)
            desc_agg = desc_df.groupby("screen_name", sort=False)[["left_ht", "right_ht"]].sum()

    agg = url_agg.join(rt_agg, how="outer").join(desc_agg, how="outer").fillna(0)
    left_totals = agg.get("left_url", 0) + agg.get("left_rt", 0) + agg.get("left_ht", 0)
    right_totals = agg.get("right_url", 0) + agg.get("right_rt", 0) + agg.get("right_ht", 0)

    handle_index = pd.Index(handles)
    left_votes = handle_index.map(left_totals).fillna(0).astype(int)
    right_votes = handle_index.map(right_totals).fillna(0).astype(int)

    has_enough = (right_votes >= min_margin) | (left_votes >= min_margin)
    not_mixed = ~((right_votes > 0) & (left_votes > 0))
    valid = has_enough & not_mixed

    y_np = np.full(len(handles), -1, dtype=np.int64)
    y_np[valid & (right_votes > 0)] = LABEL_TO_ID["right"]
    y_np[valid & (right_votes == 0)] = LABEL_TO_ID["left"]
    labeled = int(valid.sum())
    print(f"Pseudo-political labels: {labeled:,}/{len(handles):,} "
          f"({100*labeled/max(1,len(handles)):.1f}%) labeled")
    return torch.from_numpy(y_np), list(LABEL_NAMES), labeled


# ---------------------------------------------------------------------------
# Raw data loading
# ---------------------------------------------------------------------------

def load_raw_rows(csv_glob: str, max_files: int) -> pd.DataFrame:
    files = sorted(glob.glob(csv_glob))
    if max_files > 0:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"No files matched: {csv_glob}")
    chunks = []
    print(f"Found {len(files)} files")
    for i, fpath in enumerate(files, start=1):
        print(f"[{i}/{len(files)}] Loading {os.path.basename(fpath)}", flush=True)
        try:
            dfi = load_ukr_rus_file(fpath)
            if dfi.empty:
                continue
            cols = [c for c in dfi.columns if c in USECOLS]
            if cols:
                chunks.append(dfi[cols].copy())
        except Exception as exc:
            print(f"  [WARN] failed {os.path.basename(fpath)}: {exc}", flush=True)
        if i % 10 == 0 or i == len(files):
            print(f"  processed {i}/{len(files)} files", flush=True)
    if not chunks:
        raise RuntimeError("No valid rows parsed")
    df = pd.concat(chunks, ignore_index=True)
    print(f"Loaded rows: {len(df):,}")
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Build ukr_rus_twitter retweet graph with optional embeddings and temporal views."
    )
    p.add_argument("--csv", default=DEFAULT_CSV)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--embeddings", default="")
    p.add_argument("--embedding_pool", choices=["meanpool", "maxpool"], default="meanpool")
    p.add_argument("--max_files", type=int, default=0)
    p.add_argument("--max_nodes", type=int, default=0)
    p.add_argument("--strict_dates", action="store_true")
    p.add_argument("--history_fraction", type=float, default=DEFAULT_HISTORY_FRACTION)
    p.add_argument("--future_target_mode", choices=["new_only", "all_future"], default="new_only")
    p.add_argument("--no_temporal_views", action="store_true")
    p.add_argument("--keep-isolates", dest="keep_isolates",
                   action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--pseudo-political-labels",
                   action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--pseudo-label-margin", type=int, default=2)
    return p.parse_args()


def main():
    args = parse_args()

    raw = load_raw_rows(args.csv, args.max_files)
    rt = prepare_retweet_rows(raw, args.strict_dates)
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

    labeled = 0
    if args.pseudo_political_labels:
        y, label_names, labeled = build_pseudo_political_labels(
            raw, handles, min_margin=args.pseudo_label_margin
        )
    else:
        y = torch.full((len(user_ids),), -1, dtype=torch.long)
        label_names = list(LABEL_NAMES)

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
        "csv": args.csv,
        "max_files": args.max_files,
        "nodes": int(len(user_ids)),
        "edges": int(edge_index.shape[1]),
        "node_feature_dim": int(x.shape[1]),
        "edge_feature_names": EDGE_FEATURE_NAMES,
        "label_count": len(label_names),
        "labeled_nodes": labeled,
        "embeddings": args.embeddings,
        "embedding_pool": args.embedding_pool,
        **emb_stats,
        "pseudo_political_labels": args.pseudo_political_labels,
        "pseudo_label_margin": args.pseudo_label_margin,
        "keep_isolates": args.keep_isolates,
        "isolated_nodes_before_drop": isolated_before,
        "isolated_nodes_dropped": isolated_dropped,
        "temporal": temporal_stats,
    }
    save_graph(args.out, graph_obj, meta)


if __name__ == "__main__":
    main()
