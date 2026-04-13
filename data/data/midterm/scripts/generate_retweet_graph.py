"""Build the midterm retweet graph with optional embeddings and temporal views.

Dataset-specific logic: CSV loading and hashtag/URL-based weak political labels.
All shared graph-building primitives come from rapids.graph.build.
"""
import argparse
import glob
import os
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from rapids.graph.build import (
    EDGE_FEATURE_NAMES,
    aggregate_edge_features,
    build_node_features,
    build_temporal_views,
    build_user_index,
    drop_isolates_from_graph,
    maybe_attach_embeddings,
    prepare_retweet_rows,
    save_graph,
    to_edge_tensors,
    trim_rt_to_max_nodes,
)
from rapids.loaders.csv_loader import load_standard_csv

# ---------------------------------------------------------------------------
# Weak-label signal tables
# ---------------------------------------------------------------------------

REP_HASHTAGS = [
    "voteredtosaveamerica", "votered", "redwavecoming", "democratsaretheproblem",
    "2a", "1a", "fjb", "americafirst", "kag",
]
DEM_HASHTAGS = [
    "voteblue", "voteblue2022", "votebluetosavedemocracy", "votebluetosaveamerica",
    "votebluein2022", "votebluenomatterwho", "votebluefordemocracy",
    "votebluetoprotectwomen", "voteblueforwomensrights", "votebluetoprotectyourrights",
    "voteblueforsomanyreasons", "votebluetoendtheinsanity", "votebluenotq",
    "votebluedownballot", "votebluedownballotlocalstatefederal",
    "votebluetosavesocialsecurity", "votebluetosavesocialsecurityandmedicare",
    "votebluetosaveourkids", "bluewave", "bluewave2022", "bluecrew", "bluevoters",
    "ourbluevoice", "bluein22", "proudblue22", "demvoice1", "wtpblue",
    "democratsdeliver", "demsact", "voteouteveryrepublican",
    "stopvotingforrepublicans", "neverrepublicanagain", "republicansaretheproblem",
    "republicanwaronwomen", "goptraitorstodemocracy", "gopliesabouteverything",
    "magaidiots", "blm", "blacklivesmatter", "resist", "fbr",
]
DEM_MEDIA_OUTLETS = [
    "abcnews", "bbc", "buzzfeednews", "huffpost", "msnbc", "cnn",
    "nytimes", "washingtonpost", "latimes", "guardian",
]
REP_MEDIA_OUTLETS = [
    "breitbartnews", "dailycaller", "dailymail", "foxnews", "infowars", "oann", "breitbart",
]

LABEL_NAMES = ["rep", "dem"]
LABEL_TO_ID = {n: i for i, n in enumerate(LABEL_NAMES)}
HASHTAG_TO_LABEL = {t: "rep" for t in REP_HASHTAGS} | {t: "dem" for t in DEM_HASHTAGS}
URL_TO_LABEL = {d: "rep" for d in REP_MEDIA_OUTLETS} | {d: "dem" for d in DEM_MEDIA_OUTLETS}


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------

def _extract_hashtags_field(val) -> List[str]:
    s = str(val).strip()
    if s in {"", "[]", "nan", "None", "<NA>"}:
        return []
    s = s.strip("[]").replace("'", "").replace('"', "")
    return [t.strip().lstrip("#").lower() for t in s.split(",") if t.strip()]


def _extract_hashtags_text(text) -> List[str]:
    return [m[1:] for m in re.findall(r"#\w+", str(text or "").lower())]


def _score_hashtags(tags: List[str]) -> Dict[str, int]:
    counts = {"rep": 0, "dem": 0}
    for t in tags:
        label = HASHTAG_TO_LABEL.get(t)
        if label:
            counts[label] += 1
    return counts


def _score_urls(val) -> Dict[str, int]:
    import ast
    counts = {"rep": 0, "dem": 0}
    s = str(val).strip()
    if s in {"", "[]", "nan", "None"}:
        return counts
    try:
        items = ast.literal_eval(s)
    except Exception:
        return counts
    if not isinstance(items, list):
        items = [items]
    for item in items:
        url = ""
        if isinstance(item, dict):
            url = str(item.get("expanded_url") or item.get("url") or "").lower()
        elif isinstance(item, str):
            url = item.lower()
        url = url.replace("https://", "").replace("http://", "").split("/")[0]
        if url.startswith("www."):
            url = url[4:]
        label = URL_TO_LABEL.get(url)
        if label:
            counts[label] += 1
    return counts


def _map_field_scores(series: pd.Series, score_fn) -> Tuple[pd.Series, pd.Series]:
    keys = series.astype("string").fillna("")
    unique_vals = pd.unique(keys)
    rep_map, dem_map = {}, {}
    for val in unique_vals:
        c = score_fn(val)
        rep_map[val] = c.get("rep", 0)
        dem_map[val] = c.get("dem", 0)
    return keys.map(rep_map).fillna(0).astype(np.int32), keys.map(dem_map).fillna(0).astype(np.int32)


def build_political_leaning_labels(
    base: pd.DataFrame,
    id_to_idx: Dict[int, int],
    min_margin: int = 2,
) -> Tuple[torch.Tensor, List[str]]:
    label_df = base[["userid"]].copy()
    url_source = base.get("urls_list", base.get("urls", pd.Series(index=base.index, dtype=object)))
    label_df["url_source"] = url_source
    label_df["description"] = base.get("description", "").fillna("").astype(str).str.lower()
    label_df["hashtag"] = base.get("hashtag", "").astype("string").fillna("")
    label_df["rt_hashtag"] = base.get("rt_hashtag", "").astype("string").fillna("")

    rep_url, dem_url = _map_field_scores(label_df["url_source"], _score_urls)
    rep_ht, dem_ht = _map_field_scores(label_df["hashtag"],
                                        lambda v: _score_hashtags(_extract_hashtags_field(v)))
    rep_rht, dem_rht = _map_field_scores(label_df["rt_hashtag"],
                                          lambda v: _score_hashtags(_extract_hashtags_field(v)))

    agg = (
        pd.DataFrame({
            "userid": label_df["userid"],
            "rep": rep_url + rep_ht + rep_rht,
            "dem": dem_url + dem_ht + dem_rht,
        })
        .groupby("userid", sort=False)[["rep", "dem"]].sum()
    )

    desc_df = label_df[["userid", "description"]].drop_duplicates(subset=["userid", "description"])
    desc_df = desc_df[desc_df["description"].str.strip() != ""]
    if not desc_df.empty:
        scores = desc_df["description"].apply(lambda t: _score_hashtags(_extract_hashtags_text(t)))
        desc_df = desc_df.copy()
        desc_df["rep"] = scores.map(lambda d: d["rep"]).astype(np.int32)
        desc_df["dem"] = scores.map(lambda d: d["dem"]).astype(np.int32)
        desc_agg = desc_df.groupby("userid", sort=False)[["rep", "dem"]].sum()
        agg = agg.add(desc_agg, fill_value=0)

    node_ids = np.asarray(sorted(id_to_idx, key=id_to_idx.get), dtype=np.int64)
    idx = pd.Index(node_ids)
    rep_votes = idx.map(agg["rep"]).fillna(0).astype(int)
    dem_votes = idx.map(agg["dem"]).fillna(0).astype(int)

    has_enough = (rep_votes >= min_margin) | (dem_votes >= min_margin)
    not_mixed = ~((rep_votes > 0) & (dem_votes > 0))
    valid = has_enough & not_mixed

    y_np = np.full(len(node_ids), -1, dtype=np.int64)
    y_np[valid & (rep_votes > 0)] = LABEL_TO_ID["rep"]
    y_np[valid & (rep_votes == 0)] = LABEL_TO_ID["dem"]

    labeled = int(valid.sum())
    print(f"Political labels: {labeled:,}/{len(id_to_idx):,} "
          f"({100*labeled/max(1,len(id_to_idx)):.1f}%) labeled")
    return torch.from_numpy(y_np), list(LABEL_NAMES)


# ---------------------------------------------------------------------------
# Raw data loading
# ---------------------------------------------------------------------------

USECOLS = {
    "userid", "rt_userid", "date", "state",
    "followers_count", "verified", "statuses_count", "sent_vader",
    "hashtag", "rt_hashtag", "mentionsn", "media_urls",
    "rt_fav_count", "rt_reply_count", "description", "urls", "urls_list",
}


def load_raw_rows(csv_glob: str, max_files: int, strict_dates: bool = False, max_nodes: int = 0) -> pd.DataFrame:
    files = sorted(glob.glob(csv_glob))
    if max_files > 0:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"No files matched: {csv_glob}")
    chunks = []
    seen_nodes = set()
    stopped_early = False
    print(f"Found {len(files)} files")
    for i, fpath in enumerate(files, start=1):
        try:
            dfi = load_standard_csv(fpath)
            if dfi.empty:
                continue
            cols = [c for c in dfi.columns if c in USECOLS]
            if not cols:
                continue
            chunk = dfi[cols].copy()
            if max_nodes > 0:
                try:
                    file_rt = prepare_retweet_rows(chunk, strict_dates)
                except RuntimeError:
                    file_rt = pd.DataFrame(columns=["userid", "rt_userid"])
                if not file_rt.empty:
                    seen_nodes.update(file_rt["userid"].tolist())
                    seen_nodes.update(file_rt["rt_userid"].tolist())
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
                        chunks.append(chunk)
                        break
            chunks.append(chunk)
        except Exception as exc:
            print(f"[WARN] failed {os.path.basename(fpath)}: {exc}")
        if i % 10 == 0 or i == len(files):
            print(f"  processed {i}/{len(files)}")
    if not chunks:
        raise RuntimeError("No valid rows parsed")
    df = pd.concat(chunks, ignore_index=True)
    print(f"Loaded rows: {len(df):,}")
    if max_nodes > 0:
        print(
            f"Ingestion summary: cleaned participants seen={len(seen_nodes):,} "
            f"(target={max_nodes:,}, early_stop={stopped_early})",
            flush=True,
        )
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Build midterm retweet graph with optional embeddings and temporal views."
    )
    p.add_argument("--csv_glob", default="/project2/ll_774_951/midterm/*/*.csv")
    p.add_argument("--out", default="data/data/midterm/graphs/retweet_graph.pt")
    p.add_argument("--embeddings", default="")
    p.add_argument("--embedding_pool", choices=["meanpool", "maxpool"], default="meanpool")
    p.add_argument("--max_files", type=int, default=0)
    p.add_argument("--max_nodes", type=int, default=0)
    p.add_argument("--strict_dates", action="store_true")
    p.add_argument("--history_fraction", type=float, default=0.3)
    p.add_argument("--future_target_mode", choices=["new_only", "all_future"], default="new_only")
    p.add_argument("--no_temporal_views", action="store_true")
    p.add_argument("--keep-isolates", dest="keep_isolates",
                   action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--pseudo-label-margin", type=int, default=2)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    print("Configuration")
    print(f"  csv_glob: {args.csv_glob}")
    print(f"  out: {args.out}")
    print(f"  embeddings: {args.embeddings or '<none>'}")
    print(f"  embedding_pool: {args.embedding_pool}")
    print(f"  max_files: {args.max_files if args.max_files > 0 else 'all'}")
    print(f"  max_nodes: {args.max_nodes if args.max_nodes > 0 else 'all'}")
    print(f"  history_fraction: {args.history_fraction}")
    print(f"  future_target_mode: {args.future_target_mode}")
    print(f"  keep_isolates: {args.keep_isolates}")
    print(f"  pseudo_label_margin: {args.pseudo_label_margin}")
    print()

    raw = load_raw_rows(args.csv_glob, args.max_files, strict_dates=args.strict_dates, max_nodes=args.max_nodes)
    print(f"Raw frame: rows={len(raw):,} cols={len(raw.columns):,}", flush=True)
    rt = prepare_retweet_rows(raw, args.strict_dates)
    pretrim_nodes = len(set(rt["userid"].tolist()) | set(rt["rt_userid"].tolist()))
    print(f"Cleaned retweets: rows={len(rt):,} unique_nodes={pretrim_nodes:,}", flush=True)
    rt = trim_rt_to_max_nodes(rt, args.max_nodes)
    posttrim_nodes = len(set(rt["userid"].tolist()) | set(rt["rt_userid"].tolist()))
    print(f"Post-trim retweets: rows={len(rt):,} unique_nodes={posttrim_nodes:,}", flush=True)

    user_ids, u2i = build_user_index(rt)
    print(f"Nodes: {len(user_ids):,}")

    edge_all_df = aggregate_edge_features(rt)
    edge_index, edge_attr = to_edge_tensors(edge_all_df, u2i)
    print(f"Directed edges: {edge_index.shape[1]:,}")

    x, feature_names = build_node_features(rt, u2i, edge_all_df)
    y, label_names = build_political_leaning_labels(raw, u2i, args.pseudo_label_margin)

    x, feature_names, emb_stats = maybe_attach_embeddings(
        x, feature_names, user_ids, handles=None,
        embeddings_path=args.embeddings, embedding_pool=args.embedding_pool,
    )
    print(
        f"After attaching embeddings: {x.shape[1]} dims total, "
        f"matched_users={emb_stats['matched_users']:,}, embedding_dim={emb_stats['embedding_dim']}",
        flush=True,
    )

    isolated_before = int(
        ((torch.bincount(edge_index[0], minlength=len(user_ids)) == 0) &
         (torch.bincount(edge_index[1], minlength=len(user_ids)) == 0)).sum()
    )
    isolated_dropped = 0
    if not args.keep_isolates:
        x, edge_index, edge_attr, user_ids, y, _, u2i, isolated_dropped = drop_isolates_from_graph(
            x, edge_index, edge_attr, user_ids, y=y
        )
        print(f"Dropped isolated nodes: {isolated_dropped:,}")

    graph_obj = {
        "x": x, "edge_index": edge_index, "edge_attr": edge_attr,
        "edge_attr_feature_names": EDGE_FEATURE_NAMES,
        "user_ids": user_ids, "u2i": u2i,
        "feature_names": feature_names, "y": y, "label_names": label_names,
    }

    temporal_stats = {}
    if not args.no_temporal_views:
        temporal_entries, temporal_stats = build_temporal_views(
            rt, u2i, args.history_fraction, args.future_target_mode,
            edge_index, edge_attr,
        )
        graph_obj.update(temporal_entries)

    meta = {
        "csv_glob": args.csv_glob,
        "max_files": args.max_files,
        "nodes": int(len(user_ids)),
        "edges": int(edge_index.shape[1]),
        "node_feature_dim": int(x.shape[1]),
        "edge_feature_names": EDGE_FEATURE_NAMES,
        "label_count": len(label_names),
        "labeled_nodes": int((y >= 0).sum()),
        "embeddings": args.embeddings,
        "embedding_pool": args.embedding_pool,
        **emb_stats,
        "pseudo_label_margin": args.pseudo_label_margin,
        "keep_isolates": args.keep_isolates,
        "isolated_nodes_before_drop": isolated_before,
        "isolated_nodes_dropped": isolated_dropped,
        "temporal": temporal_stats,
    }
    save_graph(args.out, graph_obj, meta)


if __name__ == "__main__":
    main()
