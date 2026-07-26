#!/usr/bin/env python3
"""Reconstruct a user->user retweet edge list for TwiBot-20 (Format22).

TwiBot-20 has no native retweet edges: ``edge.csv`` only has ``follow``,
``friend`` and ``post`` (user->tweet), and tweet nodes in ``node.json`` are just
``{id, text}``. Retweets are recoverable from tweet text (~32% start with
``RT @handle:``), and the poster of each tweet is known via ``post`` edges.

This script builds directed **retweeter -> retweeted** edges:

    user --post--> tweet(text = "RT @handle: ...")   and   handle -> rt_userid
    =>  edge (userid, rt_userid), weighted by the number of such retweets.

Only retweets whose target handle resolves to a user in the dataset are kept
(handles are matched case-insensitively against the ``username`` field). Self
-retweets (userid == rt_userid) are dropped.

Output: a Parquet with columns ``userid``, ``rt_userid``, ``n_retweets``.
Intermediates deliberately live outside ``…/parquet/`` so the bio-embedding
pipeline (which globs ``parquet/**``) never ingests them.

Example
-------
    python extract_retweet_edges.py \
        --node-json /dataMeR1/phil/data/twibot20/raw/Twibot-20/node.json \
        --edge-csv  /dataMeR1/phil/data/twibot20/raw/Twibot-20/edge.csv \
        --out       /dataMeR1/phil/data/twibot20/graph_build/retweet_edges.parquet
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from collections import Counter
from pathlib import Path

import ijson
import pyarrow as pa
import pyarrow.parquet as pq

# TwiBot-20 stores classic retweets as text beginning "RT @<handle>: ...".
RT_PATTERN = re.compile(r"^RT @([A-Za-z0-9_]+):")


def build_username_map(node_json: Path) -> dict[str, str]:
    """handle (lowercased) -> userid, over all user nodes."""
    uname: dict[str, str] = {}
    with open(node_json, "rb") as handle:
        for obj in ijson.items(handle, "item"):
            node_id = str(obj.get("id", ""))
            if not node_id.startswith("u"):
                continue
            username = str(obj.get("username") or "").strip().lower()
            if username:
                uname[username] = node_id
    return uname


def build_retweet_tweet_map(node_json: Path, uname: dict[str, str]) -> dict[str, str]:
    """tweetid -> rt_userid, for retweets whose target is an in-set user."""
    rt_tweet: dict[str, str] = {}
    with open(node_json, "rb") as handle:
        for obj in ijson.items(handle, "item"):
            node_id = str(obj.get("id", ""))
            if not node_id.startswith("t"):
                continue
            text = str(obj.get("text") or "").lstrip()
            if not text.startswith("RT @"):
                continue
            match = RT_PATTERN.match(text)
            if match is None:
                continue
            target = uname.get(match.group(1).lower())
            if target is not None:
                rt_tweet[node_id] = target
    return rt_tweet


def aggregate_edges(edge_csv: Path, rt_tweet: dict[str, str]) -> Counter[tuple[str, str]]:
    """Count directed (retweeter, retweeted) pairs from `post` edges."""
    pair_counts: Counter[tuple[str, str]] = Counter()
    with open(edge_csv, newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        # Expect: source_id, relation, target_id
        for row in reader:
            if len(row) != 3 or row[1] != "post":
                continue
            retweeter, tweetid = row[0], row[2]
            target = rt_tweet.get(tweetid)
            if target is None or target == retweeter:
                continue  # not an in-set retweet, or a self-retweet
            pair_counts[(retweeter, target)] += 1
    return pair_counts


def write_edges(pair_counts: Counter[tuple[str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Deterministic order for reproducible artifacts.
    items = sorted(pair_counts.items())
    userids = [src for (src, _), _ in items]
    rt_userids = [dst for (_, dst), _ in items]
    counts = [c for _, c in items]
    table = pa.table(
        {
            "userid": pa.array(userids, type=pa.string()),
            "rt_userid": pa.array(rt_userids, type=pa.string()),
            "n_retweets": pa.array(counts, type=pa.int64()),
        }
    )
    pq.write_table(table, out_path, compression="snappy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node-json", default="/dataMeR1/phil/data/twibot20/raw/Twibot-20/node.json")
    parser.add_argument("--edge-csv", default="/dataMeR1/phil/data/twibot20/raw/Twibot-20/edge.csv")
    parser.add_argument("--out", default="/dataMeR1/phil/data/twibot20/graph_build/retweet_edges.parquet")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    node_json = Path(args.node_json)
    edge_csv = Path(args.edge_csv)
    out_path = Path(args.out)

    if not node_json.exists():
        print(f"ERROR: node.json not found: {node_json}", file=sys.stderr)
        return 1
    if not edge_csv.exists():
        print(f"ERROR: edge.csv not found: {edge_csv}", file=sys.stderr)
        return 1
    if out_path.exists() and not args.overwrite:
        print(f"ERROR: output exists (use --overwrite): {out_path}", file=sys.stderr)
        return 1

    start = time.time()
    print("building username -> userid map ...", flush=True)
    uname = build_username_map(node_json)
    print(f"  usernames={len(uname):,} ({time.time() - start:.0f}s)", flush=True)

    print("scanning tweets for in-set retweets ...", flush=True)
    rt_tweet = build_retweet_tweet_map(node_json, uname)
    print(f"  in-set retweet tweets={len(rt_tweet):,} ({time.time() - start:.0f}s)", flush=True)

    print("aggregating directed retweet edges from post edges ...", flush=True)
    pair_counts = aggregate_edges(edge_csv, rt_tweet)
    total_events = sum(pair_counts.values())
    print(
        f"  distinct edges={len(pair_counts):,}  retweet events={total_events:,} "
        f"({time.time() - start:.0f}s)",
        flush=True,
    )

    write_edges(pair_counts, out_path)
    print(f"wrote {len(pair_counts):,} edges -> {out_path} ({time.time() - start:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
