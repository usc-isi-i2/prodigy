"""
Generate labeled versions of mention_graph.pt from the Instagram parquet data.

Labels generated (one output file each):
  mention_graph_language.pt     — ru=0, uk=1, other=2  (majority languageCode per account)
  mention_graph_verified.pt     — not verified=0, verified=1
  mention_graph_overperformer.pt — under=0, normal=1, over=2  (tertile of actual/expected likes ratio)
  mention_graph_content_type.pt — video=0, photo=1, album=2   (majority post type per account)

Accounts not present in the parquet (e.g. zero-padded edge-only nodes) get label -1.

Usage:
  python generate_labeled_graphs.py \
      --parquet /path/to/instagram.parquet \
      --graph   mention_graph.pt \
      --out_dir .
"""

import argparse
import ast
import os

import numpy as np
import pandas as pd
import torch
from collections import Counter


# ── helpers ───────────────────────────────────────────────────────────────────

def parse_account(val):
    if isinstance(val, dict):
        return val
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except Exception:
            return {}
    return {}


def parse_statistics(val):
    if isinstance(val, dict):
        return val
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except Exception:
            return {}
    return {}


def majority(series):
    c = Counter(series.dropna())
    return c.most_common(1)[0][0] if c else None


def save_graph(base_ckpt, h2i, handles, y, label_names, out_path):
    data = base_ckpt["data"]
    data.y = y
    data.label_names = label_names
    torch.save({"data": data, "h2i": h2i, "handles": handles}, out_path)
    counts = {label_names[i]: (y == i).sum().item() for i in range(len(label_names))}
    labeled = (y >= 0).sum().item()
    print(f"  Saved {out_path}  |  {labeled}/{len(y)} labeled  |  {counts}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True, help="Path to Instagram parquet file")
    parser.add_argument("--graph",   default="mention_graph.pt", help="Input graph checkpoint")
    parser.add_argument("--out_dir", default=".", help="Output directory")
    args = parser.parse_args()

    # Load graph
    print(f"Loading graph from {args.graph}...")
    ckpt = torch.load(args.graph, map_location="cpu")
    data, h2i, handles = ckpt["data"], ckpt["h2i"], ckpt["handles"]
    num_nodes = data.x.shape[0]
    print(f"  {num_nodes} nodes, handles: {len(handles)}")

    # Load parquet
    print(f"\nLoading parquet from {args.parquet}...")
    df = pd.read_parquet(args.parquet)
    print(f"  {len(df):,} rows, {df['handle'].nunique():,} unique handles")

    # Parse account field to extract verified
    print("Parsing account fields...")
    df["_account"] = df["account"].apply(parse_account)
    df["_verified"] = df["_account"].apply(lambda a: bool(a.get("verified", False)))

    # Parse statistics to get overperformance ratio per post
    print("Parsing statistics fields...")
    df["_stats"] = df["statistics"].apply(parse_statistics)
    def overperf_ratio(s):
        actual = s.get("actual", {})
        expected = s.get("expected", {})
        a = float(actual.get("favoriteCount") or 0)
        e = float(expected.get("favoriteCount") or 1)
        return a / e if e > 0 else 1.0
    df["_ratio"] = df["_stats"].apply(overperf_ratio)

    # Aggregate per handle
    print("Aggregating per handle...")
    agg = df.groupby("handle").agg(
        lang        = ("languageCode", majority),
        verified    = ("_verified",    "max"),
        avg_ratio   = ("_ratio",       "mean"),
        content     = ("type",         majority),
    ).reset_index()
    handle_to_row = {row["handle"]: row for _, row in agg.iterrows()}

    # ── 1. Language ───────────────────────────────────────────────────────────
    print("\n[1] language (ru=0, uk=1, other=2)")
    y = torch.full((num_nodes,), -1, dtype=torch.long)
    for handle, idx in h2i.items():
        if idx >= num_nodes:
            continue
        row = handle_to_row.get(handle)
        if row is None:
            continue
        lang = str(row["lang"]).lower() if row["lang"] else "und"
        if lang == "ru":
            y[idx] = 0
        elif lang == "uk":
            y[idx] = 1
        else:
            y[idx] = 2
    out = os.path.join(args.out_dir, "mention_graph_language.pt")
    save_graph(ckpt, h2i, handles, y, ["ru", "uk", "other"], out)

    # ── 2. Verified ───────────────────────────────────────────────────────────
    print("\n[2] verified (0=no, 1=yes)")
    y = torch.full((num_nodes,), -1, dtype=torch.long)
    for handle, idx in h2i.items():
        if idx >= num_nodes:
            continue
        row = handle_to_row.get(handle)
        if row is None:
            continue
        y[idx] = int(bool(row["verified"]))
    out = os.path.join(args.out_dir, "mention_graph_verified.pt")
    save_graph(ckpt, h2i, handles, y, ["not_verified", "verified"], out)

    # ── 3. Overperformer ──────────────────────────────────────────────────────
    print("\n[3] overperformer (tertile: under=0, normal=1, over=2)")
    ratios = []
    handle_list = []
    for handle, idx in h2i.items():
        if idx >= num_nodes:
            continue
        row = handle_to_row.get(handle)
        if row is not None:
            ratios.append(row["avg_ratio"])
            handle_list.append((handle, idx))

    ratios_arr = np.array(ratios)
    t33, t66 = np.percentile(ratios_arr, [33, 66])
    y = torch.full((num_nodes,), -1, dtype=torch.long)
    for (handle, idx), ratio in zip(handle_list, ratios):
        if ratio <= t33:
            y[idx] = 0
        elif ratio <= t66:
            y[idx] = 1
        else:
            y[idx] = 2
    out = os.path.join(args.out_dir, "mention_graph_overperformer.pt")
    save_graph(ckpt, h2i, handles, y, ["underperformer", "normal", "overperformer"], out)

    # ── 4. Content type ───────────────────────────────────────────────────────
    print("\n[4] content_type (video=0, photo=1, album=2)")
    y = torch.full((num_nodes,), -1, dtype=torch.long)
    type_map = {"video": 0, "photo": 1, "album": 2}
    for handle, idx in h2i.items():
        if idx >= num_nodes:
            continue
        row = handle_to_row.get(handle)
        if row is None:
            continue
        t = str(row["content"]).lower() if row["content"] else ""
        label = type_map.get(t)
        if label is not None:
            y[idx] = label
    out = os.path.join(args.out_dir, "mention_graph_content_type.pt")
    save_graph(ckpt, h2i, handles, y, ["video", "photo", "album"], out)

    print("\nDone.")


if __name__ == "__main__":
    main()
