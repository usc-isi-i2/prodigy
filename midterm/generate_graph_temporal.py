# Temporal co-retweet graph
# Splits the single day (Oct 17, 2022) into 80% history / 20% future by row count.
# history edge_index: co-retweet pairs from first 80% of tweets (by timestamp)
# future_edge_index:  NEW co-retweet pairs that appear only in the last 20%
# Node features and labels are reused from graph_co_retweet/graph_data.pt

import glob
import os
import warnings
from itertools import combinations
import argparse

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings('ignore')

# %%
## Config
parser = argparse.ArgumentParser(description="Build temporal co-retweet graph.")
parser.add_argument(
    "--strict-dates",
    "--strict_dates",
    dest="strict_dates",
    action="store_true",
    help="Fail if any row has an invalid date value instead of dropping invalid rows.",
)
args = parser.parse_args()

path_pattern = "/project2/ll_774_951/midterm/*/*.csv"
SOURCE_GRAPH = "midterm/graph_co_retweet/graph_data.pt"
OUTPUT_DIR = "midterm/graph_temporal"
MAX_GROUP_SIZE = 500

os.makedirs(OUTPUT_DIR, exist_ok=True)

# %%
## Load only the columns we need
files = sorted(glob.glob(path_pattern))
print(f"Found {len(files)} CSV files")
chunks = []
for f in files:
    df = pd.read_csv(f, usecols=['userid', 'rt_userid', 'date'])
    chunks.append(df)
mdf = pd.concat(chunks, ignore_index=True)
print(f"Total rows: {len(mdf):,}  unique users: {mdf['userid'].nunique():,}")

# %%
## Parse timestamps and sort
# Some source rows can contain malformed date values (e.g. "1").
# Coerce parse failures to NaT, report them, and drop those rows.
mdf['date'] = mdf['date'].astype(str).str.strip()
mdf['timestamp'] = pd.to_datetime(
    mdf['date'],
    format='%a %b %d %H:%M:%S +0000 %Y',
    utc=True,
    errors='coerce',
)
invalid_ts = mdf['timestamp'].isna()
if invalid_ts.any():
    bad_count = int(invalid_ts.sum())
    bad_examples = mdf.loc[invalid_ts, 'date'].drop_duplicates().head(5).tolist()
    if args.strict_dates:
        raise ValueError(
            f"Invalid timestamp rows: {bad_count:,} / {len(mdf):,}. "
            f"Examples: {bad_examples}"
        )
    print(f"Invalid timestamp rows: {bad_count:,} / {len(mdf):,} (dropping)")
    print(f"Example invalid date values: {bad_examples}")
    mdf = mdf.loc[~invalid_ts].copy()

if len(mdf) == 0:
    raise ValueError("No valid timestamps after parsing 'date' column.")

mdf = mdf.sort_values('timestamp').reset_index(drop=True)

cutoff_idx = int(len(mdf) * 0.80)
cutoff_time = mdf['timestamp'].iloc[cutoff_idx]
print(f"80th-percentile cutoff: row {cutoff_idx:,}  time={cutoff_time}")

history_df = mdf.iloc[:cutoff_idx].copy()
future_df  = mdf.iloc[cutoff_idx:].copy()
print(f"History rows: {len(history_df):,}  Future rows: {len(future_df):,}")

# %%
## Helper: build edge set from a dataframe of (userid, rt_userid) rows
def build_edge_set(df):
    rt = df[df['rt_userid'].notna()][['userid', 'rt_userid']].copy()
    rt['userid']    = pd.to_numeric(rt['userid'],    errors='coerce')
    rt['rt_userid'] = pd.to_numeric(rt['rt_userid'], errors='coerce')
    rt = rt.dropna().astype(int)
    groups = rt.groupby('rt_userid')['userid'].apply(list)
    edges = set()
    for rt_uid, retweeters in groups.items():
        retweeters = list(set(retweeters))
        if MAX_GROUP_SIZE is not None and len(retweeters) > MAX_GROUP_SIZE:
            continue
        for u, v in combinations(retweeters, 2):
            edges.add((min(u, v), max(u, v)))
    return edges

print("Building history edge set...")
history_pairs = build_edge_set(history_df)
print(f"History unique undirected edges: {len(history_pairs):,}")

print("Building future edge set...")
future_pairs = build_edge_set(future_df)
print(f"Future unique undirected edges: {len(future_pairs):,}")

new_pairs = future_pairs - history_pairs
print(f"New future edges (not in history): {len(new_pairs):,}")

# %%
## Load node features from existing co-retweet graph
print(f"\nLoading source graph from {SOURCE_GRAPH} ...")
raw = torch.load(SOURCE_GRAPH, map_location='cpu')

user_ids = raw['user_ids']          # numpy array of integer user IDs
num_nodes = len(user_ids)
print(f"Source graph: {num_nodes:,} nodes")


def _to_int_user_id(uid):
    """Best-effort conversion to integer user ID; returns None if invalid."""
    if isinstance(uid, (int, np.integer)):
        return int(uid)
    if isinstance(uid, (float, np.floating)):
        if np.isfinite(uid):
            return int(uid)
        return None
    s = str(uid).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return None
    v = pd.to_numeric(s, errors='coerce')
    if pd.isna(v):
        return None
    return int(v)


# Build index only for valid numeric user IDs in the source graph.
id_to_idx = {}
invalid_uid_count = 0
duplicate_uid_count = 0
invalid_uid_examples = []
for i, uid in enumerate(user_ids):
    u = _to_int_user_id(uid)
    if u is None:
        invalid_uid_count += 1
        if len(invalid_uid_examples) < 10:
            invalid_uid_examples.append(repr(uid))
        continue
    if u in id_to_idx:
        duplicate_uid_count += 1
        continue
    id_to_idx[u] = i

print(
    f"Usable source user_ids: {len(id_to_idx):,} / {num_nodes:,} "
    f"(invalid: {invalid_uid_count:,}, duplicates: {duplicate_uid_count:,})"
)
if invalid_uid_examples:
    print("Example invalid source user_ids:", invalid_uid_examples)

# %%
## Convert edge sets to tensors (only keep edges where both endpoints are in node set)
def pairs_to_edge_index(pairs):
    srcs, dsts = [], []
    for u, v in pairs:
        if u in id_to_idx and v in id_to_idx:
            ui, vi = id_to_idx[u], id_to_idx[v]
            srcs += [ui, vi]
            dsts += [vi, ui]
    if not srcs:
        return torch.zeros((2, 0), dtype=torch.long)
    return torch.tensor([srcs, dsts], dtype=torch.long)

history_edge_index = pairs_to_edge_index(history_pairs)
future_edge_index  = pairs_to_edge_index(new_pairs)

print(f"\n=== FINAL GRAPH SUMMARY ===")
print(f"Nodes:              {num_nodes:,}")
print(f"History edges:      {history_edge_index.shape[1]:,}  ({history_edge_index.shape[1]//2:,} undirected)")
print(f"New future edges:   {future_edge_index.shape[1]:,}  ({future_edge_index.shape[1]//2:,} undirected)")

# %%
## Save
out_path = os.path.join(OUTPUT_DIR, 'graph_data.pt')
torch.save({
    'x':                raw['x'],
    'y':                raw['y'],
    'label_names':      raw['label_names'],
    'feature_names':    raw.get('feature_names', []),
    'user_ids':         user_ids,
    'edge_index':       history_edge_index,
    'future_edge_index': future_edge_index,
}, out_path)
print(f"\nSaved to {out_path}")
