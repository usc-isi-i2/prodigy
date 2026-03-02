# Co-retweet graph
# Nodes: Twitter users (reuse node features from existing graph)
# Edges: user A and user B are connected if they retweeted the same account (undirected)
# MAX_GROUP_SIZE=None means k=infinity; set e.g. 500 to cap large popular accounts

import glob
import numpy as np
import pandas as pd
from itertools import combinations
import torch
import os
import warnings
warnings.filterwarnings('ignore')

## Config
path_pattern = "/project2/ll_774_951/midterm/*/*.csv"
EXISTING_GRAPH = "midterm/graph_mention/graph_data.pt"
OUTPUT_DIR = "midterm/graph_co_retweet"
MAX_GROUP_SIZE = 500

os.makedirs(OUTPUT_DIR, exist_ok=True)

## Load existing graph (reuse node features — only edges change)
print(f"Loading existing graph from {EXISTING_GRAPH}...")
existing = torch.load(EXISTING_GRAPH, map_location='cpu')
user_ids = existing['user_ids']
uid_to_idx = {uid: i for i, uid in enumerate(user_ids)}
print(f"Nodes: {len(user_ids):,}  Features: {existing['x'].shape[1]}")

## Load CSVs (only need userid + rt_userid)
print("Loading CSVs...")
files = sorted(glob.glob(path_pattern))
rt_df = pd.concat(
    [pd.read_csv(f, usecols=['userid', 'rt_userid'], low_memory=False) for f in files],
    ignore_index=True
)
rt_df = rt_df[rt_df['rt_userid'].notna()].copy()
rt_df['rt_userid'] = pd.to_numeric(rt_df['rt_userid'], errors='coerce')
rt_df = rt_df.dropna(subset=['rt_userid'])
rt_df['rt_userid'] = rt_df['rt_userid'].astype(int)
print(f"Retweet rows: {len(rt_df):,}  Unique retweeted accounts: {rt_df['rt_userid'].nunique():,}")

## Group sizes
rt_groups = rt_df.groupby('rt_userid')['userid'].apply(list)
group_sizes = rt_groups.apply(len)
print(f"Group sizes: mean={group_sizes.mean():.1f}  median={group_sizes.median():.1f}  max={group_sizes.max()}")
print(f"Groups >100: {(group_sizes > 100).sum()}  Groups >1000: {(group_sizes > 1000).sum()}")
if MAX_GROUP_SIZE is not None:
    print(f"Groups capped (>{MAX_GROUP_SIZE}): {(group_sizes > MAX_GROUP_SIZE).sum()} skipped")

## Build co-retweet edges
print(f"\nBuilding co-retweet edges (MAX_GROUP_SIZE={MAX_GROUP_SIZE})...")
edge_set = set()
for rt_uid, retweeters in rt_groups.items():
    retweeters = list(set(retweeters))
    if MAX_GROUP_SIZE is not None and len(retweeters) > MAX_GROUP_SIZE:
        continue
    for u, v in combinations(retweeters, 2):
        edge_set.add((min(u, v), max(u, v)))

print(f"Unique undirected edges: {len(edge_set):,}")

# Map to node indices, drop edges where either node isn't in the graph
src, dst = [], []
for u, v in edge_set:
    if u in uid_to_idx and v in uid_to_idx:
        ui, vi = uid_to_idx[u], uid_to_idx[v]
        src += [ui, vi]
        dst += [vi, ui]

edge_index = torch.tensor([src, dst], dtype=torch.long)
print(f"Directed edges (both directions): {edge_index.shape[1]:,}")

degrees = torch.bincount(edge_index[0], minlength=len(user_ids))
print(f"Degree: mean={degrees.float().mean():.1f}  median={degrees.float().median():.1f}  isolated={(degrees == 0).sum().item():,}")

## Save
torch.save({
    'x': existing['x'],
    'edge_index': edge_index,
    'user_ids': user_ids,
    'feature_names': existing['feature_names'],
    'y': existing['y'],
    'label_names': existing['label_names'],
}, f'{OUTPUT_DIR}/graph_data.pt')
print(f"\nSaved to {OUTPUT_DIR}/graph_data.pt")
