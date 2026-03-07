# Direct retweet graph
# Nodes: Twitter users
# Edges: user A retweeted user B (directed, A → B)
# Node features and labels reused from graph_co_retweet/graph_data.pt

import glob
import os
import warnings

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings('ignore')

# %%
## Config
path_pattern = "/project2/ll_774_951/midterm/*/*.csv"
SOURCE_GRAPH = "midterm/graph_co_retweet/graph_data.pt"
OUTPUT_DIR = "midterm/graph_retweet"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# %%
## Load only the columns we need
files = sorted(glob.glob(path_pattern))
print(f"Found {len(files)} CSV files")
chunks = []
for f in files:
    chunks.append(pd.read_csv(f, usecols=['userid', 'rt_userid']))
mdf = pd.concat(chunks, ignore_index=True)
print(f"Total rows: {len(mdf):,}")

# %%
## Build direct retweet edge list
rt_df = mdf[mdf['rt_userid'].notna()][['userid', 'rt_userid']].copy()
rt_df['userid']    = pd.to_numeric(rt_df['userid'],    errors='coerce')
rt_df['rt_userid'] = pd.to_numeric(rt_df['rt_userid'], errors='coerce')
rt_df = rt_df.dropna().astype(int)
rt_df = rt_df.drop_duplicates()
print(f"Unique directed retweet edges: {len(rt_df):,}")
print(f"Unique retweeting users:  {rt_df['userid'].nunique():,}")
print(f"Unique retweeted accounts: {rt_df['rt_userid'].nunique():,}")

# %%
## Load node features from co-retweet graph
print(f"\nLoading source graph from {SOURCE_GRAPH} ...")
raw = torch.load(SOURCE_GRAPH, map_location='cpu')

user_ids = raw['user_ids']
num_nodes = len(user_ids)
print(f"Source graph: {num_nodes:,} nodes")

id_to_idx = {}
for i, uid in enumerate(user_ids):
    try:
        id_to_idx[int(uid)] = i
    except (ValueError, TypeError):
        pass
print(f"Usable node IDs: {len(id_to_idx):,}")

# %%
## Convert to edge_index (keep only edges where both endpoints are in node set)
srcs, dsts = [], []
for _, row in rt_df.iterrows():
    u, v = int(row['userid']), int(row['rt_userid'])
    if u in id_to_idx and v in id_to_idx:
        srcs.append(id_to_idx[u])
        dsts.append(id_to_idx[v])

edge_index = torch.tensor([srcs, dsts], dtype=torch.long)

print(f"\n=== FINAL GRAPH SUMMARY ===")
print(f"Nodes:          {num_nodes:,}")
print(f"Directed edges: {edge_index.shape[1]:,}")
print(f"Avg out-degree: {edge_index.shape[1] / num_nodes:.2f}")

# %%
## Save
out_path = os.path.join(OUTPUT_DIR, 'graph_data.pt')
torch.save({
    'x':             raw['x'],
    'y':             raw['y'],
    'label_names':   raw['label_names'],
    'feature_names': raw.get('feature_names', []),
    'user_ids':      user_ids,
    'edge_index':    edge_index,
}, out_path)
print(f"\nSaved to {out_path}")
