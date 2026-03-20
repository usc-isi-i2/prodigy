# Temporal mention graph
# Splits the dataset into 80% history / 20% future by timestamp.
# history edge_index: A→B if A @mentioned B in first 80% of tweets
# future_edge_index:  NEW mention edges that appear only in the last 20%
# Node features and labels are reused from graph_mention/graph_data.pt

import glob
import os
import warnings
import argparse

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings('ignore')

# %%
## Config
parser = argparse.ArgumentParser(description="Build temporal mention graph.")
parser.add_argument("--strict-dates", "--strict_dates", dest="strict_dates",
                    action="store_true",
                    help="Fail on invalid date values instead of dropping.")
args = parser.parse_args()

path_pattern = "/project2/ll_774_951/midterm/*/*.csv"
SOURCE_GRAPH = "midterm/graph_mention/graph_data.pt"
OUTPUT_DIR   = "midterm/graph_temporal_mention"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# %%
## Load columns needed for mention edges + timestamp resolution
files = sorted(glob.glob(path_pattern))
print(f"Found {len(files)} CSV files")
chunks = []
for f in files:
    chunks.append(pd.read_csv(
        f, usecols=lambda c: c in ['userid', 'text', 'date', 'screen_name', 'display_name']
    ))
mdf = pd.concat(chunks, ignore_index=True)
print(f"Total rows: {len(mdf):,}  unique users: {mdf['userid'].nunique():,}")

# %%
## Parse timestamps and sort
mdf['date'] = mdf['date'].astype(str).str.strip()
mdf['timestamp'] = pd.to_datetime(
    mdf['date'], format='%a %b %d %H:%M:%S +0000 %Y', utc=True, errors='coerce'
)
invalid_ts = mdf['timestamp'].isna()
if invalid_ts.any():
    bad_count = int(invalid_ts.sum())
    bad_examples = mdf.loc[invalid_ts, 'date'].drop_duplicates().head(5).tolist()
    if args.strict_dates:
        raise ValueError(f"Invalid timestamp rows: {bad_count:,}. Examples: {bad_examples}")
    print(f"Invalid timestamp rows: {bad_count:,} / {len(mdf):,} (dropping)")
    mdf = mdf.loc[~invalid_ts].copy()

if len(mdf) == 0:
    raise ValueError("No valid timestamps after parsing 'date' column.")

mdf = mdf.sort_values('timestamp').reset_index(drop=True)

cutoff_idx  = int(len(mdf) * 0.80)
cutoff_time = mdf['timestamp'].iloc[cutoff_idx]
print(f"80th-percentile cutoff: row {cutoff_idx:,}  time={cutoff_time}")

history_df = mdf.iloc[:cutoff_idx].copy()
future_df  = mdf.iloc[cutoff_idx:].copy()
print(f"History rows: {len(history_df):,}  Future rows: {len(future_df):,}")

# %%
## Build username → userid lookup from full dataset
mdf['userid'] = pd.to_numeric(mdf['userid'], errors='coerce')
valid_users = mdf[mdf['userid'].notna()].copy()
valid_users['userid'] = valid_users['userid'].astype(int)

sn_to_uid = valid_users.dropna(subset=['screen_name']).drop_duplicates('screen_name') \
                        .set_index('screen_name')['userid'].to_dict()
dn_to_uid = valid_users.dropna(subset=['display_name']).drop_duplicates('display_name') \
                        .set_index('display_name')['userid'].to_dict()

def resolve_username(name):
    if name in sn_to_uid: return sn_to_uid[name]
    if name in dn_to_uid: return dn_to_uid[name]
    return None

# %%
## Helper: build directed edge set from a dataframe
def build_mention_edge_set(df):
    df = df.copy()
    df['userid'] = pd.to_numeric(df['userid'], errors='coerce')
    df = df.dropna(subset=['userid', 'text'])
    df['userid'] = df['userid'].astype(int)
    df['mentions'] = df['text'].str.findall(r'(?<!\w)@(\w+)')
    exploded = (
        df[['userid', 'mentions']]
        .explode('mentions')
        .dropna(subset=['mentions'])
    )
    exploded['tgt_uid'] = exploded['mentions'].apply(resolve_username)
    exploded = exploded.dropna(subset=['tgt_uid'])
    exploded['tgt_uid'] = exploded['tgt_uid'].astype(int)
    # Remove self-loops and deduplicate
    exploded = exploded[exploded['userid'] != exploded['tgt_uid']]
    edges = set(zip(exploded['userid'], exploded['tgt_uid']))
    return edges

print("Building history mention edge set...")
history_pairs = build_mention_edge_set(history_df)
print(f"History unique directed edges: {len(history_pairs):,}")

print("Building future mention edge set...")
future_pairs = build_mention_edge_set(future_df)
print(f"Future unique directed edges: {len(future_pairs):,}")

new_pairs = future_pairs - history_pairs
print(f"New future edges (not in history): {len(new_pairs):,}")

# %%
## Load node features from mention graph
print(f"\nLoading source graph from {SOURCE_GRAPH} ...")
raw = torch.load(SOURCE_GRAPH, map_location='cpu')

user_ids  = raw['user_ids']
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
## Convert edge sets to tensors (directed, both endpoints must be in node set)
def pairs_to_edge_index(pairs):
    srcs, dsts = [], []
    for u, v in pairs:
        if u in id_to_idx and v in id_to_idx:
            srcs.append(id_to_idx[u])
            dsts.append(id_to_idx[v])
    if not srcs:
        return torch.zeros((2, 0), dtype=torch.long)
    return torch.tensor([srcs, dsts], dtype=torch.long)

history_edge_index = pairs_to_edge_index(history_pairs)
future_edge_index  = pairs_to_edge_index(new_pairs)

print(f"\n=== FINAL GRAPH SUMMARY ===")
print(f"Nodes:            {num_nodes:,}")
print(f"History edges:    {history_edge_index.shape[1]:,}")
print(f"New future edges: {future_edge_index.shape[1]:,}")

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
