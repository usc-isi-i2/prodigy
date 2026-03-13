"""
Generate the Instagram mention graph (base structure, no text embeddings).

Nodes : Instagram accounts (handles)
Edges : directed poster → @mentioned account
Node x: account-level stat features (see data.feature_names for column names)

Output: mention_graph.pt
    data    – PyG Data(x, edge_index, feature_names)
    h2i     – dict[handle → node index]
    handles – list[str]

Next step:
    python attach_embeddings.py \\
        --graph      mention_graph.pt \\
        --embeddings user_embeddings_minilm.pt \\
        --out        mention_graph_minilm.pt
"""

import argparse
import gc
import glob
import os
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

warnings.filterwarnings("ignore")

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_PKL     = "/project2/ll_774_951/uk_ru/Instagram_Uk_ru/*.pkl"
DEFAULT_PARQUET = "instagram_uk_ru.parquet"
DEFAULT_OUT     = "mention_graph.pt"

FEATURE_COLS = [
    "subscriber_count",   # log1p followers
    "verified",           # 0/1
    "avg_favorites",      # log1p mean likes per post
    "avg_comments",       # log1p mean comments per post
    "avg_score",          # mean engagement score
    "avg_n_hashtags",     # mean hashtags per post
    "avg_n_mentions",     # mean @mentions per post
    "avg_has_media",      # fraction of posts with media
    "post_count",         # log1p total posts
]

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--parquet", default=DEFAULT_PARQUET,
                    help="Path to instagram parquet (faster than pkl files)")
parser.add_argument("--pkl",     default=DEFAULT_PKL,
                    help="Glob pattern for raw pkl files (fallback if no parquet)")
parser.add_argument("--out",     default=DEFAULT_OUT)
args = parser.parse_args()


# ── Load data ─────────────────────────────────────────────────────────────────
if os.path.exists(args.parquet):
    print(f"Loading parquet from {args.parquet}...")
    df = pd.read_parquet(args.parquet)
    print(f"  {len(df):,} rows")
else:
    print("Parquet not found, loading pickle files...")
    files = sorted(glob.glob(args.pkl))
    if not files:
        raise FileNotFoundError(
            f"No pkl files found at: {args.pkl}  "
            f"(and no parquet at {args.parquet})"
        )
    print(f"  Found {len(files)} files")
    dfs = []
    for f in files:
        dfs.append(pd.read_pickle(f))
        gc.collect()
    dfs = [pd.DataFrame(x) for x in dfs]
    df = pd.concat(dfs, ignore_index=True)
    del dfs
    gc.collect()
    print(f"  {len(df):,} rows")


# ── Parse account fields ──────────────────────────────────────────────────────
def safe_get(d, key, default=None):
    return d.get(key, default) if isinstance(d, dict) else default

df["handle"]           = df["account"].apply(lambda x: safe_get(x, "handle", ""))
df["verified"]         = df["account"].apply(lambda x: int(bool(safe_get(x, "verified", False))))
df["subscriber_count"] = df["account"].apply(lambda x: float(safe_get(x, "subscriberCount") or 0))


# ── Parse statistics ──────────────────────────────────────────────────────────
def get_stat(stats, key):
    if isinstance(stats, dict):
        actual = stats.get("actual", {})
        if isinstance(actual, dict):
            return float(actual.get(key) or 0)
    return 0.0

df["favoriteCount"] = df["statistics"].apply(lambda x: get_stat(x, "favoriteCount"))
df["commentCount"]  = df["statistics"].apply(lambda x: get_stat(x, "commentCount"))
df["score"]         = pd.to_numeric(df["score"], errors="coerce").fillna(0.0)


# ── Extract mentions / hashtags if not already in parquet ────────────────────
mention_re = r'(?:^|(?<=\s))@([A-Za-z0-9_](?:(?:[A-Za-z0-9_]|(?:\.(?!\.))){0,28}(?:[A-Za-z0-9_]))?)'
hashtag_re = r'(?:^|(?<=\s))#([A-Za-z_][A-Za-z0-9_]{0,29})'

df["description"] = df["description"].fillna("")

if "mentions" not in df.columns:
    df["mentions"] = (
        df["description"].str.extractall(mention_re)[0].str.lower()
        .groupby(level=0).agg(list).reindex(df.index)
        .apply(lambda x: x if isinstance(x, list) else [])
    )

if "hashtags" not in df.columns:
    df["hashtags"] = (
        df["description"].str.extractall(hashtag_re)[0].str.lower()
        .groupby(level=0).agg(list).reindex(df.index)
        .apply(lambda x: x if isinstance(x, list) else [])
    )

# Drop rows with empty handle
df = df[df["handle"].notna() & df["handle"].ne("")]

df["n_hashtags"] = df["hashtags"].apply(len)
df["n_mentions"]  = df["mentions"].apply(len)
df["has_media"]   = df["media"].apply(
    lambda x: int(bool(x) and str(x) not in ("", "[]", "nan"))
)

print(f"Unique handles (posters): {df['handle'].nunique():,}")


# ── Build node set: all posters + all mentioned accounts ─────────────────────
# Preserve original handle case so h2i matches user_embeddings_minilm.pt.
# Build a lowercase → canonical-handle map for case-insensitive mention lookup.
handle_lower_to_canonical: dict[str, str] = {}

for h in df["handle"].unique():
    if h:
        handle_lower_to_canonical[h.lower()] = h   # last write wins; fine for uniqueness

for mentions in df["mentions"]:
    for m in (mentions or []):
        ml = m.lower()
        if ml not in handle_lower_to_canonical:
            handle_lower_to_canonical[ml] = m      # mention-only node; use mention text as handle

handles: list[str] = sorted(handle_lower_to_canonical.values())
h2i: dict[str, int] = {h: i for i, h in enumerate(handles)}
N = len(handles)
print(f"Total nodes (posters + mentioned): {N:,}")


# ── Account-level aggregated features ────────────────────────────────────────
print("Building node features...")

user_agg = df.groupby("handle").agg(
    subscriber_count = ("subscriber_count", "max"),
    verified         = ("verified", "max"),
    avg_favorites    = ("favoriteCount", "mean"),
    avg_comments     = ("commentCount", "mean"),
    avg_score        = ("score", "mean"),
    avg_n_hashtags   = ("n_hashtags", "mean"),
    avg_n_mentions   = ("n_mentions", "mean"),
    avg_has_media    = ("has_media", "mean"),
    post_count       = ("handle", "count"),
).reset_index()

for col in ["subscriber_count", "avg_favorites", "avg_comments", "post_count"]:
    user_agg[col] = np.log1p(user_agg[col].fillna(0).clip(lower=0))

# Zero-initialise; only posters get real features (mention-only nodes stay 0)
X = np.zeros((N, len(FEATURE_COLS)), dtype=np.float64)
user_agg["node_idx"] = user_agg["handle"].map(h2i)
has_idx = user_agg["node_idx"].notna()
rows = user_agg.loc[has_idx, "node_idx"].astype(int).values
X[rows] = user_agg.loc[has_idx, FEATURE_COLS].values
matched = int(has_idx.sum())
print(f"  Nodes with features: {matched:,} / {N:,}  "
      f"({N - matched:,} mention-only nodes stay zero)")

# StandardScaler fitted only on nodes that have at least one non-zero feature
has_feats = X.any(axis=1)
scaler = StandardScaler()
X[has_feats] = scaler.fit_transform(X[has_feats])
X = X.astype(np.float32)
print(f"  Feature matrix: {X.shape}  columns: {FEATURE_COLS}")


# ── Build directed mention edges ──────────────────────────────────────────────
print("Building mention edges...")

# Explode mentions so each (poster, mention) pair is its own row, then map to node indices
edges = (
    df[["handle", "mentions"]]
    .explode("mentions")
    .dropna(subset=["mentions"])
)
edges = edges[edges["mentions"].ne("")]
edges["mentions"] = edges["mentions"].str.lower()
edges["mentions"] = edges["mentions"].map(handle_lower_to_canonical)
edges = edges.dropna(subset=["mentions"])
edges = edges[edges["handle"] != edges["mentions"]]   # drop self-loops
edges["src"] = edges["handle"].map(h2i)
edges["dst"] = edges["mentions"].map(h2i)
edges = edges.dropna(subset=["src", "dst"])
edges = edges.drop_duplicates(subset=["src", "dst"])

edge_index = torch.tensor(
    [edges["src"].astype(int).values, edges["dst"].astype(int).values],
    dtype=torch.long,
)
print(f"  Unique directed mention edges: {edge_index.shape[1]:,}")

degrees_out = torch.bincount(edge_index[0], minlength=N)
degrees_in  = torch.bincount(edge_index[1], minlength=N)
print(f"  Out-degree: mean={degrees_out.float().mean():.2f}  "
      f"max={degrees_out.max().item()}")
print(f"  In-degree:  mean={degrees_in.float().mean():.2f}  "
      f"max={degrees_in.max().item()}")
print(f"  Isolated nodes (out-deg=0): {(degrees_out == 0).sum().item():,}")


# ── Assemble and save ─────────────────────────────────────────────────────────
x = torch.from_numpy(X)
data = Data(x=x, edge_index=edge_index)
data.feature_names = FEATURE_COLS   # list[str] — one entry per column of data.x

torch.save({"data": data, "h2i": h2i, "handles": handles}, args.out)

print(f"\nSaved to {args.out}")
print(f"  Nodes:         {N:,}")
print(f"  Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")
print(f"  Edges:         {edge_index.shape[1]:,}")
