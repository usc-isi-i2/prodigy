"""
Generate Twitter retweet graph in instagram_mention format.
Nodes : Twitter users (screen_name)
Edges : directed A → B if A retweeted B
Node x: 9 account-level stat features (same names as Instagram graph)

Output: retweet_graph.pt
    data    – PyG Data(x, edge_index, feature_names)
    h2i     – dict[screen_name → node index]
    handles – list[str]

Next steps:
    python ../ukr_ru/instagram/attach_embeddings.py \\
        --graph      retweet_graph.pt \\
        --embeddings user_embeddings_minilm.pt \\
        --out        retweet_graph_minilm.pt

    python generate_pseudo_labels.py \\
        --graph retweet_graph.pt --out_dir .
"""

import argparse
import glob
import os
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_CSV = "/project2/ll_774_951/midterm/*/*.csv"
DEFAULT_OUT = "retweet_graph.pt"

# Same feature names as Instagram — keeps feature space interpretable for transfer
FEATURE_COLS = [
    "subscriber_count",   # log1p followers_count
    "verified",           # 0/1
    "avg_favorites",      # log1p mean rt_fav_count per tweet
    "avg_comments",       # log1p mean rt_reply_count per tweet
    "avg_score",          # mean sent_vader
    "avg_n_hashtags",     # mean hashtags per tweet
    "avg_n_mentions",     # mean @mentions per tweet
    "avg_has_media",      # fraction of tweets with media
    "post_count",         # log1p statuses_count
]

USECOLS = [
    "screen_name", "userid", "rt_userid", "rt_screen",
    "followers_count", "verified", "statuses_count",
    "rt_fav_count", "rt_reply_count", "sent_vader",
    "hashtag", "mentionsn", "media_urls",
]

parser = argparse.ArgumentParser()
parser.add_argument("--csv", default=DEFAULT_CSV)
parser.add_argument("--out", default=DEFAULT_OUT)
args = parser.parse_args()

# ── Load CSVs ─────────────────────────────────────────────────────────────────
print("Loading CSV files...")
files = sorted(glob.glob(args.csv))
if not files:
    raise FileNotFoundError(f"No CSV files found at: {args.csv}")
print(f"  Found {len(files)} files")

chunks = []
for f in files:
    try:
        avail = pd.read_csv(f, nrows=0).columns.tolist()
        usecols = [c for c in USECOLS if c in avail]
        chunks.append(pd.read_csv(f, usecols=usecols, low_memory=False, on_bad_lines="skip"))
    except Exception as e:
        print(f"  Skipping {os.path.basename(f)}: {e}")

df = pd.concat(chunks, ignore_index=True)
del chunks
print(f"  {len(df):,} rows")

# ── Clean ─────────────────────────────────────────────────────────────────────
df = df[df["screen_name"].notna() & df["screen_name"].ne("")]
df["screen_name"] = df["screen_name"].str.lower()

print(f"Unique screen_names: {df['screen_name'].nunique():,}")

# ── Build node set ────────────────────────────────────────────────────────────
print("Building node set...")
all_handles = set(df["screen_name"].unique())
if "rt_screen" in df.columns:
    all_handles.update(df["rt_screen"].dropna().str.lower().unique())

handles: list[str] = sorted(all_handles)
h2i: dict[str, int] = {h: i for i, h in enumerate(handles)}
N = len(handles)
print(f"  Total nodes (tweeters + retweeted): {N:,}")

# ── Account-level features ────────────────────────────────────────────────────
print("Building node features...")

def parse_list_col(series):
    """Count items in a column that may be a stringified list or NaN."""
    def count_items(val):
        if pd.isna(val) or str(val).strip() in ("", "[]", "nan"):
            return 0
        val = str(val).strip("[]").replace("'", "").replace('"', '')
        return len([x for x in val.split(",") if x.strip()])
    return series.apply(count_items)

df["n_hashtags"] = parse_list_col(df["hashtag"]) if "hashtag" in df.columns else 0
df["n_mentions"]  = parse_list_col(df["mentionsn"]) if "mentionsn" in df.columns else 0
df["has_media"]   = df["media_urls"].apply(
    lambda x: int(pd.notna(x) and str(x).strip() not in ("", "[]", "nan"))
) if "media_urls" in df.columns else 0

agg_dict = dict(
    subscriber_count = ("followers_count",  "max"),
    verified         = ("verified",         "max"),
    avg_favorites    = ("rt_fav_count",     "mean"),
    avg_comments     = ("rt_reply_count",   "mean"),
    avg_score        = ("sent_vader",       "mean"),
    avg_n_hashtags   = ("n_hashtags",       "mean"),
    avg_n_mentions   = ("n_mentions",       "mean"),
    avg_has_media    = ("has_media",        "mean"),
    post_count       = ("statuses_count",   "max"),
)
# Only aggregate columns that exist
agg_dict = {k: v for k, v in agg_dict.items() if v[0] in df.columns}

user_agg = df.groupby("screen_name").agg(**agg_dict).reset_index()

# Fill any missing feature columns with 0
for col in FEATURE_COLS:
    if col not in user_agg.columns:
        user_agg[col] = 0.0

for col in ["subscriber_count", "avg_favorites", "avg_comments", "post_count"]:
    user_agg[col] = np.log1p(user_agg[col].fillna(0).clip(lower=0))

X = np.zeros((N, len(FEATURE_COLS)), dtype=np.float64)
user_agg["node_idx"] = user_agg["screen_name"].map(h2i)
has_idx = user_agg["node_idx"].notna()
rows = user_agg.loc[has_idx, "node_idx"].astype(int).values
X[rows] = user_agg.loc[has_idx, FEATURE_COLS].values
matched = int(has_idx.sum())
print(f"  Nodes with features: {matched:,} / {N:,}  ({N-matched:,} retweet-only nodes stay zero)")

has_feats = X.any(axis=1)
scaler = StandardScaler()
X[has_feats] = scaler.fit_transform(X[has_feats])
X = X.astype(np.float32)
print(f"  Feature matrix: {X.shape}  columns: {FEATURE_COLS}")

# ── Build retweet edges ───────────────────────────────────────────────────────
print("Building retweet edges (explode + map)...")

if "rt_screen" in df.columns:
    edges = df[["screen_name", "rt_screen"]].copy()
    edges = edges.dropna(subset=["rt_screen"])
    edges["rt_screen"] = edges["rt_screen"].str.lower()
    edges = edges[edges["screen_name"] != edges["rt_screen"]]
    edges["src"] = edges["screen_name"].map(h2i)
    edges["dst"] = edges["rt_screen"].map(h2i)
elif "rt_userid" in df.columns:
    # Fallback: resolve rt_userid → screen_name via uid→sn map
    print("  rt_screen not available, resolving via rt_userid...")
    uid_to_sn = df.dropna(subset=["userid", "screen_name"]).drop_duplicates("userid") \
                  .set_index("userid")["screen_name"].to_dict()
    edges = df[["screen_name", "rt_userid"]].copy().dropna(subset=["rt_userid"])
    edges["rt_userid"] = pd.to_numeric(edges["rt_userid"], errors="coerce")
    edges = edges.dropna(subset=["rt_userid"])
    edges["rt_screen"] = edges["rt_userid"].astype(int).map(uid_to_sn)
    edges = edges.dropna(subset=["rt_screen"])
    edges["rt_screen"] = edges["rt_screen"].str.lower()
    edges = edges[edges["screen_name"] != edges["rt_screen"]]
    edges["src"] = edges["screen_name"].map(h2i)
    edges["dst"] = edges["rt_screen"].map(h2i)
else:
    raise ValueError("Neither rt_screen nor rt_userid found in data.")

edges = edges.dropna(subset=["src", "dst"])
edges = edges.drop_duplicates(subset=["src", "dst"])
print(f"  Pairs after dedup: {len(edges):,}")

edge_index = torch.tensor(
    [edges["src"].astype(int).values, edges["dst"].astype(int).values],
    dtype=torch.long,
)
print(f"  Unique directed retweet edges: {edge_index.shape[1]:,}")

degrees_out = torch.bincount(edge_index[0], minlength=N)
degrees_in  = torch.bincount(edge_index[1], minlength=N)
print(f"  Out-degree: mean={degrees_out.float().mean():.2f}  max={degrees_out.max().item()}")
print(f"  In-degree:  mean={degrees_in.float().mean():.2f}  max={degrees_in.max().item()}")
print(f"  Isolated nodes (out-deg=0): {(degrees_out == 0).sum().item():,}")

# ── Save ──────────────────────────────────────────────────────────────────────
data = Data(x=torch.from_numpy(X), edge_index=edge_index)
data.feature_names = FEATURE_COLS

torch.save({"data": data, "h2i": h2i, "handles": handles}, args.out)
print(f"\nSaved to {args.out}")
print(f"  Nodes:         {N:,}")
print(f"  Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")
print(f"  Edges:         {edge_index.shape[1]:,}")
