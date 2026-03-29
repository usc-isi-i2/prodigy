import argparse
import glob
import os
import warnings
import csv
import time

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_CSV = "/project2/ll_774_951/uk_ru/twitter/data/*/*.csv"
DEFAULT_OUT = "retweet_graph.pt"

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
parser.add_argument("--max_files", type=int, default=None, help="Limit number of CSV files to load")
args = parser.parse_args()

# ── NEW READER ────────────────────────────────────────────────────────────────
def load_interleaved_csv(filepath):
    """
    Loads CSVs where each record can be split across two consecutive rows.
    """
    main_rows, sub_rows = [], []
    skipped_rows = 0

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
            sub_header_raw = next(reader)
        except StopIteration:
            return pd.DataFrame()

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        if sub_header_raw is not None:
            next(reader)

        pending_main = None
        for row in reader:
            if not row: continue
            
            if len(row) == 66:
                if pending_main is not None:
                    main_rows.append(pending_main)
                    sub_rows.append([""] * 11)
                pending_main = row
            elif len(row) == 11:
                if pending_main is not None:
                    main_rows.append(pending_main)
                    sub_rows.append(row)
                    pending_main = None
                else:
                    skipped_rows += 1
            else:
                skipped_rows += 1

        if pending_main is not None:
            main_rows.append(pending_main)
            sub_rows.append([""] * 11)

    sub_cols = ["sub_extra", "state", "country", "rt_state", "rt_country",
                "qtd_state", "qtd_country", "norm_country",
                "norm_rt_country", "norm_qtd_country", "acc_age"]

    df_main = pd.DataFrame(main_rows, columns=header)
    df_sub = pd.DataFrame(sub_rows, columns=sub_cols).drop(columns=["sub_extra"], errors="ignore")
    df = pd.concat([df_main.reset_index(drop=True), df_sub.reset_index(drop=True)], axis=1)

    # Logging internal reconstruct stats
    # print(f"    [Parser] Reconstructed {len(df)} records. (Skipped {skipped_rows} malformed rows)")
    
    return df

# ── Load CSVs ─────────────────────────────────────────────────────────────────
print(f"{'='*60}\nSTEP 1: Loading Data\n{'='*60}")
files = sorted(glob.glob(args.csv))
if args.max_files is not None:
    files = files[:args.max_files]

if not files:
    raise FileNotFoundError(f"No CSV files found at: {args.csv}")

print(f"Found {len(files)} files to process...")

chunks = []
start_time = time.time()
for i, f in enumerate(files):
    try:
        print(f"Processing file {i+1}/{len(files)}: {os.path.basename(f)}")
        df_file = load_interleaved_csv(f)
        if df_file.empty:
            continue
            
        usecols = [c for c in USECOLS if c in df_file.columns]
        if not usecols:
            continue

        df_sel = df_file[usecols].copy()
        chunks.append(df_sel)
        
        if (i + 1) % 10 == 0 or (i + 1) == len(files):
            elapsed = time.time() - start_time
            print(f"  Processed {i+1}/{len(files)} files... ({elapsed:.1f}s)")

    except Exception as e:
        print(f"  [ERROR] Skipping {os.path.basename(f)}: {e}")

if not chunks:
    raise FileNotFoundError(f"No valid CSV content found.")

df = pd.concat(chunks, ignore_index=True)
del chunks
print(f"Total raw rows loaded: {len(df):,}")

# ── Clean ─────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}\nSTEP 2: Cleaning and Node Mapping\n{'='*60}")
initial_len = len(df)
df = df[df["screen_name"].notna() & df["screen_name"].ne("")]
df["screen_name"] = df["screen_name"].str.lower()
print(f"Dropped {initial_len - len(df):,} rows with missing screen_names.")

# ── Build node set ───────────────────────────────────────────────────────────
print("Building node set...")
all_handles = set(df["screen_name"].unique())
if "rt_screen" in df.columns:
    rt_handles = set(df["rt_screen"].dropna().str.lower().unique())
    print(f"  Unique tweeters: {len(all_handles):,}")
    print(f"  Unique retweeted targets: {len(rt_handles):,}")
    all_handles.update(rt_handles)

handles = sorted(all_handles)
h2i = {h: i for i, h in enumerate(handles)}
N = len(handles)
print(f"  Total unique nodes in graph: {N:,}")

# ── Account-level features ────────────────────────────────────────────────────
print(f"\n{'='*60}\nSTEP 3: Feature Engineering\n{'='*60}")

def parse_list_col(series):
    def count_items(val):
        if pd.isna(val) or str(val).strip() in ("", "[]", "nan"):
            return 0
        val = str(val).strip("[]").replace("'", "").replace('"', '')
        return len([x for x in val.split(",") if x.strip()])
    return series.apply(count_items)

print("Calculating averages and counts per user...")
df["n_hashtags"] = parse_list_col(df["hashtag"]) if "hashtag" in df.columns else 0
df["n_mentions"] = parse_list_col(df["mentionsn"]) if "mentionsn" in df.columns else 0
df["has_media"]  = df["media_urls"].apply(
    lambda x: int(pd.notna(x) and str(x).strip() not in ("", "[]", "nan"))
) if "media_urls" in df.columns else 0

for col in ["followers_count", "verified", "rt_fav_count", "rt_reply_count", "sent_vader", "statuses_count"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

agg_dict = {
    "subscriber_count": ("followers_count", "max"),
    "verified": ("verified", "max"),
    "avg_favorites": ("rt_fav_count", "mean"),
    "avg_comments": ("rt_reply_count", "mean"),
    "avg_score": ("sent_vader", "mean"),
    "avg_n_hashtags": ("n_hashtags", "mean"),
    "avg_n_mentions": ("n_mentions", "mean"),
    "avg_has_media": ("has_media", "mean"),
    "post_count": ("statuses_count", "max"),
}
agg_dict = {k: v for k, v in agg_dict.items() if v[0] in df.columns}

user_agg = df.groupby("screen_name").agg(**agg_dict).reset_index()

# Log missing feature coverage
for col in FEATURE_COLS:
    if col not in user_agg.columns:
        user_agg[col] = 0.0
    # Prevent NaNs from sparse/missing aggregates from leaking into node features
    user_agg[col] = user_agg[col].fillna(0.0)

print("Applying log transformations and scaling...")
for col in ["subscriber_count", "avg_favorites", "avg_comments", "post_count"]:
    user_agg[col] = np.log1p(user_agg[col].fillna(0).clip(lower=0))

X = np.zeros((N, len(FEATURE_COLS)), dtype=np.float64)
user_agg["node_idx"] = user_agg["screen_name"].map(h2i)
has_idx = user_agg["node_idx"].notna()
rows = user_agg.loc[has_idx, "node_idx"].astype(int).values
X[rows] = user_agg.loc[has_idx, FEATURE_COLS].values

matched = int(has_idx.sum())
print(f"  Nodes with primary features: {matched:,} / {N:,}")
print(f"  Retweet-only nodes (zero-initialized): {N-matched:,}")

has_feats = X.any(axis=1)
if has_feats.any():
    scaler = StandardScaler()
    X[has_feats] = scaler.fit_transform(X[has_feats])
X = np.nan_to_num(X, nan=0.0).astype(np.float32)

# ── Build retweet edges ───────────────────────────────────────────────────────
print(f"\n{'='*60}\nSTEP 4: Edge Construction\n{'='*60}")

if "rt_screen" in df.columns:
    edges = df[["screen_name", "rt_screen"]].dropna(subset=["rt_screen"]).copy()
    edges["rt_screen"] = edges["rt_screen"].str.lower()
elif "rt_userid" in df.columns:
    print("  Resolving rt_userid to screen_names...")
    uid_to_sn = df.dropna(subset=["userid", "screen_name"]).drop_duplicates("userid").set_index("userid")["screen_name"].to_dict()
    edges = df[["screen_name", "rt_userid"]].dropna(subset=["rt_userid"]).copy()
    edges["rt_userid"] = pd.to_numeric(edges["rt_userid"], errors="coerce")
    edges = edges.dropna(subset=["rt_userid"])
    edges["rt_screen"] = edges["rt_userid"].astype(int).map(uid_to_sn)
    edges = edges.dropna(subset=["rt_screen"]).str.lower()

# Edge filtering logs
raw_edge_count = len(edges)
edges = edges[edges["screen_name"] != edges["rt_screen"]]
print(f"  Removed {raw_edge_count - len(edges):,} self-retweets (loops).")

edges["src"] = edges["screen_name"].map(h2i)
edges["dst"] = edges["rt_screen"].map(h2i)
edges = edges.dropna(subset=["src", "dst"])
edges = edges.drop_duplicates(subset=["src", "dst"])

edge_index = torch.tensor(
    [edges["src"].astype(int).values, edges["dst"].astype(int).values],
    dtype=torch.long,
)
print(f"  Final unique directed edges: {edge_index.shape[1]:,}")

# Degree Statistics
degrees_out = torch.bincount(edge_index[0], minlength=N)
degrees_in  = torch.bincount(edge_index[1], minlength=N)
print(f"  Graph Density Stats:")
print(f"    - Avg Out-degree: {degrees_out.float().mean():.2f}")
print(f"    - Max In-degree (most retweeted): {degrees_in.max().item()}")
print(f"    - Isolated nodes: {(degrees_out == 0).sum().item():,}")

# ── Save ──────────────────────────────────────────────────────────────────────
data = Data(x=torch.from_numpy(X), edge_index=edge_index)
data.feature_names = FEATURE_COLS

torch.save({"data": data, "h2i": h2i, "handles": handles}, args.out)
print(f"\n{'='*60}\nSUCCESS: Saved to {args.out}\n{'='*60}")
