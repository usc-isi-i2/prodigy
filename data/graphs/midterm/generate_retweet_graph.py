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
import csv

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
parser.add_argument("--max_files", type=int, default=None, help="Limit number of CSV files to load (for debugging)")
args = parser.parse_args()

# ── NEW READER ────────────────────────────────────────────────────────────────
def load_interleaved_csv(filepath):
    """
    Loads CSVs where each record can be split across two consecutive rows:
      - a "main" row (len 66 in your example)
      - optionally followed by a "sub" row (len 11)
    This reconstructs a single DataFrame with columns from the main header
    plus a set of sub columns merged on the right.
    """
    main_rows, sub_rows = [], []

    # read headers (two header-like lines at the top)
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError(f"File {filepath} appears empty or malformed (no header).")
        # Attempt to read a second header/sub-header line if present; otherwise keep as-is.
        try:
            sub_header_raw = next(reader)
        except StopIteration:
            sub_header_raw = None

    # read and pair rows
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        # skip the header lines we already consumed above (if present)
        next(reader)  # skip header
        # If there was a second header/sub-header, skip it too.
        if sub_header_raw is not None:
            next(reader)

        pending_main = None
        for row in reader:
            # ensure we only consider non-empty rows
            if not row:
                continue
            if len(row) == 66:
                # start or replace pending main
                if pending_main is not None:
                    # previous main had no sub; record it with empty sub
                    main_rows.append(pending_main)
                    sub_rows.append([""] * 11)
                pending_main = row
            elif len(row) == 11:
                if pending_main is not None:
                    main_rows.append(pending_main)
                    sub_rows.append(row)
                    pending_main = None
                else:
                    # sub-row without a pending main — skip or treat as standalone
                    # we'll skip to avoid misalignment
                    continue
            else:
                # row length doesn't match either expected pattern; ignore
                # (could optionally try to coerce, but keep conservative)
                continue
        if pending_main is not None:
            # last main without sub
            main_rows.append(pending_main)
            sub_rows.append([""] * 11)

    # define sub columns (drop the first 'sub_extra' column later if unused)
    sub_cols = ["sub_extra", "state", "country", "rt_state", "rt_country",
                "qtd_state", "qtd_country", "norm_country",
                "norm_rt_country", "norm_qtd_country", "acc_age"]

    # create dataframes
    df_main = pd.DataFrame(main_rows, columns=header)
    df_sub = pd.DataFrame(sub_rows, columns=sub_cols).drop(columns=["sub_extra"], errors="ignore")
    df = pd.concat([df_main.reset_index(drop=True), df_sub.reset_index(drop=True)], axis=1)

    # basic coercions and cleaning
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], format="%a %b %d %H:%M:%S %z %Y", errors="coerce")
    if "acc_age" in df.columns:
        df["acc_age"] = pd.to_numeric(df["acc_age"], errors="coerce")
    if "tweetid" in df.columns:
        df["tweetid"] = pd.to_numeric(df["tweetid"], errors="coerce")
    if "userid" in df.columns:
        df["userid"] = pd.to_numeric(df["userid"], errors="coerce")

    # strip whitespace from object columns
    str_cols = df.select_dtypes(include="object").columns
    if len(str_cols) > 0:
        df[str_cols] = df[str_cols].apply(lambda x: x.str.strip())

    # replace empty strings with NA
    df.replace("", pd.NA, inplace=True)

    return df

# ── Load CSVs ─────────────────────────────────────────────────────────────────
print("Loading CSV files...")
files = sorted(glob.glob(args.csv))

if args.max_files is not None:
    files = files[:args.max_files]

if not files:
    raise FileNotFoundError(f"No CSV files found at: {args.csv}")
print(f"  Found {len(files)} files")

chunks = []
for f in files:
    try:
        # Use the new reader to load the file (it returns a DataFrame)
        df_file = load_interleaved_csv(f)

        # Keep only the columns we care about (if present)
        usecols = [c for c in USECOLS if c in df_file.columns]
        if not usecols:
            print(f"  Skipping {os.path.basename(f)}: no usable columns found")
            continue

        # select the requested columns (will preserve NaNs for missing ones)
        df_sel = df_file[usecols].copy()
        chunks.append(df_sel)
    except Exception as e:
        print(f"  Skipping {os.path.basename(f)}: {e}")

if not chunks:
    raise FileNotFoundError(f"No valid CSV content found in: {args.csv}")

df = pd.concat(chunks, ignore_index=True)
del chunks
print(f"  {len(df):,} rows")

# ── Clean ─────────────────────────────────────────────────────────────────────
df = df[df["screen_name"].notna() & df["screen_name"].ne("")]
df["screen_name"] = df["screen_name"].str.lower()

print(f"Unique screen_names: {df['screen_name'].nunique():,}")

# ── Build node set ───────────────────────────────────────────────────────────
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

# Coerce numeric columns that may have been read as strings
for col in ["followers_count", "verified", "rt_fav_count", "rt_reply_count",
            "sent_vader", "statuses_count"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

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
if has_feats.any():
    X[has_feats] = scaler.fit_transform(X[has_feats])
X = np.nan_to_num(X, nan=0.0).astype(np.float32)
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