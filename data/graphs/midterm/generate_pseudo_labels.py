"""
Attach pseudo labels to retweet_graph.pt (instagram_mention format).

Labels generated:
  retweet_graph_political.pt   – pro_republican=0, pro_democrat=1, other_political=2
  retweet_graph_follower.pt    – follower tier quintiles: nano=0 … mega=4

Usage:
    python generate_pseudo_labels.py \\
        --csv  /project2/ll_774_951/midterm/*/*.csv \\
        --graph retweet_graph.pt \\
        --out_dir .
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
import torch

# ── Hashtag-based political groups ───────────────────────────────────────────
POLITICAL_GROUPS = {
    "pro_republican": [
        "maga", "trump2024", "letsgobrandon", "fjb", "redwave",
        "republicanparty", "gop", "trump", "saveamerica",
    ],
    "pro_democrat": [
        "voteblue", "democraticparty", "biden2024", "bluewave",
        "democrats", "votedem", "bidensamerica",
    ],
    "other_political": [
        "prochoice", "mybodymychoice", "roevwade", "abortionrights",
        "prolife", "endabortion", "prolifegeneration",
        "blacklivesmatter", "blm", "stopthesteal", "jan6",
    ],
}
MIN_SCORE = 2   # minimum hashtag hits to assign a label

parser = argparse.ArgumentParser()
parser.add_argument("--csv",     default="/project2/ll_774_951/midterm/*/*.csv")
parser.add_argument("--graph",   default="retweet_graph.pt")
parser.add_argument("--out_dir", default=".")
args = parser.parse_args()


def save_graph(ckpt, y, label_names, out_path):
    data = ckpt["data"]
    data.y = y
    data.label_names = label_names
    torch.save({"data": data, "h2i": ckpt["h2i"], "handles": ckpt["handles"]}, out_path)
    labeled = (y >= 0).sum().item()
    counts = {label_names[i]: (y == i).sum().item() for i in range(len(label_names))}
    print(f"  Saved {out_path}  |  {labeled}/{len(y)} labeled  |  {counts}")


# ── Load graph ────────────────────────────────────────────────────────────────
print(f"Loading graph from {args.graph}...")
ckpt = torch.load(args.graph, map_location="cpu")
h2i, handles = ckpt["h2i"], ckpt["handles"]
num_nodes = ckpt["data"].x.shape[0]
print(f"  {num_nodes:,} nodes")

# ── Load CSVs ─────────────────────────────────────────────────────────────────
print("Loading CSVs...")
files = sorted(glob.glob(args.csv))
USECOLS = ["screen_name", "userid", "hashtag", "rt_hashtag", "followers_count"]
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
df = df[df["screen_name"].notna()]
df["screen_name"] = df["screen_name"].str.lower()
print(f"  {len(df):,} rows, {df['screen_name'].nunique():,} unique users")


# ── 1. Political pseudo labels ────────────────────────────────────────────────
print("\n[1] political (pro_republican=0, pro_democrat=1, other_political=2)")

def parse_hashtags(val):
    if pd.isna(val):
        return set()
    val = str(val).lower().replace("'", "").replace('"', '').strip("[]")
    return {t.strip() for t in val.split(",") if t.strip()}

label_names_pol = list(POLITICAL_GROUPS.keys())
label2idx = {name: i for i, name in enumerate(label_names_pol)}

from collections import defaultdict
user_scores = defaultdict(lambda: defaultdict(int))

for _, row in df.iterrows():
    sn = row["screen_name"]
    tags = parse_hashtags(row.get("hashtag")) | parse_hashtags(row.get("rt_hashtag"))
    for label, keywords in POLITICAL_GROUPS.items():
        hits = sum(1 for t in keywords if t in tags)
        if hits:
            user_scores[sn][label] += hits

sn_to_label = {}
for sn, scores in user_scores.items():
    best = max(scores, key=scores.get)
    if scores[best] >= MIN_SCORE:
        sn_to_label[sn] = label2idx[best]

y = torch.full((num_nodes,), -1, dtype=torch.long)
for handle, idx in h2i.items():
    if idx < num_nodes and handle in sn_to_label:
        y[idx] = sn_to_label[handle]

out = os.path.join(args.out_dir, "retweet_graph_political.pt")
save_graph(ckpt, y, label_names_pol, out)


# ── 2. Follower tier (quintiles — balanced by construction) ──────────────────
print("\n[2] follower_tier (quintiles: nano=0, micro=1, mid=2, macro=3, mega=4)")

user_followers = (
    df.groupby("screen_name")["followers_count"]
    .max()
    .reset_index()
)
user_followers["followers_count"] = pd.to_numeric(
    user_followers["followers_count"], errors="coerce"
).fillna(0)

log_followers = np.log1p(user_followers["followers_count"].values)
quintile_labels = pd.qcut(log_followers, q=5, labels=False, duplicates="drop")
user_followers["tier"] = quintile_labels

sn_to_tier = dict(zip(user_followers["screen_name"], user_followers["tier"]))

y = torch.full((num_nodes,), -1, dtype=torch.long)
for handle, idx in h2i.items():
    if idx < num_nodes:
        tier = sn_to_tier.get(handle)
        if tier is not None and not np.isnan(tier):
            y[idx] = int(tier)

out = os.path.join(args.out_dir, "retweet_graph_follower.pt")
save_graph(ckpt, y, ["nano", "micro", "mid", "macro", "mega"], out)

print("\nDone.")
