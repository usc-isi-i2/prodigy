# Co-hashtag graph for Instagram
# Nodes: Instagram accounts
# Edges (undirected): Account A and Account B connected if they used the same hashtag
#         (analogous to co-retweet on Twitter)
# Edges (directed):   Account A mentioned Account B in a post description
# Node features: account stats + text embeddings (all-MiniLM-L6-v2, PCA 64d)
# Labels: post language majority vote → ru=0, uk=1, other=2
# MAX_GROUP_SIZE caps large/generic hashtag groups (set None for no cap)

import glob
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
import torch
import os
import warnings
warnings.filterwarnings('ignore')

## ── Config ────────────────────────────────────────────────────────────────────
PKL_PATTERN   = "/project2/ll_774_951/uk_ru/Instagram_Uk_ru/*.pkl"
OUTPUT_DIR    = "graph_instagram_co_hashtag"
SBERT_MODEL   = "all-MiniLM-L6-v2"
TEXT_DIM      = 64
MAX_GROUP_SIZE = 500   # skip hashtags with more than this many unique accounts
BATCH_SIZE    = 256
## ─────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

## Load all pkl files
print("Loading pickle files...")
files = sorted(glob.glob(PKL_PATTERN))
if not files:
    raise FileNotFoundError(f"No pkl files found at: {PKL_PATTERN}")
print(f"Found {len(files)} files")

dfs = []
for f in files:
    print(f"  reading {f}")
    dfs.append(pd.read_pickle(f))
dfs = [pd.DataFrame(x) for x in dfs]
df = pd.concat(dfs, ignore_index=True)
del dfs
print(f"Total rows: {len(df):,}")

## Parse account field
def safe_get(d, key, default=None):
    return d.get(key, default) if isinstance(d, dict) else default

df['account_id']      = df['account'].apply(lambda x: safe_get(x, 'id'))
df['account_name']    = df['account'].apply(lambda x: safe_get(x, 'name', ''))
df['handle']          = df['account'].apply(lambda x: safe_get(x, 'handle', ''))
df['verified']        = df['account'].apply(lambda x: int(bool(safe_get(x, 'verified', False))))
df['subscriber_count'] = df['account'].apply(lambda x: float(safe_get(x, 'subscriberCount') or 0))

## Parse statistics
def get_stat(stats, key):
    if isinstance(stats, dict):
        actual = stats.get('actual', {})
        if isinstance(actual, dict):
            return float(actual.get(key) or 0)
    return 0.0

df['favoriteCount'] = df['statistics'].apply(lambda x: get_stat(x, 'favoriteCount'))
df['commentCount']  = df['statistics'].apply(lambda x: get_stat(x, 'commentCount'))
df['score']         = pd.to_numeric(df['score'], errors='coerce').fillna(0.0)

## Extract hashtags and mentions from description
hashtag_re = r'(?:^|(?<=\s))#([A-Za-z_][A-Za-z0-9_]{0,29})'
mention_re  = r'(?:^|(?<=\s))@([A-Za-z0-9_](?:(?:[A-Za-z0-9_]|(?:\.(?!\.))){0,28}(?:[A-Za-z0-9_]))?)'

df['description'] = df['description'].fillna('')
df['hashtags'] = (
    df['description'].str.extractall(hashtag_re).groupby(level=0).agg(list)
    .reindex(df.index).apply(lambda x: [h.lower() for h in x] if isinstance(x, list) else [])
)
df['mentions'] = (
    df['description'].str.extractall(mention_re).groupby(level=0).agg(list)
    .reindex(df.index).apply(lambda x: [m.lower() for m in x] if isinstance(x, list) else [])
)

## Drop rows with no account_id
df = df.dropna(subset=['account_id'])
df['account_id'] = df['account_id'].astype(int)
df['languageCode'] = df['languageCode'].fillna('und')
df['n_hashtags'] = df['hashtags'].apply(len)
df['n_mentions']  = df['mentions'].apply(len)
df['has_media']   = df['media'].apply(lambda x: int(bool(x) and str(x) not in ('', '[]', 'nan')))

print(f"Unique accounts: {df['account_id'].nunique():,}")
print(f"Posts with hashtags: {df['n_hashtags'].gt(0).sum():,}")
print(f"Posts with mentions: {df['n_mentions'].gt(0).sum():,}")

## Account-level aggregated features
print("\nBuilding node features...")
user_agg = df.groupby('account_id').agg(
    subscriber_count = ('subscriber_count', 'max'),
    verified         = ('verified', 'max'),
    avg_favorites    = ('favoriteCount', 'mean'),
    avg_comments     = ('commentCount', 'mean'),
    avg_score        = ('score', 'mean'),
    avg_n_hashtags   = ('n_hashtags', 'mean'),
    avg_n_mentions   = ('n_mentions', 'mean'),
    avg_has_media    = ('has_media', 'mean'),
    post_count       = ('account_id', 'count'),
    language         = ('languageCode', lambda x: x.value_counts().index[0]),
    handle           = ('handle', 'first'),
).reset_index()

# Log-transform skewed counts
for col in ['subscriber_count', 'avg_favorites', 'avg_comments', 'post_count']:
    user_agg[col] = np.log1p(user_agg[col].fillna(0).clip(lower=0))

print(f"Unique accounts: {len(user_agg):,}")

## Language labels: ru=0, uk=1, other=2
def lang_to_label(lang):
    if lang == 'ru': return 0
    if lang == 'uk': return 1
    return 2

user_agg['label'] = user_agg['language'].apply(lang_to_label)
label_names = ['ru', 'uk', 'other']
print(f"Label distribution: {dict(user_agg['label'].value_counts().sort_index())}")

## Text embeddings — average per account, then PCA
print(f"\nLoading sentence transformer: {SBERT_MODEL}...")
sbert = SentenceTransformer(SBERT_MODEL)

df['text_input'] = (df['account_name'].fillna('') + '. ' + df['description']).str[:512]
texts = df['text_input'].tolist()
print(f"Encoding {len(texts):,} posts...")
embeddings = sbert.encode(texts, show_progress_bar=True, batch_size=BATCH_SIZE)

emb_cols = [f'emb_{i}' for i in range(embeddings.shape[1])]
emb_df = pd.DataFrame(embeddings, columns=emb_cols)
emb_df['account_id'] = df['account_id'].values
emb_agg = emb_df.groupby('account_id')[emb_cols].mean().reset_index()

pca = PCA(n_components=TEXT_DIM, random_state=42)
emb_reduced = pca.fit_transform(emb_agg[emb_cols].values)
print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")

text_cols = [f'text_{i}' for i in range(TEXT_DIM)]
emb_agg_reduced = pd.DataFrame(emb_reduced, columns=text_cols)
emb_agg_reduced['account_id'] = emb_agg['account_id'].values

## Merge node features
node_features = user_agg.merge(emb_agg_reduced, on='account_id', how='left').fillna(0)
user_ids  = node_features['account_id'].values
uid_to_idx = {uid: i for i, uid in enumerate(user_ids)}

feature_cols = [c for c in node_features.columns
                if c not in ('account_id', 'language', 'label', 'handle')]
X = node_features[feature_cols].values.astype(float)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"Node feature matrix: {X_scaled.shape}  (accounts × features)")

## Build co-hashtag edges (undirected, analogous to co-retweet)
print("\nBuilding co-hashtag edges...")
hashtag_to_accounts: dict = {}
for _, row in df.iterrows():
    aid = row['account_id']
    for ht in row['hashtags']:
        hashtag_to_accounts.setdefault(ht, set()).add(aid)

group_sizes = pd.Series({k: len(v) for k, v in hashtag_to_accounts.items()})
print(f"Unique hashtags: {len(hashtag_to_accounts):,}")
print(f"Group sizes: mean={group_sizes.mean():.1f}  median={group_sizes.median():.1f}  max={group_sizes.max()}")
if MAX_GROUP_SIZE is not None:
    print(f"Groups capped (>{MAX_GROUP_SIZE}): {(group_sizes > MAX_GROUP_SIZE).sum()} skipped")

edge_set: set = set()
for ht, accounts in hashtag_to_accounts.items():
    accounts = list(accounts)
    if MAX_GROUP_SIZE is not None and len(accounts) > MAX_GROUP_SIZE:
        continue
    for u, v in combinations(accounts, 2):
        edge_set.add((min(u, v), max(u, v)))

print(f"Unique undirected co-hashtag pairs: {len(edge_set):,}")

src, dst = [], []
for u, v in edge_set:
    if u in uid_to_idx and v in uid_to_idx:
        ui, vi = uid_to_idx[u], uid_to_idx[v]
        src += [ui, vi]
        dst += [vi, ui]

print(f"Co-hashtag directed edges (both directions): {len(src):,}")

## Build mention edges (directed: poster → mentioned account, within-graph only)
print("\nBuilding mention edges...")
handle_to_uid = (
    node_features[node_features['handle'] != '']
    .set_index('handle')['account_id']
    .to_dict()
)

men_src, men_dst = [], []
for _, row in df.iterrows():
    poster_uid = row['account_id']
    if poster_uid not in uid_to_idx:
        continue
    for mention in row['mentions']:
        mentioned_uid = handle_to_uid.get(mention)
        if mentioned_uid and mentioned_uid in uid_to_idx and mentioned_uid != poster_uid:
            men_src.append(uid_to_idx[poster_uid])
            men_dst.append(uid_to_idx[mentioned_uid])

print(f"Mention directed edges: {len(men_src):,}")

## Combine
all_src = src + men_src
all_dst = dst + men_dst
edge_index = torch.tensor([all_src, all_dst], dtype=torch.long)
print(f"Total directed edges: {edge_index.shape[1]:,}")

degrees = torch.bincount(edge_index[0], minlength=len(user_ids))
print(f"Degree: mean={degrees.float().mean():.1f}  "
      f"median={degrees.float().median():.1f}  "
      f"isolated={(degrees == 0).sum().item():,}")

## Save
x = torch.tensor(X_scaled, dtype=torch.float)
y = torch.tensor(node_features['label'].values, dtype=torch.long)

out_path = f'{OUTPUT_DIR}/graph_data.pt'
torch.save({
    'x':             x,
    'edge_index':    edge_index,
    'user_ids':      user_ids,
    'feature_names': feature_cols,
    'y':             y,
    'label_names':   label_names,
}, out_path)
print(f"\nSaved to {out_path}")
print(f"  Nodes:    {x.shape[0]:,}")
print(f"  Features: {x.shape[1]}")
print(f"  Edges:    {edge_index.shape[1]:,}")
print(f"  Labels:   {dict(zip(label_names, y.bincount(minlength=3).tolist()))}")
