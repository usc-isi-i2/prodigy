# Co-retweet graph
# Nodes: Twitter users
# Edges: user A and user B are connected if they retweeted the same account (undirected)
# k=infinity by default (MAX_GROUP_SIZE=None): no limit on neighborhood size

# %%
import glob
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import torch
import os
import warnings
warnings.filterwarnings('ignore')

# %%
## Config
path_pattern = "/project2/ll_774_951/midterm/*/*.csv"
OUTPUT_DIR = "graph_co_retweet"
MAX_GROUP_SIZE = None  # None = infinity; set e.g. 500 to cap large popular accounts

os.makedirs(OUTPUT_DIR, exist_ok=True)

# %%
## Load data
files = sorted(glob.glob(path_pattern))
mdf = pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)
print(f"Loaded {len(mdf):,} rows, {mdf['userid'].nunique():,} unique users")

# %%
## Build co-retweet edge list
rt_df = mdf[mdf['rt_userid'].notna()][['userid', 'rt_userid']].copy()
rt_df['rt_userid'] = pd.to_numeric(rt_df['rt_userid'], errors='coerce')
rt_df = rt_df.dropna(subset=['rt_userid'])
rt_df['rt_userid'] = rt_df['rt_userid'].astype(int)

print(f"Retweet rows: {len(rt_df):,}")
print(f"Unique retweeted accounts: {rt_df['rt_userid'].nunique():,}")

rt_groups = rt_df.groupby('rt_userid')['userid'].apply(list)
group_sizes = rt_groups.apply(len)
print(f"\nGroup sizes (users per retweeted account):")
print(f"  mean={group_sizes.mean():.1f}  median={group_sizes.median():.1f}  max={group_sizes.max()}")
print(f"  groups >100:  {(group_sizes > 100).sum()}")
print(f"  groups >1000: {(group_sizes > 1000).sum()}")
if MAX_GROUP_SIZE is not None:
    skipped = (group_sizes > MAX_GROUP_SIZE).sum()
    print(f"  groups capped at {MAX_GROUP_SIZE}: {skipped} groups skipped")

# Build all pairs within each group
print(f"\nBuilding co-retweet edges (MAX_GROUP_SIZE={MAX_GROUP_SIZE})...")
edge_set = set()
for rt_uid, retweeters in rt_groups.items():
    retweeters = list(set(retweeters))
    if MAX_GROUP_SIZE is not None and len(retweeters) > MAX_GROUP_SIZE:
        continue
    for u, v in combinations(retweeters, 2):
        edge_set.add((min(u, v), max(u, v)))

print(f"Unique undirected edges: {len(edge_set):,}")
edge_list = [(u, v) for u, v in edge_set] + [(v, u) for u, v in edge_set]
edgelist = pd.DataFrame(edge_list, columns=['source', 'target'])
print(f"Directed edges (both directions): {len(edgelist):,}")

out_degree = edgelist.groupby('source').size().rename('out_degree')
in_degree = edgelist.groupby('target').size().rename('in_degree')
print(f"\nDegree stats: mean={out_degree.mean():.1f}  median={out_degree.median():.1f}  max={out_degree.max()}")

# %%
## User-level static features
numeric_cols = ['followers_count', 'friends_count', 'listed_count',
                'favourites_count', 'statuses_count', 'acc_age',
                'rt_rt_count', 'rt_fav_count', 'rt_reply_count', 'rt_qtd_count',
                'sent_vader']
for col in numeric_cols:
    if col in mdf.columns:
        mdf[col] = pd.to_numeric(mdf[col], errors='coerce')

mdf['verified_int'] = mdf['verified'].map({True: 1, False: 0, 'True': 1, 'False': 0}).fillna(0).astype(int)

user_agg = mdf.groupby('userid').agg(
    followers_count=('followers_count', 'max'),
    friends_count=('friends_count', 'max'),
    listed_count=('listed_count', 'max'),
    favourites_count=('favourites_count', 'max'),
    statuses_count=('statuses_count', 'max'),
    acc_age=('acc_age', 'max'),
    verified=('verified_int', 'max'),
).reset_index()

user_agg['ff_ratio'] = (user_agg['followers_count'] + 1) / (user_agg['friends_count'] + 1)
log_cols = ['followers_count', 'friends_count', 'listed_count',
            'favourites_count', 'statuses_count', 'acc_age', 'ff_ratio']
for c in log_cols:
    user_agg[c] = np.log1p(user_agg[c].fillna(0).clip(lower=0))

# State labels
state_per_user = (
    mdf[mdf['state'].notna()]
    .groupby('userid')['state']
    .first()
    .rename('state')
)
all_states = sorted(state_per_user.unique())
state_to_idx = {s: i for i, s in enumerate(all_states)}
print(f"States: {len(all_states)}, users with state label: {len(state_per_user):,} / {len(user_agg):,}")

# %%
## Behavioral features
mdf['mentions'] = mdf['text'].str.findall(r'(?<!\w)@(\w+)')
tweet_type_dummies = pd.get_dummies(mdf['tweet_type'], prefix='tt')
mdf_behav = pd.concat([mdf[['userid']], tweet_type_dummies], axis=1)
mdf_behav['sentiment'] = mdf['sent_vader'].fillna(0)

def count_list(x):
    if isinstance(x, list): return len(x)
    if isinstance(x, str):
        try: return len(eval(x))
        except: return 0
    return 0

mdf_behav['n_hashtags'] = mdf['hashtag'].apply(count_list)
mdf_behav['n_mentions'] = mdf['mentions'].apply(count_list)
mdf_behav['has_media'] = mdf['media_urls'].apply(lambda x: int(count_list(x) > 0))
mdf_behav['has_urls'] = mdf['urls_list'].apply(lambda x: int(count_list(x) > 0))
for col in ['rt_rt_count', 'rt_fav_count', 'rt_reply_count', 'rt_qtd_count']:
    mdf_behav[col] = np.log1p(pd.to_numeric(mdf[col], errors='coerce').fillna(0).clip(lower=0))

behav_agg = mdf_behav.groupby('userid').mean().reset_index()
tweet_counts = mdf.groupby('userid').size().rename('tweet_count').reset_index()
tweet_counts['tweet_count'] = np.log1p(tweet_counts['tweet_count'])
behav_agg = behav_agg.merge(tweet_counts, on='userid', how='left')

# %%
## Text embeddings
print("Loading sentence transformer...")
model = SentenceTransformer('all-MiniLM-L6-v2')
texts = mdf['token'].fillna('').astype(str).tolist()
print(f"Encoding {len(texts):,} tweets...")
embeddings = model.encode(texts, show_progress_bar=True, batch_size=256)

mdf_emb = pd.DataFrame({'userid': mdf['userid']})
emb_cols = [f'emb_{i}' for i in range(embeddings.shape[1])]
emb_df = pd.DataFrame(embeddings, columns=emb_cols)
mdf_emb = pd.concat([mdf_emb, emb_df], axis=1)
emb_agg = mdf_emb.groupby('userid')[emb_cols].mean().reset_index()

TEXT_DIM = 64
pca = PCA(n_components=TEXT_DIM, random_state=42)
emb_reduced = pca.fit_transform(emb_agg[emb_cols].values)
print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")

emb_reduced_cols = [f'text_{i}' for i in range(TEXT_DIM)]
emb_agg_reduced = pd.DataFrame(emb_reduced, columns=emb_reduced_cols)
emb_agg_reduced['userid'] = emb_agg['userid'].values

# %%
## Merge node features
node_features = user_agg.copy()
node_features = node_features.merge(behav_agg, on='userid', how='left')
node_features = node_features.merge(emb_agg_reduced, on='userid', how='left')
node_features = node_features.merge(
    out_degree.reset_index().rename(columns={'source': 'userid'}), on='userid', how='left')
node_features = node_features.merge(
    in_degree.reset_index().rename(columns={'target': 'userid'}), on='userid', how='left')
node_features['out_degree'] = np.log1p(node_features['out_degree'].fillna(0))
node_features['in_degree'] = np.log1p(node_features['in_degree'].fillna(0))
node_features = node_features.fillna(0)

user_ids = node_features['userid'].values
feature_cols = [c for c in node_features.columns if c != 'userid']
X = node_features[feature_cols].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"\n=== FINAL NODE FEATURES ===")
print(f"Nodes: {X_scaled.shape[0]:,}  Features: {X_scaled.shape[1]}")

# %%
## Build tensors and save
id_to_idx = {uid: i for i, uid in enumerate(user_ids)}
edgelist_idx = edgelist.copy()
edgelist_idx['source_idx'] = edgelist_idx['source'].map(id_to_idx)
edgelist_idx['target_idx'] = edgelist_idx['target'].map(id_to_idx)
edgelist_idx = edgelist_idx.dropna(subset=['source_idx', 'target_idx'])
edgelist_idx[['source_idx', 'target_idx']] = edgelist_idx[['source_idx', 'target_idx']].astype(int)

edge_index = torch.tensor(edgelist_idx[['source_idx', 'target_idx']].values.T, dtype=torch.long)
node_feats_tensor = torch.tensor(X_scaled, dtype=torch.float)

node_state_labels = torch.full((len(user_ids),), -1, dtype=torch.long)
for i, uid in enumerate(user_ids):
    if uid in state_per_user.index:
        node_state_labels[i] = state_to_idx[state_per_user[uid]]
labeled = (node_state_labels >= 0).sum().item()

print(f"\n=== GRAPH SUMMARY ===")
print(f"Nodes:          {node_feats_tensor.shape[0]:,}")
print(f"Edges:          {edge_index.shape[1]:,}")
print(f"Features:       {node_feats_tensor.shape[1]}")
print(f"Labeled nodes:  {labeled:,} ({100*labeled/len(user_ids):.1f}%)")
print(f"States:         {len(all_states)}")

torch.save({
    'x': node_feats_tensor,
    'edge_index': edge_index,
    'user_ids': user_ids,
    'feature_names': feature_cols,
    'y': node_state_labels,
    'label_names': all_states,
}, f'{OUTPUT_DIR}/graph_data.pt')
print(f"\nSaved to {OUTPUT_DIR}/graph_data.pt")
