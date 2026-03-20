# %%
import glob
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Load data
# 1. Define the pattern
path_pattern = "/project2/ll_774_951/midterm/*/*.csv"

# 2. Get the files and SORT them immediately
# sorted() handles strings alphabetically: 2022-10-17 comes before 2022-10-18
files = sorted(glob.glob(path_pattern))
# files = files[:100]

# %%

# 3. Read and combine
mdf = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

# Extract @mentions from text
mdf['mentions'] = mdf['text'].str.findall(r'(?<!\w)@(\w+)')

# Build edge list: author handle → mentioned handle
# Adjust 'username' to whatever your author-handle column is actually called
edges = (
    mdf[['display_name', 'mentions']]
    .explode('mentions')
    .dropna(subset=['mentions'])
    .rename(columns={'display_name': 'source', 'mentions': 'target'})
)
edges = edges[edges['source'] != edges['target']]  # drop self-loops

# %%

# Sample a subgraph around 10 random accounts
sample_nodes = edges['source'].drop_duplicates().sample(10)
mask = edges['source'].isin(sample_nodes) | edges['target'].isin(sample_nodes)
# mask = edges['source'].isin(sample_nodes) & edges['target'].isin(sample_nodes)
G = nx.from_pandas_edgelist(edges[mask], source='source', target='target', create_using=nx.DiGraph())

# Draw
pos = nx.spring_layout(G, k=0.3, iterations=30)
plt.figure(figsize=(8, 8))
nx.draw_networkx_nodes(G, pos, node_size=20, alpha=0.6)
nx.draw_networkx_edges(G, pos, width=0.3, alpha=0.2, arrows=True)
clean_labels = {node: node.encode('ascii', 'ignore').decode() for node in G.nodes}
nx.draw_networkx_labels(G, pos, labels=clean_labels, font_size=6, alpha=0.7)
plt.axis('off')
plt.tight_layout()
plt.show()

# %%
# n egdes
print(f"Number of edges: {len(edges)}")

# n nodes
print(f"Number of unique nodes: {len(set(edges['source']).union(set(edges['target'])))}")

# mean degree
degree_counts = pd.concat([edges['source'], edges['target']]).value_counts()
mean_degree = degree_counts.mean()
print(f"Mean degree: {mean_degree:.2f}")

# n isolated nodes
all_nodes = set(edges['source']).union(set(edges['target']))
connected_nodes = set(edges['source']).union(set(edges['target']))
isolated_nodes = all_nodes - connected_nodes
print(f"Number of isolated nodes: {len(isolated_nodes)}")

# degree distribution
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 5))
plt.hist(degree_counts, bins=50, log=True)
plt.title("Degree Distribution (log scale)")
plt.xlabel("Degree")
plt.ylabel("Count (log scale)")
plt.show()

# plot graph sample
sample_nodes = edges['source'].drop_duplicates().sample(10)
mask = edges['source'].isin(sample_nodes) | edges['target'].isin(sample_nodes)
G = nx.from_pandas_edgelist(edges[mask], source='source', target='target', create_using=nx.DiGraph())
pos = nx.spring_layout(G, k=0.3, iterations=30)
plt.figure(figsize=(8, 8))
nx.draw_networkx_nodes(G, pos, node_size=20, alpha=0.6)
nx.draw_networkx_edges(G, pos, width=0.3, alpha=0.2, arrows=True)
clean_labels = {node: node.encode('ascii', 'ignore').decode() for node in G.nodes}
nx.draw_networkx_labels(G, pos, labels=clean_labels, font_size=6, alpha=0.7)
plt.axis('off')
plt.tight_layout()
plt.show()

# %%
## Cell 1: Imports and setup
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import torch
import warnings
warnings.filterwarnings('ignore')

# %%
## Cell 2: Use existing edges df, map to userid

# Build lookup: screen_name -> userid, display_name -> userid
sn_to_uid = mdf.drop_duplicates('screen_name').set_index('screen_name')['userid'].to_dict()
dn_to_uid = mdf.drop_duplicates('display_name').set_index('display_name')['userid'].to_dict()

def resolve_to_userid(name):
    if name in sn_to_uid:
        return sn_to_uid[name]
    if name in dn_to_uid:
        return dn_to_uid[name]
    return None

edgelist = edges.copy()
edgelist['source_uid'] = edgelist['source'].apply(resolve_to_userid)
edgelist['target_uid'] = edgelist['target'].apply(resolve_to_userid)

print(f"Total edges: {len(edgelist)}")
print(f"Source resolved: {edgelist['source_uid'].notna().sum()}")
print(f"Target resolved: {edgelist['target_uid'].notna().sum()}")

# Keep only edges where both nodes resolve
edgelist = edgelist.dropna(subset=['source_uid', 'target_uid'])
edgelist = edgelist[['source_uid', 'target_uid']].rename(
    columns={'source_uid': 'source', 'target_uid': 'target'}
).drop_duplicates()

print(f"Internal edges (both resolved): {len(edgelist)}")

# Compute degree
out_degree = edgelist.groupby('source').size().rename('out_degree')
in_degree = edgelist.groupby('target').size().rename('in_degree')

# %%

## Cell 3: User-level static features (take max/last per user)
user_static_cols = ['followers_count', 'friends_count', 'listed_count',
                    'favourites_count', 'statuses_count', 'acc_age', 'verified']

mdf['verified_int'] = mdf['verified'].map({True: 1, False: 0, 'True': 1, 'False': 0}).fillna(0).astype(int)

numeric_cols = ['followers_count', 'friends_count', 'listed_count',
                'favourites_count', 'statuses_count', 'acc_age',
                'rt_rt_count', 'rt_fav_count', 'rt_reply_count', 'rt_qtd_count',
                'sent_vader']

for col in numeric_cols:
    if col in mdf.columns:
        mdf[col] = pd.to_numeric(mdf[col], errors='coerce')

user_agg = mdf.groupby('userid').agg(
    followers_count=('followers_count', 'max'),
    friends_count=('friends_count', 'max'),
    listed_count=('listed_count', 'max'),
    favourites_count=('favourites_count', 'max'),
    statuses_count=('statuses_count', 'max'),
    acc_age=('acc_age', 'max'),
    verified=('verified_int', 'max'),
).reset_index()

# Derived: follower/friend ratio
user_agg['ff_ratio'] = (user_agg['followers_count'] + 1) / (user_agg['friends_count'] + 1)

# Log-transform skewed counts
log_cols = ['followers_count', 'friends_count', 'listed_count',
            'favourites_count', 'statuses_count', 'acc_age', 'ff_ratio']
for c in log_cols:
    user_agg[c] = np.log1p(user_agg[c].fillna(0).clip(lower=0))

print(f"Unique users: {len(user_agg)}")
print(f"Static feature cols: {list(user_agg.columns[1:])}")

# User-level state label: take first non-null state per user
state_per_user = (
    mdf[mdf['state'].notna()]
    .groupby('userid')['state']
    .first()
    .rename('state')
)
# Encode states as integers; users with no state get -1
all_states = sorted(state_per_user.unique())
state_to_idx = {s: i for i, s in enumerate(all_states)}
print(f"States found: {len(all_states)} — {all_states[:10]}...")
print(f"Users with state label: {len(state_per_user)} / {len(user_agg)}")


# %%

## Cell 4: Tweet-level behavioral features (aggregate per user)
# One-hot tweet type
tweet_type_dummies = pd.get_dummies(mdf['tweet_type'], prefix='tt')
mdf_behav = pd.concat([mdf[['userid']], tweet_type_dummies], axis=1)

# Sentiment
mdf_behav['sentiment'] = mdf['sent_vader'].fillna(0)

# Hashtag count
def count_list(x):
    if isinstance(x, list):
        return len(x)
    if isinstance(x, str):
        try:
            return len(eval(x))
        except:
            return 0
    return 0

mdf_behav['n_hashtags'] = mdf['hashtag'].apply(count_list)
mdf_behav['n_mentions'] = mdf['mentions'].apply(count_list)
mdf_behav['has_media'] = mdf['media_urls'].apply(lambda x: int(count_list(x) > 0))
mdf_behav['has_urls'] = mdf['urls_list'].apply(lambda x: int(count_list(x) > 0))

# RT engagement features (log-transformed)
for col in ['rt_rt_count', 'rt_fav_count', 'rt_reply_count', 'rt_qtd_count']:
    mdf_behav[col] = np.log1p(pd.to_numeric(mdf[col], errors='coerce').fillna(0).clip(lower=0))

# Aggregate per user (mean across their tweets)
behav_agg = mdf_behav.groupby('userid').mean().reset_index()
# Also add tweet count per user
tweet_counts = mdf.groupby('userid').size().rename('tweet_count').reset_index()
tweet_counts['tweet_count'] = np.log1p(tweet_counts['tweet_count'])
behav_agg = behav_agg.merge(tweet_counts, on='userid', how='left')

print(f"Behavioral feature cols: {list(behav_agg.columns[1:])}")

# %%

## Cell 5: Text embeddings (average per user)
print("Loading sentence transformer...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Use the preprocessed token column
texts = mdf['token'].fillna('').astype(str).tolist()
print(f"Encoding {len(texts)} tweets...")
embeddings = model.encode(texts, show_progress_bar=True, batch_size=256)

# Average embeddings per user
mdf_emb = pd.DataFrame({'userid': mdf['userid']})
emb_cols = [f'emb_{i}' for i in range(embeddings.shape[1])]
emb_df = pd.DataFrame(embeddings, columns=emb_cols)
mdf_emb = pd.concat([mdf_emb, emb_df], axis=1)
emb_agg = mdf_emb.groupby('userid')[emb_cols].mean().reset_index()

# PCA reduction: 384 -> 64 dims
TEXT_DIM = 64
pca = PCA(n_components=TEXT_DIM, random_state=42)
emb_reduced = pca.fit_transform(emb_agg[emb_cols].values)
print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")

emb_reduced_cols = [f'text_{i}' for i in range(TEXT_DIM)]
emb_agg_reduced = pd.DataFrame(emb_reduced, columns=emb_reduced_cols)
emb_agg_reduced['userid'] = emb_agg['userid'].values

print(f"Text embedding dims: {TEXT_DIM}")


# %%

## Cell 6: Merge everything into final node features
node_features = user_agg.copy()
node_features = node_features.merge(behav_agg, on='userid', how='left')
node_features = node_features.merge(emb_agg_reduced, on='userid', how='left')

# Add structural features
node_features = node_features.merge(
    out_degree.reset_index().rename(columns={'source': 'userid'}),
    on='userid', how='left'
)
node_features = node_features.merge(
    in_degree.reset_index().rename(columns={'target': 'userid'}),
    on='userid', how='left'
)
node_features['out_degree'] = np.log1p(node_features['out_degree'].fillna(0))
node_features['in_degree'] = np.log1p(node_features['in_degree'].fillna(0))

# Fill any remaining NaN
node_features = node_features.fillna(0)

# Separate userid and feature matrix
user_ids = node_features['userid'].values
feature_cols = [c for c in node_features.columns if c != 'userid']
X = node_features[feature_cols].values

# Standardize (skip text embeddings which are already normalized-ish, but scaling all is fine)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"\n=== FINAL NODE FEATURES ===")
print(f"Nodes: {X_scaled.shape[0]}")
print(f"Features: {X_scaled.shape[1]}")
print(f"Feature groups:")
static_cols = [c for c in feature_cols if c in user_agg.columns]
behav_cols_final = [c for c in feature_cols if c in behav_agg.columns]
text_cols = [c for c in feature_cols if c.startswith('text_')]
struct_cols = ['in_degree', 'out_degree']
print(f"  User static:    {len(static_cols)} dims")
print(f"  Behavioral:     {len(behav_cols_final)} dims")
print(f"  Text embedding: {len(text_cols)} dims")
print(f"  Structural:     {len(struct_cols)} dims")


# %%

## Cell 7: Save
# Save as numpy + mapping
# np.save('node_features.npy', X_scaled)
# np.save('node_ids.npy', user_ids)
t = pd.DataFrame({'userid': user_ids, 'idx': range(len(user_ids))})
# t.to_csv('node_id_map.csv', index=False)

# Also save edgelist indexed by node position (for PyG / DGL)
id_to_idx = {uid: i for i, uid in enumerate(user_ids)}
edgelist_idx = edgelist.copy()
edgelist_idx['source_idx'] = edgelist_idx['source'].map(id_to_idx)
edgelist_idx['target_idx'] = edgelist_idx['target'].map(id_to_idx)
edgelist_idx = edgelist_idx.dropna(subset=['source_idx', 'target_idx'])
edgelist_idx[['source_idx', 'target_idx']] = edgelist_idx[['source_idx', 'target_idx']].astype(int)

edge_index = torch.tensor(edgelist_idx[['source_idx', 'target_idx']].values.T, dtype=torch.long)
node_feats_tensor = torch.tensor(X_scaled, dtype=torch.float)

# Build state label tensor (-1 = no state available)
node_state_labels = torch.full((len(user_ids),), -1, dtype=torch.long)
for i, uid in enumerate(user_ids):
    if uid in state_per_user.index:
        node_state_labels[i] = state_to_idx[state_per_user[uid]]
labeled = (node_state_labels >= 0).sum().item()
print(f"Nodes with state label: {labeled} / {len(user_ids)}")

# torch.save({
#     'x': node_feats_tensor,
#     'edge_index': edge_index,
#     'user_ids': user_ids,
#     'feature_names': feature_cols,
# }, 'graph_data.pt')

print(f"\nSaved:")
print(f"  node_features.npy  - ({X_scaled.shape})")
print(f"  node_ids.npy       - ({user_ids.shape})")
print(f"  node_id_map.csv")
print(f"  graph_data.pt      - PyG-ready (x, edge_index)")
print(f"  edge_index shape:  {edge_index.shape}")

# %%
np.save('graph/node_features.npy', X_scaled)
np.save('graph/node_ids.npy', user_ids)
t.to_csv('graph/node_id_map.csv', index=False)
torch.save({
    'x': node_feats_tensor,
    'edge_index': edge_index,
    'user_ids': user_ids,
    'feature_names': feature_cols,
    'y': node_state_labels,
    'label_names': all_states,
}, 'graph/graph_data.pt')

# %%
edges


