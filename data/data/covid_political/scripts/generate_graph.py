"""
Build the covid_political retweet_graph.pt from graph.pickle + user_data.csv.

Node identifiers in the DiGraph are integers matching user_data row indices (0..N-1).

Output schema (same as midterm / covid19_twitter):
    x                  : float32 tensor (N, F+D)  -- node features + optional embedding
    edge_index         : long tensor   (2, E)
    edge_attr          : float32 tensor (E, 2)    -- [rt_weight_log1p, mn_weight_log1p]
    edge_attr_feature_names : list[str]
    y                  : long tensor   (N,)        -- label_conservative {0,1}
    label_names        : list[str]
    feature_names      : list[str]
    user_ids           : list[int]
    u2i                : dict[int, int]           -- node_id -> row index (identity here)
"""
import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data


DEFAULT_GRAPH = "/scratch1/eibl/data/social_llm_covid/graph.pickle"
DEFAULT_CSV = "/scratch1/eibl/data/social_llm_covid/user_data.csv"
DEFAULT_OUT = "data/data/covid_political/graphs/retweet_graph.pt"

# Columns to log1p-transform (count-like)
LOG1P_FEATURES = [
    "init_followers_count",
    "final_followers_count",
    "n_posts",
    "n_days_active",
    "n_original_posts",
]
# Columns to keep as-is (already bounded / binary)
RAW_FEATURES = ["verified"]

NODE_FEATURE_NAMES = LOG1P_FEATURES + RAW_FEATURES

EDGE_FEATURE_NAMES = ["rt_weight", "mn_weight"]

LABEL_NAMES = ["not_conservative", "conservative"]


def parse_args():
    p = argparse.ArgumentParser(
        description="Build covid_political graph.pt from graph.pickle + user_data.csv."
    )
    p.add_argument("--graph", default=DEFAULT_GRAPH)
    p.add_argument("--csv", default=DEFAULT_CSV)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument(
        "--embeddings",
        default="",
        help="Optional path to user_embeddings_*.pt (user_ids + meanpool). "
             "Appended to node features when provided.",
    )
    p.add_argument(
        "--embedding_pool",
        choices=["meanpool", "maxpool"],
        default="meanpool",
    )
    return p.parse_args()


def load_graph(path):
    with open(path, "rb") as f:
        G = pickle.load(f)
    print(f"Loaded DiGraph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    return G


def build_edge_tensors(G, n_nodes):
    """Convert DiGraph edges to COO edge_index + edge_attr (rt_weight, mn_weight)."""
    src_list, dst_list, rt_w, mn_w = [], [], [], []
    for u, v, data in G.edges(data=True):
        src_list.append(int(u))
        dst_list.append(int(v))
        rt_w.append(float(data.get("rt_weight") or 0.0))
        mn_w.append(float(data.get("mn_weight") or 0.0))

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)

    rt_arr = np.log1p(np.array(rt_w, dtype=np.float32))
    mn_arr = np.log1p(np.array(mn_w, dtype=np.float32))
    edge_attr = torch.tensor(np.stack([rt_arr, mn_arr], axis=1), dtype=torch.float)

    print(f"Edges: {edge_index.shape[1]:,}  edge_attr shape: {edge_attr.shape}")
    return edge_index, edge_attr


def build_node_features(user_data, n_nodes):
    """Build (N, F) feature matrix from user_data, ordered by row index."""
    feat = user_data[NODE_FEATURE_NAMES].copy()

    for col in LOG1P_FEATURES:
        feat[col] = np.log1p(feat[col].fillna(0).clip(lower=0))

    feat["verified"] = feat["verified"].fillna(0).astype(np.float32)

    x_np = feat.to_numpy(dtype=np.float32)

    # StandardScaler on non-zero rows (guard against all-zero cols)
    nonzero = np.any(x_np != 0, axis=1)
    if nonzero.any():
        scaler = StandardScaler()
        x_np[nonzero] = scaler.fit_transform(x_np[nonzero]).astype(np.float32)

    x = torch.tensor(x_np, dtype=torch.float)
    print(f"Node features: {x.shape}  names={NODE_FEATURE_NAMES}")
    return x


def maybe_attach_embeddings(x, feature_names, user_ids, embeddings_path, embedding_pool):
    if not embeddings_path:
        return x, feature_names, {"matched_users": 0, "embedding_dim": 0}

    emb = torch.load(embeddings_path, map_location="cpu")
    emb_mat = emb.get(embedding_pool)
    if emb_mat is None:
        raise KeyError(f"Embeddings file must contain '{embedding_pool}'")
    emb_dim = int(emb_mat.shape[1])
    extra = torch.zeros((len(user_ids), emb_dim), dtype=torch.float)

    # Join by user_id (row index in this dataset).
    emb_ids = np.asarray(emb["user_ids"], dtype=np.int64)
    order = np.argsort(emb_ids)
    sorted_ids = emb_ids[order]
    query = np.asarray(user_ids, dtype=np.int64)
    pos = np.searchsorted(sorted_ids, query)
    pos_clipped = np.clip(pos, 0, len(sorted_ids) - 1)
    hit = (pos < len(sorted_ids)) & (sorted_ids[pos_clipped] == query)
    tgt_rows = np.where(hit)[0]
    src_rows = order[pos[hit]]
    if len(tgt_rows):
        extra[torch.from_numpy(tgt_rows)] = emb_mat[torch.from_numpy(src_rows)].float()
    matched = int(hit.sum())

    x_out = torch.cat([x, extra], dim=1)
    names_out = feature_names + [f"emb_{k}" for k in range(emb_dim)]
    print(f"Embeddings attached: matched={matched:,}/{len(user_ids):,} dim={emb_dim}")
    return x_out, names_out, {"matched_users": matched, "embedding_dim": emb_dim}


def build_labels(user_data):
    y_np = user_data["label_conservative"].fillna(-1).to_numpy(dtype=np.int64)
    y = torch.from_numpy(y_np)
    labeled = int((y_np >= 0).sum())
    print(f"Labels: {labeled:,} labeled  dist={dict(zip(*np.unique(y_np[y_np>=0], return_counts=True)))}")
    return y


def main():
    args = parse_args()

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"Loading graph from {args.graph}")
    G = load_graph(args.graph)

    print(f"Loading user_data from {args.csv}")
    user_data = pd.read_csv(args.csv)
    print(f"user_data shape: {user_data.shape}")

    n_nodes = len(user_data)
    user_ids = user_data.index.tolist()          # [0, 1, ..., N-1]
    u2i = {uid: i for i, uid in enumerate(user_ids)}  # identity for this dataset

    edge_index, edge_attr = build_edge_tensors(G, n_nodes)
    x = build_node_features(user_data, n_nodes)
    feature_names = list(NODE_FEATURE_NAMES)

    x, feature_names, emb_stats = maybe_attach_embeddings(
        x, feature_names, user_ids, args.embeddings, args.embedding_pool
    )

    y = build_labels(user_data)

    graph_obj = {
        "x": x,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "edge_attr_feature_names": EDGE_FEATURE_NAMES,
        "y": y,
        "label_names": LABEL_NAMES,
        "feature_names": feature_names,
        "user_ids": user_ids,
        "u2i": u2i,
    }

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data.feature_names = feature_names
    data.edge_attr_feature_names = EDGE_FEATURE_NAMES
    data.label_names = LABEL_NAMES
    data.user_ids = user_ids
    graph_obj["data"] = data

    torch.save(graph_obj, args.out)

    meta = {
        "graph": args.graph,
        "csv": args.csv,
        "nodes": int(n_nodes),
        "edges": int(edge_index.shape[1]),
        "node_feature_dim": int(x.shape[1]),
        "node_feature_names": feature_names,
        "edge_feature_names": EDGE_FEATURE_NAMES,
        "label_names": LABEL_NAMES,
        "labeled_nodes": int((y.numpy() >= 0).sum()),
        "embeddings": args.embeddings,
        "embedding_pool": args.embedding_pool,
        "embedding_dim": emb_stats["embedding_dim"],
        "embedding_matched_users": emb_stats["matched_users"],
    }
    meta_path = args.out.replace(".pt", ".meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved graph: {args.out}")
    print(f"Saved meta:  {meta_path}")


if __name__ == "__main__":
    main()
