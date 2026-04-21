"""
Generic graph builder for social_llm_* datasets.

All datasets share the same DiGraph edge format (rt_weight, mn_weight).
Node features and label columns are auto-detected from user_data.csv, or
can be specified explicitly via CLI args.

Usage:
    python generate_graph.py \
        --graph /path/to/graph.pickle \
        --csv   /path/to/user_data.csv \
        --out   data/data/<dataset>/graphs/retweet_graph.pt \
        [--label_col label_toxicity] \
        [--embeddings /path/to/user_embeddings_minilm.pt]
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


EDGE_FEATURE_NAMES = ["rt_weight", "mn_weight"]

# Columns that are never used as node features
_SKIP_COLS = {"profile", "account_creation_date", "created_at"}

# Substrings that indicate a column should be log1p-transformed
_LOG1P_PATTERNS = [
    "_count", "n_tweets", "n_orig", "n_rt", "n_qtd", "n_replies",
    "n_posts", "n_days", "acc_age", "statuses", "listed", "friends",
    "favourites", "following",
]


def _is_log1p(col: str) -> bool:
    col_l = col.lower()
    return any(p in col_l for p in _LOG1P_PATTERNS)


def _detect_label_cols(df: pd.DataFrame):
    return [c for c in df.columns if c.startswith("label_")]


def _detect_feature_cols(df: pd.DataFrame, label_cols):
    skip = _SKIP_COLS | set(label_cols)
    return [c for c in df.columns if c not in skip and pd.api.types.is_numeric_dtype(df[c])]


def _label_names_from_col(col: str):
    base = col[len("label_"):]
    return [f"non_{base}", base]


def parse_args():
    p = argparse.ArgumentParser(
        description="Build a social_llm retweet_graph.pt from graph.pickle + user_data.csv."
    )
    p.add_argument("--graph", required=True, help="Path to graph.pickle (NetworkX DiGraph).")
    p.add_argument("--csv", required=True, help="Path to user_data.csv.")
    p.add_argument("--out", required=True, help="Output .pt path.")
    p.add_argument(
        "--label_col", default="",
        help="Label column to use as y. Auto-detected (first label_* col) if empty.",
    )
    p.add_argument(
        "--feature_cols", default="",
        help="Comma-sep feature columns. Auto-detected from numeric non-label cols if empty.",
    )
    p.add_argument("--embeddings", default="", help="Optional user_embeddings_*.pt path.")
    p.add_argument("--embedding_pool", choices=["meanpool", "maxpool"], default="meanpool")
    return p.parse_args()


def load_graph(path):
    with open(path, "rb") as f:
        G = pickle.load(f)
    print(f"Loaded DiGraph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    return G


def build_edge_tensors(G):
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
    print(f"Edges: {edge_index.shape[1]:,}")
    return edge_index, edge_attr


def build_node_features(user_data, feature_cols):
    if not feature_cols:
        print("No feature cols detected — node features will come from embeddings only.")
        return torch.zeros((len(user_data), 1), dtype=torch.float), []

    feat = user_data[feature_cols].copy()
    for col in feature_cols:
        if _is_log1p(col):
            feat[col] = np.log1p(feat[col].fillna(0).clip(lower=0))
        else:
            feat[col] = feat[col].fillna(0)

    x_np = feat.to_numpy(dtype=np.float32)
    nonzero = np.any(x_np != 0, axis=1)
    if nonzero.any():
        scaler = StandardScaler()
        x_np[nonzero] = scaler.fit_transform(x_np[nonzero]).astype(np.float32)

    print(f"Node features: {x_np.shape}  log1p={[c for c in feature_cols if _is_log1p(c)]}")
    return torch.tensor(x_np, dtype=torch.float), list(feature_cols)


def maybe_attach_embeddings(x, feature_names, user_ids, embeddings_path, embedding_pool):
    if not embeddings_path:
        return x, feature_names, {"matched_users": 0, "embedding_dim": 0}

    emb = torch.load(embeddings_path, map_location="cpu")
    emb_mat = emb.get(embedding_pool)
    if emb_mat is None:
        raise KeyError(f"Embeddings file must contain '{embedding_pool}'")
    emb_dim = int(emb_mat.shape[1])
    extra = torch.zeros((len(user_ids), emb_dim), dtype=torch.float)

    emb_ids = np.asarray(emb["user_ids"], dtype=np.int64)
    order = np.argsort(emb_ids)
    sorted_ids = emb_ids[order]
    query = np.asarray(user_ids, dtype=np.int64)
    pos = np.searchsorted(sorted_ids, query)
    pos_clipped = np.clip(pos, 0, len(sorted_ids) - 1)
    hit = (pos < len(sorted_ids)) & (sorted_ids[pos_clipped] == query)
    tgt_rows, src_rows = np.where(hit)[0], order[pos[hit]]
    if len(tgt_rows):
        extra[torch.from_numpy(tgt_rows)] = emb_mat[torch.from_numpy(src_rows)].float()
    matched = int(hit.sum())

    # If there were no structural features, replace the zero placeholder entirely.
    x_out = extra if not feature_names else torch.cat([x, extra], dim=1)
    names_out = feature_names + [f"emb_{k}" for k in range(emb_dim)]
    print(f"Embeddings attached: matched={matched:,}/{len(user_ids):,} dim={emb_dim}")
    return x_out, names_out, {"matched_users": matched, "embedding_dim": emb_dim}


def build_labels(user_data, label_col):
    y_np = user_data[label_col].fillna(-1).to_numpy(dtype=np.int64)
    y = torch.from_numpy(y_np)
    labeled = int((y_np >= 0).sum())
    vals, counts = np.unique(y_np[y_np >= 0], return_counts=True)
    print(f"Labels ({label_col}): {labeled:,} labeled  dist={dict(zip(vals.tolist(), counts.tolist()))}")
    return y


def main():
    args = parse_args()
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"Loading graph: {args.graph}")
    G = load_graph(args.graph)

    print(f"Loading user_data: {args.csv}")
    user_data = pd.read_csv(args.csv)
    print(f"Shape: {user_data.shape}  columns: {user_data.columns.tolist()}")

    all_label_cols = _detect_label_cols(user_data)
    label_col = args.label_col or (all_label_cols[0] if all_label_cols else None)
    if not label_col:
        raise ValueError("No label_* column found and --label_col not specified.")

    feature_cols = (
        [c.strip() for c in args.feature_cols.split(",") if c.strip()]
        if args.feature_cols else _detect_feature_cols(user_data, all_label_cols)
    )
    print(f"Label col: {label_col}")
    print(f"Feature cols: {feature_cols}")

    n_nodes = len(user_data)
    user_ids = list(range(n_nodes))

    edge_index, edge_attr = build_edge_tensors(G)
    x, feature_names = build_node_features(user_data, feature_cols)
    x, feature_names, emb_stats = maybe_attach_embeddings(
        x, feature_names, user_ids, args.embeddings, args.embedding_pool
    )
    label_names = _label_names_from_col(label_col)
    y = build_labels(user_data, label_col)

    graph_obj = {
        "x": x,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "edge_attr_feature_names": EDGE_FEATURE_NAMES,
        "y": y,
        "label_names": label_names,
        "feature_names": feature_names,
        "user_ids": user_ids,
        "u2i": {i: i for i in range(n_nodes)},
    }
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data.feature_names = feature_names
    data.edge_attr_feature_names = EDGE_FEATURE_NAMES
    data.label_names = label_names
    data.user_ids = user_ids
    graph_obj["data"] = data
    torch.save(graph_obj, args.out)

    meta = {
        "graph": args.graph,
        "csv": args.csv,
        "nodes": int(n_nodes),
        "edges": int(edge_index.shape[1]),
        "feature_cols": feature_cols,
        "label_col": label_col,
        "label_names": label_names,
        "labeled_nodes": int((y.numpy() >= 0).sum()),
        "all_label_cols": all_label_cols,
        "embeddings": args.embeddings,
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
