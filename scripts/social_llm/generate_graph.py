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
        [--embeddings /path/to/user_bio_embeddings_gte_multilingual_base.pt]
"""
import argparse
import io
import json
import os
import pickle
import shutil
from pathlib import Path

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
    feature_cols = []
    for col in df.columns:
        if col in skip:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().any():
            feature_cols.append(col)
    return feature_cols


def _label_names_from_col(col: str):
    base = col[len("label_"):]
    return [f"non_{base}", base]


def _label_stem(label_col: str) -> str:
    return label_col[len("label_"):] if label_col.startswith("label_") else label_col


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
        "--label_cols", default="",
        help="Comma-separated label columns to emit. Overrides --label_col when set.",
    )
    p.add_argument(
        "--all_label_cols", action="store_true",
        help="Emit one graph per detected label_* column.",
    )
    p.add_argument(
        "--out_dir", default="",
        help="Directory for --label_cols/--all_label_cols outputs. Defaults to dirname(--out).",
    )
    p.add_argument(
        "--write_default_copy", action="store_true",
        help="Also copy the first emitted label graph to --out / retweet_graph.pt.",
    )
    p.add_argument(
        "--feature_cols", default="",
        help="Comma-sep feature columns. Auto-detected from numeric non-label cols if empty.",
    )
    p.add_argument(
        "--embeddings-only",
        action="store_true",
        help="Use only attached embedding features and skip numeric user_data features.",
    )
    p.add_argument("--embeddings", default="", help="Optional user_embeddings_*.pt path.")
    p.add_argument("--embedding_feature_prefix", default="emb")
    p.add_argument("--embedding_pool", choices=["meanpool", "maxpool"], default="meanpool")
    p.add_argument(
        "--edge_view_aliases",
        default="retweet_all,temporal_history",
        help="Comma-separated aliases that should resolve to the default static edge graph.",
    )
    return p.parse_args()


def read_user_data_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (UnicodeDecodeError, pd.errors.ParserError):
        raw = Path(path).read_bytes().replace(b"\x00", b"")
        text = raw.decode("utf-8", errors="replace")
        return pd.read_csv(io.StringIO(text), engine="python")


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


def validate_graph_node_ids(G, n_nodes: int):
    if G.number_of_nodes() == 0:
        raise ValueError("Graph has no nodes.")
    nodes = np.asarray(list(G.nodes()), dtype=np.int64)
    min_node = int(nodes.min())
    max_node = int(nodes.max())
    if min_node < 0 or max_node >= n_nodes:
        raise ValueError(
            f"Graph node ids must be row indices within user_data.csv. "
            f"Got min={min_node}, max={max_node}, rows={n_nodes}."
        )
    missing = n_nodes - len(np.unique(nodes))
    print(
        f"Graph node id check: min={min_node} max={max_node} "
        f"unique_graph_nodes={len(np.unique(nodes)):,} csv_rows={n_nodes:,} "
        f"csv_rows_without_graph_node={missing:,}"
    )


def build_node_features(user_data, feature_cols):
    if not feature_cols:
        print("No feature cols detected — node features will come from embeddings only.")
        return torch.zeros((len(user_data), 1), dtype=torch.float), []

    feat = user_data[feature_cols].copy()
    for col in feature_cols:
        feat[col] = pd.to_numeric(feat[col], errors="coerce")
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


def maybe_attach_embeddings(x, feature_names, user_ids, embeddings_path, embedding_pool, feature_prefix):
    if not embeddings_path:
        return x, feature_names, {"matched_users": 0, "embedding_dim": 0}

    try:
        emb = torch.load(embeddings_path, map_location="cpu", weights_only=False)
    except TypeError:
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
    names_out = feature_names + [f"{feature_prefix}_{k}" for k in range(emb_dim)]
    print(f"Embeddings attached: matched={matched:,}/{len(user_ids):,} dim={emb_dim}")
    return x_out, names_out, {"matched_users": matched, "embedding_dim": emb_dim}


def _numeric_label_series(series: "pd.Series") -> "pd.Series":
    return pd.to_numeric(series, errors="coerce")


def _is_continuous(series: "pd.Series") -> bool:
    numeric = _numeric_label_series(series)
    observed = numeric.dropna().to_numpy()
    if observed.size == 0:
        return False
    unique = set(np.unique(observed).tolist())
    return not unique.issubset({0, 1})


def build_labels(user_data, label_col):
    s = _numeric_label_series(user_data[label_col])
    if _is_continuous(user_data[label_col]):
        y_np = s.to_numpy(dtype=np.float32, na_value=np.nan)
        y = torch.from_numpy(y_np)
        labeled = int(np.isfinite(y_np).sum())
        if labeled > 0:
            min_val = float(np.nanmin(y_np))
            max_val = float(np.nanmax(y_np))
            mean_val = float(np.nanmean(y_np))
        else:
            min_val = float("nan")
            max_val = float("nan")
            mean_val = float("nan")
        print(f"Labels ({label_col}): regression  "
              f"min={min_val:.4f}  max={max_val:.4f}  "
              f"mean={mean_val:.4f}  labeled={labeled:,}  "
              f"nulls={int(s.isna().sum())}")
    else:
        y_np = s.fillna(-1).to_numpy(dtype=np.int64)
        y = torch.from_numpy(y_np)
        labeled = int((y_np >= 0).sum())
        vals, counts = np.unique(y_np[y_np >= 0], return_counts=True)
        print(f"Labels ({label_col}): {labeled:,} labeled  "
              f"dist={dict(zip(vals.tolist(), counts.tolist()))}")
    return y


def add_edge_view_aliases(graph_obj, edge_index, edge_attr, aliases):
    aliases = [alias.strip() for alias in aliases.split(",") if alias.strip()]
    if not aliases:
        return
    graph_obj["edge_index_views"] = {alias: edge_index for alias in aliases}
    graph_obj["edge_attr_views"] = {alias: edge_attr for alias in aliases}
    graph_obj["edge_attr_feature_names_views"] = {
        alias: EDGE_FEATURE_NAMES for alias in aliases
    }


def save_graph_for_label(
    args,
    label_col,
    out_path,
    *,
    edge_index,
    edge_attr,
    x,
    feature_names,
    user_ids,
    user_data,
    all_label_cols,
    emb_stats,
):
    y = build_labels(user_data, label_col)
    is_reg = _is_continuous(user_data[label_col])
    label_names = (
        [_label_stem(label_col)]
        if is_reg
        else _label_names_from_col(label_col)
    )
    label_type = "regression" if is_reg else "classification"

    graph_obj = {
        "x": x,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "edge_attr_feature_names": EDGE_FEATURE_NAMES,
        "y": y,
        "label_type": label_type,
        "label_names": label_names,
        "feature_names": feature_names,
        "user_ids": user_ids,
        "u2i": {i: i for i in range(len(user_ids))},
    }
    add_edge_view_aliases(graph_obj, edge_index, edge_attr, args.edge_view_aliases)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data.feature_names = feature_names
    data.edge_attr_feature_names = EDGE_FEATURE_NAMES
    data.label_names = label_names
    data.label_type = label_type
    data.user_ids = user_ids
    graph_obj["data"] = data

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(graph_obj, out_path)

    meta = {
        "graph": args.graph,
        "csv": args.csv,
        "nodes": int(len(user_ids)),
        "edges": int(edge_index.shape[1]),
        "feature_cols": [
            name for name in feature_names if not name.startswith("emb_")
        ],
        "feature_dim": int(x.shape[1]),
        "label_col": label_col,
        "label_names": label_names,
        "label_type": label_type,
        "labeled_nodes": int(np.isfinite(y.numpy()).sum())
        if is_reg else int((y.numpy() >= 0).sum()),
        "all_label_cols": all_label_cols,
        "embeddings": args.embeddings,
        "embedding_pool": args.embedding_pool,
        "embedding_dim": emb_stats["embedding_dim"],
        "embedding_matched_users": emb_stats["matched_users"],
        "edge_view_aliases": [
            alias.strip() for alias in args.edge_view_aliases.split(",") if alias.strip()
        ],
    }
    meta_path = out_path.replace(".pt", ".meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved graph: {out_path}")
    print(f"Saved meta:  {meta_path}")


def resolve_label_cols(args, all_label_cols):
    if args.all_label_cols:
        return all_label_cols
    if args.label_cols:
        return [c.strip() for c in args.label_cols.split(",") if c.strip()]
    label_col = args.label_col or (all_label_cols[0] if all_label_cols else None)
    return [label_col] if label_col else []


def output_path_for_label(args, label_col, multiple):
    if not multiple:
        return args.out
    out_dir = args.out_dir or os.path.dirname(args.out) or "."
    return os.path.join(out_dir, f"retweet_graph_{_label_stem(label_col)}.pt")


def main():
    args = parse_args()
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
    elif os.path.dirname(args.out):
        os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print(f"Loading graph: {args.graph}")
    G = load_graph(args.graph)

    print(f"Loading user_data: {args.csv}")
    user_data = read_user_data_csv(args.csv)
    print(f"Shape: {user_data.shape}  columns: {user_data.columns.tolist()}")
    validate_graph_node_ids(G, len(user_data))

    all_label_cols = _detect_label_cols(user_data)
    label_cols = resolve_label_cols(args, all_label_cols)
    if not label_cols:
        raise ValueError("No label_* column found and --label_col not specified.")
    unknown = sorted(set(label_cols) - set(user_data.columns))
    if unknown:
        raise ValueError(f"Unknown label columns: {unknown}")

    if args.embeddings_only:
        if not args.embeddings:
            raise ValueError("--embeddings-only requires --embeddings")
        feature_cols = []
    else:
        feature_cols = (
            [c.strip() for c in args.feature_cols.split(",") if c.strip()]
            if args.feature_cols else _detect_feature_cols(user_data, all_label_cols)
        )
    print(f"Label cols: {label_cols}")
    print(f"Feature cols: {feature_cols}")

    n_nodes = len(user_data)
    user_ids = list(range(n_nodes))

    edge_index, edge_attr = build_edge_tensors(G)
    x, feature_names = build_node_features(user_data, feature_cols)
    x, feature_names, emb_stats = maybe_attach_embeddings(
        x,
        feature_names,
        user_ids,
        args.embeddings,
        args.embedding_pool,
        args.embedding_feature_prefix,
    )

    written = []
    multiple = len(label_cols) > 1 or args.all_label_cols or bool(args.label_cols)
    for label_col in label_cols:
        out_path = output_path_for_label(args, label_col, multiple)
        save_graph_for_label(
            args,
            label_col,
            out_path,
            edge_index=edge_index,
            edge_attr=edge_attr,
            x=x,
            feature_names=feature_names,
            user_ids=user_ids,
            user_data=user_data,
            all_label_cols=all_label_cols,
            emb_stats=emb_stats,
        )
        written.append(out_path)

    if args.write_default_copy and written:
        default_out = args.out
        default_meta = default_out.replace(".pt", ".meta.json")
        if os.path.abspath(default_out) != os.path.abspath(written[0]):
            shutil.copyfile(written[0], default_out)
            shutil.copyfile(written[0].replace(".pt", ".meta.json"), default_meta)
            print(f"Saved default copy: {default_out}")


if __name__ == "__main__":
    main()
