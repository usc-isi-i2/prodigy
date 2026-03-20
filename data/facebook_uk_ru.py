import os
import re
import shutil
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Data

from experiments.sampler import NeighborSampler
from models.sentence_embedding import SentenceEmb
from .augment import get_aug
from .dataloader import MulticlassTask, ParamSampler, BatchSampler, Collator, NeighborTask
from .dataset import SubgraphDataset


DEFAULT_FACEBOOK_PKL = "facebook_2022-06-24_2022-06-25_part1.pkl"
DEFAULT_FACEBOOK_EDGES_CSV = "facebook_2022-06-24_2022-06-25_part1_edges.csv"
DEFAULT_FACEBOOK_NODE_FEATURES_CSV = "facebook_2022-06-24_2022-06-25_part1_node_features.csv"


def _cache_token(value: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value))
    return clean[:80]


def _resolve_pickle_path(
        root: str,
        pkl_filename: str,
        source_pkl_path: str = ""
) -> str:
    os.makedirs(root, exist_ok=True)

    local_path = os.path.join(root, pkl_filename)
    if os.path.exists(local_path):
        return local_path

    processed_under_root = os.path.join(root, "processed", os.path.basename(pkl_filename))
    if os.path.exists(processed_under_root):
        return processed_under_root

    if os.path.exists(pkl_filename):
        return pkl_filename

    default_repo_processed = os.path.join("facebook_uk_ru", "processed", os.path.basename(pkl_filename))
    if os.path.exists(default_repo_processed):
        return default_repo_processed

    if os.path.isabs(pkl_filename) and os.path.exists(pkl_filename):
        return pkl_filename

    if source_pkl_path:
        if not os.path.exists(source_pkl_path):
            raise FileNotFoundError(f"source_pkl_path not found: {source_pkl_path}")
        if not os.path.exists(local_path):
            print(f"Copying dataset file to {local_path}")
            shutil.copy2(source_pkl_path, local_path)
        return local_path

    raise FileNotFoundError(
        f"Could not find pickle file. Looked for:\n"
        f"- {local_path}\n"
        f"- absolute path from pkl_filename ({pkl_filename})"
    )


def _resolve_data_path(root: str, filename: str) -> str:
    candidates = [
        os.path.join(root, filename),
        os.path.join(root, "processed", os.path.basename(filename)),
        filename,
        os.path.join("facebook_uk_ru", "processed", os.path.basename(filename)),
    ]
    if os.path.isabs(filename):
        candidates.insert(0, filename)
    for path in candidates:
        if path and os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"Could not find file '{filename}'. Checked:\n- " + "\n- ".join(candidates)
    )


def _parse_timestamp(date_val: Any) -> float:
    if date_val is None:
        return 0.0
    try:
        ts = pd.to_datetime(date_val, utc=True, errors="coerce")
        if pd.isna(ts):
            return 0.0
        return float(ts.timestamp())
    except Exception:
        return 0.0


def _extract_group_id(post_url: str) -> str:
    if not isinstance(post_url, str):
        return ""
    m = re.search(r"/groups/(\d+)", post_url)
    if m:
        return m.group(1)
    return ""


def _adjust_feature_dim(x: torch.Tensor, target_dim: int) -> torch.Tensor:
    if target_dim <= 0 or x.size(1) == target_dim:
        return x
    if x.size(1) > target_dim:
        return x[:, :target_dim]
    repeats = target_dim // x.size(1) + 1
    return x.repeat(1, repeats)[:, :target_dim]


def _load_precomputed_embeddings(
        embeddings_path: str,
        records: List[Dict[str, Any]],
        embedding_ids_path: str = "",
) -> Optional[torch.Tensor]:
    if not embeddings_path:
        return None
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")

    if embeddings_path.endswith(".npy"):
        emb = np.load(embeddings_path)
    else:
        emb = torch.load(embeddings_path, map_location="cpu")

    post_ids = [str(r["id"]) for r in records]

    if isinstance(emb, dict):
        rows = []
        missing = 0
        for pid in post_ids:
            if pid in emb:
                rows.append(torch.as_tensor(emb[pid]).float())
            else:
                missing += 1
                rows.append(None)

        if missing == len(post_ids):
            raise ValueError("No matching post ids found in embeddings dict.")

        first = next(x for x in rows if x is not None)
        dim = int(first.numel())
        fixed = []
        for row in rows:
            if row is None:
                fixed.append(torch.zeros(dim))
            else:
                fixed.append(row.reshape(-1))
        return torch.stack(fixed, dim=0)

    emb_tensor = torch.as_tensor(emb).float()
    if emb_tensor.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got shape {tuple(emb_tensor.shape)}")

    if embedding_ids_path:
        if not os.path.exists(embedding_ids_path):
            raise FileNotFoundError(f"Embedding ids file not found: {embedding_ids_path}")
        emb_ids = np.load(embedding_ids_path, allow_pickle=True)
        emb_ids = [str(v) for v in emb_ids.tolist()]
        if len(emb_ids) != emb_tensor.size(0):
            raise ValueError(
                f"Embedding ids count ({len(emb_ids)}) does not match embedding rows ({emb_tensor.size(0)})."
            )
        id_to_row = {pid: i for i, pid in enumerate(emb_ids)}
        missing = 0
        rows = []
        dim = emb_tensor.size(1)
        for pid in post_ids:
            idx = id_to_row.get(pid)
            if idx is None:
                missing += 1
                rows.append(torch.zeros(dim))
            else:
                rows.append(emb_tensor[idx])
        if missing == len(post_ids):
            raise ValueError("No matching ids found between dataset and embedding_ids file.")
        return torch.stack(rows, dim=0)

    if emb_tensor.size(0) != len(records):
        raise ValueError(
            f"Embedding rows ({emb_tensor.size(0)}) must match number of records ({len(records)})."
        )
    return emb_tensor


def _auto_detect_embeddings(
        pkl_path: str,
        explicit_embeddings_path: str,
        explicit_embedding_ids_path: str,
) -> Tuple[str, str]:
    if explicit_embeddings_path:
        return explicit_embeddings_path, explicit_embedding_ids_path

    base = os.path.splitext(os.path.basename(pkl_path))[0]
    parent = os.path.dirname(pkl_path)
    npy_path = os.path.join(parent, f"{base}_embeddings.npy")
    ids_path = os.path.join(parent, f"{base}_embedding_ids.npy")
    if os.path.exists(npy_path):
        return npy_path, ids_path if os.path.exists(ids_path) else ""
    return "", ""


def _numeric_features(records: List[Dict[str, Any]]) -> torch.Tensor:
    rows = []
    for r in records:
        account = r.get("account", {}) or {}
        stats = r.get("statistics", {}) or {}
        actual = stats.get("actual", {}) if isinstance(stats, dict) else {}
        if not isinstance(actual, dict):
            actual = {}

        msg_len = len(r.get("message", "") or "")
        has_link = 1.0 if (r.get("expandedLinks") and len(r.get("expandedLinks")) > 0) else 0.0
        feat = [
            float(account.get("subscriberCount") or 0),
            float(actual.get("commentCount") or 0),
            float(actual.get("likeCount") or 0),
            float(actual.get("shareCount") or 0),
            float(actual.get("loveCount") or 0),
            float(actual.get("wowCount") or 0),
            float(actual.get("hahaCount") or 0),
            float(actual.get("sadCount") or 0),
            float(actual.get("angryCount") or 0),
            float(actual.get("careCount") or 0),
            float(msg_len),
            has_link,
        ]
        rows.append(feat)

    x = torch.tensor(rows, dtype=torch.float)
    x = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-8)
    return x


def _text_features(
        records: List[Dict[str, Any]],
        model_name: str,
        device: Union[str, torch.device]
) -> torch.Tensor:
    model = SentenceEmb(model_name, device=device)
    texts = []
    for r in records:
        account = r.get("account", {}) or {}
        title = str(account.get("name") or "Unknown Account")
        message = str(r.get("message") or "")
        text = f"{title}. {message}"[:512]
        texts.append(text)
    return model.get_sentence_embeddings(texts).cpu()


def _make_records(
        raw: List[Dict[str, Any]],
        label_type: str,
        filter_to_uk_ru: bool,
        max_posts: Optional[int]
) -> List[Dict[str, Any]]:
    records = []
    for i, d in enumerate(raw):
        lang = str(d.get("languageCode") or "und").lower()
        if label_type == "uk_ru" and filter_to_uk_ru and lang not in {"uk", "ru"}:
            continue

        account = d.get("account", {}) or {}
        rec = {
            "id": str(d.get("id") or f"post_{i}"),
            "platform_id": str(d.get("platformId") or ""),
            "account_id": str(account.get("id") or ""),
            "group_id": _extract_group_id(str(d.get("postUrl") or "")),
            "timestamp": _parse_timestamp(d.get("date") or d.get("updated") or datetime.utcnow()),
            "message": str(d.get("message") or ""),
            "language": lang,
            "account": account,
            "statistics": d.get("statistics") or {},
            "expandedLinks": d.get("expandedLinks") or [],
        }
        records.append(rec)

        if max_posts is not None and len(records) >= max_posts:
            break

    if not records:
        raise ValueError("No records available after filtering.")
    return records


def _build_post_edges(records: List[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor]:
    by_account = defaultdict(list)
    by_group = defaultdict(list)

    for idx, rec in enumerate(records):
        if rec["account_id"]:
            by_account[rec["account_id"]].append((rec["timestamp"], idx))
        if rec["group_id"]:
            by_group[rec["group_id"]].append((rec["timestamp"], idx))

    edges: List[List[int]] = []
    edge_types: List[int] = []

    def chain_edges(groups: Dict[str, List[Tuple[float, int]]], edge_type: int) -> None:
        for _, items in groups.items():
            if len(items) < 2:
                continue
            items.sort(key=lambda x: x[0])
            for i in range(len(items) - 1):
                a = items[i][1]
                b = items[i + 1][1]
                if a == b:
                    continue
                edges.append([a, b])
                edge_types.append(edge_type)
                edges.append([b, a])
                edge_types.append(edge_type)

    # 0: consecutive post from same account, 1: consecutive post in same group
    chain_edges(by_account, edge_type=0)
    chain_edges(by_group, edge_type=1)

    if len(edges) == 0:
        # Fallback chain so the graph is not empty.
        for i in range(len(records) - 1):
            edges.append([i, i + 1])
            edge_types.append(0)

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_types, dtype=torch.long)
    return edge_index, edge_attr


def _create_labels(records: List[Dict[str, Any]], label_type: str) -> Tuple[List[int], List[str]]:
    if label_type == "uk_ru":
        labels = []
        for r in records:
            if r["language"] == "ru":
                labels.append(0)
            elif r["language"] == "uk":
                labels.append(1)
            else:
                labels.append(2)
        label_names = ["ru", "uk", "other"]
    elif label_type == "lang":
        langs = sorted(list({r["language"] for r in records}))
        l2i = {lang: i for i, lang in enumerate(langs)}
        labels = [l2i[r["language"]] for r in records]
        label_names = langs
    else:
        raise ValueError(f"Unknown label_type: {label_type}")
    return labels, label_names


def _graph_cache_name(
        label_type: str,
        max_posts: Optional[int],
        filter_to_uk_ru: bool,
        embeddings_path: str,
        text_emb_model: str,
        target_dim: int,
        data_source: str = "pkl",
        edges_filename: str = "",
        node_features_filename: str = "",
        use_edge_features: bool = False,
) -> str:
    parts = [
        "facebook_graph",
        _cache_token(data_source),
        _cache_token(label_type),
        f"filt{int(filter_to_uk_ru)}",
        f"tdim{target_dim}",
    ]
    if max_posts:
        parts.append(f"max{max_posts}")
    if embeddings_path:
        parts.append(f"emb_{_cache_token(os.path.basename(embeddings_path))}")
    elif text_emb_model:
        parts.append(f"txt_{_cache_token(text_emb_model)}")
    else:
        parts.append("num")
    if edges_filename:
        parts.append(f"edges_{_cache_token(os.path.basename(edges_filename))}")
    if node_features_filename:
        parts.append(f"nodes_{_cache_token(os.path.basename(node_features_filename))}")
    parts.append(f"edgefeat{int(use_edge_features)}")
    return "_".join(parts) + ".pt"


def _build_undirected_edges(
        edge_df: pd.DataFrame,
        use_edge_features: bool = False,
        edge_feature_columns: Optional[List[str]] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[Any, int]]:
    if edge_df.shape[1] < 2:
        raise ValueError("Edge CSV must have at least two columns for source/destination.")

    src_col, dst_col = list(edge_df.columns[:2])
    sources = edge_df[src_col]
    destinations = edge_df[dst_col]
    node_ids = pd.unique(pd.concat([sources, destinations], ignore_index=True))
    node_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    edges: Set[Tuple[int, int]] = set()
    edge_feats_by_edge: Dict[Tuple[int, int], List[float]] = {}

    selected_edge_cols: List[str] = []
    if use_edge_features:
        if isinstance(edge_feature_columns, str):
            edge_feature_columns = [c.strip() for c in edge_feature_columns.split(",") if c.strip()]
        if edge_feature_columns:
            selected_edge_cols = [c for c in edge_feature_columns if c in edge_df.columns]
        else:
            selected_edge_cols = list(edge_df.columns[2:])

    for _, row in edge_df.iterrows():
        src_raw = row[src_col]
        dst_raw = row[dst_col]
        src = node_to_idx[src_raw]
        dst = node_to_idx[dst_raw]
        if src == dst:
            continue
        a, b = (src, dst) if src < dst else (dst, src)
        edges.add((a, b))

        if selected_edge_cols:
            values: List[float] = []
            for c in selected_edge_cols:
                val = row[c]
                try:
                    values.append(float(val))
                except Exception:
                    values.append(0.0)
            edge_feats_by_edge[(a, b)] = values

    if not edges:
        raise ValueError("No valid edges found in edge CSV.")

    directed_edges: List[List[int]] = []
    directed_feats: List[List[float]] = []
    for a, b in sorted(edges):
        directed_edges.append([a, b])
        directed_edges.append([b, a])
        if selected_edge_cols:
            feat = edge_feats_by_edge.get((a, b), [0.0] * len(selected_edge_cols))
            directed_feats.append(feat)
            directed_feats.append(feat)

    edge_index = torch.tensor(directed_edges, dtype=torch.long).t().contiguous()
    edge_attr = None
    if selected_edge_cols:
        edge_attr = torch.tensor(directed_feats, dtype=torch.float)
    return edge_index, edge_attr, node_to_idx


def _node_features_to_tensor(
        node_df: pd.DataFrame,
        node_to_idx: Dict[Any, int],
        target_dim: int,
        max_nodes: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    max_edge_idx = max(node_to_idx.values()) if node_to_idx else -1
    num_nodes = max(max_edge_idx + 1, len(node_df))
    if max_nodes is not None:
        num_nodes = min(num_nodes, max_nodes)

    all_columns = list(node_df.columns)
    id_columns = [c for c in all_columns if c.lower() in {"accountid", "account_id", "id", "node_id"}]
    feature_columns = [c for c in all_columns if c not in id_columns]
    if not feature_columns:
        raise ValueError("Node features CSV has no usable numeric feature columns.")

    # Use row order as default mapping; many preprocessing pipelines export node features in node-index order.
    id_to_row: Dict[Any, int] = {}
    if id_columns:
        id_col = id_columns[0]
        id_to_row = {row_id: idx for idx, row_id in enumerate(node_df[id_col].tolist())}

    x = torch.zeros((num_nodes, len(feature_columns)), dtype=torch.float)
    labels = torch.full((num_nodes,), 2, dtype=torch.long)  # default "other"

    ru_col = "language_dist_ru" if "language_dist_ru" in node_df.columns else None
    uk_col = "language_dist_uk" if "language_dist_uk" in node_df.columns else None

    for original_node_id, node_idx in node_to_idx.items():
        if node_idx >= num_nodes:
            continue

        row_idx = None
        if id_to_row:
            row_idx = id_to_row.get(original_node_id)
        if row_idx is None and isinstance(original_node_id, (int, np.integer)):
            if 0 <= int(original_node_id) < len(node_df):
                row_idx = int(original_node_id)
        if row_idx is None and 0 <= node_idx < len(node_df):
            row_idx = node_idx
        if row_idx is None or row_idx >= len(node_df):
            continue

        row = node_df.iloc[row_idx]
        values = pd.to_numeric(row[feature_columns], errors="coerce").fillna(0.0).astype(float).to_numpy()
        x[node_idx] = torch.from_numpy(values).float()

        if ru_col and uk_col:
            ru_val = float(row.get(ru_col, 0.0) or 0.0)
            uk_val = float(row.get(uk_col, 0.0) or 0.0)
            if ru_val > uk_val and ru_val > 0:
                labels[node_idx] = 0
            elif uk_val > ru_val and uk_val > 0:
                labels[node_idx] = 1
            else:
                labels[node_idx] = 2

    x = _adjust_feature_dim(x, target_dim=target_dim)
    return x, labels


def load_facebook_uk_ru_csv(
        edge_csv_path: str,
        node_features_csv_path: str,
        use_edge_features: bool = False,
        edge_feature_columns: Optional[List[str]] = None,
        target_dim: int = 768,
        max_nodes: Optional[int] = None,
) -> Data:
    print(f"Loading Facebook edges from {edge_csv_path}...")
    edge_df = pd.read_csv(edge_csv_path)
    print(f"Loading Facebook node features from {node_features_csv_path}...")
    node_df = pd.read_csv(node_features_csv_path)

    edge_index, edge_attr, node_to_idx = _build_undirected_edges(
        edge_df=edge_df,
        use_edge_features=use_edge_features,
        edge_feature_columns=edge_feature_columns,
    )
    x, y = _node_features_to_tensor(
        node_df=node_df,
        node_to_idx=node_to_idx,
        target_dim=target_dim,
        max_nodes=max_nodes,
    )

    graph = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        num_nodes=x.size(0),
    )
    return graph


def load_facebook_uk_ru_pkl(
        pkl_path: str,
        label_type: str = "uk_ru",
        filter_to_uk_ru: bool = True,
        max_posts: Optional[int] = None,
        embeddings_path: str = "",
        embedding_ids_path: str = "",
        text_emb_model: str = "",
        bert: Optional[Any] = None,
        bert_device: Union[str, torch.device] = "cpu",
        target_dim: int = 768
) -> Data:
    if label_type == "verified":
        label_type = "uk_ru"

    print(f"Loading Facebook pickle from {pkl_path}...")
    raw = pd.read_pickle(pkl_path)
    if not isinstance(raw, list):
        raise ValueError(f"Expected pickled list[dict], got {type(raw)}")

    records = _make_records(
        raw=raw,
        label_type=label_type,
        filter_to_uk_ru=filter_to_uk_ru,
        max_posts=max_posts,
    )
    print(f"Loaded {len(records)} posts after filtering")

    edge_index, edge_attr = _build_post_edges(records)
    labels, label_names = _create_labels(records, label_type)

    x: Optional[torch.Tensor] = _load_precomputed_embeddings(
        embeddings_path,
        records,
        embedding_ids_path=embedding_ids_path,
    )
    if x is None:
        text_model_name = text_emb_model
        if not text_model_name and isinstance(bert, str) and bert.strip():
            text_model_name = bert

        if text_model_name:
            print(f"Generating text embeddings with model: {text_model_name}")
            x = _text_features(records, text_model_name, bert_device)
        elif bert is not None and hasattr(bert, "get_sentence_embeddings"):
            texts = [f"{r.get('message', '')}"[:512] for r in records]
            x = bert.get_sentence_embeddings(texts).cpu()
        else:
            print("Using numeric fallback features")
            x = _numeric_features(records)

    x = _adjust_feature_dim(x.float(), target_dim=target_dim)
    y = torch.tensor(labels, dtype=torch.long)

    graph = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        num_nodes=len(records),
    )
    return graph


def facebook_uk_ru_task(
        labels: np.ndarray,
        num_classes: int,
        split: str,
        label_set: Optional[Set[int]] = None,
        split_labels: bool = True,
        train_cap: Optional[int] = None,
        linear_probe: bool = False,
) -> MulticlassTask:
    if label_set is not None:
        chosen_label_set = set(label_set)
    elif split_labels:
        all_labels = list(range(num_classes))
        n_train = max(1, int(num_classes * 0.6))
        n_val = max(1, int(num_classes * 0.2))
        train_labels = all_labels[:n_train]
        val_labels = all_labels[n_train:n_train + n_val]
        test_labels = all_labels[n_train + n_val:]
        if not test_labels and val_labels:
            test_labels = [val_labels.pop()]
        if not val_labels and train_labels:
            val_labels = [train_labels.pop()]

        if split == "train":
            chosen_label_set = set(train_labels)
        elif split == "val":
            chosen_label_set = set(val_labels)
        elif split == "test":
            chosen_label_set = set(test_labels)
        else:
            raise ValueError(f"Invalid split: {split}")
    else:
        chosen_label_set = set(range(num_classes))

    train_label = None
    if train_cap is not None and split == "train":
        train_label = labels.copy()
        for i in range(num_classes):
            idx = np.where(labels == i)[0]
            if len(idx) > train_cap:
                disabled_idx = idx[train_cap:]
                train_label[disabled_idx] = -1 - i

    return MulticlassTask(labels, chosen_label_set, train_label, linear_probe)


def get_facebook_uk_ru_dataset(
        root: str,
        pkl_filename: str = DEFAULT_FACEBOOK_PKL,
        facebook_edges_filename: str = DEFAULT_FACEBOOK_EDGES_CSV,
        facebook_node_features_filename: str = DEFAULT_FACEBOOK_NODE_FEATURES_CSV,
        facebook_use_edge_features: bool = False,
        facebook_edge_feature_columns: Optional[List[str]] = None,
        facebook_data_source: str = "csv",
        source_pkl_path: str = "",
        label_type: str = "uk_ru",
        n_hop: int = 2,
        bert: Optional[Any] = None,
        bert_device: Union[str, torch.device] = "cpu",
        original_features: bool = False,
        max_posts: Optional[int] = None,
        max_users: Optional[int] = None,
        facebook_embeddings_path: str = "",
        facebook_embedding_ids_path: str = "",
        facebook_text_emb_model: str = "",
        facebook_target_dim: int = 768,
        facebook_filter_to_uk_ru: bool = True,
        **kwargs
) -> SubgraphDataset:
    del max_users  # unused, kept for interface compatibility with existing call sites.
    if label_type == "verified":
        label_type = "uk_ru"

    data_source = str(facebook_data_source or "csv").strip().lower()
    if data_source not in {"csv", "pkl"}:
        raise ValueError(f"facebook_data_source must be 'csv' or 'pkl', got: {facebook_data_source}")

    pkl_path = ""
    edge_csv_path = ""
    node_features_csv_path = ""
    if data_source == "pkl":
        pkl_path = _resolve_pickle_path(root, pkl_filename, source_pkl_path=source_pkl_path)
        facebook_embeddings_path, facebook_embedding_ids_path = _auto_detect_embeddings(
            pkl_path=pkl_path,
            explicit_embeddings_path=facebook_embeddings_path,
            explicit_embedding_ids_path=facebook_embedding_ids_path,
        )
        if original_features:
            facebook_text_emb_model = ""
            facebook_embeddings_path = ""
            facebook_embedding_ids_path = ""
    else:
        edge_csv_path = _resolve_data_path(root, facebook_edges_filename)
        node_features_csv_path = _resolve_data_path(root, facebook_node_features_filename)

    cache_name = _graph_cache_name(
        label_type=label_type,
        max_posts=max_posts,
        filter_to_uk_ru=facebook_filter_to_uk_ru,
        embeddings_path=facebook_embeddings_path if data_source == "pkl" else "",
        text_emb_model=facebook_text_emb_model if data_source == "pkl" else "",
        target_dim=facebook_target_dim,
        data_source=data_source,
        edges_filename=edge_csv_path,
        node_features_filename=node_features_csv_path,
        use_edge_features=facebook_use_edge_features,
    )
    cache_path = os.path.join(root, cache_name)

    if os.path.exists(cache_path):
        print(f"Loading cached graph from {cache_path}")
        graph = torch.load(cache_path)
    else:
        if data_source == "csv":
            graph = load_facebook_uk_ru_csv(
                edge_csv_path=edge_csv_path,
                node_features_csv_path=node_features_csv_path,
                use_edge_features=facebook_use_edge_features,
                edge_feature_columns=facebook_edge_feature_columns,
                target_dim=facebook_target_dim,
                max_nodes=max_posts,
            )
        else:
            graph = load_facebook_uk_ru_pkl(
                pkl_path=pkl_path,
                label_type=label_type,
                filter_to_uk_ru=facebook_filter_to_uk_ru,
                max_posts=max_posts,
                embeddings_path=facebook_embeddings_path,
                embedding_ids_path=facebook_embedding_ids_path,
                text_emb_model=facebook_text_emb_model,
                bert=bert,
                bert_device=bert_device,
                target_dim=facebook_target_dim,
            )
        print(f"Caching graph to {cache_path}")
        torch.save(graph, cache_path)

    print(f"Graph: {graph.num_nodes} nodes, {graph.edge_index.shape[1]} edges")
    neighbor_sampler = NeighborSampler(graph, num_hops=n_hop)
    return SubgraphDataset(graph, neighbor_sampler, bidirectional=False)


def get_facebook_uk_ru_dataloader(
        dataset: SubgraphDataset,
        split: str,
        node_split: str,
        batch_size: Union[int, range],
        n_way: Union[int, range],
        n_shot: Union[int, range],
        n_query: Union[int, range],
        batch_count: int,
        root: str,
        bert: Optional[Any],
        num_workers: int,
        aug: str,
        aug_test: bool,
        split_labels: bool,
        train_cap: Optional[int],
        linear_probe: bool,
        label_set: Optional[Set[int]] = None,
        facebook_text_emb_model: str = "",
        **kwargs
) -> DataLoader:
    del node_split, root
    task_name = kwargs.get("task_name", "classification")
    seed = sum(ord(c) for c in split)

    graph = dataset.graph
    labels = graph.y.numpy()
    if task_name == "neighbor_matching":
        sampler = BatchSampler(
            batch_count,
            NeighborTask(dataset.neighbor_sampler, graph.num_nodes, "inout"),
            ParamSampler(batch_size, n_way, n_shot, n_query, 1),
            seed=seed,
        )
        label_embeddings = torch.zeros(1, 768).expand(graph.num_nodes, -1)
    elif task_name == "classification":
        label_names = list(getattr(graph, "label_names", ["ru", "uk", "other"]))
        num_classes = len(label_names)

        if bert is not None:
            label_embeddings = bert.get_sentence_embeddings(label_names)
        elif facebook_text_emb_model:
            tmp_bert = SentenceEmb(facebook_text_emb_model, device="cpu")
            label_embeddings = tmp_bert.get_sentence_embeddings(label_names)
        else:
            label_embeddings = torch.randn(len(label_names), 768)

        task = facebook_uk_ru_task(
            labels=labels,
            num_classes=num_classes,
            split=split,
            label_set=label_set,
            split_labels=split_labels,
            train_cap=train_cap,
            linear_probe=linear_probe,
        )

        sampler = BatchSampler(
            batch_count,
            task,
            ParamSampler(batch_size, n_way, n_shot, n_query, 1),
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown task for Facebook UK/RU: {task_name}")

    aug_fn = get_aug(aug, dataset.graph.x) if (split == "train" or aug_test) else get_aug("")

    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        collate_fn=Collator(label_embeddings, aug=aug_fn),
    )
