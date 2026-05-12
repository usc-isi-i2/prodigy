"""Graph evaluation utilities: node splitting, label downsampling, feature subsetting.

These functions are dataset-agnostic and operate on raw numpy arrays or PyTorch
tensors.  They were previously duplicated inside dataset-specific modules; place
them here so any dataset script or baseline can import them from a single location.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
from torch_geometric.data import Data


# ---------------------------------------------------------------------------
# Train / val / test split
# ---------------------------------------------------------------------------

def build_stratified_node_splits(
    labels: np.ndarray,
    *,
    seed: int = 0,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
) -> Dict[str, np.ndarray]:
    """Return stratified train/val/test index arrays for a node-classification task.

    Parameters
    ----------
    labels:
        Integer label array of shape ``(n_nodes,)``.  Nodes with label ``< 0``
        are treated as unlabelled and excluded from all splits.
    seed:
        RNG seed for reproducible shuffling.
    train_frac, val_frac:
        Target fractions for train and val sets; the remainder goes to test.

    Returns
    -------
    dict with keys ``"train"``, ``"val"``, ``"test"`` mapping to int64 index arrays.
    """
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    split_members: Dict[str, list] = {"train": [], "val": [], "test": []}

    classes = sorted(int(c) for c in np.unique(labels) if int(c) >= 0)
    for cls in classes:
        idx = np.where(labels == cls)[0].copy()
        if idx.size == 0:
            continue
        rng.shuffle(idx)

        n = int(idx.size)
        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))

        if n >= 3:
            n_train = min(max(1, n_train), n - 2)
            n_val = min(max(1, n_val), n - n_train - 1)
        elif n == 2:
            n_train, n_val = 1, 0
        else:
            n_train, n_val = 1, 0

        n_test = n - n_train - n_val
        if n >= 3 and n_test <= 0:
            n_test = 1
            if n_val > 1:
                n_val -= 1
            else:
                n_train -= 1

        split_members["train"].append(idx[:n_train])
        split_members["val"].append(idx[n_train : n_train + n_val])
        split_members["test"].append(idx[n_train + n_val :])

    return {
        key: (np.concatenate(vals) if vals else np.empty(0, dtype=np.int64))
        for key, vals in split_members.items()
    }


# ---------------------------------------------------------------------------
# Label downsampling
# ---------------------------------------------------------------------------

def apply_label_downsample(
    labels: torch.Tensor,
    label_names: List[str],
    ratio_spec: str,
    *,
    seed: int = 0,
) -> torch.Tensor:
    """Downsample per-class labels to a target ratio.

    Parameters
    ----------
    labels:
        Integer label tensor; ``-1`` for unlabelled nodes.
    label_names:
        Ordered list of class names (length must match number of classes).
    ratio_spec:
        Colon-separated target ratios, e.g. ``"50:50"`` or ``"20:80"``.
        An empty string is a no-op.
    seed:
        RNG seed.

    Returns
    -------
    New label tensor with excess nodes masked to ``-1``.
    """
    spec = (ratio_spec or "").strip()
    if spec == "":
        return labels

    parts = [p.strip() for p in spec.replace("/", ":").split(":") if p.strip()]
    if len(parts) < 2:
        raise ValueError(
            f"Invalid label_downsample='{ratio_spec}'. "
            "Use colon-separated ratios like '50:50' or '20:80'."
        )

    try:
        target = np.asarray([float(p) for p in parts], dtype=np.float64)
    except ValueError as ex:
        raise ValueError(
            f"Invalid label_downsample='{ratio_spec}'. "
            "Ratios must be numeric, e.g. '50:50'."
        ) from ex

    if np.any(target <= 0):
        raise ValueError(
            f"Invalid label_downsample='{ratio_spec}'. "
            "All ratio entries must be > 0."
        )

    num_classes = len(label_names)
    if target.size != num_classes:
        raise ValueError(
            f"label_downsample='{ratio_spec}' has {target.size} entries, "
            f"but graph has {num_classes} labels: {list(label_names)}"
        )

    labels_np = labels.detach().cpu().numpy().copy()
    present_classes = [cls for cls in range(num_classes) if np.any(labels_np == cls)]
    if len(present_classes) != num_classes:
        raise ValueError(
            "apply_label_downsample requires all classes to be present in labels. "
            f"Present classes: {present_classes}, expected 0..{num_classes - 1}."
        )

    counts = np.asarray(
        [int(np.sum(labels_np == cls)) for cls in range(num_classes)], dtype=np.int64
    )
    target_norm = target / target.sum()
    max_total = min(counts[i] / target_norm[i] for i in range(num_classes))
    target_counts = np.floor(max_total * target_norm + 1e-9).astype(np.int64)
    target_counts = np.minimum(target_counts, counts)
    if np.any(target_counts <= 0):
        raise ValueError(
            f"label_downsample='{ratio_spec}' produced invalid target counts "
            f"{target_counts.tolist()} from observed counts {counts.tolist()}."
        )

    rng = np.random.default_rng(seed)
    keep_idx: list = []
    for cls, keep_n in enumerate(target_counts):
        cls_idx = np.where(labels_np == cls)[0]
        rng.shuffle(cls_idx)
        keep_idx.append(cls_idx[:keep_n])
    keep_idx_arr = np.sort(np.concatenate(keep_idx))

    out = np.full_like(labels_np, -1)
    out[keep_idx_arr] = labels_np[keep_idx_arr]

    before = ", ".join(f"{label_names[i]}={counts[i]}" for i in range(num_classes))
    after = ", ".join(f"{label_names[i]}={target_counts[i]}" for i in range(num_classes))
    print(
        f"Applied label downsampling '{ratio_spec}' with seed={seed}: "
        f"{before} -> {after}."
    )
    return torch.as_tensor(out, dtype=labels.dtype)


# ---------------------------------------------------------------------------
# Feature subsetting
# ---------------------------------------------------------------------------

def apply_feature_subset(graph: Data, subset_spec: str) -> Data:
    """Select a subset of node features in-place on a PyG ``Data`` object.

    Parameters
    ----------
    graph:
        PyG ``Data`` object with ``x``, and optionally ``feature_names`` and
        ``label_names`` attributes.
    subset_spec:
        One of:

        - ``"all"`` — keep all features (no-op).
        - ``"constant1"`` — replace features with a single all-ones column.
        - ``"stats_only"`` — keep non-embedding features (any without ``emb_`` prefix).
        - ``"emb_only"`` — keep only embedding features (``emb_`` prefix).
        - ``"emb_only_plus_label"`` — embedding features concatenated with a
          one-hot label-leak column (for upper-bound experiments).
        - ``"label_only"`` — one-hot label-leak column only.
        - ``"keep:<f1>,<f2>"`` — keep named features.
        - ``"drop:<f1>,<f2>"`` — drop named features.

    Returns
    -------
    The same ``graph`` object with ``x`` and ``feature_names`` updated.
    """
    spec = (subset_spec or "all").strip()
    if spec in {"", "all"}:
        print(f"Using all features ({graph.x.shape[1]} dims).")
        return graph

    if spec == "constant1":
        graph.x = torch.ones(
            (graph.x.shape[0], 1), dtype=graph.x.dtype, device=graph.x.device
        )
        graph.feature_names = ["const_1"]
        print("Using constant node feature: 1 dim (all ones).")
        return graph

    feature_names = list(getattr(graph, "feature_names", []))
    x_dim = graph.x.shape[1]
    if not feature_names or len(feature_names) != x_dim:
        feature_names = [f"f{i}" for i in range(x_dim)]

    emb_mask = [name.startswith("emb_") for name in feature_names]
    stats_idx = [i for i, is_emb in enumerate(emb_mask) if not is_emb]
    emb_idx = [i for i, is_emb in enumerate(emb_mask) if is_emb]
    label_names = list(getattr(graph, "label_names", []))
    n_label = max(1, len(label_names))

    def _build_label_leak():
        leak = torch.zeros(
            (graph.x.shape[0], n_label), dtype=graph.x.dtype, device=graph.x.device
        )
        y = getattr(graph, "y", None)
        if y is not None:
            labeled_mask = (y >= 0) & (y < n_label)
            if labeled_mask.any():
                leak[labeled_mask, y[labeled_mask].long()] = 1.0
        leak_names = [f"label_leak_{name}" for name in label_names] or [
            f"label_leak_{i}" for i in range(n_label)
        ]
        return leak, leak_names

    if spec == "label_only":
        leak, leak_names = _build_label_leak()
        graph.x = leak
        graph.feature_names = leak_names
        print(f"Applied feature subset 'label_only': {x_dim} -> {graph.x.shape[1]} dims.")
        return graph

    if spec == "emb_only_plus_label":
        if not emb_idx:
            raise ValueError(
                "apply_feature_subset: spec='emb_only_plus_label' requires embedding features."
            )
        leak, leak_names = _build_label_leak()
        graph.x = torch.cat([graph.x[:, emb_idx], leak], dim=1)
        graph.feature_names = [feature_names[i] for i in emb_idx] + leak_names
        print(
            f"Applied feature subset 'emb_only_plus_label': {x_dim} -> {graph.x.shape[1]} dims."
        )
        return graph

    indices = None
    if spec == "stats_only":
        indices = stats_idx
    elif spec == "emb_only":
        indices = emb_idx
    elif spec.startswith("keep:"):
        keep_set = {s.strip() for s in spec.split(":", 1)[1].split(",") if s.strip()}
        indices = [i for i, name in enumerate(feature_names) if name in keep_set]
    elif spec.startswith("drop:"):
        drop_set = {s.strip() for s in spec.split(":", 1)[1].split(",") if s.strip()}
        indices = [i for i, name in enumerate(feature_names) if name not in drop_set]
    else:
        raise ValueError(
            f"Unsupported feature_subset='{subset_spec}'. "
            "Use one of: all, constant1, stats_only, emb_only, emb_only_plus_label, "
            "label_only, keep:<...>, drop:<...>."
        )

    if not indices:
        raise ValueError(
            f"apply_feature_subset: spec='{subset_spec}' selected 0 columns out of {x_dim}."
        )

    graph.x = graph.x[:, indices]
    graph.feature_names = [feature_names[i] for i in indices]
    print(f"Applied feature subset '{subset_spec}': {x_dim} -> {graph.x.shape[1]} dims.")
    return graph
