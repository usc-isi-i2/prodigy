"""Shared graph validation logic for all retweet graph artifacts."""
import sys

import torch


REQUIRED_KEYS = ["x", "edge_index", "user_ids", "feature_names", "y", "label_names"]


def _fail(msg: str) -> None:
    print(f"[FAIL] {msg}")
    sys.exit(1)


def validate_graph(raw: dict) -> None:
    """Validate schema and integrity of a saved graph dict.

    Checks:
    - Required top-level keys are present.
    - Tensor shapes are consistent (nodes, edges, features, labels).
    - No NaN values in node or edge features.
    - Edge indices are within valid range.
    - Temporal view tensors (if present) are well-formed.
    """
    if not isinstance(raw, dict):
        _fail("top-level object must be a dict")

    for k in REQUIRED_KEYS:
        if k not in raw:
            _fail(f"missing required key: {k!r}")

    x = raw["x"]
    ei = raw["edge_index"]
    y = raw["y"]
    user_ids = raw["user_ids"]
    feature_names = raw["feature_names"]

    if x.dim() != 2:
        _fail("x must be 2-D")
    if ei.dim() != 2 or ei.shape[0] != 2:
        _fail("edge_index must have shape [2, E]")
    if y.dim() != 1:
        _fail("y must be 1-D")
    if len(user_ids) != x.shape[0]:
        _fail(f"len(user_ids)={len(user_ids)} must equal x.shape[0]={x.shape[0]}")
    if len(feature_names) != x.shape[1]:
        _fail(f"len(feature_names)={len(feature_names)} must equal x.shape[1]={x.shape[1]}")
    if y.shape[0] != x.shape[0]:
        _fail(f"y.shape[0]={y.shape[0]} must equal x.shape[0]={x.shape[0]}")
    if x.dtype not in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
        _fail(f"x must be floating-point, got {x.dtype}")
    if torch.isnan(x).any():
        _fail("x contains NaN")

    if ei.numel() > 0:
        if int(ei.min().item()) < 0:
            _fail("edge_index has negative node index")
        if int(ei.max().item()) >= x.shape[0]:
            _fail("edge_index references out-of-range node index")

    edge_attr = raw.get("edge_attr")
    if edge_attr is not None:
        if edge_attr.shape[0] != ei.shape[1]:
            _fail("edge_attr rows must match edge_index columns")
        if torch.isnan(edge_attr).any():
            _fail("edge_attr contains NaN")

    u2i = raw.get("u2i", {})
    if u2i and len(u2i) != len(user_ids):
        _fail(f"len(u2i)={len(u2i)} must equal len(user_ids)={len(user_ids)}")

    for name, vei in raw.get("edge_index_views", {}).items():
        if vei.dim() != 2 or vei.shape[0] != 2:
            _fail(f"edge_index_views[{name!r}] invalid shape")
        va = raw.get("edge_attr_views", {}).get(name)
        if va is not None and va.shape[0] != vei.shape[1]:
            _fail(f"edge_attr_views[{name!r}] row count != edge count")

    print("[PASS] graph is valid")
