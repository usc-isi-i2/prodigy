import argparse
import sys

import torch


REQUIRED_TOP_LEVEL = ["data", "user_ids"]


def parse_args():
    p = argparse.ArgumentParser(description="Validate covid19_twitter retweet graph schema and integrity")
    p.add_argument("--graph", default="data/data/covid19_twitter/graphs/retweet_graph.pt")
    return p.parse_args()


def fail(msg: str):
    print(f"[FAIL] {msg}")
    sys.exit(1)


def main():
    args = parse_args()
    raw = torch.load(args.graph, map_location="cpu")

    if not isinstance(raw, dict):
        fail("top-level object must be a dict")
    for key in REQUIRED_TOP_LEVEL:
        if key not in raw:
            fail(f"missing required key: {key}")

    data = raw["data"]
    user_ids = raw["user_ids"]
    u2i = raw.get("u2i", {})
    handles = raw.get("handles", [])
    x = data.x
    edge_index = data.edge_index
    feature_names = list(getattr(data, "feature_names", []))

    if x is None or edge_index is None:
        fail("data.x and data.edge_index are required")
    if x.dim() != 2:
        fail("data.x must be 2D")
    if edge_index.dim() != 2 or edge_index.shape[0] != 2:
        fail("data.edge_index must have shape [2, E]")
    if not isinstance(user_ids, (list, tuple)):
        fail("raw['user_ids'] must be a list/tuple")
    if len(user_ids) != x.shape[0]:
        fail("len(user_ids) must equal data.x.shape[0]")
    if handles and (not isinstance(handles, list) or len(handles) != x.shape[0]):
        fail("if present, raw['handles'] must be a list aligned to data.x.shape[0]")
    if len(feature_names) != x.shape[1]:
        fail("len(data.feature_names) must equal data.x.shape[1]")
    if u2i and len(u2i) != len(user_ids):
        fail("if present, len(u2i) must equal len(user_ids)")
    if torch.isnan(x).any():
        fail("data.x contains NaN")
    if edge_index.numel() > 0:
        max_idx = int(edge_index.max().item())
        min_idx = int(edge_index.min().item())
        if min_idx < 0:
            fail("edge_index has negative node index")
        if max_idx >= x.shape[0]:
            fail("edge_index references out-of-range node index")

    for i, user_id in enumerate(user_ids):
        try:
            user_id_int = int(user_id)
        except Exception:
            fail(f"user_ids[{i}] is not int-like: {user_id!r}")
        if u2i and u2i.get(user_id_int) != i:
            fail(f"u2i[{user_id_int!r}] mismatch at index {i}")

    print("[PASS] graph is valid")


if __name__ == "__main__":
    main()
