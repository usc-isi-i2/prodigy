import argparse
import sys

import torch


REQUIRED_TOP_LEVEL = ["data", "h2i", "handles"]


def parse_args():
    p = argparse.ArgumentParser(description="Validate ukr_rus_twitter retweet graph schema and integrity")
    p.add_argument("--graph", default="data/data/ukr_rus_twitter/graphs/retweet_graph.pt")
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
    h2i = raw["h2i"]
    handles = raw["handles"]

    if not hasattr(data, "x") or not hasattr(data, "edge_index"):
        fail("raw['data'] must expose x and edge_index")

    x = data.x
    edge_index = data.edge_index
    feature_names = list(getattr(data, "feature_names", []))

    if x is None:
        fail("data.x is missing")
    if edge_index is None:
        fail("data.edge_index is missing")
    if x.dim() != 2:
        fail("data.x must be 2D")
    if edge_index.dim() != 2 or edge_index.shape[0] != 2:
        fail("data.edge_index must have shape [2, E]")
    if not isinstance(h2i, dict):
        fail("raw['h2i'] must be a dict")
    if not isinstance(handles, list):
        fail("raw['handles'] must be a list")
    if len(handles) != x.shape[0]:
        fail("len(handles) must equal data.x.shape[0]")
    if len(feature_names) != x.shape[1]:
        fail("len(data.feature_names) must equal data.x.shape[1]")
    if len(h2i) != len(handles):
        fail("len(h2i) must equal len(handles)")

    if torch.isnan(x).any():
        fail("data.x contains NaN")
    if x.dtype not in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
        fail(f"data.x must be floating point, got {x.dtype}")

    if edge_index.numel() > 0:
        max_idx = int(edge_index.max().item())
        min_idx = int(edge_index.min().item())
        if min_idx < 0:
            fail("edge_index has negative node index")
        if max_idx >= x.shape[0]:
            fail("edge_index references out-of-range node index")

    seen_handles = set()
    for i, handle in enumerate(handles):
        if not isinstance(handle, str) or not handle:
            fail(f"handles[{i}] must be a non-empty string")
        if handle in seen_handles:
            fail(f"duplicate handle in handles: {handle}")
        seen_handles.add(handle)

        mapped = h2i.get(handle)
        if mapped != i:
            fail(f"h2i[{handle!r}]={mapped} but expected {i}")

    for handle, idx in h2i.items():
        if not isinstance(handle, str) or not handle:
            fail(f"invalid h2i key: {handle!r}")
        if not isinstance(idx, int):
            fail(f"h2i[{handle!r}] must map to int, got {type(idx).__name__}")
        if idx < 0 or idx >= len(handles):
            fail(f"h2i[{handle!r}] index {idx} out of range")
        if handles[idx] != handle:
            fail(f"h2i[{handle!r}] points to handles[{idx}]={handles[idx]!r}")

    print("[PASS] graph is valid")


if __name__ == "__main__":
    main()
