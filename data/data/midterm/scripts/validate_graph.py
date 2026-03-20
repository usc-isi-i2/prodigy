import argparse
import sys

import torch


REQUIRED = ["x", "edge_index", "user_ids", "feature_names", "y", "label_names"]


def parse_args():
    p = argparse.ArgumentParser(description="Validate graph schema and integrity")
    p.add_argument("--graph", default="data/data/midterm/graphs/retweet_graph.pt")
    return p.parse_args()


def fail(msg: str):
    print(f"[FAIL] {msg}")
    sys.exit(1)


def main():
    args = parse_args()
    raw = torch.load(args.graph, map_location="cpu")

    for k in REQUIRED:
        if k not in raw:
            fail(f"missing required key: {k}")

    x = raw["x"]
    ei = raw["edge_index"]
    y = raw["y"]
    user_ids = raw["user_ids"]
    feature_names = raw["feature_names"]

    if x.dim() != 2:
        fail("x must be 2D")
    if ei.dim() != 2 or ei.shape[0] != 2:
        fail("edge_index must have shape [2, E]")
    if y.dim() != 1:
        fail("y must be 1D")
    if len(user_ids) != x.shape[0]:
        fail("len(user_ids) must equal x.shape[0]")
    if len(feature_names) != x.shape[1]:
        fail("len(feature_names) must equal x.shape[1]")
    if y.shape[0] != x.shape[0]:
        fail("y.shape[0] must equal x.shape[0]")

    if ei.numel() > 0:
        max_idx = int(ei.max().item())
        min_idx = int(ei.min().item())
        if min_idx < 0:
            fail("edge_index has negative node index")
        if max_idx >= x.shape[0]:
            fail("edge_index references out-of-range node index")

    if torch.isnan(x).any():
        fail("x contains NaN")
    edge_attr = raw.get("edge_attr")
    if edge_attr is not None:
        if edge_attr.shape[0] != ei.shape[1]:
            fail("edge_attr rows must match edge_index columns")
        if torch.isnan(edge_attr).any():
            fail("edge_attr contains NaN")

    views = raw.get("edge_index_views", {})
    attr_views = raw.get("edge_attr_views", {})
    for name, vei in views.items():
        if vei.dim() != 2 or vei.shape[0] != 2:
            fail(f"edge_index_views[{name}] invalid shape")
        va = attr_views.get(name)
        if va is not None and va.shape[0] != vei.shape[1]:
            fail(f"edge_attr_views[{name}] rows != edge count")

    print("[PASS] graph is valid")


if __name__ == "__main__":
    main()
