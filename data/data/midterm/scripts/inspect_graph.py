import argparse
import torch


def parse_args():
    p = argparse.ArgumentParser(description="Inspect a graph_data.pt/retweet_graph.pt artifact")
    p.add_argument("--graph", default="data/data/midterm/graphs/retweet_graph.pt")
    return p.parse_args()


def main():
    args = parse_args()
    raw = torch.load(args.graph, map_location="cpu")

    print(f"Loaded: {args.graph}")
    print("Keys:", sorted(raw.keys()))

    for key in ["x", "y", "edge_index", "edge_attr", "future_edge_index"]:
        val = raw.get(key)
        if hasattr(val, "shape"):
            print(f"{key}: shape={tuple(val.shape)}, dtype={val.dtype}")

    print("feature_names:", len(raw.get("feature_names", [])))
    print("edge_attr_feature_names:", raw.get("edge_attr_feature_names", []))
    print("label_names:", len(raw.get("label_names", [])))

    print("edge_index_views:", list(raw.get("edge_index_views", {}).keys()))
    print("edge_attr_views:", list(raw.get("edge_attr_views", {}).keys()))
    print("target_edge_index_views:", list(raw.get("target_edge_index_views", {}).keys()))


if __name__ == "__main__":
    main()
