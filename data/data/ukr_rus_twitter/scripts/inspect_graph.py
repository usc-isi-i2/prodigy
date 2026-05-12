"""Inspect a ukr_rus_twitter retweet graph artifact."""
import argparse

import torch

from rapids.graph.inspect import inspect_graph


def parse_args():
    p = argparse.ArgumentParser(description="Inspect a retweet graph artifact")
    p.add_argument("--graph", default="data/data/ukr_rus_twitter/graphs/retweet_graph.pt")
    p.add_argument("--topk", type=int, default=10)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    raw = torch.load(args.graph, map_location="cpu", weights_only=False)
    print(f"Loaded: {args.graph}")
    inspect_graph(raw, topk=args.topk)
