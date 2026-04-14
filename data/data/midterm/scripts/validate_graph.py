"""Validate a midterm retweet graph artifact."""
import argparse

import torch

from rapids.graph.validate import validate_graph


def parse_args():
    p = argparse.ArgumentParser(description="Validate graph schema and integrity")
    p.add_argument("--graph", default="data/data/midterm/graphs/retweet_graph.pt")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    raw = torch.load(args.graph, map_location="cpu")
    validate_graph(raw)
