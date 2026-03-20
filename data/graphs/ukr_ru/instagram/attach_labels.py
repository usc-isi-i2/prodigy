"""
Copy labels (data.y, data.label_names) from a labeled graph onto the bge-features graph.

Usage:
    python attach_labels.py \
        --features mention_graph_bge.pt \
        --labels   mention_graph_overperformer.pt \
        --out      mention_graph_overperformer_bge.pt
"""

import argparse
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="mention_graph_bge.pt")
    parser.add_argument("--labels",   required=True)
    parser.add_argument("--out",      required=True)
    args = parser.parse_args()

    feat_ckpt  = torch.load(args.features, map_location="cpu")
    label_ckpt = torch.load(args.labels,   map_location="cpu")

    data = feat_ckpt["data"]
    data.y           = label_ckpt["data"].y
    data.label_names = label_ckpt["data"].label_names

    torch.save({"data": data, "h2i": feat_ckpt["h2i"], "handles": feat_ckpt["handles"]}, args.out)
    labeled = (data.y >= 0).sum().item()
    print(f"Saved {args.out}  |  x: {data.x.shape}  |  labeled nodes: {labeled}/{data.x.shape[0]}")


if __name__ == "__main__":
    main()
