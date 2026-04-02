import argparse

import torch


def parse_args():
    p = argparse.ArgumentParser(description="Inspect a covid19_twitter retweet_graph.pt artifact")
    p.add_argument("--graph", default="data/data/covid19_twitter/graphs/retweet_graph.pt")
    p.add_argument("--topk", type=int, default=10)
    return p.parse_args()


def pct(n: int, d: int) -> str:
    if d <= 0:
        return "n/a"
    return f"{100.0 * float(n) / float(d):.1f}%"


def describe_degree(name: str, deg: torch.Tensor):
    if deg.numel() == 0:
        print(f"{name}: empty")
        return
    degf = deg.to(torch.float32)
    q = torch.quantile(degf, torch.tensor([0.5, 0.9, 0.99]))
    print(
        f"{name}: min={int(deg.min())} max={int(deg.max())} "
        f"mean={degf.mean().item():.2f} median={q[0].item():.1f} "
        f"p90={q[1].item():.1f} p99={q[2].item():.1f}"
    )


def print_topk(title: str, deg: torch.Tensor, handles: list[str], k: int):
    if deg.numel() == 0 or k <= 0:
        return
    k = min(k, deg.numel())
    vals, idxs = torch.topk(deg, k=k)
    print(title)
    for rank, (idx, val) in enumerate(zip(idxs.tolist(), vals.tolist()), start=1):
        print(f"  {rank:>2}. {handles[idx]}: {int(val)}")
    print()


def main():
    args = parse_args()
    raw = torch.load(args.graph, map_location="cpu")
    x = raw["x"]
    edge_index = raw["edge_index"]
    handles = raw["handles"]
    feature_names = list(raw.get("feature_names", []))

    print("Keys:", sorted(raw.keys()))
    print("Shapes")
    print(f"  x: shape={tuple(x.shape)}, dtype={x.dtype}")
    print(f"  edge_index: shape={tuple(edge_index.shape)}, dtype={edge_index.dtype}")
    print()

    n_nodes = int(x.shape[0])
    n_edges = int(edge_index.shape[1])
    print("Graph")
    print(f"  nodes: {n_nodes:,}")
    print(f"  directed edges: {n_edges:,}")
    if n_nodes > 0:
        print(f"  avg out-degree: {n_edges / n_nodes:.2f}")
        print(f"  avg in-degree: {n_edges / n_nodes:.2f}")
    print()

    src = edge_index[0]
    dst = edge_index[1]
    out_deg = torch.bincount(src, minlength=n_nodes) if n_nodes > 0 else torch.zeros(0, dtype=torch.long)
    in_deg = torch.bincount(dst, minlength=n_nodes) if n_nodes > 0 else torch.zeros(0, dtype=torch.long)
    total_deg = in_deg + out_deg

    print("Connectivity")
    print(f"  zero out-degree: {int((out_deg == 0).sum().item()):,} ({pct(int((out_deg == 0).sum().item()), n_nodes)})")
    print(f"  zero in-degree: {int((in_deg == 0).sum().item()):,} ({pct(int((in_deg == 0).sum().item()), n_nodes)})")
    print(f"  isolated: {int((total_deg == 0).sum().item()):,} ({pct(int((total_deg == 0).sum().item()), n_nodes)})")
    describe_degree("  out-degree", out_deg)
    describe_degree("  in-degree", in_deg)
    print()

    print("Features")
    print(f"  feature dims: {len(feature_names)}")
    zero_rows = int((x.abs().sum(dim=1) == 0).sum().item()) if n_nodes > 0 else 0
    print(f"  zero-feature rows: {zero_rows:,} ({pct(zero_rows, n_nodes)})")
    print()

    print_topk("Top retweeted handles (in-degree)", in_deg, handles, args.topk)
    print_topk("Top retweeters (out-degree)", out_deg, handles, args.topk)


if __name__ == "__main__":
    main()
