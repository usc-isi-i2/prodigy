import argparse

import torch


def parse_args():
    p = argparse.ArgumentParser(description="Inspect a ukr_rus_twitter retweet_graph.pt artifact")
    p.add_argument("--graph", default="data/data/ukr_rus_twitter/graphs/retweet_graph.pt")
    p.add_argument("--topk", type=int, default=10, help="How many high-degree users to print")
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


def print_topk(title: str, deg: torch.Tensor, names: list[str], k: int):
    if deg.numel() == 0 or k <= 0:
        return
    k = min(k, deg.numel())
    vals, idxs = torch.topk(deg, k=k)
    print(title)
    for rank, (idx, val) in enumerate(zip(idxs.tolist(), vals.tolist()), start=1):
        print(f"  {rank:>2}. {names[idx]}: {int(val)}")
    print()


def main():
    args = parse_args()
    raw = torch.load(args.graph, map_location="cpu")
    data = raw["data"]
    user_ids = list(raw.get("user_ids", []))
    handles = list(raw.get("handles", []))
    u2i = raw.get("u2i", {})

    x = data.x
    edge_index = data.edge_index
    feature_names = list(getattr(data, "feature_names", []))
    names = []
    for i in range(x.shape[0]):
        handle = handles[i] if i < len(handles) else None
        user_id = user_ids[i] if i < len(user_ids) else i
        names.append(f"{handle} [{user_id}]" if handle else str(user_id))

    print(f"Loaded: {args.graph}")
    print("Keys:", sorted(raw.keys()))
    print()

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

    zero_out = int((out_deg == 0).sum().item())
    zero_in = int((in_deg == 0).sum().item())
    isolated = int((total_deg == 0).sum().item())
    target_only = int(((in_deg > 0) & (out_deg == 0)).sum().item())
    source_only = int(((out_deg > 0) & (in_deg == 0)).sum().item())
    both_sides = int(((in_deg > 0) & (out_deg > 0)).sum().item())

    print("Connectivity")
    print(f"  zero out-degree: {zero_out:,} ({pct(zero_out, n_nodes)})")
    print(f"  zero in-degree: {zero_in:,} ({pct(zero_in, n_nodes)})")
    print(f"  isolated: {isolated:,} ({pct(isolated, n_nodes)})")
    print(f"  target-only nodes: {target_only:,} ({pct(target_only, n_nodes)})")
    print(f"  source-only nodes: {source_only:,} ({pct(source_only, n_nodes)})")
    print(f"  both in/out-degree > 0: {both_sides:,} ({pct(both_sides, n_nodes)})")
    describe_degree("  out-degree", out_deg)
    describe_degree("  in-degree", in_deg)
    print()

    print("Features")
    print(f"  feature dims: {len(feature_names)}")
    if feature_names:
        print(f"  feature names: {feature_names}")
    zero_rows = int((x.abs().sum(dim=1) == 0).sum().item()) if n_nodes > 0 else 0
    nan_rows = int(torch.isnan(x).any(dim=1).sum().item()) if n_nodes > 0 else 0
    print(f"  zero-feature rows: {zero_rows:,} ({pct(zero_rows, n_nodes)})")
    print(f"  rows with NaN: {nan_rows:,} ({pct(nan_rows, n_nodes)})")
    print()

    print("Mapping")
    print(f"  user_ids: {len(user_ids):,}")
    print(f"  u2i size: {len(u2i):,}")
    if handles:
        print(f"  handles: {len(handles):,}")
    print()

    print_topk("Top retweeted users (in-degree)", in_deg, names, args.topk)
    print_topk("Top retweeters (out-degree)", out_deg, names, args.topk)


if __name__ == "__main__":
    main()
