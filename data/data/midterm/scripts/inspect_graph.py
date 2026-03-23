import argparse
import math

import torch


def parse_args():
    p = argparse.ArgumentParser(description="Inspect a graph_data.pt/retweet_graph.pt artifact")
    p.add_argument("--graph", default="data/data/midterm/graphs/retweet_graph.pt")
    return p.parse_args()


def pct(n: int, d: int) -> str:
    if d <= 0:
        return "n/a"
    return f"{100.0 * float(n) / float(d):.1f}%"


def tensor_shape_str(val) -> str:
    if hasattr(val, "shape"):
        return f"shape={tuple(val.shape)}, dtype={val.dtype}"
    return str(type(val))


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


def main():
    args = parse_args()
    raw = torch.load(args.graph, map_location="cpu")
    x = raw.get("x")
    y = raw.get("y")
    edge_index = raw.get("edge_index")
    edge_attr = raw.get("edge_attr")
    future_edge_index = raw.get("future_edge_index")
    feature_names = list(raw.get("feature_names", []))
    label_names = list(raw.get("label_names", []))
    user_ids = raw.get("user_ids")
    edge_views = raw.get("edge_index_views", {})
    edge_attr_views = raw.get("edge_attr_views", {})
    target_views = raw.get("target_edge_index_views", {})

    print(f"Loaded: {args.graph}")
    print("Keys:", sorted(raw.keys()))
    print()

    print("Shapes")
    for key, val in [
        ("x", x),
        ("y", y),
        ("edge_index", edge_index),
        ("edge_attr", edge_attr),
        ("future_edge_index", future_edge_index),
    ]:
        if val is not None:
            print(f"  {key}: {tensor_shape_str(val)}")
    print()

    n_nodes = int(x.shape[0]) if x is not None else 0
    n_edges = int(edge_index.shape[1]) if edge_index is not None else 0
    n_future_edges = int(future_edge_index.shape[1]) if future_edge_index is not None else 0

    print("Graph")
    print(f"  nodes: {n_nodes:,}")
    print(f"  directed edges: {n_edges:,}")
    if n_nodes > 0:
        print(f"  avg out-degree: {n_edges / n_nodes:.2f}")
        print(f"  avg in-degree: {n_edges / n_nodes:.2f}")
    if future_edge_index is not None:
        print(f"  future edges: {n_future_edges:,}")
    print()

    if edge_index is not None and n_nodes > 0:
        src = edge_index[0]
        dst = edge_index[1]
        out_deg = torch.bincount(src, minlength=n_nodes)
        in_deg = torch.bincount(dst, minlength=n_nodes)
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

        if y is not None:
            labeled_mask = y >= 0
            unlabeled_mask = ~labeled_mask
            labeled = int(labeled_mask.sum().item())
            unlabeled = int(unlabeled_mask.sum().item())

            print("Labels")
            print(f"  label names: {label_names}")
            print(f"  labeled nodes: {labeled:,} ({pct(labeled, n_nodes)})")
            print(f"  unlabeled nodes: {unlabeled:,} ({pct(unlabeled, n_nodes)})")
            for i, name in enumerate(label_names):
                cls_n = int((y == i).sum().item())
                print(f"  class {i} ({name}): {cls_n:,} ({pct(cls_n, labeled)})")
            unknown_values = sorted(int(v) for v in torch.unique(y[(y < 0) & (y != -1)]).tolist())
            if unknown_values:
                print(f"  extra negative label ids: {unknown_values}")

            labeled_target_only = int((labeled_mask & (in_deg > 0) & (out_deg == 0)).sum().item())
            unlabeled_target_only = int((unlabeled_mask & (in_deg > 0) & (out_deg == 0)).sum().item())
            labeled_source_only = int((labeled_mask & (out_deg > 0) & (in_deg == 0)).sum().item())
            unlabeled_source_only = int((unlabeled_mask & (out_deg > 0) & (in_deg == 0)).sum().item())

            print(f"  labeled target-only nodes: {labeled_target_only:,} ({pct(labeled_target_only, labeled)})")
            print(f"  unlabeled target-only nodes: {unlabeled_target_only:,} ({pct(unlabeled_target_only, unlabeled)})")
            print(f"  labeled source-only nodes: {labeled_source_only:,} ({pct(labeled_source_only, labeled)})")
            print(f"  unlabeled source-only nodes: {unlabeled_source_only:,} ({pct(unlabeled_source_only, unlabeled)})")
            print()

    print("Features")
    print(f"  total feature dims: {len(feature_names)}")
    emb_names = [name for name in feature_names if name.startswith("emb_")]
    stat_names = [name for name in feature_names if not name.startswith("emb_")]
    print(f"  non-embedding dims: {len(stat_names)}")
    print(f"  embedding dims: {len(emb_names)}")
    if stat_names:
        print(f"  first non-embedding features: {stat_names[:8]}")
    if raw.get("edge_attr_feature_names") is not None:
        print(f"  edge features: {raw.get('edge_attr_feature_names', [])}")
    if x is not None and emb_names:
        emb_start = feature_names.index(emb_names[0])
        emb_block = x[:, emb_start:emb_start + len(emb_names)]
        zero_emb = int((emb_block.abs().sum(dim=1) == 0).sum().item())
        print(f"  zero embedding rows: {zero_emb:,} ({pct(zero_emb, n_nodes)})")
    print()

    print("Views")
    print(f"  edge_index_views: {list(edge_views.keys())}")
    for name, vei in edge_views.items():
        print(f"    {name}: {vei.shape[1]:,} edges")
    print(f"  edge_attr_views: {list(edge_attr_views.keys())}")
    print(f"  target_edge_index_views: {list(target_views.keys())}")
    for name, vei in target_views.items():
        print(f"    {name}: {vei.shape[1]:,} edges")
    print()

    if user_ids is not None:
        print("IDs")
        print(f"  user_ids: {len(user_ids):,}")


if __name__ == "__main__":
    main()
