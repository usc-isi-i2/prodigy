"""Shared graph inspection utilities for all retweet graph artifacts."""
import torch


def pct(n: int, d: int) -> str:
    if d <= 0:
        return "n/a"
    return f"{100.0 * float(n) / float(d):.1f}%"


def describe_degree(name: str, deg: torch.Tensor) -> None:
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


def print_topk(title: str, deg: torch.Tensor, names: list, k: int) -> None:
    if deg.numel() == 0 or k <= 0:
        return
    k = min(k, deg.numel())
    vals, idxs = torch.topk(deg, k=k)
    print(title)
    for rank, (idx, val) in enumerate(zip(idxs.tolist(), vals.tolist()), start=1):
        print(f"  {rank:>2}. {names[idx]}: {int(val)}")
    print()


def inspect_graph(raw: dict, topk: int = 10) -> None:
    """Print a detailed summary of a saved graph dict."""
    x = raw.get("x")
    y = raw.get("y")
    edge_index = raw.get("edge_index")
    edge_attr = raw.get("edge_attr")
    future_edge_index = raw.get("future_edge_index")
    feature_names = list(raw.get("feature_names", []))
    label_names = list(raw.get("label_names", []))
    user_ids = raw.get("user_ids", [])
    handles = raw.get("handles", [])
    edge_views = raw.get("edge_index_views", {})
    edge_attr_views = raw.get("edge_attr_views", {})
    target_views = raw.get("target_edge_index_views", {})

    # Build display names: "handle [userid]" or just userid
    names = []
    if x is not None:
        for i in range(x.shape[0]):
            handle = handles[i] if i < len(handles) else None
            uid = user_ids[i] if i < len(user_ids) else i
            names.append(f"{handle} [{uid}]" if handle else str(uid))

    print("Keys:", sorted(raw.keys()))
    print()

    print("Shapes")
    for key, val in [
        ("x", x), ("y", y), ("edge_index", edge_index),
        ("edge_attr", edge_attr), ("future_edge_index", future_edge_index),
    ]:
        if val is not None and hasattr(val, "shape"):
            print(f"  {key}: shape={tuple(val.shape)}, dtype={val.dtype}")
    print()

    n_nodes = int(x.shape[0]) if x is not None else 0
    n_edges = int(edge_index.shape[1]) if edge_index is not None else 0
    n_future = int(future_edge_index.shape[1]) if future_edge_index is not None else 0

    print("Graph")
    print(f"  nodes: {n_nodes:,}")
    print(f"  directed edges: {n_edges:,}")
    if n_nodes > 0:
        print(f"  avg degree: {n_edges / n_nodes:.2f}")
    if future_edge_index is not None:
        print(f"  future edges: {n_future:,}")
    print()

    if edge_index is not None and n_nodes > 0:
        src, dst = edge_index[0], edge_index[1]
        out_deg = torch.bincount(src, minlength=n_nodes)
        in_deg = torch.bincount(dst, minlength=n_nodes)
        total_deg = in_deg + out_deg

        zero_out = int((out_deg == 0).sum())
        zero_in = int((in_deg == 0).sum())
        isolated = int((total_deg == 0).sum())
        target_only = int(((in_deg > 0) & (out_deg == 0)).sum())
        source_only = int(((out_deg > 0) & (in_deg == 0)).sum())
        both_sides = int(((in_deg > 0) & (out_deg > 0)).sum())

        print("Connectivity")
        print(f"  zero out-degree: {zero_out:,} ({pct(zero_out, n_nodes)})")
        print(f"  zero in-degree:  {zero_in:,} ({pct(zero_in, n_nodes)})")
        print(f"  isolated:        {isolated:,} ({pct(isolated, n_nodes)})")
        print(f"  target-only:     {target_only:,} ({pct(target_only, n_nodes)})")
        print(f"  source-only:     {source_only:,} ({pct(source_only, n_nodes)})")
        print(f"  both in+out > 0: {both_sides:,} ({pct(both_sides, n_nodes)})")
        describe_degree("  out-degree", out_deg)
        describe_degree("  in-degree", in_deg)
        print()

        if y is not None:
            labeled_mask = y >= 0
            labeled = int(labeled_mask.sum())
            unlabeled = n_nodes - labeled
            print("Labels")
            print(f"  label names:      {label_names}")
            print(f"  labeled nodes:    {labeled:,} ({pct(labeled, n_nodes)})")
            print(f"  unlabeled nodes:  {unlabeled:,} ({pct(unlabeled, n_nodes)})")
            for i, name in enumerate(label_names):
                cls_n = int((y == i).sum())
                print(f"  class {i} ({name}): {cls_n:,} ({pct(cls_n, labeled)})")
            print()

        if topk > 0 and names:
            print_topk("Top retweeted (in-degree)", in_deg, names, topk)
            print_topk("Top retweeters (out-degree)", out_deg, names, topk)

    print("Features")
    print(f"  total dims: {len(feature_names)}")
    emb_names = [n for n in feature_names if n.startswith("emb_")]
    stat_names = [n for n in feature_names if not n.startswith("emb_")]
    print(f"  stat dims:  {len(stat_names)}: {stat_names}")
    print(f"  emb dims:   {len(emb_names)}")
    if raw.get("edge_attr_feature_names"):
        print(f"  edge features: {raw['edge_attr_feature_names']}")
    if x is not None and emb_names:
        emb_start = feature_names.index(emb_names[0])
        emb_block = x[:, emb_start: emb_start + len(emb_names)]
        zero_emb = int((emb_block.abs().sum(dim=1) == 0).sum())
        print(f"  zero emb rows: {zero_emb:,} ({pct(zero_emb, n_nodes)})")
    print()

    print("Views")
    print(f"  edge_index_views:        {list(edge_views.keys())}")
    for name, vei in edge_views.items():
        print(f"    {name}: {vei.shape[1]:,} edges")
    print(f"  edge_attr_views:         {list(edge_attr_views.keys())}")
    print(f"  target_edge_index_views: {list(target_views.keys())}")
    for name, vei in target_views.items():
        print(f"    {name}: {vei.shape[1]:,} edges")
    print()

    print("IDs")
    print(f"  user_ids: {len(user_ids):,}")
