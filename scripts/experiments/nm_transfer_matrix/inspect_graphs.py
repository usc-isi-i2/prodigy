#!/usr/bin/env python3
"""Pre-train sanity check for the NM transfer-matrix graphs.

Loads the three training graphs (ukr, covid, merged) and reports the things that
would silently invalidate the experiment:

  * feature dim must be IDENTICAL across all three  -> else cross-eval is
    meaningless / the model can't even load on another domain;
  * feature_names should match (same feature semantics);
  * NaN / Inf / all-zero feature rows;
  * node/edge counts and the covid-vs-ukr split inside the merged graph
    (quantifies the proportional-sampling / per-domain-exposure caveat);
  * merged provenance should equal the actual single-source counts (disjoint
    block concat: merged == ukr + covid);
  * degree distribution per graph (NM learns local neighborhoods);
  * the edge view the training config will actually use (edge_view: default).

Usage (in the prodigy conda env, on Tucker):
    python inspect_graphs.py
    python inspect_graphs.py --ukr <path> --covid <path> --merged <path>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

DEFAULTS = {
    "ukr": "/dataMeR1/phil/data/ukr_rus_twitter/graphs/retweet_graph_parquet.pt",
    "covid": "/dataMeR1/phil/data/covid19_twitter/graphs/retweet_graph_parquet.pt",
    "merged": "/dataMeR1/phil/data/merged/graphs/ukr_rus_covid_retweet_graph.pt",
}


def load_graph(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def get(raw: dict, key: str):
    """Fetch a field from the dict, falling back to the PyG Data object."""
    val = raw.get(key)
    if val is None and "data" in raw:
        val = getattr(raw["data"], key, None)
    return val


def fmt(n) -> str:
    return f"{n:,}" if isinstance(n, int) else str(n)


def degree_stats(edge_index: torch.Tensor, num_nodes: int) -> dict:
    # total degree treating both endpoints (retweet graphs are directed; we
    # report combined touch count so isolated nodes are visible).
    deg = torch.zeros(num_nodes, dtype=torch.long)
    deg += torch.bincount(edge_index[0], minlength=num_nodes)
    deg += torch.bincount(edge_index[1], minlength=num_nodes)
    deg_f = deg.float()
    return {
        "isolated_nodes": int((deg == 0).sum()),
        "deg_min": int(deg.min()),
        "deg_mean": float(deg_f.mean()),
        "deg_median": int(deg.median()),
        "deg_max": int(deg.max()),
    }


def feature_stats(x: torch.Tensor) -> dict:
    xf = x.float()
    nan = int(torch.isnan(xf).sum())
    inf = int(torch.isinf(xf).sum())
    clean = xf[~torch.isnan(xf) & ~torch.isinf(xf)]
    zero_rows = int((xf == 0).all(dim=1).sum())
    # columns that are constant across all nodes (dead features)
    const_cols = int((xf.max(dim=0).values == xf.min(dim=0).values).sum())
    return {
        "dtype": str(x.dtype),
        "nan": nan,
        "inf": inf,
        "all_zero_rows": zero_rows,
        "constant_cols": const_cols,
        "mean": float(clean.mean()) if clean.numel() else float("nan"),
        "std": float(clean.std()) if clean.numel() else float("nan"),
        "min": float(clean.min()) if clean.numel() else float("nan"),
        "max": float(clean.max()) if clean.numel() else float("nan"),
    }


def inspect_one(name: str, path: Path) -> dict:
    print(f"\n{'='*70}\n{name}: {path}")
    if not path.exists():
        print("  !! MISSING FILE")
        return {"missing": True}
    raw = load_graph(path)

    x = get(raw, "x")
    edge_index = get(raw, "edge_index")
    n_nodes = int(x.shape[0]) if x is not None else -1
    feat_dim = int(x.shape[1]) if x is not None and x.dim() == 2 else -1
    n_edges = int(edge_index.shape[1]) if edge_index is not None else -1

    print(f"  nodes={fmt(n_nodes)}  edges={fmt(n_edges)}  feature_dim={feat_dim}")

    fs = feature_stats(x) if x is not None else {}
    if fs:
        print(f"  features: dtype={fs['dtype']} mean={fs['mean']:.4f} std={fs['std']:.4f} "
              f"range=[{fs['min']:.3f},{fs['max']:.3f}]")
        flags = []
        if fs["nan"]:
            flags.append(f"NaN={fs['nan']}")
        if fs["inf"]:
            flags.append(f"Inf={fs['inf']}")
        if fs["all_zero_rows"]:
            flags.append(f"all-zero-rows={fmt(fs['all_zero_rows'])}")
        if fs["constant_cols"]:
            flags.append(f"constant-cols={fs['constant_cols']}/{feat_dim}")
        print(f"  feature flags: {', '.join(flags) if flags else 'none'}")

    if edge_index is not None and n_nodes > 0:
        ds = degree_stats(edge_index, n_nodes)
        print(f"  degree: min={ds['deg_min']} median={ds['deg_median']} "
              f"mean={ds['deg_mean']:.2f} max={fmt(ds['deg_max'])}  "
              f"isolated_nodes={fmt(ds['isolated_nodes'])}")
        self_loops = int((edge_index[0] == edge_index[1]).sum())
        print(f"  self_loops={fmt(self_loops)}")

    # edge views (config uses edge_view: default -> 'edge_index')
    views = raw.get("edge_index_views") or {}
    tviews = raw.get("target_edge_index_views") or {}
    print(f"  edge_index_views: {sorted(views) or '(none, default uses edge_index)'}")
    if tviews:
        print(f"  target_edge_index_views: {sorted(tviews)}")

    # labels (not used by NM, but worth seeing)
    label_names = raw.get("label_names") or []
    print(f"  label_names: {len(label_names)}")

    # provenance (merged only)
    snc = raw.get("source_node_counts")
    sec = raw.get("source_edge_counts")
    if snc:
        total = sum(snc.values()) or 1
        print("  merged provenance:")
        for src in snc:
            frac = 100.0 * snc[src] / total
            print(f"    {src:>10}: nodes={fmt(int(snc[src]))} ({frac:4.1f}%)"
                  f"  edges={fmt(int(sec[src])) if sec else '?'}")

    feat_names = get(raw, "feature_names")
    return {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "feat_dim": feat_dim,
        "feature_names": list(feat_names) if feat_names is not None else None,
        "source_node_counts": {k: int(v) for k, v in (snc or {}).items()},
        "source_edge_counts": {k: int(v) for k, v in (sec or {}).items()},
    }


def cross_checks(info: dict) -> int:
    print(f"\n{'='*70}\nCROSS-GRAPH CHECKS")
    rc = 0

    def ok(cond, msg):
        nonlocal rc
        print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")
        if not cond:
            rc = 1

    present = {k: v for k, v in info.items() if not v.get("missing")}
    if len(present) < 3:
        print("  (skipping cross-checks: not all three graphs loaded)")
        return 1

    dims = {k: v["feat_dim"] for k, v in present.items()}
    ok(len(set(dims.values())) == 1,
       f"feature_dim identical across graphs: {dims}  <-- required for cross-eval")

    # feature_names equality (if available)
    names = {k: v["feature_names"] for k, v in present.items() if v["feature_names"]}
    if len(names) == 3:
        ref = names["ukr"]
        ok(all(names[k] == ref for k in names),
           "feature_names identical across graphs")
    else:
        print("  [warn] feature_names missing on >=1 graph; skipping name check")

    # merged == ukr + covid (disjoint block concat)
    if {"ukr", "covid", "merged"} <= present.keys():
        u, c, m = present["ukr"], present["covid"], present["merged"]
        ok(m["n_nodes"] == u["n_nodes"] + c["n_nodes"],
           f"merged nodes == ukr+covid ({fmt(m['n_nodes'])} == "
           f"{fmt(u['n_nodes'])}+{fmt(c['n_nodes'])})")
        ok(m["n_edges"] == u["n_edges"] + c["n_edges"],
           f"merged edges == ukr+covid ({fmt(m['n_edges'])} == "
           f"{fmt(u['n_edges'])}+{fmt(c['n_edges'])})")

        snc = m["source_node_counts"]
        if snc:
            tot = sum(snc.values()) or 1
            split = {k: f"{100*v/tot:.1f}%" for k, v in snc.items()}
            print(f"  [info] merged domain split (per-node): {split}")
            print("         -> under uniform NM center sampling this is each "
                  "domain's expected episode share;")
            print("            the smaller domain is under-exposed vs its "
                  "single-source run even at 2x epochs.")
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k}", default=v)
    args = ap.parse_args()

    info = {}
    for name in ("ukr", "covid", "merged"):
        info[name] = inspect_one(name, Path(getattr(args, name)))

    rc = cross_checks(info)
    print(f"\n{'='*70}\n{'ALL CHECKS PASSED' if rc == 0 else 'CHECKS FAILED — review above before training'}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
