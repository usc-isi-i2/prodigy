#!/usr/bin/env python3
"""Stage-1 diagnostic for the directed3 input-scaling hypothesis (topology_feature_ssl).

Tests whether the directed3 structural INPUT features that E1/E2/E2b/E4 feed the encoder
are pathologically distributed: z-scoring RAW power-law degree counts (in_deg, out_deg)
leaves a tiny bulk near 0 with rare hubs at |z| ~ 100. That heavy tail — not the average
scale — is the proposed reason E1 < B0 on the pretext val_roc_auc (see FINDINGS + the
"just learn a smaller weight can't compress a tail" argument).

Two paths:
  * default (cache): load the z-scored directed3 cache the trainer already wrote
      (<graph>.structural_directed3.pt) and report the per-column distribution of EXACTLY
      what the encoder saw. Fast — no graph load. This is the necessary-condition check.
  * --from-graph: load the merged graph, recompute RAW in/out degrees, and report
      (a) raw-degree stats, (b) the CURRENT z-scored columns, and (c) the PROPOSED FIX
      (log1p in/out -> z-score) — before/after in one run. Needs the full graph in RAM
      (~large node; the default cache path avoids this).

Verdict: current z-scored in_deg/out_deg with max|z| >> 4 and a tiny bulk IQR => the
pathological input is confirmed. max|z| ~ 4-6 => hypothesis falsified cheaply.

Run on Tucker (prodigy env):
    conda activate prodigy
    python scripts/experiments/analysis/objectives/topology_vs_features/topology_feature_ssl/check_directed3_distribution.py
    # with raw degrees + fix preview (needs the graph in memory):
    python scripts/experiments/analysis/objectives/topology_vs_features/topology_feature_ssl/check_directed3_distribution.py --from-graph
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

# Locate the repository by its canonical instruction file; fall back to cwd when piped.
REPO_ROOT = (
    next(p for p in Path(__file__).resolve().parents if (p / "AGENTS.md").is_file())
    if "__file__" in globals()
    else Path.cwd()
)
sys.path.insert(0, str(REPO_ROOT))
from data.structural_features import (  # noqa: E402
    _degrees,
    compute_structural_features,
    structural_feature_names,
)

DEFAULT_ROOT = "/dataMeR1/phil/data/merged/graphs"
DEFAULT_GRAPH = "ukr_rus_covid_midterm_retweet_graph.pt"


def _torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # older torch without the kwarg
        return torch.load(path, map_location="cpu")


def _summarize(v: np.ndarray) -> dict:
    a = np.asarray(v, dtype=np.float64).ravel()
    absa = np.abs(a)
    p = np.percentile(a, [25, 50, 75, 99, 99.9, 100])
    return {
        "mean": a.mean(), "std": a.std(),
        "p50": p[1], "iqr": p[2] - p[0], "p99": p[3], "p99_9": p[4], "max": p[5],
        "maxabs": absa.max(),
        "frac_gt4": float((absa > 4).mean()),
        "frac_gt10": float((absa > 10).mean()),
    }


def _print_table(title: str, col_names: list[str], mat: np.ndarray) -> list[dict]:
    print(f"\n=== {title}   (N={mat.shape[0]:,}) ===")
    print(f"{'col':>9} {'mean':>9} {'std':>9} {'p50':>9} {'IQR':>9} "
          f"{'p99.9':>11} {'max':>12} {'maxabs':>10} {'%>4':>8} {'%>10':>8}")
    rows = []
    for j, nm in enumerate(col_names):
        s = _summarize(mat[:, j])
        rows.append(s)
        print(f"{nm:>9} {s['mean']:9.3f} {s['std']:9.3f} {s['p50']:9.3f} {s['iqr']:9.3f} "
              f"{s['p99_9']:11.3f} {s['max']:12.2f} {s['maxabs']:10.2f} "
              f"{100 * s['frac_gt4']:8.4f} {100 * s['frac_gt10']:8.4f}")
    return rows


def _zscore(feats: torch.Tensor) -> torch.Tensor:
    mean = feats.mean(dim=0, keepdim=True)
    std = feats.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (feats - mean) / std


def _load_graph(graph_path: str):
    # mmap keeps peak RAM to a few GB on the 34M-node / 104GB-x graph: only edge_index
    # (for degrees) and the sampled x rows get paged in. Fall back to a full load only
    # if mmap is unsupported for this .pt format.
    print(f"[graph] loading {graph_path} (mmap) ...", flush=True)
    try:
        obj = torch.load(graph_path, map_location="cpu", mmap=True, weights_only=False)
    except Exception as exc:  # noqa: BLE001 - mmap unsupported / old format
        print(f"[graph] mmap load failed ({exc}); falling back to full load (large RAM)", flush=True)
        obj = _torch_load(graph_path)
    if hasattr(obj, "edge_index"):
        ei = obj.edge_index
        x = getattr(obj, "x", None)
    elif isinstance(obj, dict):
        ei = obj.get("edge_index")
        x = obj.get("x")
        if ei is None:
            raise SystemExit(f"no 'edge_index' key in {graph_path}; keys={list(obj)[:20]}")
    else:
        raise SystemExit(f"unrecognized graph object type {type(obj)}")
    n = int(x.shape[0]) if x is not None else int(ei.max()) + 1
    return ei, n, x


def _verdict(zscored_rows: list[dict], col_names: list[str]) -> None:
    # in_deg / out_deg are the raw-count columns; log_deg is already logged (healthy).
    raw_cols = [r for nm, r in zip(col_names, zscored_rows) if nm in ("in_deg", "out_deg")]
    worst = max((r["maxabs"] for r in raw_cols), default=0.0)
    tiny_iqr = min((r["iqr"] for r in raw_cols), default=1.0)
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    print(f"  worst raw-count column max|z| = {worst:.1f}")
    print(f"  tightest raw-count bulk IQR   = {tiny_iqr:.4f}")
    if worst >= 20:
        print("  --> CONFIRMED: heavy-tailed z-scored input (max|z| >> 4).")
        print("      The bulk is squished while rare hubs spike; a linear input")
        print("      weight cannot compress this. Fix = log1p before z-score.")
    elif worst >= 8:
        print("  --> LIKELY: moderate heavy tail (max|z| in [8,20)). Worth the E1 re-run.")
    else:
        print("  --> NOT SUPPORTED: max|z| < 8. The input is not badly heavy-tailed;")
        print("      the E1<B0 gap is probably NOT this. Look elsewhere.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--graph-filename", default=DEFAULT_GRAPH)
    ap.add_argument("--graph", default=None, help="full path to the graph .pt (overrides root/filename)")
    ap.add_argument("--cache", default=None, help="explicit directed3 cache path")
    ap.add_argument("--mode", default="directed3", choices=["directed3", "directed6"])
    ap.add_argument("--from-graph", action="store_true",
                    help="recompute from the graph: raw degrees + current z-score + log1p-fix preview")
    ap.add_argument("--sample-bio", type=int, default=200_000,
                    help="with --from-graph, sample this many nodes' bio |x| for scale context (0=off)")
    args = ap.parse_args()

    graph_path = args.graph or os.path.join(args.root, args.graph_filename)
    cache_path = args.cache or f"{graph_path}.structural_{args.mode}.pt"
    col_names = structural_feature_names(args.mode)

    if not args.from_graph:
        # ---- fast path: read exactly what the trainer cached / the encoder saw ----
        if not os.path.exists(cache_path):
            print(f"[cache] not found: {cache_path}\n"
                  f"        pass --from-graph to recompute from {graph_path}, "
                  f"or --cache <path>.", file=sys.stderr)
            return 2
        blob = _torch_load(cache_path)
        feats = blob["feats"] if isinstance(blob, dict) else blob
        print(f"[cache] loaded {cache_path}")
        print(f"[cache] mode={blob.get('mode') if isinstance(blob, dict) else '?'} "
              f"standardize={blob.get('standardize') if isinstance(blob, dict) else '?'} "
              f"n_edges={blob.get('n_edges') if isinstance(blob, dict) else '?'}")
        rows = _print_table("CURRENT directed3 input (z-scored — exactly what the encoder saw)",
                            col_names, np.asarray(feats, dtype=np.float32))
        _verdict(rows, col_names)
        print("\n(Run with --from-graph to also see raw degrees and the log1p-fix preview.)")
        return 0

    # ---- full path: recompute from the graph, show raw + current + fix ----
    ei, n, x = _load_graph(graph_path)
    in_deg, out_deg = _degrees(ei, n)
    log_deg = torch.log1p(in_deg + out_deg)

    raw = torch.stack([in_deg, out_deg, log_deg], dim=1)
    _print_table("RAW directed3 (pre-standardize): in/out are counts, log_deg=log1p(in+out)",
                 col_names[:3], raw.numpy())

    current = compute_structural_features(ei, n, mode="directed3", standardize=True)
    rows = _print_table("CURRENT (z-scored raw counts — what E1/E2/E2b/E4 feed the encoder)",
                        col_names[:3], current.numpy())

    fixed = _zscore(torch.stack([torch.log1p(in_deg), torch.log1p(out_deg), log_deg], dim=1))
    _print_table("PROPOSED FIX (log1p in/out -> z-score all three)", col_names[:3], fixed.numpy())

    if x is not None and args.sample_bio > 0:
        m = min(args.sample_bio, int(x.shape[0]))
        idx = torch.randperm(int(x.shape[0]))[:m]
        bio = x[idx, :768] if x.shape[1] >= 768 else x[idx]
        absb = bio.abs().float().numpy().ravel()
        p = np.percentile(absb, [50, 99, 99.9, 100])
        print(f"\n=== BIO block |x| (sampled {m:,} nodes, {bio.shape[1]} dims) ===")
        print(f"  median |x|={p[0]:.4f}  p99={p[1]:.4f}  p99.9={p[2]:.4f}  max={p[3]:.4f}")
        print("  (compare max|x| here to the degree max|z| above: that ratio is how")
        print("   much the raw-count columns dominate the bio dims for hub nodes.)")

    _verdict(rows, col_names[:3])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
