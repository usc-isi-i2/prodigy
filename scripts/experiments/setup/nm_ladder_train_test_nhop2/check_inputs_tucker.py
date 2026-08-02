#!/usr/bin/env python3
"""Audit source and merged static views before split-aware NM training."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
import sys

import torch

from make_configs import SOURCES


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
MERGED = Path(
    "/dataMeR1/phil/data/merged/graphs/"
    "ukr_rus_covid_midterm_all8_static_split_retweet_graph.pt"
)
INPUTS = {
    "ukr_rus": Path("/dataMeR1/phil/data/ukr_rus_twitter/graphs/retweet_graph_parquet.pt"),
    "covid": Path("/dataMeR1/phil/data/covid19_twitter/graphs/retweet_graph_parquet.pt"),
    "midterm": Path("/dataMeR1/phil/data/midterm/graphs/retweet_graph_parquet.pt"),
    "covid_political": Path("/dataMeR1/phil/data/covid_political/graphs/retweet_graph.pt"),
    "election2020": Path("/dataMeR1/phil/data/election2020/graphs/retweet_graph.pt"),
    "ukr_rus_suspended": Path("/dataMeR1/phil/data/ukr_rus_suspended/graphs/retweet_graph.pt"),
    "twibot20": Path("/dataMeR1/phil/data/twibot20/graphs/retweet_graph.pt"),
    "cp_hk": Path("/dataMeR1/phil/data/cp_hk_twitter/graphs/retweet_graph.pt"),
}


def split_views(raw, path: Path):
    try:
        background = raw["edge_index_views"]["static_background"]
        holdout = raw["target_edge_index_views"]["static_holdout"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"{path}: missing static_background/static_holdout") from exc
    return background, holdout


def ensure_split_views(raw, name: str):
    background = raw.get("edge_index_views", {}).get("static_background")
    holdout = raw.get("target_edge_index_views", {}).get("static_holdout")
    if (background is None) != (holdout is None):
        raise ValueError(f"{name}: partial static split views")
    if background is not None:
        return "stored"
    from scripts.graph_construction.benchmark_targets import build_static_edge_split
    split = build_static_edge_split(raw["edge_index"], holdout_frac=0.15, seed=0)
    raw.setdefault("edge_index_views", {})["static_background"] = split.background_edge_index
    raw.setdefault("target_edge_index_views", {})["static_holdout"] = split.holdout_edge_index
    raw.setdefault("benchmark_target_stats", {})["static_split"] = split.stats
    return "derived-in-memory"


def eligible_count(edge_index: torch.Tensor, num_nodes: int, minimum: int = 7) -> int:
    degree = torch.bincount(edge_index[0], minlength=num_nodes)
    degree += torch.bincount(edge_index[1], minlength=num_nodes)
    return int((degree >= minimum).sum())


def audit_one(name: str, path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = torch.load(path, map_location="cpu")
    split_origin = ensure_split_views(raw, name)
    x = raw["x"]
    if x.dim() != 2 or x.shape[1] != 768:
        raise ValueError(f"{path}: expected x shape [N,768], got {tuple(x.shape)}")
    background, holdout = split_views(raw, path)
    n = int(x.shape[0])
    bg_eligible = eligible_count(background, n)
    ho_eligible = eligible_count(holdout, n)
    if min(bg_eligible, ho_eligible) < 30:
        raise ValueError(
            f"{name}: insufficient 30-way centers: background={bg_eligible}, "
            f"holdout={ho_eligible}"
        )
    stats = raw.get(
        "static_split_stats", raw.get("benchmark_target_stats", {}).get("static_split")
    )
    if not stats:
        raise ValueError(f"{path}: no static split provenance/stats")
    print(
        f"OK {name:18s} split={split_origin:17s} nodes={n:,} bg_edges={background.shape[1]:,} "
        f"holdout_edges={holdout.shape[1]:,} eligible(bg/ho)={bg_eligible:,}/{ho_eligible:,}"
    )
    del raw, x, background, holdout
    gc.collect()


def audit_merged(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = torch.load(path, map_location="cpu")
    names = [key for key, _ in SOURCES]
    if raw.get("source_graph_names") != names:
        raise ValueError(
            f"merged source order {raw.get('source_graph_names')} != expected {names}"
        )
    if raw.get("preserved_edge_views") != ["static_background"]:
        raise ValueError("merged artifact did not record static_background preservation")
    if raw.get("preserved_target_edge_views") != ["static_holdout"]:
        raise ValueError("merged artifact did not record static_holdout preservation")
    if not all(raw.get("source_static_split_stats", {}).get(name) for name in names):
        raise ValueError("merged artifact lacks per-source static split provenance")
    background, holdout = split_views(raw, path)
    graph_id = raw["graph_id"].long()
    for source_id, name in enumerate(names):
        mask = graph_id == source_id
        nodes = torch.nonzero(mask, as_tuple=False).flatten()
        lo, hi = int(nodes[0]), int(nodes[-1]) + 1
        for label, view in (("background", background), ("holdout", holdout)):
            local = view[:, (view[0] >= lo) & (view[0] < hi)] - lo
            count = eligible_count(local, hi - lo)
            if count < 30:
                raise ValueError(f"merged {name} {label}: only {count} eligible centers")
        print(f"OK merged source {source_id}: {name} nodes={hi-lo:,}")
    print(
        f"OK merged artifact nodes={raw['x'].shape[0]:,} "
        f"background={background.shape[1]:,} holdout={holdout.shape[1]:,}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", action="store_true", help="audit inputs before merge")
    parser.add_argument("--merged", type=Path, default=MERGED)
    args = parser.parse_args()
    if args.sources:
        for name, _ in SOURCES:
            audit_one(name, INPUTS[name])
    else:
        audit_merged(args.merged)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
