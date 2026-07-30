#!/usr/bin/env python3
"""Attach profile-regression targets to an experimental Ukraine-suspended graph copy.

The canonical graph was built from ``user_data.csv`` but stores only the suspended
classification label. Rows in the CSV are the node ids used by the graph builder, so
the profile columns can be attached without rebuilding edges or node features.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch


TARGET_COLUMNS = {
    "followers_count": "followers_count",
    "statuses_count": "statuses_count",
    "account_age_days": "acc_age",
}


def load_graph(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--user-csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.out.exists() and not args.overwrite:
        raise SystemExit(f"output exists (pass --overwrite): {args.out}")

    raw = load_graph(args.graph)
    x = raw.get("x")
    if x is None or x.ndim != 2:
        raise SystemExit(f"graph has no two-dimensional x tensor: {args.graph}")
    num_nodes = int(x.shape[0])

    frame = pd.read_csv(args.user_csv, usecols=list(TARGET_COLUMNS.values()))
    if len(frame) != num_nodes:
        raise SystemExit(
            f"row alignment failed: graph has {num_nodes} nodes, CSV has {len(frame)} rows"
        )

    node_targets = dict(raw.get("node_targets") or {})
    stats = dict(raw.get("benchmark_target_stats") or {})
    for target, column in TARGET_COLUMNS.items():
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float32)
        tensor = torch.from_numpy(values)
        finite = torch.isfinite(tensor)
        if int(finite.sum()) < 100:
            raise SystemExit(f"{target}: only {int(finite.sum())} finite values")
        node_targets[target] = tensor
        stats[target] = {
            "finite": int(finite.sum()),
            "missing": int((~finite).sum()),
            "min": float(tensor[finite].min()),
            "max": float(tensor[finite].max()),
            "source_column": column,
        }

    raw["node_targets"] = node_targets
    raw["node_target_names"] = list(TARGET_COLUMNS)
    raw["benchmark_target_stats"] = stats

    args.out.parent.mkdir(parents=True, exist_ok=True)
    temp = args.out.with_suffix(args.out.suffix + ".tmp")
    if temp.exists():
        temp.unlink()
    torch.save(raw, temp)

    check = load_graph(temp)
    check_targets = check.get("node_targets") or {}
    for target in TARGET_COLUMNS:
        if target not in check_targets or len(check_targets[target]) != num_nodes:
            raise SystemExit(f"saved graph validation failed for {target}")
    temp.replace(args.out)

    print(f"wrote experimental enriched graph: {args.out}")
    for target in TARGET_COLUMNS:
        target_stats = stats[target]
        print(
            f"{target}: finite={target_stats['finite']} "
            f"missing={target_stats['missing']} source={target_stats['source_column']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
