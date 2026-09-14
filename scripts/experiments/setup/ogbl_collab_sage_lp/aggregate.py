#!/usr/bin/env python3
"""Validate and aggregate the three-seed matched GraphSAGE campaign."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


EXPECTED_SEEDS = (0, 1, 2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    rows = []
    for seed in EXPECTED_SEEDS:
        path = args.root / f"seed{seed}" / "results.json"
        if not path.exists():
            raise SystemExit(f"campaign incomplete; missing seed {seed}: {path}")
        row = json.loads(path.read_text())
        if not row.get("complete") or row.get("seed") != seed:
            raise ValueError(f"invalid seed payload: {path}")
        rows.append(row)
    revisions = {row["revision"] for row in rows}
    datasets = {row["dataset_fingerprint"] for row in rows}
    graphs = {row["graph_fingerprint"] for row in rows}
    if len(revisions) != 1 or len(datasets) != 1 or len(graphs) != 1:
        raise ValueError("incompatible revision, dataset, or graph provenance")
    keys = rows[0]["official_test"]
    aggregate = {
        "complete": True,
        "seeds": list(EXPECTED_SEEDS),
        "revision": next(iter(revisions)),
        "dataset_fingerprint": next(iter(datasets)),
        "graph_fingerprint": next(iter(graphs)),
        "metrics": {
            key: {
                "mean": float(np.mean([row["official_test"][key] for row in rows])),
                "sample_std": float(np.std([row["official_test"][key] for row in rows], ddof=1)),
                "n": len(rows),
            }
            for key in keys
        },
    }
    text = json.dumps(aggregate, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
