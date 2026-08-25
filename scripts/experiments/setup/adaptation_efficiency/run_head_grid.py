#!/usr/bin/env python3
"""Run the complete label-budget x optimizer-update grid over feature caches."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from .protocol import (
    LABEL_BUDGETS,
    LABEL_SEEDS,
    load_feature_cache,
    run_curve,
    standardize_and_pad,
    stratified_node_splits,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--label-seeds", default=",".join(map(str, LABEL_SEEDS)))
    parser.add_argument("--label-budgets", default=",".join(map(str, LABEL_BUDGETS)))
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    seeds = tuple(int(value) for value in args.label_seeds.split(",") if value)
    budgets = tuple(int(value) for value in args.label_budgets.split(",") if value)
    if set(budgets) - set(LABEL_BUDGETS):
        raise ValueError(f"budgets must be drawn from {LABEL_BUDGETS}")

    output_rows: list[dict[str, object]] = []
    for path in args.cache:
        cache = load_feature_cache(path)
        splits_global = stratified_node_splits(cache.labels, seed=args.split_seed)
        lookup = {int(node): index for index, node in enumerate(cache.node_ids)}
        missing = {
            split: [int(node) for node in nodes if int(node) not in lookup][:5]
            for split, nodes in splits_global.items()
        }
        missing = {split: nodes for split, nodes in missing.items() if nodes}
        if missing:
            raise ValueError(f"cache {path} lacks split nodes: {missing}")
        splits = {
            split: np.asarray([lookup[int(node)] for node in nodes], dtype=np.int64)
            for split, nodes in splits_global.items()
        }
        labels = cache.labels[cache.node_ids]
        kind = "mlp" if cache.model_id == "raw_mlp" else "linear"
        features = cache.features
        if kind == "linear":
            features = standardize_and_pad(features, splits["train"])
        else:
            features = standardize_and_pad(features, splits["train"], output_dim=features.shape[1])
        for seed in seeds:
            for budget in budgets:
                output_rows.extend(
                    run_curve(
                        features,
                        labels,
                        splits,
                        model_id=cache.model_id,
                        target=cache.target,
                        label_seed=seed,
                        budget=budget,
                        head_kind=kind,
                    )
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"wrote {len(output_rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
