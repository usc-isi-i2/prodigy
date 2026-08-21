from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .config import load_config


def rows(config: dict, seeds: list[int]) -> list[dict[str, str | int]]:
    names = list(config["graphs"])
    result = []
    for seed in seeds:
        for target in names:
            result.append({"run_id": f"specialist_{target}_s{seed}", "kind": "specialist", "target": target,
                           "sources": target, "seed": seed})
            sources = ",".join(name for name in names if name != target)
            result.append({"run_id": f"loo_{target}_s{seed}", "kind": "leave_one_out", "target": target,
                           "sources": sources, "seed": seed})
    return result


def ladder_rows(config: dict, seeds: list[int]) -> list[dict[str, str | int]]:
    """New intermediate held-out mixtures using one fixed global source order.

    For every target, remove that graph from the catalog order and take prefixes of
    lengths 2..N-2. Endpoints are deliberately omitted because k=1 specialists
    and k=N-1 leave-one-out models are already trained by the primary design.
    Duplicate physical source mixtures are represented once in the training
    manifest and mapped to every applicable target separately.
    """
    names = list(config["graphs"])
    train_by_key: dict[tuple[int, tuple[str, ...]], dict[str, str | int]] = {}
    evaluations: list[dict[str, str | int]] = []
    for seed in seeds:
        for target in names:
            ordered_sources = [name for name in names if name != target]
            for size in range(2, len(ordered_sources)):
                sources = tuple(ordered_sources[:size])
                key = (seed, sources)
                run_id = f"ladder_k{size}_{'-'.join(sources)}_s{seed}"
                train_by_key.setdefault(key, {
                    "run_id": run_id, "kind": f"ladder_k{size}",
                    "target": "_", "sources": ",".join(sources), "seed": seed,
                })
                evaluations.append({
                    "run_id": run_id, "kind": f"ladder_k{size}",
                    "target": target, "sources": ",".join(sources), "seed": seed,
                })
    return list(train_by_key.values()), evaluations


def matrix_rows(config: dict, seeds: list[int]) -> list[dict[str, str | int]]:
    """Evaluate every primary checkpoint family on every graph."""
    names = list(config["graphs"])
    evaluations = []
    for row in rows(config, seeds):
        for target in names:
            evaluations.append({**row, "target": target})
    return evaluations


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/graphs.yaml")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--output", default="manifests/primary.tsv")
    parser.add_argument("--design", choices=("primary", "ladder", "matrix"), default="primary")
    parser.add_argument("--eval-output")
    args = parser.parse_args()
    config = load_config(args.config)
    seeds = [int(item) for item in args.seeds.split(",")]
    if args.design == "primary":
        result = rows(config, seeds)
        eval_result = None
    elif args.design == "ladder":
        result, eval_result = ladder_rows(config, seeds)
    else:
        result = matrix_rows(config, seeds)
        eval_result = None
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result[0]), delimiter="\t")
        writer.writeheader(); writer.writerows(result)
    print(f"wrote {output}: {len(result)} runs")
    if eval_result is not None:
        eval_output = Path(args.eval_output or output.with_name(output.stem + "_eval.tsv"))
        eval_output.parent.mkdir(parents=True, exist_ok=True)
        with eval_output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(eval_result[0]), delimiter="\t")
            writer.writeheader(); writer.writerows(eval_result)
        print(f"wrote {eval_output}: {len(eval_result)} evaluations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
