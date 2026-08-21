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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/graphs.yaml")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--output", default="manifests/primary.tsv")
    args = parser.parse_args()
    config = load_config(args.config)
    result = rows(config, [int(item) for item in args.seeds.split(",")])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result[0]), delimiter="\t")
        writer.writeheader(); writer.writerows(result)
    print(f"wrote {output}: {len(result)} runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

