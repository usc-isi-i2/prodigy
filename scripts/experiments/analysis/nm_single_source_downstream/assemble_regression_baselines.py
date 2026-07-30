#!/usr/bin/env python3
"""Assemble multi-seed two-dataset regression floors."""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from statistics import mean, stdev


DATASETS = ["ukr_rus_suspended", "twibot20"]
TARGETS = ["followers_count", "statuses_count", "account_age_days"]
BASELINES = ["raw_features", "raw_degree", "random_init"]
RANDOM_INIT_RE = re.compile(r"^random_init_s(?P<seed>\d+)$")


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def seed_from_path(path: Path) -> int:
    match = re.search(r"_seed(?P<seed>\d+)\.csv$", path.name)
    if match is None:
        raise ValueError(f"Could not parse seed from {path}")
    return int(match.group("seed"))


def raw_floor_paths(data_dir: Path, baseline: str) -> list[tuple[Path, int]]:
    paths = sorted(data_dir.glob(f"regression_baseline_{baseline}_seed*.csv"))
    if paths:
        return [(path, seed_from_path(path)) for path in paths]
    legacy = data_dir / f"regression_baseline_{baseline}.csv"
    return [(legacy, 0)] if legacy.is_file() else []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds",
        default="",
        help="Optional comma-separated seed set; rejects missing cells and ignores stale runs.",
    )
    args = parser.parse_args()
    requested_seeds = (
        [int(value) for value in args.seeds.split(",") if value.strip()]
        if args.seeds
        else []
    )

    here = Path(__file__).resolve().parent
    data_dir = here / "data"
    cells: dict[tuple[str, str, str, int], float] = {}

    for baseline in ("raw_features", "raw_degree"):
        paths = (
            [
                (
                    data_dir / f"regression_baseline_{baseline}_seed{seed}.csv",
                    seed,
                )
                for seed in requested_seeds
            ]
            if requested_seeds
            else raw_floor_paths(data_dir, baseline)
        )
        for path, file_seed in paths:
            if not path.is_file():
                continue
            for row in read_csv(path):
                if row["dataset"] not in DATASETS or row["target"] not in TARGETS:
                    continue
                # The CSV's seed is the actual episode RNG seed (448 + run seed);
                # the filename carries the cross-method experiment seed used by
                # random_init_s<seed>.
                seed = file_seed
                cells[(baseline, row["dataset"], row["target"], seed)] = float(
                    row["spearman"]
                )

    random_path = (
        data_dir / "regression_baseline_random_init_parsed"
        / "node_regression/data/node_regression.csv"
    )
    for row in read_csv(random_path):
        match = RANDOM_INIT_RE.fullmatch(row["model"])
        if match is None and row["model"] != "random_init":
            continue
        if (
            row["dataset"] in DATASETS
            and row["target"] in TARGETS
            and row["split"] == "test"
            and row["shots"] == "10"
        ):
            seed = int(match.group("seed")) if match is not None else 0
            if requested_seeds and seed not in requested_seeds:
                continue
            cells[("random_init", row["dataset"], row["target"], seed)] = float(
                row["spearman"]
            )

    seeds = requested_seeds or sorted({seed for *_, seed in cells})
    missing = [
        (baseline, dataset, target, seed)
        for baseline in BASELINES
        for dataset in DATASETS
        for target in TARGETS
        for seed in seeds
        if (baseline, dataset, target, seed) not in cells
    ]
    if not seeds or missing:
        if not seeds:
            print("missing: no seeds found")
        for item in missing:
            print("missing:", " / ".join(map(str, item)))
        return 1

    seed_rows = [
        {
            "baseline": baseline,
            "dataset": dataset,
            "target": target,
            "seed": seed,
            "spearman": cells[(baseline, dataset, target, seed)],
        }
        for baseline in BASELINES
        for dataset in DATASETS
        for target in TARGETS
        for seed in seeds
    ]
    seed_out = data_dir / "regression_baseline_seeds.csv"
    with seed_out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["baseline", "dataset", "target", "seed", "spearman"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(seed_rows)

    summary_rows = []
    for baseline in BASELINES:
        for dataset in DATASETS:
            for target in TARGETS:
                values = [
                    cells[(baseline, dataset, target, seed)]
                    for seed in seeds
                ]
                summary_rows.append(
                    {
                        "baseline": baseline,
                        "dataset": dataset,
                        "target": target,
                        "mean": mean(values),
                        "std": stdev(values) if len(values) > 1 else 0.0,
                        "n": len(values),
                    }
                )

    summary_out = data_dir / "regression_baselines.csv"
    with summary_out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["baseline", "dataset", "target", "mean", "std", "n"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(
        f"wrote {seed_out} ({len(seed_rows)} rows) and "
        f"{summary_out} ({len(summary_rows)} rows), seeds={seeds}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
