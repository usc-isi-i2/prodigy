#!/usr/bin/env python3
"""Require a complete, validity-clean static-LP model matrix."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


DEFAULT_DATASETS = (
    "ukr_rus_twitter",
    "covid19_twitter",
    "midterm",
    "twibot20",
    "cp_hk_twitter",
)
EXPECTED_FLOORS = {
    "common_neighbors",
    "adamic_adar",
    "preferential_attachment",
    "jaccard",
    "raw_feature_cosine",
}


def model_names(path: Path) -> list[str]:
    return [
        line.split()[0]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-dir", type=Path, required=True)
    parser.add_argument("--model-list", type=Path, required=True)
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--negative-kind", default="degree_matched")
    args = parser.parse_args()

    expected = set(model_names(args.model_list))
    errors: list[str] = []
    for dataset in args.datasets.split(","):
        path = args.pair_dir / f"{dataset}__pair_lp.csv"
        if not path.is_file():
            errors.append(f"missing {path}")
            continue
        seen: dict[str, int] = {}
        floors: dict[str, int] = {}
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("negative_kind") != args.negative_kind:
                    continue
                if row.get("model") == "__floor__":
                    scorer = row.get("scorer", "")
                    floors[scorer] = floors.get(scorer, 0) + 1
                    continue
                if row.get("scorer") != "encoder_cosine":
                    continue
                model = row.get("model", "")
                if model not in expected:
                    continue
                seen[model] = seen.get(model, 0) + 1
                leak = float(row["leakage_edges"])
                sensitivity = float(row["endpoint_sensitivity"])
                permutation = float(row["endpoint_permutation_auc"])
                if leak != 0 or sensitivity < 0.99 or abs(permutation - 0.5) >= 0.05:
                    errors.append(
                        f"invalid {dataset}/{model}: leak={leak} "
                        f"sensitivity={sensitivity} permutation_auc={permutation}"
                    )
        missing = sorted(expected - set(seen))
        duplicates = sorted(model for model, count in seen.items() if count != 1)
        if missing:
            errors.append(f"{dataset}: missing {len(missing)} model(s): {', '.join(missing)}")
        if duplicates:
            errors.append(f"{dataset}: duplicate model rows: {', '.join(duplicates)}")
        missing_floors = sorted(EXPECTED_FLOORS - set(floors))
        duplicate_floors = sorted(name for name, count in floors.items() if count != 1)
        if missing_floors:
            errors.append(f"{dataset}: missing floors: {', '.join(missing_floors)}")
        if duplicate_floors:
            errors.append(f"{dataset}: duplicate floors: {', '.join(duplicate_floors)}")
        print(f"{dataset}: {len(seen)}/{len(expected)} clean model rows")

    if errors:
        print(f"ERROR: {len(errors)} pair-LP completeness/validity failure(s)", file=sys.stderr)
        for error in errors:
            print(f"  {error}", file=sys.stderr)
        return 1
    print(f"pair-LP gate passed: {len(expected)} models x {len(args.datasets.split(','))} graphs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
