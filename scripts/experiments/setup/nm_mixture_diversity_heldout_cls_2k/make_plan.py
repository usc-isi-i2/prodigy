#!/usr/bin/env python3
"""Build the exhaustive held-out-target mixture-diversity plan."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path


HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "manifest.tsv"
EVALUATION_MANIFEST = HERE / "evaluation_manifest.tsv"

# graph_id order and names from merge_ukr_rus_covid_midterm_all8.yaml.
SOURCES = (
    ("ukr", "ukr_rus", "ukr_rus_twitter"),
    ("cov", "covid", "covid19_twitter"),
    ("mid", "midterm", "midterm"),
    ("cp", "covid_political", "covid_political"),
    ("elec", "election2020", "election2020"),
    ("susp", "ukr_rus_suspended", "ukr_rus_suspended"),
    ("bot", "twibot20", "twibot20"),
    ("hk", "cp_hk", "cp_hk_twitter"),
)
SOURCE_BY_SUBSET = {subset: (code, dataset) for code, subset, dataset in SOURCES}
TARGET_SUBSETS = ("covid_political", "election2020", "ukr_rus_suspended", "twibot20")


def rows() -> list[dict[str, object]]:
    """Return unique physical training runs; checkpoints are reusable across targets."""
    plan: list[dict[str, object]] = []
    for size in range(1, 5):
        for combination in itertools.combinations(SOURCES, size):
            donor_codes = tuple(source[0] for source in combination)
            donor_subsets = tuple(source[1] for source in combination)
            heldout_targets = tuple(target for target in TARGET_SUBSETS if target not in donor_subsets)
            plan.append(
                {
                    "mixture_size": size,
                    "donor_codes": donor_codes,
                    "donors": donor_subsets,
                    "heldout_targets": heldout_targets,
                    "prefix": f"mixdiv2k_k{size}_{'-'.join(donor_codes)}",
                }
            )
    return plan


def render_manifest(plan: list[dict[str, object]] | None = None) -> str:
    lines = ["mixture_size\tdonor_codes\tdonors\theldout_targets\tprefix"]
    for row in plan or rows():
        lines.append(
            "\t".join(
                (
                    str(row["mixture_size"]),
                    ",".join(row["donor_codes"]),
                    ",".join(row["donors"]),
                    ",".join(row["heldout_targets"]),
                    str(row["prefix"]),
                )
            )
        )
    return "\n".join(lines) + "\n"


def evaluation_rows(plan: list[dict[str, object]] | None = None) -> list[dict[str, object]]:
    associations = []
    for row in plan or rows():
        for target in row["heldout_targets"]:
            target_code, target_dataset = SOURCE_BY_SUBSET[target]
            associations.append(
                {
                    **row,
                    "target": target,
                    "target_code": target_code,
                    "target_dataset": target_dataset,
                }
            )
    return associations


def render_evaluation_manifest(plan: list[dict[str, object]] | None = None) -> str:
    lines = [
        "target\ttarget_code\ttarget_dataset\tmixture_size\tdonor_codes\tdonors\tprefix"
    ]
    for row in evaluation_rows(plan):
        lines.append(
            "\t".join(
                (
                    str(row["target"]), str(row["target_code"]),
                    str(row["target_dataset"]), str(row["mixture_size"]),
                    ",".join(row["donor_codes"]), ",".join(row["donors"]),
                    str(row["prefix"]),
                )
            )
        )
    return "\n".join(lines) + "\n"


def validate(plan: list[dict[str, object]] | None = None) -> None:
    plan = plan or rows()
    assert len(plan) == 162
    assert len({row["prefix"] for row in plan}) == len(plan)
    associations = evaluation_rows(plan)
    assert len(associations) == 392
    for target in TARGET_SUBSETS:
        target_rows = [row for row in associations if row["target"] == target]
        assert len(target_rows) == 98
        for size, expected in ((1, 7), (2, 21), (3, 35), (4, 35)):
            size_rows = [row for row in target_rows if row["mixture_size"] == size]
            assert len(size_rows) == expected
            assert all(target not in row["donors"] for row in size_rows)
            # Exhaustive combinations balance every eligible donor at a fixed k.
            counts = {
                source[1]: sum(source[1] in row["donors"] for row in size_rows)
                for source in SOURCES
                if source[1] != target
            }
            assert len(set(counts.values())) == 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--list", action="store_true", help="print the TSV plan")
    args = parser.parse_args()
    plan = rows()
    validate(plan)
    expected = render_manifest(plan)
    expected_eval = render_evaluation_manifest(plan)
    if args.list:
        print(expected, end="")
        return 0
    if args.check:
        stale = [
            path for path, contents in ((MANIFEST, expected), (EVALUATION_MANIFEST, expected_eval))
            if not path.is_file() or path.read_text(encoding="utf-8") != contents
        ]
        if stale:
            print(f"ERROR: missing or stale: {', '.join(map(str, stale))}")
            return 1
        print("OK: 162 physical models, 392 held-out target-mixture evaluations")
        return 0
    MANIFEST.write_text(expected, encoding="utf-8")
    EVALUATION_MANIFEST.write_text(expected_eval, encoding="utf-8")
    print(f"wrote 162 training rows and 392 target-mixture evaluation rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
