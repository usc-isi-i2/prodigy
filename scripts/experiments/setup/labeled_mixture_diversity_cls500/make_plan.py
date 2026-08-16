#!/usr/bin/env python3
"""Plan every nonempty proper subset of the five labeled source graphs."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRAIN_MANIFEST = HERE / "manifest.tsv"
EVAL_MANIFEST = HERE / "evaluation_manifest.tsv"

SOURCES = (
    ("cp", "covid_political"),
    ("elec", "election2020"),
    ("fb", "facebook_page_reference"),
    ("susp", "ukr_rus_suspended"),
    ("bot", "twibot20"),
)
TARGETS = tuple(source for _, source in SOURCES)
ALL_FIVE_PREFIX = "labmix500_k5_all"


def rows() -> list[dict[str, object]]:
    plan = []
    for size in range(1, len(SOURCES)):
        for combination in itertools.combinations(SOURCES, size):
            codes = tuple(code for code, _ in combination)
            donors = tuple(source for _, source in combination)
            heldout = tuple(target for target in TARGETS if target not in donors)
            plan.append({
                "mixture_size": size,
                "donor_codes": codes,
                "donors": donors,
                "heldout_targets": heldout,
                "prefix": f"labmix500_k{size}_{'-'.join(codes)}",
            })
    return plan


def evaluation_rows(plan=None):
    return [
        {**row, "target": target}
        for row in (plan or rows())
        for target in row["heldout_targets"]
    ]


def all_five_row() -> dict[str, object]:
    return {
        "mixture_size": 5,
        "donor_codes": tuple(code for code, _ in SOURCES),
        "donors": TARGETS,
        "heldout_targets": (),
        "prefix": ALL_FIVE_PREFIX,
    }


def control_evaluation_rows(plan=None):
    plan = plan or rows()
    controls = []
    full = all_five_row()
    for target in TARGETS:
        singleton = next(row for row in plan if row["donors"] == (target,))
        controls.append({**singleton, "target": target, "endpoint": "target_only"})
        controls.append({**full, "target": target, "endpoint": "all_five"})
    return controls


def render_train(plan=None) -> str:
    lines = ["mixture_size\tdonor_codes\tdonors\theldout_targets\tprefix"]
    for row in plan or rows():
        lines.append("\t".join((
            str(row["mixture_size"]), ",".join(row["donor_codes"]),
            ",".join(row["donors"]), ",".join(row["heldout_targets"]),
            str(row["prefix"]),
        )))
    return "\n".join(lines) + "\n"


def render_eval(plan=None) -> str:
    lines = ["target\tmixture_size\tdonor_codes\tdonors\tprefix"]
    for row in evaluation_rows(plan):
        lines.append("\t".join((
            str(row["target"]), str(row["mixture_size"]),
            ",".join(row["donor_codes"]), ",".join(row["donors"]),
            str(row["prefix"]),
        )))
    return "\n".join(lines) + "\n"


def validate(plan=None) -> None:
    plan = plan or rows()
    assert len(plan) == 30
    assert len({row["prefix"] for row in plan}) == 30
    assert {k: sum(row["mixture_size"] == k for row in plan) for k in range(1, 5)} == {
        1: 5, 2: 10, 3: 10, 4: 5,
    }
    associations = evaluation_rows(plan)
    assert len(associations) == 75
    for target in TARGETS:
        target_rows = [row for row in associations if row["target"] == target]
        assert len(target_rows) == 15
        assert {k: sum(row["mixture_size"] == k for row in target_rows) for k in range(1, 5)} == {
            1: 4, 2: 6, 3: 4, 4: 1,
        }
        assert all(target not in row["donors"] for row in target_rows)
    controls = control_evaluation_rows(plan)
    assert len(controls) == 10
    assert all(row["target"] in row["donors"] for row in controls)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    plan = rows()
    validate(plan)
    expected = ((TRAIN_MANIFEST, render_train(plan)), (EVAL_MANIFEST, render_eval(plan)))
    if args.check:
        stale = [path for path, text in expected if not path.is_file() or path.read_text() != text]
        if stale:
            print("ERROR stale: " + ", ".join(map(str, stale)))
            return 1
        print("OK: 30 physical models, 75 held-out CLS evaluations")
        return 0
    for path, text in expected:
        path.write_text(text, encoding="utf-8")
    print("wrote 30-model training and 75-cell evaluation manifests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
