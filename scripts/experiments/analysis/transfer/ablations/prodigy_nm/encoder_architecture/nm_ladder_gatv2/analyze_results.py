#!/usr/bin/env python3
"""Assemble the 8x8 GATv2 NM ladder and compare it with GraphSAGE.

The eval wrapper produces directories named
``eval_nm_ladder_gatv2_r<R>_<N>src_to_<dataset>_nm_3shot_30way_<timestamp>``.
This script selects the newest complete value per (rung, test graph), refuses a
partial 8x8 table by default, and writes only experiment-dedicated outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path


CANON = [
    "ukr_rus_twitter",
    "covid19_twitter",
    "midterm",
    "covid_political",
    "election2020",
    "ukr_rus_suspended",
    "twibot20",
    "cp_hk_twitter",
]
SHORT = {
    "ukr_rus_twitter": "ukr",
    "covid19_twitter": "covid",
    "midterm": "midterm",
    "covid_political": "cov_pol",
    "election2020": "elec20",
    "ukr_rus_suspended": "ukr_susp",
    "twibot20": "twibot20",
    "cp_hk_twitter": "cp_hk",
}
RUNGS = [
    (1, "ukr", "ukr_rus_twitter"),
    (2, "ukr+cov", "covid19_twitter"),
    (3, "ukr+cov+mid", "midterm"),
    (4, "+cov_pol", "covid_political"),
    (5, "+elec20", "election2020"),
    (6, "+ukr_susp", "ukr_rus_suspended"),
    (7, "+twibot20", "twibot20"),
    (8, "all8", "cp_hk_twitter"),
]
ENTRY_RUNG = {dataset: rung for rung, _, dataset in RUNGS}
PRIMARY_EVENTS = (4, 5, 6, 7, 8)
RUN_RE = re.compile(
    r"^eval_nm_ladder_gatv2_r(?P<rung>[1-8])_(?P<n_sources>[1-8])src_to_"
    r"(?P<test>.+?)_nm_(?P<shots>\d+)shot_(?P<nway>\d+)way(?:_|$)"
)
STEP_RE = re.compile(r"_step(?P<step>\d+)\.json$")


@dataclass(frozen=True)
class Cell:
    auc: float
    run_dir: Path
    metrics_path: Path
    mtime: float


def metrics_step(path: Path) -> int:
    match = STEP_RE.search(path.name)
    return int(match["step"]) if match else -1


def read_auc(run_dir: Path) -> tuple[float, Path] | None:
    paths = sorted(
        (run_dir / "data").glob("metrics_test*.json"),
        key=metrics_step,
        reverse=True,
    )
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        value = payload.get("test_roc_auc")
        if value is not None:
            return float(value), path
    return None


def collect_cells(log_root: Path, shots: int = 3, n_way: int = 30) -> dict[tuple[int, str], Cell]:
    cells: dict[tuple[int, str], Cell] = {}
    if not log_root.is_dir():
        return cells
    for run_dir in log_root.glob("eval_nm_ladder_gatv2_r*_to_*_nm_*shot_*way*"):
        if not run_dir.is_dir():
            continue
        match = RUN_RE.match(run_dir.name)
        if match is None:
            continue
        rung = int(match["rung"])
        if int(match["n_sources"]) != rung:
            continue
        if int(match["shots"]) != shots or int(match["nway"]) != n_way:
            continue
        test = match["test"]
        if test not in CANON:
            continue
        result = read_auc(run_dir)
        if result is None:
            continue
        auc, metrics_path = result
        cell = Cell(auc=auc, run_dir=run_dir, metrics_path=metrics_path, mtime=run_dir.stat().st_mtime)
        key = (rung, test)
        previous = cells.get(key)
        if previous is None or (cell.mtime, str(cell.run_dir)) > (previous.mtime, str(previous.run_dir)):
            cells[key] = cell
    return cells


def missing_cells(cells: dict[tuple[int, str], Cell]) -> list[tuple[int, str]]:
    return [
        (rung, dataset)
        for rung, _, _ in RUNGS
        for dataset in CANON
        if (rung, dataset) not in cells
    ]


def load_sage(path: Path) -> dict[tuple[int, str], float]:
    values: dict[tuple[int, str], float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rung = int(row["rung"])
            for dataset in CANON:
                values[(rung, dataset)] = float(row[dataset])
    expected = {(rung, dataset) for rung, _, _ in RUNGS for dataset in CANON}
    absent = sorted(expected - values.keys())
    if absent:
        raise ValueError(f"GraphSAGE baseline is incomplete: {absent}")
    return values


def table_from_cells(cells: dict[tuple[int, str], Cell]) -> dict[tuple[int, str], float]:
    return {key: cell.auc for key, cell in cells.items()}


def summarize(values: dict[tuple[int, str], float]) -> dict[str, object]:
    events: list[dict[str, object]] = []
    for rung, _, added in RUNGS[1:]:
        before = values[(rung - 1, added)]
        after = values[(rung, added)]
        events.append(
            {
                "rung": rung,
                "dataset": added,
                "before": before,
                "after": after,
                "entry_delta": after - before,
                "primary": rung in PRIMARY_EVENTS,
            }
        )

    trajectories: list[dict[str, object]] = []
    for dataset in CANON:
        entry = ENTRY_RUNG[dataset]
        pre = [values[(rung, dataset)] for rung in range(1, entry)]
        at_entry = values[(entry, dataset)]
        final = values[(8, dataset)]
        trajectories.append(
            {
                "dataset": dataset,
                "entry_rung": entry,
                "pre_entry_range": max(pre) - min(pre) if len(pre) > 1 else 0.0,
                "at_entry": at_entry,
                "final": final,
                "post_entry_retention": final - at_entry,
            }
        )

    primary = [event for event in events if event["primary"]]
    substantial_without_twibot = [
        event
        for event in primary
        if event["dataset"] != "twibot20"
    ]
    return {
        "entry_events": events,
        "trajectories": trajectories,
        "registered_checks": {
            "positive_primary_entry_deltas": sum(event["entry_delta"] > 0 for event in primary),
            "primary_entry_event_count": len(primary),
            "non_twibot_primary_deltas_over_0_02": sum(
                event["entry_delta"] > 0.02 for event in substantial_without_twibot
            ),
            "non_twibot_primary_event_count": len(substantial_without_twibot),
        },
    }


def write_outputs(
    values: dict[tuple[int, str], float],
    sage: dict[tuple[int, str], float],
    out_dir: Path,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    wide_path = out_dir / "nm_ladder_gatv2.csv"
    with wide_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rung", "n_sources", "train_graph", "added", "sampling", "backbone"] + CANON)
        for rung, label, added in RUNGS:
            writer.writerow(
                [rung, rung, label, added, "within_balanced", "gatv2"]
                + [f"{values[(rung, dataset)]:.6f}" for dataset in CANON]
            )

    comparison_path = out_dir / "nm_ladder_backbone_comparison.csv"
    with comparison_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "rung",
            "n_sources",
            "test_graph",
            "entry_rung",
            "in_training_merge",
            "sage_auc",
            "gatv2_auc",
            "gatv2_minus_sage",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rung, _, _ in RUNGS:
            for dataset in CANON:
                gatv2 = values[(rung, dataset)]
                sage_auc = sage[(rung, dataset)]
                writer.writerow(
                    {
                        "rung": rung,
                        "n_sources": rung,
                        "test_graph": dataset,
                        "entry_rung": ENTRY_RUNG[dataset],
                        "in_training_merge": int(rung >= ENTRY_RUNG[dataset]),
                        "sage_auc": f"{sage_auc:.6f}",
                        "gatv2_auc": f"{gatv2:.6f}",
                        "gatv2_minus_sage": f"{gatv2 - sage_auc:+.6f}",
                    }
                )

    summary = summarize(values)
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def print_summary(summary: dict[str, object]) -> None:
    print("\nEntry-aligned deltas:")
    for event in summary["entry_events"]:
        marker = "PRIMARY" if event["primary"] else "context"
        print(
            f"  r{event['rung']} {SHORT[event['dataset']]:9s}: "
            f"{event['before']:.3f} -> {event['after']:.3f} "
            f"({event['entry_delta']:+.3f}) [{marker}]"
        )
    checks = summary["registered_checks"]
    print("\nRegistered checks (descriptive; seed 0 only):")
    print(
        "  positive primary deltas: "
        f"{checks['positive_primary_entry_deltas']}/{checks['primary_entry_event_count']}"
    )
    print(
        "  non-twibot primary deltas > .02: "
        f"{checks['non_twibot_primary_deltas_over_0_02']}/"
        f"{checks['non_twibot_primary_event_count']}"
    )


def main() -> int:
    here = Path(__file__).resolve().parent
    repo_root = next(p for p in here.parents if (p / "AGENTS.md").is_file())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-root", type=Path, default=repo_root / "log")
    parser.add_argument(
        "--sage-csv",
        type=Path,
        default=repo_root / "scripts/experiments/analysis/transfer/ladders/prodigy_nm/baseline/nm_ladder/data/nm_ladder_full.csv",
    )
    parser.add_argument("--out-dir", type=Path, default=here / "data")
    parser.add_argument("--shots", type=int, default=3)
    parser.add_argument("--n-way", type=int, default=30)
    args = parser.parse_args()

    cells = collect_cells(args.log_root, args.shots, args.n_way)
    missing = missing_cells(cells)
    print(f"found {len(cells)}/64 GATv2 ladder cells under {args.log_root}")
    if missing:
        for rung, dataset in missing:
            print(f"MISSING r{rung} -> {dataset}")
        print("refusing to write a partial ladder; complete the missing evaluations")
        return 2

    values = table_from_cells(cells)
    sage = load_sage(args.sage_csv)
    summary = write_outputs(values, sage, args.out_dir)
    print(f"wrote {args.out_dir / 'nm_ladder_gatv2.csv'}")
    print(f"wrote {args.out_dir / 'nm_ladder_backbone_comparison.csv'}")
    print(f"wrote {args.out_dir / 'summary.json'}")
    print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
