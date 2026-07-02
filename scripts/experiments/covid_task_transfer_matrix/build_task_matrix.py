#!/usr/bin/env python3
"""Aggregate COVID task-transfer eval logs into train-task x eval-task tables.

Reads zero-adaptation eval dirs produced by eval_task_matrix_tucker.sh:

    eval_covid_task_transfer_<train>_to_<eval>_<timestamp>/

or fixed-adaptation dirs produced by adapt_task_matrix_tucker.sh:

    adapt_covid_task_transfer_<train>_to_<eval>_<timestamp>/

For NM/CL, classifier metrics are read from metrics_test_step0.json. For FP,
the primary score is read from scores_test_step0.json (`test_score`, negative
MSE, higher is better).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


RUN_RE_TEMPLATE = r"^{prefix}_(?P<train>scratch|nm|cl|fp)_to_(?P<eval>nm|cl|fp)(?:_(?P<ts>\d.*))?$"
TASKS = ["nm", "cl", "fp"]
ROW_TASKS = ["scratch", "nm", "cl", "fp"]
TASK_LABELS = {"scratch": "scratch", "nm": "NM", "cl": "CL", "fp": "FP"}


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def step_of(path: Path) -> int:
    match = re.search(r"_step(\d+)\.json$", path.name)
    if match:
        return int(match.group(1))
    return -1


def latest_file(run_dir: Path, pattern: str) -> Path | None:
    files = sorted((run_dir / "data").glob(pattern), key=step_of)
    return files[-1] if files else None


def collect(log_root: Path, run_prefix: str) -> dict[tuple[str, str], dict[str, float | str]]:
    run_re = re.compile(RUN_RE_TEMPLATE.format(prefix=re.escape(run_prefix)))
    cells: dict[tuple[str, str], dict[str, float | str]] = {}
    for run_dir in sorted(log_root.glob(f"{run_prefix}_*_to_*")):
        if not run_dir.is_dir():
            continue
        match = run_re.match(run_dir.name)
        if not match:
            continue

        train = match["train"]
        eval_task = match["eval"]
        metrics_path = latest_file(run_dir, "metrics_test*.json")
        scores_path = latest_file(run_dir, "scores_test*.json")
        metrics = read_json(metrics_path) if metrics_path else {}
        scores = read_json(scores_path) if scores_path else {}

        cell: dict[str, float | str] = {"run_dir": run_dir.name}
        for key in ("test_accuracy", "test_f1", "test_roc_auc"):
            if key in metrics:
                cell[key.removeprefix("test_")] = float(metrics[key])
        for key in ("test_score", "test_score_std", "test_loss", "test_aux_loss"):
            if key in scores:
                cell[key.removeprefix("test_")] = float(scores[key])

        if eval_task == "fp":
            primary = cell.get("score")
        else:
            primary = cell.get("roc_auc", cell.get("accuracy"))
        if primary is not None:
            cell["primary"] = float(primary)
        cells[(train, eval_task)] = cell
    return cells


def print_matrix(cells: dict[tuple[str, str], dict[str, float | str]], metric: str) -> None:
    print(f"\n=== {metric} (train rows / eval cols) ===")
    header = "train\\eval".ljust(12) + "".join(TASK_LABELS[t].ljust(12) for t in TASKS)
    print(header)
    print("-" * len(header))
    rows = [row for row in ROW_TASKS if any(key[0] == row for key in cells)]
    for train in rows:
        line = TASK_LABELS[train].ljust(12)
        for eval_task in TASKS:
            value = cells.get((train, eval_task), {}).get(metric)
            line += (f"{value:.6f}" if isinstance(value, float) else "  -   ").ljust(12)
        print(line)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-root", default="log")
    parser.add_argument("--out-csv", default=None)
    parser.add_argument(
        "--protocol",
        default="zero",
        choices=["zero", "adapt"],
        help="Read zero-adaptation eval dirs or fixed-adaptation dirs.",
    )
    parser.add_argument(
        "--metric",
        default="all",
        choices=["primary", "accuracy", "f1", "roc_auc", "score", "loss", "all"],
    )
    args = parser.parse_args()

    log_root = Path(args.log_root)
    if not log_root.is_dir():
        raise SystemExit(f"log-root not found: {log_root}")

    run_prefix = {
        "zero": "eval_covid_task_transfer",
        "adapt": "adapt_covid_task_transfer",
    }[args.protocol]

    cells = collect(log_root, run_prefix)
    if not cells:
        raise SystemExit(f"No task-transfer eval dirs found under {log_root}")

    metrics = ["primary", "accuracy", "f1", "roc_auc", "score", "loss"] if args.metric == "all" else [args.metric]
    for metric in metrics:
        if any(metric in cell for cell in cells.values()):
            print_matrix(cells, metric)

    if args.out_csv:
        csv_rows: list[list[str]] = []
        task_rows = [row for row in ROW_TASKS if any(key[0] == row for key in cells)]
        for train in task_rows:
            for eval_task in TASKS:
                cell = cells.get((train, eval_task))
                if not cell:
                    continue
                for key, value in cell.items():
                    if key == "run_dir" or not isinstance(value, float):
                        continue
                    csv_rows.append([train, eval_task, key, f"{value:.10g}", str(cell["run_dir"])])
        out = Path(args.out_csv)
        with out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["train_task", "eval_task", "metric", "value", "run_dir"])
            writer.writerows(csv_rows)
        print(f"\nwrote {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
