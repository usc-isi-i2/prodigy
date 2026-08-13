#!/usr/bin/env python3
"""Recover fixed-test metrics printed by the PRODIGY production workers.

The original fixed-test result JSONs retained accuracy and loss, while the
trainer also printed accuracy, macro-F1, and ROC-AUC to the worker logs.  This
script reads those logs from Tucker without modifying the cluster and creates
one local, keyed physical-cell table.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
import subprocess
import sys


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "data/prodigy_final_core/fixed_test/results"
EXACT_AUC = HERE / "data/prodigy_final_core/auc/results"
DEFAULT_OUTPUT = (
    HERE
    / "data/prodigy_final_core/log_recovered_metrics/physical_metrics.tsv"
)

REMOTE_LOG_GLOBS = (
    "/dataMeR1/phil/gfm/prodigy-final-core-cache/log/"
    "final_core_cached_test/queue/production_bs32_worker*.log",
    "/dataMeR1/phil/gfm/prodigy-final-core-cache/log/"
    "final_core_cached_test/recovery/worker*.log",
    "/dataMeR1/phil/gfm/prodigy-final-core-fixed-test/log/"
    "final_core_fixed_test/queue/production_bs32_worker*.log",
)

METRIC_RE = re.compile(
    r"\[metrics\]\s+"
    r"test_(?P<target>[a-z0-9_]+)_accuracy=(?P<accuracy>[0-9.]+)\s+"
    r"test_(?P=target)_f1=(?P<f1>[0-9.]+)\s+"
    r"test_(?P=target)_roc_auc=(?P<auc>[0-9.]+)"
)
DONE_RE = re.compile(
    r"DONE model=(?P<model>\S+) seed=(?P<seed>\d+) "
    r"target=(?P<target>\S+) score=(?P<score>[0-9.]+)"
)

FIELDS = (
    "physical_result_id",
    "seed",
    "model_id",
    "target",
    "accuracy_logged",
    "f1_macro_logged",
    "roc_auc_ovr_macro_logged",
    "printed_decimal_places",
    "source_log",
)


def fetch_metric_lines(host: str) -> str:
    globs = " ".join(REMOTE_LOG_GLOBS)
    remote = (
        f"for f in {globs}; do "
        "printf '@@FILE\\t%s\\n' \"$f\"; "
        "LC_ALL=C grep -aE '\\[metrics\\]|DONE model=' \"$f\"; "
        "done"
    )
    completed = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", host, remote],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def parse_records(text: str) -> list[dict[str, str]]:
    current_log = ""
    pending: dict[str, str] | None = None
    records: list[dict[str, str]] = []
    for line in text.splitlines():
        if line.startswith("@@FILE\t"):
            if pending is not None:
                raise ValueError(f"unpaired metric before {line}")
            current_log = line.split("\t", 1)[1]
            continue
        metric = METRIC_RE.search(line)
        if metric:
            if pending is not None:
                raise ValueError(f"two metrics without DONE in {current_log}")
            pending = metric.groupdict()
            continue
        done = DONE_RE.search(line)
        if not done:
            continue
        if pending is None:
            raise ValueError(f"DONE without preceding metric in {current_log}: {line}")
        identity = done.groupdict()
        if identity["target"] != pending["target"]:
            raise ValueError(
                f"metric/DONE target mismatch in {current_log}: "
                f"{pending['target']} versus {identity['target']}"
            )
        records.append(
            {
                "physical_result_id": (
                    f"prodigy|seed={identity['seed']}|model={identity['model']}|"
                    f"target={identity['target']}|checkpoint=2500"
                ),
                "seed": identity["seed"],
                "model_id": identity["model"],
                "target": identity["target"],
                "accuracy_logged": pending["accuracy"],
                "f1_macro_logged": pending["f1"],
                "roc_auc_ovr_macro_logged": pending["auc"],
                "printed_decimal_places": "4",
                "source_log": current_log,
            }
        )
        pending = None
    if pending is not None:
        raise ValueError(f"final metric is unpaired in {current_log}")
    return records


def validate(records: list[dict[str, str]]) -> None:
    if len(records) != 837:
        raise ValueError(f"recovered {len(records)} records, expected 837")
    keys = {(row["seed"], row["model_id"], row["target"]) for row in records}
    if len(keys) != 837:
        raise ValueError(f"recovered only {len(keys)} unique physical cells")

    raw_paths = list(RESULTS.glob("seed_*/*/*.json"))
    if len(raw_paths) != 837:
        raise ValueError(f"archive has {len(raw_paths)} fixed-test JSONs, expected 837")
    expected = {
        (path.parents[1].name.removeprefix("seed_"), path.parent.name, path.stem): path
        for path in raw_paths
    }
    if keys != set(expected):
        raise ValueError(
            f"logged/archive key mismatch: missing={len(set(expected) - keys)}, "
            f"extra={len(keys - set(expected))}"
        )

    by_key = {(row["seed"], row["model_id"], row["target"]): row for row in records}
    tolerance = 5.1e-5
    for key, path in expected.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        logged = float(by_key[key]["accuracy_logged"])
        if not math.isclose(logged, float(payload["score"]), abs_tol=tolerance):
            raise ValueError(f"logged accuracy does not match {path}: {logged}")

    exact_paths = list(EXACT_AUC.glob("seed_*/*/*.json"))
    if len(exact_paths) != 243:
        raise ValueError(f"archive has {len(exact_paths)} exact AUC JSONs, expected 243")
    for path in exact_paths:
        key = (
            path.parents[1].name.removeprefix("seed_"),
            path.parent.name,
            path.stem,
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        logged = float(by_key[key]["roc_auc_ovr_macro_logged"])
        if not math.isclose(
            logged, float(payload["roc_auc_ovr_macro"]), abs_tol=tolerance
        ):
            raise ValueError(f"logged AUC does not match exact replay {path}: {logged}")


def write_table(records: list[dict[str, str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=FIELDS,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(
            sorted(
                records,
                key=lambda row: (
                    int(row["seed"]),
                    row["model_id"],
                    row["target"],
                ),
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="tucker")
    parser.add_argument(
        "--stdin",
        action="store_true",
        help="read prefiltered worker-log text from standard input instead of SSH",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    source_text = sys.stdin.read() if args.stdin else fetch_metric_lines(args.host)
    records = parse_records(source_text)
    validate(records)
    write_table(records, args.output)
    print(
        f"LOGGED_PRODIGY_METRICS_OK physical_cells={len(records)} "
        f"precision=4dp output={args.output}"
    )


if __name__ == "__main__":
    main()
