#!/usr/bin/env python3
"""Regression test: parsing must not destroy the shared per-task CSVs.

On 2026-07-27 a sweep run from a per-experiment worktree truncated
``static_link_prediction.csv`` from 149 rows to zero and dropped every historical arm
from the other two CSVs. Cause: the parser rebuilt each CSV from ``--log-root`` alone
and wrote unconditionally, but a worktree's ``log/`` holds only its own sweep.

This test reproduces that exact shape against the REAL committed CSVs: a sweep that ran
only regression must leave classification and static LP untouched, and must add only its
own rows to regression. It also pins the properties the fix depends on -- idempotence,
in-place supersession of a re-evaluated config, byte-level preservation of carried-through
rows, LF endings, and that ``--overwrite`` still performs the destructive rebuild for a
genuine full reparse.

    /opt/homebrew/bin/python3.11 scripts/harness/benchmark_tasks/tests/test_parse_merge.py

Needs pandas; no cluster, no GPU.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
PARSER = REPO_ROOT / "scripts/harness/benchmark_tasks/parse_benchmark_eval_logs.py"
ANALYSIS = REPO_ROOT / "scripts/experiments/analysis"

TASKS = {"node_regression": "node_regression.csv",
         "node_classification": "node_classification.csv",
         "static_link_prediction": "static_link_prediction.csv"}


class Harness:
    def __init__(self, tmp: Path):
        self.tmp = tmp
        (tmp / "log").mkdir(parents=True)
        for name, fn in TASKS.items():
            d = tmp / "analysis" / name / "data"
            d.mkdir(parents=True)
            shutil.copy(ANALYSIS / name / "data" / fn, d / fn)

    def make_run(self, tag, model, dataset, target=None, shots="10",
                 ts="02_02_2026_03_04_05"):
        tgt = f"_{target}" if target else ""
        d = self.tmp / "log" / f"eval_{model}_to_{dataset}_{tag}_{shots}shot{tgt}_{ts}" / "data"
        d.mkdir(parents=True, exist_ok=True)
        (d / "metrics_test.json").write_text(json.dumps({
            "test_roc_auc": 0.777, "test_accuracy": 0.7, "test_f1": 0.7,
            "test_spearman": 0.777, "test_rmse": 1.0, "test_mae": 1.0,
            "test_r2": 0.5, "test_mse": 1.0}))

    def parse(self, *extra):
        r = subprocess.run([sys.executable, str(PARSER),
                            "--log-root", str(self.tmp / "log"),
                            "--out-dir", str(self.tmp / "analysis"), *extra],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout, r.stderr, file=sys.stderr)
            raise AssertionError("parser exited nonzero")

    def path(self, name):
        return self.tmp / "analysis" / name / "data" / TASKS[name]

    def n_rows(self, name):
        return sum(1 for line in self.path(name).read_text().splitlines() if line.strip()) - 1


def check(cond, label, failures):
    print(f"  {'OK  ' if cond else 'FAIL'} {label}")
    if not cond:
        failures.append(label)


def main() -> int:
    failures: list[str] = []
    with tempfile.TemporaryDirectory() as td:
        h = Harness(Path(td))
        base = {n: h.n_rows(n) for n in TASKS}
        print(f"baseline rows: {base}")

        # A sweep that ran ONLY regression -- the failure case.
        h.make_run("reg", "__TESTARM__", "midterm", target="followers_count")
        h.parse()

        check(h.n_rows("static_link_prediction") == base["static_link_prediction"],
              "static LP untouched by a regression-only parse", failures)
        check(h.n_rows("node_classification") == base["node_classification"],
              "classification untouched by a regression-only parse", failures)
        check(h.n_rows("node_regression") == base["node_regression"] + 1,
              "regression gained exactly the new arm", failures)

        txt = h.path("node_regression").read_text()
        check("__TESTARM__" in txt, "new arm present", failures)
        check("\r\n" not in txt, "LF line endings", failures)

        # Carried-through rows must be byte-identical, or `comm` review is useless.
        orig = set((ANALYSIS / "node_regression/data/node_regression.csv")
                   .read_text().splitlines()[1:])
        now = set(txt.splitlines()[1:])
        check(not (orig - now), "every original row preserved byte-for-byte", failures)

        after = {n: h.n_rows(n) for n in TASKS}
        h.parse()
        check({n: h.n_rows(n) for n in TASKS} == after, "idempotent on re-parse", failures)

        # A rerun of the same config supersedes rather than duplicating.
        h.make_run("reg", "__TESTARM__", "midterm", target="followers_count",
                   ts="03_03_2026_03_04_05")
        h.parse()
        check(h.n_rows("node_regression") == after["node_regression"],
              "rerun of a config superseded in place", failures)

        # --overwrite must still rebuild from --log-root alone.
        h.parse("--overwrite")
        check(h.n_rows("node_regression") < base["node_regression"],
              "--overwrite still performs the destructive rebuild", failures)

    print("\nRESULT:", "ALL PASS" if not failures else f"{len(failures)} FAILURE(S)")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
