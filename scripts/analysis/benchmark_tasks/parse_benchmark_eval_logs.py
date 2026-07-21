#!/usr/bin/env python3
"""Collect node-regression and static-LP eval metrics into tidy CSVs.

Parses eval log directories produced by the shared runner
(``eval_<model>_to_<dataset>_<tag>_..._<shots>shot_<timestamp>/``) and their
``metrics_<split>[_step<N>].json`` files (same layout as
``scripts/analysis/export_eval_results_csv.py``), keeping only the two new tasks:

  * ``reg`` -> node_regression.csv  (spearman, rmse, mae, r2)
  * ``slp`` -> static_link_prediction.csv  (roc_auc, accuracy, f1)

Usage (laptop or Tucker, needs pandas):

    python scripts/analysis/benchmark_tasks/parse_benchmark_eval_logs.py \
        --log-root /dataMeR2/phil/gfm/prodigy/log \
        --out-dir scripts/experiments/analysis
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path
from typing import Any, Optional

# Run dirs: eval_<model>_to_<dataset>_reg_<shots>shot_<target>_<DD_MM_YYYY_HH_MM_SS>
#           eval_<model>_to_<dataset>_slp_<shots>shot_<DD_MM_YYYY_HH_MM_SS>
_TS = r"_\d{2}_\d{2}_\d{4}_\d{2}_\d{2}_\d{2}$"
_TS_RE = re.compile(r"_(\d{2})_(\d{2})_(\d{4})_(\d{2})_(\d{2})_(\d{2})$")


def _run_timestamp(run_name: str) -> tuple:
    """Sortable key from the trailing _DD_MM_YYYY_HH_MM_SS so the latest run of a
    config wins (the log dir can hold reruns from several sweeps)."""
    m = _TS_RE.search(run_name)
    if not m:
        return (0,) * 6
    dd, mm, yyyy, hh, mi, ss = (int(g) for g in m.groups())
    return (yyyy, mm, dd, hh, mi, ss)
REG_RE = re.compile(
    r"^eval_(?P<model>.+?)_to_(?P<dataset>.+?)_reg_(?P<shots>\d+)shot_(?P<target>.+?)" + _TS
)
SLP_RE = re.compile(
    r"^eval_(?P<model>.+?)_to_(?P<dataset>.+?)_slp_(?P<shots>\d+)shot" + _TS
)
# few-shot classification (pl); skip the probe_* capability datasets (their own parser)
PL_RE = re.compile(
    r"^eval_(?P<model>.+?)_to_(?P<dataset>(?!probe_).+?)_pl_(?P<shots>\d+)shot" + _TS
)

REG_METRICS = ("spearman", "rmse", "mae", "r2", "mse")
SLP_METRICS = ("roc_auc", "accuracy", "f1")
PL_METRICS = ("roc_auc", "accuracy", "f1")


def _split_of(path: Path) -> Optional[str]:
    m = re.match(r"metrics_(?P<split>.+?)(?:_step\d+)?\.json$", path.name)
    return m.group("split") if m else None


def _numeric_metrics(payload: dict[str, Any], split: str) -> dict[str, float]:
    out: dict[str, float] = {}
    prefix = f"{split}_"
    for key, value in payload.items():
        if not isinstance(value, (int, float)):
            continue
        out[key[len(prefix):] if key.startswith(prefix) else key] = float(value)
    return out


def _latest_metrics(run_dir: Path, split: str) -> dict[str, float]:
    """Return the highest-step metrics for a split (falls back to unstepped).

    Eval writes metrics under ``<run_dir>/data/metrics_<split>[_step<N>].json``.
    """
    best_step = -1
    best: dict[str, float] = {}
    for path in (run_dir / "data").glob(f"metrics_{split}*.json"):
        if _split_of(path) != split:
            continue
        m = re.search(r"_step(\d+)\.json$", path.name)
        step = int(m.group(1)) if m else 0
        if step >= best_step:
            try:
                payload = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            best_step, best = step, _numeric_metrics(payload, split)
    return best


def _rows_for_run(run_dir: Path) -> list[dict[str, Any]]:
    name = run_dir.name
    reg = REG_RE.match(name)
    slp = SLP_RE.match(name)
    pl = PL_RE.match(name)
    rows: list[dict[str, Any]] = []
    for split in ("test", "val"):
        metrics = _latest_metrics(run_dir, split)
        if not metrics:
            continue
        if reg:
            row = {**reg.groupdict(), "task": "regression", "split": split, "run": name}
            row["shots"] = int(row["shots"])
            for k in REG_METRICS:
                row[k] = metrics.get(k)
            rows.append(row)
        elif slp:
            row = {**slp.groupdict(), "target": "", "task": "static_link_prediction",
                   "split": split, "run": name}
            row["shots"] = int(row["shots"])
            for k in SLP_METRICS:
                row[k] = metrics.get(k)
            rows.append(row)
        elif pl:
            row = {**pl.groupdict(), "target": "", "task": "classification",
                   "split": split, "run": name}
            row["shots"] = int(row["shots"])
            for k in PL_METRICS:
                row[k] = metrics.get(k)
            rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log-root", required=True, help="Directory containing eval_* run dirs.")
    ap.add_argument("--out-dir", default="scripts/experiments/analysis",
                    help="Base dir; CSVs land under <out-dir>/{node_regression,static_link_prediction}/data.")
    ap.add_argument("--reg-glob", default="eval_*_reg_*")
    ap.add_argument("--slp-glob", default="eval_*_slp_*")
    ap.add_argument("--pl-glob", default="eval_*_pl_*")
    args = ap.parse_args()

    import pandas as pd

    log_root = Path(args.log_root)
    run_dirs = sorted(
        {Path(p) for p in glob.glob(str(log_root / args.reg_glob))}
        | {Path(p) for p in glob.glob(str(log_root / args.slp_glob))}
        | {Path(p) for p in glob.glob(str(log_root / args.pl_glob))}
    )
    print(f"[parse] {len(run_dirs)} candidate run dirs under {log_root}")

    task_rows = {"regression": [], "static_link_prediction": [], "classification": []}
    for run_dir in run_dirs:
        if not run_dir.is_dir():
            continue
        for row in _rows_for_run(run_dir):
            task_rows[row["task"]].append(row)

    out_base = Path(args.out_dir)
    written = []
    dedup_keys = {
        "node_regression": ["model", "dataset", "target", "shots", "split"],
        "static_link_prediction": ["model", "dataset", "shots", "split"],
        "node_classification": ["model", "dataset", "shots", "split"],
    }
    for name, rows in (("node_regression", task_rows["regression"]),
                       ("static_link_prediction", task_rows["static_link_prediction"]),
                       ("node_classification", task_rows["classification"])):
        data_dir = out_base / name / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        csv_path = data_dir / f"{name}.csv"
        df = pd.DataFrame(rows)
        if not df.empty:
            # keep the latest run per config (reruns from earlier sweeps are dropped)
            df["_ts"] = df["run"].map(_run_timestamp)
            df = (df.sort_values("_ts")
                    .drop_duplicates(dedup_keys[name], keep="last")
                    .drop(columns="_ts"))
        df.to_csv(csv_path, index=False)
        written.append((csv_path, len(df)))
    for path, n in written:
        print(f"[parse] wrote {n:>4} rows -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
