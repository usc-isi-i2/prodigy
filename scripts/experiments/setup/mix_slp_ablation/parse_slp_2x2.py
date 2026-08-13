#!/usr/bin/env python3
"""Collect the mix_slp_ablation 2x2 static-LP runs into one tidy CSV.

Reads each eval run dir's ``data/metrics_test.json`` (and ``metrics_val`` if
present) directly instead of going through
``scripts/harness/benchmark_tasks/parse_benchmark_eval_logs.py``, whose SLP_RE
does not match the ablation-tagged run dirs.

Run-dir naming (from scripts/eval/eval_ckpts_all_graph_tasks_tucker.py):

    eval_<model>_to_<dataset>_slp[_abl<TAG>]_<shots>shot_<DD_MM_YYYY_HH_MM_SS>

    <TAG>: E = --ablate-edges rewire, P = --ablate-features permute,
           PE = both (feature tag letter precedes the edge tag letter).

Conditions: '' -> none, 'E' -> rewire, 'P' -> permute, 'PE' -> both.
When a (model, dataset, condition, shots, split) cell has several runs, the
latest timestamp wins.

Usage:
    python3 scripts/experiments/setup/mix_slp_ablation/parse_slp_2x2.py \
        --log-root log \
        --out scripts/experiments/analysis/archive/mix_slp_ablation/data/slp_ablation_2x2.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

RUN_RE = re.compile(
    r"^eval_(?P<model>.+?)_to_(?P<dataset>.+?)_slp(?:_abl(?P<abl>[ZPNE]+))?"
    r"_(?P<shots>\d+)shot_(?P<ts>\d{2}_\d{2}_\d{4}_\d{2}_\d{2}_\d{2})$"
)
CONDITION = {None: "none", "E": "rewire", "P": "permute", "PE": "both"}
METRICS = ("roc_auc", "accuracy", "f1")


def ts_key(ts: str) -> tuple:
    dd, mm, yyyy, hh, mi, ss = (int(g) for g in ts.split("_"))
    return (yyyy, mm, dd, hh, mi, ss)


def latest_metrics(run_dir: Path, split: str) -> dict:
    """Highest-step metrics_<split>[_step<N>].json under <run_dir>/data."""
    best_step, best = -1, {}
    for path in (run_dir / "data").glob(f"metrics_{split}*.json"):
        m = re.fullmatch(rf"metrics_{split}(?:_step(\d+))?\.json", path.name)
        if not m:
            continue
        step = int(m.group(1)) if m.group(1) else 0
        if step >= best_step:
            try:
                payload = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            prefix = f"{split}_"
            best_step = step
            best = {
                (k[len(prefix):] if k.startswith(prefix) else k): v
                for k, v in payload.items()
                if isinstance(v, (int, float))
            }
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default="NM,MIX",
                    help="Comma-separated model keys to keep (default NM,MIX).")
    args = ap.parse_args()

    keep_models = {m.strip() for m in args.models.split(",") if m.strip()}
    cells: dict[tuple, tuple[tuple, dict]] = {}
    for run_dir in sorted(Path(args.log_root).glob("eval_*_slp*shot_*")):
        m = RUN_RE.match(run_dir.name)
        if not m:
            continue
        abl = m.group("abl")
        if abl is not None and abl not in CONDITION:
            continue  # foreign ablation tags (e.g. Z/N from other experiments)
        if keep_models and m.group("model") not in keep_models:
            continue
        for split in ("test", "val"):
            metrics = latest_metrics(run_dir, split)
            if not metrics:
                continue
            key = (m.group("model"), m.group("dataset"), CONDITION[abl],
                   int(m.group("shots")), split)
            entry = (ts_key(m.group("ts")), {"run": run_dir.name, **metrics})
            if key not in cells or entry[0] > cells[key][0]:
                cells[key] = entry

    if not cells:
        raise SystemExit(f"no matching slp runs under {args.log_root}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cond_order = {"none": 0, "rewire": 1, "permute": 2, "both": 3}
    with out_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["model", "dataset", "condition", "shots", "split",
                         *METRICS, "run"])
        for key in sorted(cells, key=lambda k: (k[0], k[1], cond_order[k[2]],
                                                k[3], k[4])):
            model, dataset, condition, shots, split = key
            metrics = cells[key][1]
            writer.writerow([model, dataset, condition, shots, split,
                             *(metrics.get(k, "") for k in METRICS),
                             metrics["run"]])
    print(f"wrote {out_path} ({len(cells)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
