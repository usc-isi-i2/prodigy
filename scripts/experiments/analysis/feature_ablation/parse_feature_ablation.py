#!/usr/bin/env python3
"""Parse a feature-ablation sweep into a long results table.

Reads eval run dirs written by eval_ckpts_all_graph_tasks_tucker.py under
``<log-root>/eval_*/data/metrics_test_step0.json`` and emits one row per
(config, condition), where condition is intact / zero / permute / noise and the
columns are the raw accuracy / roc_auc for that condition. Intact is included as
its own row; the intact-minus-condition gap is printed as a summary (and is
trivially derivable from the table).

Usage:
  python3 parse_feature_ablation.py --log-root log --out feature_ablation_results.csv

No dependency on where features live; only reads the metrics JSONs.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

# trailing run timestamp: _DD_MM_YYYY_HH_MM_SS
TIMESTAMP_RE = re.compile(r"_\d{2}_\d{2}_\d{4}_\d{2}_\d{2}_\d{2}$")
ABL_RE = re.compile(r"_abl([ZPN])")
ABL_NAME = {"Z": "zero", "P": "permute", "N": "noise"}


def config_key_and_mode(run_name: str) -> tuple[str, str]:
    """Strip timestamp + ablation token -> (pairing key, mode)."""
    base = TIMESTAMP_RE.sub("", run_name)
    m = ABL_RE.search(base)
    mode = ABL_NAME[m.group(1)] if m else "intact"
    key = ABL_RE.sub("", base)  # remove the _ablZ/_ablP so intact & ablated share a key
    return key, mode


def read_metrics(run_dir: Path, split: str) -> dict | None:
    f = run_dir / "data" / f"metrics_{split}_step0.json"
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log-root", default="log",
                    help="Directory holding eval_* run dirs (Tucker: /dataMeR1/phil/gfm/prodigy/log).")
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--metrics", default="accuracy,roc_auc",
                    help="Comma-separated metric names (without the split prefix).")
    ap.add_argument("--out", default="feature_ablation_results.csv")
    args = ap.parse_args()

    metric_names = [m.strip() for m in args.metrics.split(",") if m.strip()]
    log_root = Path(args.log_root)
    if not log_root.is_dir():
        raise SystemExit(f"--log-root not found: {log_root}")

    # key -> mode -> {metric: value}
    configs: dict[str, dict[str, dict[str, float]]] = {}
    for run_dir in sorted(log_root.glob("eval_*")):
        if not run_dir.is_dir():
            continue
        payload = read_metrics(run_dir, args.split)
        if payload is None:
            continue
        key, mode = config_key_and_mode(run_dir.name)
        vals = {m: payload.get(f"{args.split}_{m}") for m in metric_names}
        # keep the latest run per (key, mode): later dirs sort after earlier ones
        configs.setdefault(key, {})[mode] = vals

    # long format: one row per (config, condition), intact included as its own row
    condition_order = ["intact", "zero", "permute", "noise"]
    rows = []
    for key, by_mode in sorted(configs.items()):
        if "intact" not in by_mode:
            continue  # no baseline recorded for this config
        for cond in condition_order:
            vals = by_mode.get(cond)
            if vals is None:
                continue
            row = {"config": key, "condition": cond}
            for m in metric_names:
                row[m] = vals.get(m)
            rows.append(row)

    if not rows:
        print(f"No intact runs found under {log_root}. "
              "Did the sweep include the intact ('none') pass?")
        return 1

    fieldnames = ["config", "condition"] + metric_names
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # human-readable summary: biggest feature-reliance gap (intact - condition) first
    primary = metric_names[0]
    intact_by_cfg = {r["config"]: r[primary] for r in rows if r["condition"] == "intact"}
    gaps = []
    for r in rows:
        if r["condition"] == "intact":
            continue
        iv, av = intact_by_cfg.get(r["config"]), r[primary]
        gap = (iv - av) if (iv is not None and av is not None) else None
        gaps.append((gap, r["condition"], r["config"]))
    gaps.sort(key=lambda g: (g[0] is not None, g[0] or 0), reverse=True)
    print(f"Wrote {len(rows)} rows -> {args.out}")
    print(f"\nFeature-reliance gap (intact - condition {primary}), largest first:")
    for gap, cond, cfg in gaps:
        gap_s = f"{gap:+.4f}" if gap is not None else "   n/a "
        print(f"  {gap_s}  [{cond:7s}]  {cfg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
