#!/usr/bin/env python3
"""Parse a feature-ablation sweep into an intact-vs-ablated gap table.

Reads eval run dirs written by eval_ckpts_all_graph_tasks_tucker.py under
``<log-root>/eval_*/data/metrics_test_step0.json`` and pairs, for each
(model, dataset, task, shots, ...) configuration, the INTACT run with its ZEROED
(_ablZ) and PERMUTED (_ablP) counterparts. The intact-minus-ablated gap in
accuracy / roc_auc measures how much that task relies on node features.

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

    rows = []
    for key, by_mode in sorted(configs.items()):
        intact = by_mode.get("intact")
        if intact is None:
            continue  # no baseline to compare against
        for mode in ("zero", "permute", "noise"):
            ablated = by_mode.get(mode)
            if ablated is None:
                continue
            row = {"config": key, "ablation": mode}
            for m in metric_names:
                iv, av = intact.get(m), ablated.get(m)
                row[f"intact_{m}"] = iv
                row[f"ablated_{m}"] = av
                row[f"gap_{m}"] = (iv - av) if (iv is not None and av is not None) else None
            rows.append(row)

    if not rows:
        print(f"No intact/ablated pairs found under {log_root}. "
              "Did the sweep run all of --modes 'none zero permute'?")
        return 1

    fieldnames = ["config", "ablation"]
    for m in metric_names:
        fieldnames += [f"intact_{m}", f"ablated_{m}", f"gap_{m}"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # human-readable summary, biggest feature-reliance gap first
    primary = metric_names[0]
    rows_sorted = sorted(rows, key=lambda r: (r[f"gap_{primary}"] is not None, r.get(f"gap_{primary}") or 0),
                         reverse=True)
    print(f"Wrote {len(rows)} rows -> {args.out}")
    print(f"\nFeature-reliance gap (intact - ablated {primary}), largest first:")
    for r in rows_sorted:
        gap = r[f"gap_{primary}"]
        gap_s = f"{gap:+.4f}" if gap is not None else "   n/a "
        print(f"  {gap_s}  [{r['ablation']:7s}]  {r['config']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
