#!/usr/bin/env python3
"""Collect capability-probe linear-probe AUCs into the T3 table.

Reads the classification linear-probe eval run dirs
(eval_<arm>_to_probe_<rule>_pl_<shots>shot_lp[<cap>]_<TS>/) and builds:
  rows = arm, cols = rule, cell = test ROC-AUC (the probe score).

Reading (README T3): mean-agg B0 ≈ chance (0.5) on count/degree; E1 passes the
degree rules by structural passthrough; E2 (sum/PNA + directed) lifts
count/existence/in-out; conjunction (cross-neighbor binding) is the hardest.

Usage:
    python parse_capability_probes.py --log-root /dataMeR1/phil/gfm/prodigy/log \
        --model-list scripts/experiments/topology_feature_ssl/model_list.txt \
        --out scripts/experiments/analysis/objectives/topology_vs_features/topology_feature_ssl/data/capability_probes.csv
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

RULES = ["count_threshold", "in_degree", "out_degree", "existence", "conjunction"]
_TS = r"\d{2}_\d{2}_\d{4}_\d{2}_\d{2}_\d{2}"
# remainder after eval_<arm>_to_probe_<rule>_pl : _<shots>shot_lp[<cap>]_<TS>
_REMAINDER = re.compile(r"^_(?P<shots>\d+)shot_lp\d*_(?P<ts>" + _TS + r")$")


def _read_test_auc(run_dir: Path) -> float | None:
    best_step, best = -1, None
    for path in (run_dir / "data").glob("metrics_test*.json"):
        m = re.search(r"_step(\d+)\.json$", path.name)
        step = int(m.group(1)) if m else 0
        if step < best_step:
            continue
        try:
            payload = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        best_step, best = step, payload
    if best is None:
        return None
    for key in ("test_roc_auc", "roc_auc", "test_accuracy", "accuracy"):
        if key in best and isinstance(best[key], (int, float)):
            return float(best[key])
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log-root", required=True)
    ap.add_argument("--model-list", required=True)
    ap.add_argument("--out", default="scripts/experiments/analysis/objectives/topology_vs_features/topology_feature_ssl/data/capability_probes.csv")
    args = ap.parse_args()

    arms = [ln.split()[0] for ln in Path(args.model_list).read_text().splitlines()
            if ln.strip() and not ln.startswith("#")]
    log_root = Path(args.log_root)

    rows = []
    for arm in arms:
        for rule in RULES:
            prefix = f"eval_{arm}_to_probe_{rule}_pl"
            best_ts, auc = "", None
            for run_dir in log_root.glob(prefix + "*"):
                if not run_dir.is_dir():
                    continue
                m = _REMAINDER.match(run_dir.name[len(prefix):])
                if not m:
                    continue
                if m.group("ts") >= best_ts:  # latest run wins
                    val = _read_test_auc(run_dir)
                    if val is not None:
                        best_ts, auc = m.group("ts"), val
            if auc is not None:
                rows.append({"arm": arm, "rule": rule, "roc_auc": auc})

    if not rows:
        raise SystemExit("[T3] no probe runs found. Build graphs (make_probe_graphs.py) "
                         "and run run_capability_probes.sh first.")
    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[T3] wrote {out} ({len(df)} rows)")

    t3 = df.pivot_table(index="arm", columns="rule", values="roc_auc").reindex(columns=RULES)
    print("\nT3 — capability-probe linear-probe AUC (chance = 0.50)\n")
    print(t3.to_string(float_format=lambda x: f"{x:.2f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
