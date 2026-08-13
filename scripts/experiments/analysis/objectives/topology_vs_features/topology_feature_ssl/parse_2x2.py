#!/usr/bin/env python3
"""Join intact vs. ablated eval runs into the 2x2 retained-fraction table (T2).

Reads the eval run dirs produced by the shared runner (eval_<model>_to_<dataset>_
<task>[_abl<X>]_<shots>shot[_<target>]_<TS>/) for the topology_feature_ssl arms and
computes, per (arm, dataset, task, target), the fraction of the intact (real feat ·
real edge) test metric retained under each corruption:

  random_feat  (_ablN)  = --ablate-features noise
  rewired_edge (_ablE)  = --ablate-edges rewire
  both         (_ablNE) = both

Primary metric per task: reg -> spearman, slp -> roc_auc, pl -> roc_auc|accuracy.

"Learns both" ⇒ the FEATURE tasks (reg, pl) drop materially under BOTH halves.
NM's signature (to break): survives rewired_edge (~1.0), collapses under random_feat.

Usage:
    python parse_2x2.py --log-root /dataMeR1/phil/gfm/prodigy/log \
        --model-list scripts/experiments/topology_feature_ssl/model_list.txt \
        --out scripts/experiments/analysis/objectives/topology_vs_features/topology_feature_ssl/data/ablation_2x2.csv
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

DATASETS = ["midterm", "ukr_rus_twitter", "covid19_twitter", "twibot20", "election2020",
            "cp_hk_twitter", "merged_ukr_rus_covid", "merged_covid_midterm"]
TASK_METRIC = {"reg": "spearman", "slp": "roc_auc", "pl": "roc_auc"}
PL_FALLBACK = "accuracy"
ABL_CONDITION = {"": "intact", "N": "random_feat", "E": "rewired_edge", "NE": "both"}
FEATURE_TASKS = ("reg", "pl")  # the discriminating half of the 2x2
_TS = r"\d{2}_\d{2}_\d{4}_\d{2}_\d{2}_\d{2}"
_REMAINDER = re.compile(
    r"^(?:_abl(?P<abl>[ZPNE]+))?_(?P<shots>\d+)shot(?:_(?P<target>.+?))?_(?P<ts>" + _TS + r")$"
)


def _read_test_metric(run_dir: Path, metric: str, fallback: str | None = None) -> float | None:
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
    for key in (f"test_{metric}", metric, f"test_{fallback}" if fallback else None, fallback):
        if key and key in best and isinstance(best[key], (int, float)):
            return float(best[key])
    return None


def _parse_runs(log_root: Path, arms: list[str]) -> pd.DataFrame:
    rows = []
    for arm in arms:
        for dataset in DATASETS:
            for task, metric in TASK_METRIC.items():
                prefix = f"eval_{arm}_to_{dataset}_{task}"
                for run_dir in log_root.glob(prefix + "*"):
                    if not run_dir.is_dir():
                        continue
                    m = _REMAINDER.match(run_dir.name[len(prefix):])
                    if not m:
                        continue
                    abl = m.group("abl") or ""
                    if "Z" in abl or "P" in abl:
                        continue  # zero/permute feature ablations aren't part of this 2x2
                    condition = ABL_CONDITION.get(abl)
                    if condition is None:
                        continue
                    val = _read_test_metric(run_dir, metric,
                                            PL_FALLBACK if task == "pl" else None)
                    if val is None:
                        continue
                    rows.append({
                        "arm": arm, "dataset": dataset, "task": task,
                        "target": m.group("target") or "",
                        "condition": condition, "ts": m.group("ts"),
                        "metric": val,
                    })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # latest run wins per (arm, dataset, task, target, condition)
    df = (df.sort_values("ts")
            .drop_duplicates(["arm", "dataset", "task", "target", "condition"], keep="last")
            .drop(columns="ts"))
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log-root", required=True)
    ap.add_argument("--model-list", required=True,
                    help="Arm-keyed model list; arm names (col 1) label the CSV rows.")
    ap.add_argument("--out", default="scripts/experiments/analysis/objectives/topology_vs_features/topology_feature_ssl/data/ablation_2x2.csv")
    args = ap.parse_args()

    arms = [ln.split()[0] for ln in Path(args.model_list).read_text().splitlines()
            if ln.strip() and not ln.startswith("#")]
    df = _parse_runs(Path(args.log_root), arms)
    if df.empty:
        raise SystemExit("[2x2] no eval runs found. Run run_eval_sweep.sh (intact) "
                         "and run_2x2_ablation.sh (corrupted) first.")

    # retained fraction vs. the intact reference of the same (arm,dataset,task,target)
    intact = (df[df.condition == "intact"]
              .rename(columns={"metric": "intact"})
              .drop(columns="condition"))
    merged = df[df.condition != "intact"].merge(
        intact, on=["arm", "dataset", "task", "target"], how="left")
    merged["retained"] = merged.apply(
        lambda r: r["metric"] / r["intact"]
        if pd.notna(r["intact"]) and abs(r["intact"]) > 1e-6 else pd.NA, axis=1)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.sort_values(["arm", "task", "dataset", "target", "condition"]).to_csv(out, index=False)
    print(f"[2x2] wrote {out} ({len(merged)} rows)")

    # ---- T2 headline: mean retained over FEATURE tasks (reg+pl), rows=arm ----
    feat = merged[merged.task.isin(FEATURE_TASKS)].copy()
    if not feat.empty:
        t2 = (feat.pivot_table(index="arm", columns="condition", values="retained",
                               aggfunc="mean")
                  .reindex(columns=["random_feat", "rewired_edge", "both"]))
        t2.insert(0, "real·real", 1.00)
        print("\nT2 — fraction of real/real benchmark retained (feature tasks reg+pl)\n")
        print(t2.to_string(float_format=lambda x: f"{x:.2f}"))
        print("\n  read: NM/features-only ⇒ rewired_edge ≈ 1.0, random_feat ≈ low.")
        print("        learns-both ⇒ BOTH random_feat and rewired_edge materially < 1.0.")
    # slp sanity (LP is topology; every arm should collapse under rewired_edge)
    slp = merged[(merged.task == "slp") & (merged.condition == "rewired_edge")]
    if not slp.empty:
        print("\n  [sanity] static-LP retained under rewired_edge (expect low for all): "
              f"{slp.groupby('arm')['retained'].mean().round(2).to_dict()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
