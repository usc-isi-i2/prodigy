#!/usr/bin/env python3
"""Aggregate the multitask_ssl_rotation eval sweep into the T1 transfer table.

Reads the three shared benchmark CSVs (keyed by ``model`` = arm) produced by
``run_eval_sweep.sh`` -> ``parse_benchmark_eval_logs.py``:

    <plotting-root>/node_classification/data/node_classification.csv   (roc_auc)
    <plotting-root>/node_regression/data/node_regression.csv           (spearman)
    <plotting-root>/static_link_prediction/data/static_link_prediction.csv (roc_auc)

and prints the headline reading (README T1):

  * per-arm per-task vector (mean over datasets), for arms NM/CL/FP/MIX
  * MIX - max(NM,CL,FP) per task
  * the joint min(feature, topological) generalist bar, feature = classification
    AUC, topological = static-LP AUC (regression is reported as a secondary,
    noisy feature axis and is NOT folded into the min bar -- see FINDINGS.md)
  * the static-LP win broken out per dataset (the headline), with the
    across-dataset spread of MIX - max(control)

Single seed: "spread" is the dispersion ACROSS eval datasets, not across seeds.
Stdlib only (no pandas), so it runs anywhere. Test split only.

Usage:
    python scripts/experiments/multitask_ssl_rotation/aggregate_results.py \
        --plotting-root scripts/plotting
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

ARMS = ["NM", "CL", "FP", "MIX"]
CONTROLS = ["NM", "CL", "FP"]
CHANCE_AUC = 0.5


def _read(path: Path, metric: str, split: str = "test") -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open() as fh:
        for r in csv.DictReader(fh):
            if r.get("split") != split:
                continue
            val = r.get(metric, "")
            if val in ("", None):
                continue
            try:
                r[metric] = float(val)
            except ValueError:
                continue
            rows.append(r)
    return rows


def _by_arm(rows: list[dict], metric: str) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {a: [] for a in ARMS}
    for r in rows:
        m = r.get("model")
        if m in out:
            out[m].append(r[metric])
    return out


def _mean(xs: list[float]) -> float | None:
    return statistics.fmean(xs) if xs else None


def _fmt(x: float | None, nd: int = 3) -> str:
    return "  n/a" if x is None else f"{x:.{nd}f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plotting-root", default="scripts/plotting")
    args = ap.parse_args()
    root = Path(args.plotting_root)

    cls = _read(root / "node_classification/data/node_classification.csv", "roc_auc")
    reg = _read(root / "node_regression/data/node_regression.csv", "spearman")
    slp = _read(root / "static_link_prediction/data/static_link_prediction.csv", "roc_auc")

    cls_by = _by_arm(cls, "roc_auc")
    reg_by = _by_arm(reg, "spearman")
    slp_by = _by_arm(slp, "roc_auc")

    cls_m = {a: _mean(cls_by[a]) for a in ARMS}
    reg_m = {a: _mean(reg_by[a]) for a in ARMS}
    slp_m = {a: _mean(slp_by[a]) for a in ARMS}

    n_cls = len({r["dataset"] for r in cls})
    n_reg_ds = len({r["dataset"] for r in reg})
    n_slp = len({r["dataset"] for r in slp})

    print("=" * 68)
    print("T1 - frozen-encoder transfer (mean over datasets, test split)")
    print(f"    classification: {n_cls} datasets | regression: {n_reg_ds} datasets x 3 "
          f"targets | static-LP: {n_slp} datasets")
    print("=" * 68)
    print(f"{'arm':<5} {'cls AUC':>9} {'reg rho':>9} {'sLP AUC':>9}   note")
    for a in ARMS:
        note = "rotation (treatment)" if a == "MIX" else "single-objective control"
        print(f"{a:<5} {_fmt(cls_m[a]):>9} {_fmt(reg_m[a]):>9} {_fmt(slp_m[a]):>9}   {note}")

    # MIX - max(control) per task
    def delta(m: dict[str, float | None]) -> tuple[float, str]:
        best_ctrl = max(CONTROLS, key=lambda c: (m[c] if m[c] is not None else -9))
        return (m["MIX"] - m[best_ctrl]), best_ctrl

    print("\n" + "-" * 68)
    print("MIX - max(NM,CL,FP)  [+ => rotation wins the task]")
    print("-" * 68)
    for name, m in (("classification (AUC)", cls_m), ("regression (rho)", reg_m),
                    ("static-LP  (AUC)", slp_m)):
        d, ctrl = delta(m)
        print(f"  {name:<22} {d:+.3f}   (best control = {ctrl} @ {_fmt(m[ctrl])})")

    # joint generalist bar: min(feature=cls AUC, topological=sLP AUC)
    print("\n" + "-" * 68)
    print("Joint generalist bar  min(feature=cls AUC, topological=sLP AUC)")
    print("  (chance = 0.50 on both axes; regression excluded - see FINDINGS)")
    print("-" * 68)
    mins = {}
    for a in ARMS:
        mins[a] = min(cls_m[a], slp_m[a])
        weak = "cls" if cls_m[a] <= slp_m[a] else "sLP"
        flag = "  <-- generalist" if mins[a] > 0.6 else ("  (at chance)" if mins[a] <= 0.53 else "")
        print(f"  {a:<5} min = {mins[a]:.3f}  (bottleneck: {weak}){flag}")
    best_ctrl_min = max(CONTROLS, key=lambda c: mins[c])
    print(f"\n  MIX min - best-control min = {mins['MIX'] - mins[best_ctrl_min]:+.3f} "
          f"(best control on the joint bar = {best_ctrl_min} @ {mins[best_ctrl_min]:.3f})")

    # static-LP per dataset (the headline) + across-dataset spread of the win
    print("\n" + "=" * 68)
    print("HEADLINE - static-LP ROC-AUC per dataset (0-shot)")
    print("=" * 68)
    ds_order = sorted({r["dataset"] for r in slp})
    slp_ds = {a: {r["dataset"]: r["roc_auc"] for r in slp if r["model"] == a} for a in ARMS}
    hdr = f"{'dataset':<18}" + "".join(f"{a:>8}" for a in ARMS) + f"{'MIX-maxCtl':>12}"
    print(hdr)
    wins = []
    for ds in ds_order:
        vals = {a: slp_ds[a].get(ds) for a in ARMS}
        ctl = max((vals[c] for c in CONTROLS if vals[c] is not None), default=None)
        win = (vals["MIX"] - ctl) if (vals["MIX"] is not None and ctl is not None) else None
        if win is not None:
            wins.append(win)
        row = f"{ds:<18}" + "".join(f"{_fmt(vals[a]):>8}" for a in ARMS)
        row += f"{('%+.3f' % win) if win is not None else 'n/a':>12}"
        print(row)
    if wins:
        print("-" * 68)
        print(f"MIX - max(control) on static-LP: mean {statistics.fmean(wins):+.3f} | "
              f"range [{min(wins):+.3f}, {max(wins):+.3f}] over {len(wins)} datasets "
              f"(all {'positive' if min(wins) > 0 else 'MIXED'})")
        mix_lp = [slp_ds['MIX'][d] for d in ds_order if slp_ds['MIX'].get(d) is not None]
        sd = statistics.pstdev(mix_lp) if len(mix_lp) > 1 else 0.0
        print(f"MIX static-LP AUC: mean {statistics.fmean(mix_lp):.3f} +/- {sd:.3f} "
              f"(sd across datasets), range [{min(mix_lp):.3f}, {max(mix_lp):.3f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
