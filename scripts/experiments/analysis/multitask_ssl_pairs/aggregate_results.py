#!/usr/bin/env python3
"""Aggregate the multitask_ssl_pairs eval sweep into the subset-lattice table.

Merges the 3 PAIR arms (NMCL, NMFP, CLFP) with the single-objective controls
(NM, CL, FP) and the 3-way rotation (MIX) into ONE table over the full non-empty
subset lattice of {nm, cl, fp}:  3 singles -> 3 pairs -> 1 triple = 7 arms, all at
matched 40k-episode pretraining compute.

Reads the three shared benchmark CSVs (keyed by ``model`` = arm) produced by
``run_eval_sweep.sh`` -> ``parse_benchmark_eval_logs.py``:

    <plotting-root>/node_classification/data/node_classification.csv        (roc_auc)
    <plotting-root>/node_regression/data/node_regression.csv                (spearman)
    <plotting-root>/static_link_prediction/data/static_link_prediction.csv  (roc_auc)

To populate all 7 rows in one place, run the eval sweep with a model_list that
covers every arm (make_model_list.sh ARMS="NMCL NMFP CLFP NM CL FP MIX"). Arms
with no row in a CSV are printed as n/a.

The reading extends the rotation's headline: the 3-way MIX was the ONLY arm to
show emergent static-LP (topological transfer none of nm/cl/fp achieve alone).
The pairs answer WHICH combination is responsible -- e.g. does any 2-objective
rotation already unlock LP, or is all three necessary; is one objective (nm? fp?)
the driver.

Single seed: "spread" is dispersion ACROSS eval datasets, not across seeds.
Stdlib only (no pandas). Test split only.

Usage:
    python scripts/experiments/analysis/multitask_ssl_pairs/aggregate_results.py \
        --plotting-root scripts/experiments/analysis
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

SINGLES = ["NM", "CL", "FP"]
PAIRS = ["NMCL", "NMFP", "CLFP"]
TRIPLE = ["MIX"]
ARMS = SINGLES + PAIRS + TRIPLE
# which base objectives each arm rotates over (for the "which objective drives it" read)
CONTAINS = {
    "NM": {"nm"}, "CL": {"cl"}, "FP": {"fp"},
    "NMCL": {"nm", "cl"}, "NMFP": {"nm", "fp"}, "CLFP": {"cl", "fp"},
    "MIX": {"nm", "cl", "fp"},
}
K_OF = {a: len(CONTAINS[a]) for a in ARMS}   # number of objectives in the rotation
GENERALIST_BAR = 0.6   # min(cls, slp) above this = a genuine dual-capability generalist
CHANCE_AUC = 0.53      # <= this on an AUC axis = at chance


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


def _group_mean(m: dict[str, float | None], arms: list[str]) -> float | None:
    xs = [m[a] for a in arms if m.get(a) is not None]
    return _mean(xs)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plotting-root", default="scripts/experiments/analysis")
    args = ap.parse_args()
    root = Path(args.plotting_root)

    cls = _read(root / "node_classification/data/node_classification.csv", "roc_auc")
    reg = _read(root / "node_regression/data/node_regression.csv", "spearman")
    slp = _read(root / "static_link_prediction/data/static_link_prediction.csv", "roc_auc")

    cls_m = {a: _mean(v) for a, v in _by_arm(cls, "roc_auc").items()}
    reg_m = {a: _mean(v) for a, v in _by_arm(reg, "spearman").items()}
    slp_m = {a: _mean(v) for a, v in _by_arm(slp, "roc_auc").items()}
    mins = {a: (min(cls_m[a], slp_m[a]) if cls_m[a] is not None and slp_m[a] is not None else None)
            for a in ARMS}

    n_cls = len({r["dataset"] for r in cls})
    n_reg = len({r["dataset"] for r in reg})
    n_slp = len({r["dataset"] for r in slp})

    # ---- T1: full lattice table ----------------------------------------------
    print("=" * 74)
    print("T1 - frozen-encoder transfer over the {nm,cl,fp} subset lattice (test split)")
    print(f"    cls: {n_cls} datasets | reg: {n_reg} datasets x 3 targets | sLP: {n_slp} datasets")
    print("    all arms = merged 3-way corpus, bio-768/mean-SAGE, matched 40k episodes")
    print("=" * 74)
    print(f"{'arm':<6} {'k':>2} {'cls AUC':>9} {'reg rho':>9} {'sLP AUC':>9} {'min(cls,sLP)':>13}   group")
    for grp, arms in (("single", SINGLES), ("pair", PAIRS), ("triple", TRIPLE)):
        for a in arms:
            print(f"{a:<6} {K_OF[a]:>2} {_fmt(cls_m[a]):>9} {_fmt(reg_m[a]):>9} "
                  f"{_fmt(slp_m[a]):>9} {_fmt(mins[a]):>13}   {grp}")

    # ---- does capability scale with # objectives? ----------------------------
    print("\n" + "-" * 74)
    print("Capability vs. # objectives in the rotation (mean over arms at each k)")
    print("-" * 74)
    print(f"{'k':>2} {'#arms':>6} {'cls AUC':>9} {'reg rho':>9} {'sLP AUC':>9} {'min bar':>9}")
    for k, arms in ((1, SINGLES), (2, PAIRS), (3, TRIPLE)):
        print(f"{k:>2} {len(arms):>6} {_fmt(_group_mean(cls_m, arms)):>9} "
              f"{_fmt(_group_mean(reg_m, arms)):>9} {_fmt(_group_mean(slp_m, arms)):>9} "
              f"{_fmt(_group_mean(mins, arms)):>9}")

    # ---- joint generalist bar: min(feature=cls, topological=sLP) -------------
    print("\n" + "-" * 74)
    print("Joint generalist bar  min(feature=cls AUC, topological=sLP AUC), ranked")
    print(f"  (chance=0.50 both axes; > {GENERALIST_BAR} = genuine dual-capability generalist;")
    print("   regression excluded - noisy secondary feature axis, see rotation FINDINGS)")
    print("-" * 74)
    ranked = sorted((a for a in ARMS if mins[a] is not None), key=lambda a: mins[a], reverse=True)
    for a in ranked:
        weak = "cls" if cls_m[a] <= slp_m[a] else "sLP"
        flag = "  <-- generalist" if mins[a] > GENERALIST_BAR else (
            "  (at chance)" if mins[a] <= CHANCE_AUC else "")
        print(f"  {a:<6} (k={K_OF[a]})  min = {mins[a]:.3f}  (bottleneck: {weak}){flag}")
    best_single = max((a for a in SINGLES if mins[a] is not None), key=lambda a: mins[a], default=None)
    best_pair = max((a for a in PAIRS if mins[a] is not None), key=lambda a: mins[a], default=None)
    if best_single and best_pair and mins.get("MIX") is not None:
        print(f"\n  best single = {best_single} ({mins[best_single]:.3f}) | "
              f"best pair = {best_pair} ({mins[best_pair]:.3f}) | MIX = {mins['MIX']:.3f}")
        print(f"  best-pair - best-single = {mins[best_pair] - mins[best_single]:+.3f}   "
              f"MIX - best-pair = {mins['MIX'] - mins[best_pair]:+.3f}")

    # ---- HEADLINE: emergent static-LP, and which objective drives it ---------
    print("\n" + "=" * 74)
    print("HEADLINE - static-LP ROC-AUC (the emergent topological transfer)")
    print("=" * 74)
    for grp, arms in (("single", SINGLES), ("pair", PAIRS), ("triple", TRIPLE)):
        cells = "  ".join(f"{a}={_fmt(slp_m[a])}" for a in arms)
        print(f"  {grp:<7} {cells}")
    clears = [a for a in ARMS if slp_m[a] is not None and slp_m[a] > GENERALIST_BAR]
    print(f"\n  arms clearing sLP > {GENERALIST_BAR}: {clears or 'NONE'}")
    # which base objective correlates with LP: mean sLP with vs without each objective
    print("  marginal sLP by objective (mean over arms that DO vs DO NOT rotate it):")
    for obj in ("nm", "cl", "fp"):
        withx = [slp_m[a] for a in ARMS if obj in CONTAINS[a] and slp_m[a] is not None]
        without = [slp_m[a] for a in ARMS if obj not in CONTAINS[a] and slp_m[a] is not None]
        dw = (_mean(withx) - _mean(without)) if withx and without else None
        print(f"    {obj}: with={_fmt(_mean(withx))} without={_fmt(_mean(without))} "
              f"delta={('%+.3f' % dw) if dw is not None else ' n/a'}")

    # ---- per-dataset static-LP for all 7 arms --------------------------------
    print("\n" + "-" * 74)
    print("static-LP ROC-AUC per dataset (0-shot), all arms")
    print("-" * 74)
    ds_order = sorted({r["dataset"] for r in slp})
    slp_ds = {a: {r["dataset"]: r["roc_auc"] for r in slp if r["model"] == a} for a in ARMS}
    print(f"{'dataset':<16}" + "".join(f"{a:>7}" for a in ARMS))
    for ds in ds_order:
        print(f"{ds:<16}" + "".join(f"{_fmt(slp_ds[a].get(ds)):>7}" for a in ARMS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
