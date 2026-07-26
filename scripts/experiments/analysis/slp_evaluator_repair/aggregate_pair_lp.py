"""Aggregate the pair-conditioned LP rescore into the headline table.

Reads the per-dataset CSVs written by scripts/eval/pair_link_sweep.py and reports,
per arm: mean AUC across datasets, and the margin over the best topology heuristic
on the same pair set. The margin is the number that matters -- an encoder that only
matches common-neighbour has learned nothing a $0 CPU baseline does not already do.

    python aggregate_pair_lp.py --results-dir ../multitask_ssl/data/pair_lp [--negative-kind degree_matched]
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
from collections import defaultdict
from typing import Dict, List

HEURISTIC_FLOORS = ("common_neighbors", "adamic_adar", "preferential_attachment", "jaccard")

# Published numbers from the OLD (invalid) episodic evaluator, for the contrast.
# Source: scripts/experiments/analysis/multitask_ssl_rotation/FINDINGS.md
OLD_EVAL = {"mtr_NM": 0.467, "mtr_CL": 0.332, "mtr_FP": 0.449, "mtr_MIX": 0.759}


def load(results_dir: str) -> List[dict]:
    rows: List[dict] = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*__pair_lp.csv"))):
        with open(path) as fh:
            rows.extend(csv.DictReader(fh))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="../multitask_ssl/data/pair_lp")
    ap.add_argument("--negative-kind", default="degree_matched")
    args = ap.parse_args()

    rows = [r for r in load(args.results_dir) if r["negative_kind"] == args.negative_kind]
    if not rows:
        print(f"no rows for negative_kind={args.negative_kind}")
        return 1

    datasets = sorted({r["dataset"] for r in rows})

    # best heuristic floor per dataset
    floor: Dict[str, float] = {}
    for ds in datasets:
        vals = [float(r["auc"]) for r in rows
                if r["dataset"] == ds and r["model"] == "__floor__"
                and r["scorer"] in HEURISTIC_FLOORS]
        floor[ds] = max(vals) if vals else float("nan")

    per_model: Dict[str, Dict[str, float]] = defaultdict(dict)
    for r in rows:
        if r["model"] == "__floor__":
            continue
        per_model[r["model"]][r["dataset"]] = float(r["auc"])

    print(f"Pair-conditioned static LP -- negatives: {args.negative_kind}")
    print(f"Datasets ({len(datasets)}): {', '.join(datasets)}")
    print(f"Best heuristic floor per dataset: "
          f"{', '.join(f'{d}={floor[d]:.3f}' for d in datasets)}\n")

    header = f"{'arm':<16}" + "".join(f"{d[:11]:>12}" for d in datasets) + \
             f"{'mean':>9}{'vs floor':>10}{'old eval':>10}"
    print(header)
    print("-" * len(header))

    def sort_key(item):
        vals = [v for v in item[1].values()]
        return -(sum(vals) / len(vals)) if vals else 0.0

    for model, by_ds in sorted(per_model.items(), key=sort_key):
        vals = [by_ds.get(d) for d in datasets]
        present = [v for v in vals if v is not None]
        if not present:
            continue
        mean = sum(present) / len(present)
        margins = [by_ds[d] - floor[d] for d in datasets
                   if d in by_ds and floor[d] == floor[d]]
        margin = sum(margins) / len(margins) if margins else float("nan")
        old = OLD_EVAL.get(model)
        cells = "".join(f"{v:>12.3f}" if v is not None else f"{'-':>12}" for v in vals)
        print(f"{model:<16}{cells}{mean:>9.3f}{margin:>+10.3f}"
              f"{(f'{old:.3f}' if old else '-'):>10}")

    print("\nfloors (mean across datasets):")
    for h in HEURISTIC_FLOORS + ("raw_feature_cosine",):
        vals = [float(r["auc"]) for r in rows
                if r["model"] == "__floor__" and r["scorer"] == h]
        if vals:
            print(f"  {h:<26}{sum(vals)/len(vals):.3f}")

    print("\ngates:")
    # Two distinct conditions share this statistic and must not be conflated:
    #   sensitivity == 0      -> the scorer ignores an endpoint. Fatal; the row is
    #                            void (this is precisely the old evaluator's bug).
    #   0 < sensitivity < 1   -> the scorer is pair-conditioned, but the ENCODER
    #                            is directionally collapsed, so cosine ties. Not an
    #                            evaluator fault; a property of that encoder, and a
    #                            finding in its own right. Measured on midterm:
    #                            mtr_FP spans 536 distinct directions over 4568
    #                            nodes vs mtr_NM's 4429.
    fatal, collapsed = [], []
    for r in rows:
        if r["model"] == "__floor__":
            continue
        if r["leakage_edges"] and float(r["leakage_edges"]) > 0:
            fatal.append((r, "leakage"))
            continue
        s = r["endpoint_sensitivity"]
        if s == "":
            continue
        s = float(s)
        if s <= 1e-9:
            fatal.append((r, "endpoint-blind"))
        elif s < 0.99:
            collapsed.append((r, s))
    if fatal:
        for r, why in fatal[:10]:
            print(f"  VOID {r['dataset']}/{r['model']}: {why}")
    else:
        print("  no voided rows (no endpoint-blind scoring, no leakage)")
    if collapsed:
        by_model: Dict[str, List[float]] = defaultdict(list)
        for r, s in collapsed:
            by_model[r["model"]].append(s)
        print("  encoder-collapse flags (pair-conditioned, but embeddings tie):")
        for m, ss in sorted(by_model.items()):
            print(f"    {m:<18} sensitivity {min(ss):.2f}-{max(ss):.2f} "
                  f"across {len(ss)} dataset(s)")
    perms = [float(r["endpoint_permutation_auc"]) for r in rows
             if r["model"] != "__floor__" and r["endpoint_permutation_auc"]]
    if perms:
        print(f"  endpoint-permutation AUC: mean={sum(perms)/len(perms):.3f} "
              f"min={min(perms):.3f} max={max(perms):.3f} (must sit near 0.5)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
