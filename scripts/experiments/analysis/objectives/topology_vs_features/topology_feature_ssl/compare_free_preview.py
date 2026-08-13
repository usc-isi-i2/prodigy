#!/usr/bin/env python3
"""Free-preview reading: NM vs (masked) feature-prediction on node regression.

Reads the tidy node_regression.csv produced by
scripts/harness/benchmark_tasks/parse_benchmark_eval_logs.py and prints a
per-(dataset, target) table of test Spearman for the two existing covid
task_transfer checkpoints, with the fp - nm delta.

fp approximates E3's masked-feature-reconstruction objective. A positive
mean(fp - nm) on regression pre-validates E3's core hypothesis (a generative
objective helps regression, where NM is weak) for ~zero cost.

Usage:
    python compare_free_preview.py --csv scripts/experiments/analysis/evaluation/task_tables/node_regression/data/node_regression.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

NM = "task_transfer_covid_nm"
FP = "task_transfer_covid_fp"
HELD_OUT = {"twibot20", "election2020"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default="scripts/experiments/analysis/evaluation/task_tables/node_regression/data/node_regression.csv")
    ap.add_argument("--split", default="test")
    ap.add_argument("--shots", type=int, default=10)
    ap.add_argument("--metric", default="spearman")
    args = ap.parse_args()

    path = Path(args.csv)
    if not path.exists():
        raise SystemExit(
            f"[free-preview] {path} not found. Run run_free_preview.sh first "
            "(it evals the checkpoints and parses the logs into this CSV)."
        )
    df = pd.read_csv(path)
    df = df[(df["split"] == args.split) & (df["shots"] == args.shots)]
    df = df[df["model"].isin([NM, FP])]
    if df.empty:
        raise SystemExit(
            f"[free-preview] no {args.split} rows at {args.shots}-shot for "
            f"{NM}/{FP} in {path}. Did the eval run complete?"
        )

    wide = (
        df.pivot_table(index=["dataset", "target"], columns="model",
                       values=args.metric, aggfunc="last")
        .reset_index()
    )
    for col in (NM, FP):
        if col not in wide.columns:
            wide[col] = pd.NA
    wide["delta_fp_minus_nm"] = wide[FP] - wide[NM]
    wide["domain"] = wide["dataset"].apply(lambda d: "held-out" if d in HELD_OUT else "in-domain")
    wide = wide.sort_values(["domain", "dataset", "target"])

    pd.set_option("display.width", 120)
    pd.set_option("display.max_rows", None)
    print(f"\nNM vs FP — node regression {args.metric} ({args.split}, {args.shots}-shot)\n")
    show = wide[["domain", "dataset", "target", NM, FP, "delta_fp_minus_nm"]].rename(
        columns={NM: "nm", FP: "fp", "delta_fp_minus_nm": "fp-nm"}
    )
    print(show.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    valid = wide["delta_fp_minus_nm"].dropna()
    if len(valid):
        print("\n--- summary (fp - nm Spearman) ---")
        for label, mask in (
            ("all", wide["delta_fp_minus_nm"].notna()),
            ("in-domain", wide["domain"] == "in-domain"),
            ("held-out", wide["domain"] == "held-out"),
        ):
            d = wide.loc[mask, "delta_fp_minus_nm"].dropna()
            if len(d):
                print(f"  {label:>10}: mean {d.mean():+.3f}  "
                      f"(fp wins {int((d > 0).sum())}/{len(d)})")
        verdict = "fp BEATS nm -> E3 pre-validated" if valid.mean() > 0 else "fp does NOT beat nm"
        print(f"\n  verdict: mean(fp-nm)={valid.mean():+.3f} -> {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
