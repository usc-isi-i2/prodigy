#!/usr/bin/env python3
"""Downstream ladder — is the regression channel measuring anything at all?

An entry-aligned Δ on a metric that never clears its own floor is not evidence about
pre-training; it is drift in a quantity with no signal. The static-LP panel already
carries this check ("best rung beats the best heuristic floor by +0.046 … +0.185"), and
this is its regression analogue: for each (dataset, target), how the 21 encoders compare
against a ridge probe fitted on the raw 768-d input features over the SAME episodes.

Reads ``nm_ladder_downstream_long.csv`` (encoder cells) and
``nm_ladder_downstream_reg_floors.csv`` (the ``__features_only__`` rows), both written by
``assemble_downstream_tables.py``. Prints a markdown table; writes nothing.

Read the "clears" column first. If encoders sit at or below the floor everywhere, the
regression panel is a non-measurement and no Δ computed on it should be quoted --
which is exactly what the void episodic path did before the probe replaced it.
"""
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def main():
    long = pd.read_csv(os.path.join(DATA, "nm_ladder_downstream_long.csv"))
    floors = pd.read_csv(os.path.join(DATA, "nm_ladder_downstream_reg_floors.csv"))
    reg = long[(long["task"] == "reg") & (long["primary"] == 1)]

    # 24 rows collapse to 21 distinct encoders; dedupe so a reused checkpoint is not
    # counted two or three times in "n clearing the floor".
    reg = reg.drop_duplicates(subset=["model_key", "dataset", "target"])

    fl = {(r.dataset, r.target): r.features_only_spearman
          for r in floors.itertuples()}

    print("| dataset | target | floor (raw x) | best rung | median rung | "
          "clears floor | best − floor |")
    print("|---|---|---|---:|---:|---:|---:|")
    rows = []
    for (ds, tgt), g in reg.groupby(["dataset", "target"]):
        floor = fl.get((ds, tgt))
        if floor is None:
            continue
        best, med = g["value"].max(), g["value"].median()
        n_clear, n = int((g["value"] > floor).sum()), len(g)
        rows.append((ds, tgt, floor, best, med, n_clear, n, best - floor))
        print(f"| {ds} | {tgt} | {floor:+.4f} | {best:+.4f} | {med:+.4f} | "
              f"{n_clear}/{n} | {best - floor:+.4f} |")

    if rows:
        cleared = sum(1 for r in rows if r[5] > 0)
        any_above = sum(1 for r in rows if r[7] > 0)
        print(f"\n{any_above}/{len(rows)} cells have a best rung above the raw-feature "
              f"floor; {cleared}/{len(rows)} have at least one rung above it.")
        print("A cell where no rung clears the floor carries no encoder signal, and an "
              "entry Δ measured on it is not interpretable.")


if __name__ == "__main__":
    main()
