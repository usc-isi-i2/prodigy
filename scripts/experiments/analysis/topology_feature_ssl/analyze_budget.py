#!/usr/bin/env python3
"""Transfer-vs-pretrain-step tables from the budget sweep (run_budget_sweep.sh).

Reads the shared benchmark CSVs, filters to the budget variants (models named
`<arm>_step<N>`), and pivots the test metric by pretrain step per arm/task — so
you can see where the DOWNSTREAM metric plateaus (vs the NM training metric,
which saturates ~30-40k). Writes budget_sweep.csv.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
PLOT = REPO / "scripts" / "plotting"
STEP_RE = re.compile(r"^(?P<arm>B0|B1|E1)_step(?P<step>\d+)$")
TASKS = [("node_regression", "spearman"), ("node_classification", "roc_auc"),
         ("static_link_prediction", "roc_auc")]


def _load(task):
    p = PLOT / task / "data" / f"{task}.csv"
    return pd.read_csv(p) if p.exists() else None


def main() -> int:
    frames = []
    for task, metric in TASKS:
        df = _load(task)
        if df is None:
            continue
        d = df[df.split == "test"].copy()
        m = d.model.astype(str).str.extract(STEP_RE)
        d = d.assign(arm=m["arm"], step=m["step"])
        d = d[d.arm.notna()]
        if d.empty:
            continue
        d["step"] = d["step"].astype(int)
        g = d.groupby(["arm", "step"])[metric].mean().reset_index()
        g["task"] = task
        frames.append(g.rename(columns={metric: "score"}))

    if not frames:
        print("[budget] no *_step* budget rows yet — run run_budget_sweep.sh first")
        return 0
    allr = pd.concat(frames, ignore_index=True)
    out = PLOT / "topology_feature_ssl" / "data" / "budget_sweep.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    allr.to_csv(out, index=False)

    print("\nTransfer performance vs pretrain step (test) — where does it plateau?\n")
    for task in allr.task.unique():
        piv = allr[allr.task == task].pivot_table(index="arm", columns="step", values="score")
        print(f"--- {task} ---")
        print(piv.round(3).to_string())
        # Δ from the max step, per arm, to eyeball the plateau
        for arm, row in piv.iterrows():
            steps = sorted(row.dropna().index)
            if len(steps) >= 2:
                top = row[steps[-1]]
                deltas = {s: round(row[s] - top, 3) for s in steps}
                print(f"    {arm} Δ vs {steps[-1]}: {deltas}")
        print()
    print(f"[budget] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
