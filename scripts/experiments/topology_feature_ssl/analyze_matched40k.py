#!/usr/bin/env python3
"""Matched-40k arm comparison: B0/B1/E1/E2 at the same (~40k) budget, full 6-target
profile panel + static-LP + cls + 2x2 + probes + shot-matched leakage.

Reads the shared benchmark CSVs (filtered to the `<arm>_40k` models) plus the
_40k diagnostic CSVs, and writes a glance-able RESULTS_matched40k.md + prints the
tables. This is the fair E2-vs-{B0,B1,E1} read (matched budget; 110k is degraded).
"""
from __future__ import annotations
from pathlib import Path
import re
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
PLOT = REPO / "scripts" / "plotting"
OUT = Path(__file__).resolve().parent / "RESULTS_matched40k.md"
ARMS = ["B0_40k", "B1_40k", "E1_40k", "E2_40k"]
REG6 = ["followers_count", "friends_count", "statuses_count",
        "favourites_count", "listed_count", "account_age_days"]


def bload(task):
    p = PLOT / task / "data" / f"{task}.csv"
    return pd.read_csv(p) if p.exists() else None


def tload(name):
    p = PLOT / "topology_feature_ssl" / "data" / f"{name}.csv"
    return pd.read_csv(p) if p.exists() else None


def _fmt(df):
    return f"```\n{df.round(3).to_string()}\n```\n"


def main() -> int:
    parts = ["# topology_feature_ssl — matched-40k results (B0/B1/E1/E2)\n",
             "_All arms at the same ~40k-episode budget (regression peak). Full 6-target "
             "profile panel. E2 = multi-aggregation encoder + multi-readout._\n"]

    # --- T1: classification / regression(6) / static-LP ---
    clf, reg, slp = bload("node_classification"), bload("node_regression"), bload("static_link_prediction")
    leak = tload("leakage_baseline_6panel")
    if reg is not None:
        r = reg[(reg.split == "test") & (reg.model.isin(ARMS))]
        if not r.empty:
            piv = r.pivot_table(index="model", columns="target", values="spearman").reindex(ARMS)[
                [t for t in REG6 if t in r.target.unique()]]
            parts.append("### Regression Spearman (test, mean over datasets) — full panel\n" + _fmt(piv))
            if leak is not None:
                lk = leak.groupby("target")["spearman"].mean().reindex(piv.columns)
                parts.append("shot-matched leakage ceiling (raw directed3 -> target, mean over datasets):\n"
                             + _fmt(lk.to_frame("leakage").T))
    if clf is not None:
        c = clf[(clf.split == "test") & (clf.model.isin(ARMS))]
        if not c.empty:
            parts.append("### Classification ROC-AUC (test)\n"
                         + _fmt(c.pivot_table(index="model", columns="dataset", values="roc_auc").reindex(ARMS)))
    if slp is not None:
        s = slp[(slp.split == "test") & (slp.model.isin(ARMS))]
        if not s.empty:
            parts.append("### Static-LP ROC-AUC (test)\n"
                         + _fmt(s.pivot_table(index="model", columns="dataset", values="roc_auc").reindex(ARMS)))

    # --- T2 / T3 (40k) ---
    ab = tload("ablation_2x2_40k")
    if ab is not None and not ab.empty:
        feat = ab[ab.task.isin(["reg", "pl"])]
        t2 = feat.pivot_table(index="arm", columns="condition", values="retained", aggfunc="mean").reindex(
            columns=["random_feat", "rewired_edge", "both"])
        parts.append("### T2 — 2x2 retained (feature tasks)\n" + _fmt(t2))
    pr = tload("capability_probes_40k")
    if pr is not None and not pr.empty:
        rules = ["count_threshold", "in_degree", "out_degree", "existence", "conjunction"]
        parts.append("### T3 — capability probes (AUC, chance 0.50)\n"
                     + _fmt(pr.pivot_table(index="arm", columns="rule", values="roc_auc").reindex(columns=rules)))

    OUT.write_text("\n".join(parts))
    print(f"[matched40k] wrote {OUT}\n")
    print("\n".join(parts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
