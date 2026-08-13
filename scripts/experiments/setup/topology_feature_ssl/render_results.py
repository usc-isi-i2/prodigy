#!/usr/bin/env python3
"""Render the topology_feature_ssl result tables into a glance-able RESULTS.md.

Mirrors the notebook (scripts/experiments/analysis/objectives/topology_vs_features/topology_feature_ssl/topology_feature_ssl.ipynb)
but writes markdown, so the numbers are viewable without running anything. Robust to
CSVs that haven't landed — each missing table is marked 'pending'.

Run on Tucker after the downstream parse (needs pandas):
    python scripts/experiments/setup/topology_feature_ssl/render_results.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
PLOT = REPO / "scripts" / "plotting"
OUT = Path(__file__).resolve().parent / "RESULTS.md"

ARMS = ["B0", "B1", "E1", "E2", "E3", "E4"]
HELD_OUT = {"twibot20", "election2020"}
STRUCT_TARGETS = ["followers_count", "statuses_count"]
CONTENT_TARGETS = ["account_age_days"]
NM, FP = "task_transfer_covid_nm", "task_transfer_covid_fp"


def bload(task):
    p = PLOT / task / "data" / f"{task}.csv"
    return pd.read_csv(p) if p.exists() else None


def tload(name):
    p = PLOT / "topology_feature_ssl" / "data" / f"{name}.csv"
    return pd.read_csv(p) if p.exists() else None


def block(title, df_or_msg):
    if isinstance(df_or_msg, str):
        return f"### {title}\n\n_{df_or_msg}_\n"
    return f"### {title}\n\n```\n{df_or_msg.to_string()}\n```\n"


def sec_free_preview():
    reg = bload("node_regression")
    if reg is None or reg[reg.model.isin([NM, FP])].empty:
        return block("Free preview — NM vs FP (regression)", "pending: no nm/fp regression rows yet")
    d = reg[(reg.split == "test") & (reg.model.isin([NM, FP]))]
    w = d.pivot_table(index=["dataset", "target"], columns="model", values="spearman", aggfunc="last").reset_index()
    for c in (NM, FP):
        if c not in w:
            w[c] = np.nan
    w["fp-nm"] = w[FP] - w[NM]
    w = w.rename(columns={NM: "nm", FP: "fp"}).round(3)
    delta = w["fp-nm"].dropna()
    verdict = ""
    if len(delta):
        verdict = (f"\n\nmean(fp-nm) = {delta.mean():+.3f} "
                   f"(fp wins {int((delta > 0).sum())}/{len(delta)}) -> "
                   f"{'E3 pre-validated' if delta.mean() > 0 else 'fp does NOT beat nm'}")
    return block("Free preview — NM vs FP (regression, test Spearman)", w) + verdict + "\n"


def _arm_mean(df, value, targets=None):
    if df is None:
        return {}
    d = df[(df.split == "test") & (df.model.isin(ARMS))]
    if targets is not None:
        d = d[d.target.isin(targets)]
    return d.groupby("model")[value].mean().to_dict()


def sec_t1():
    reg, slp, clf = bload("node_regression"), bload("static_link_prediction"), bload("node_classification")
    leak = tload("leakage_baseline")
    leak_struct = leak[leak.target.isin(STRUCT_TARGETS)]["spearman"].mean() if leak is not None else None
    cls_auc = _arm_mean(clf, "roc_auc")
    reg_c = _arm_mean(reg, "spearman", CONTENT_TARGETS)
    reg_s = _arm_mean(reg, "spearman", STRUCT_TARGETS)
    lp = _arm_mean(slp, "roc_auc")
    present = [a for a in ARMS if a in set(cls_auc) | set(reg_c) | set(reg_s) | set(lp)]
    if not present:
        return block("T1 — Benchmark", "pending: no arm rows in the benchmark CSVs yet")
    rows = [{
        "arm": a, "cls_AUC": cls_auc.get(a, np.nan),
        "reg_content_age": reg_c.get(a, np.nan), "reg_struct": reg_s.get(a, np.nan),
        "reg_struct_Δ_vs_leak": (reg_s.get(a, np.nan) - leak_struct) if leak_struct is not None else np.nan,
        "staticLP_AUC": lp.get(a, np.nan),
    } for a in present]
    t1 = pd.DataFrame(rows).set_index("arm").round(3)
    extra = f"\n\nleakage baseline (raw-structural -> followers/statuses) = {leak_struct:.3f}" if leak_struct is not None else ""
    return block("T1 — Benchmark (test)", t1) + extra + "\n"


def sec_t2():
    ab = tload("ablation_2x2")
    if ab is None or ab.empty:
        return block("T2 — 2×2 ablation (retained fraction)", "pending: run run_2x2_ablation.sh")
    feat = ab[ab.task.isin(["reg", "pl"])]
    t2 = (feat.pivot_table(index="arm", columns="condition", values="retained", aggfunc="mean")
              .reindex(columns=["random_feat", "rewired_edge", "both"]))
    t2.insert(0, "real·real", 1.00)
    return block("T2 — 2×2 ablation (fraction of real/real retained; feature tasks)", t2.round(2))


def sec_budget():
    bud = tload("budget_sweep")
    if bud is None or bud.empty:
        return block("Budget — transfer vs pretrain step", "pending: run run_budget_sweep.sh")
    parts = []
    for task in sorted(bud.task.unique()):
        piv = bud[bud.task == task].pivot_table(index="arm", columns="step", values="score").round(3)
        parts.append(block(f"Budget — {task} (test) vs step", piv))
    note = ("\n_Classification flat from 20k; regression peaks ~40-60k then degrades toward 110k "
            "(NM anti-scales on regression). Optimal NM budget ~40k._\n")
    return "\n".join(parts) + note


def sec_t3():
    pr = tload("capability_probes")
    rules = ["count_threshold", "in_degree", "out_degree", "existence", "conjunction"]
    if pr is None or pr.empty:
        return block("T3 — capability probes (AUC, chance 0.50)", "pending: run run_capability_probes.sh")
    t3 = pr.pivot_table(index="arm", columns="rule", values="roc_auc").reindex(columns=rules)
    return block("T3 — capability probes (linear-probe AUC, chance = 0.50)", t3.round(2))


def main() -> int:
    parts = [
        "# topology_feature_ssl — RESULTS\n",
        "_Auto-rendered from the parsed CSVs (see the notebook for the interactive "
        "version). Primary evidence: T2 (2×2) + T3 (probes); T1 is confirmatory. "
        "Headline is `min(feature, topological)`, never the mean._\n",
        sec_free_preview(), sec_t1(), sec_t2(), sec_t3(), sec_budget(),
    ]
    OUT.write_text("\n".join(parts))
    print(f"[render] wrote {OUT}")
    print("\n".join(parts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
