#!/usr/bin/env python3
"""Matched-40k arm comparison vs the trivial floor: B0/B1/E1/E2 at the same (~30k)
budget, full 6-target profile panel + static-LP + cls + 2x2 + probes, anchored
against raw_feat (bio features, no encoder) and raw_degree (leakage). Writes a
glance-able RESULTS_matched40k.md + prints the tables. This is the read that can
finally say whether any arm beats "just use the features" / "just read degree".
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

REPO = next(p for p in Path(__file__).resolve().parents if (p / "AGENTS.md").is_file())
PLOT = REPO / "scripts/experiments/analysis/evaluation/task_tables"
OUT = Path(__file__).resolve().parent / "RESULTS_matched40k.md"
ARMS = ["B0_40k", "B1_40k", "E1_40k", "E2_40k", "E2b_40k", "E4_40k", "E4r_40k"]
REG6 = ["followers_count", "friends_count", "statuses_count",
        "favourites_count", "listed_count", "account_age_days"]


def bload(task):
    p = PLOT / task / "data" / f"{task}.csv"
    return pd.read_csv(p) if p.exists() else None


def tload(name):
    p = Path(__file__).resolve().parent / "data" / f"{name}.csv"
    return pd.read_csv(p) if p.exists() else None


def _fmt(df):
    return f"```\n{df.round(3).to_string()}\n```\n"


def main() -> int:
    parts = ["# topology_feature_ssl — matched-40k results (B0/B1/E1/E2/E2b vs trivial floor)\n",
             "_All arms at a matched 40k-episode budget (true state_dict_40000). Anchored against `raw_feat` (bio "
             "features, no encoder) and `raw_degree` (leakage), both 10-shot. An arm only "
             "'improves performance' if it beats these._\n"]
    reg, clf, slp = bload("node_regression"), bload("node_classification"), bload("static_link_prediction")
    tb = tload("trivial_baselines")          # raw_feat (reg + cls)
    leak = tload("leakage_baseline_6panel")  # raw_degree (reg)

    # ---- Regression: baselines + arms ----
    if reg is not None:
        r = reg[(reg.split == "test") & (reg.model.isin(ARMS))]
        cols = [t for t in REG6 if not r.empty and t in r.target.unique()]
        frames = []
        if leak is not None:
            frames.append(leak.groupby("target")["spearman"].mean().to_frame().T.assign(_m="raw_degree").set_index("_m"))
        if tb is not None:
            rf = tb[tb.task == "regression"].groupby("target")["metric"].mean().to_frame().T
            rf.index = ["raw_feat"]
            frames.append(rf)
        if not r.empty:
            frames.append(r.pivot_table(index="model", columns="target", values="spearman").reindex(ARMS))
        if frames:
            tab = pd.concat(frames)
            tab = tab[[c for c in cols if c in tab.columns]] if cols else tab
            parts.append("### Regression Spearman (test, mean over datasets) — full panel\n" + _fmt(tab))
            parts.append("_An arm 'learned structure' only if it beats raw_degree on followers/statuses; "
                         "'beats features' if it tops raw_feat._\n")

    # ---- Classification: baselines + arms ----
    if clf is not None:
        c = clf[(clf.split == "test") & (clf.model.isin(ARMS))]
        if not c.empty:
            tab = c.pivot_table(index="model", columns="dataset", values="roc_auc").reindex(ARMS)
            if tb is not None:
                rf = tb[tb.task == "classification"].pivot_table(index="model", columns="dataset", values="metric")
                tab = pd.concat([rf, tab])
            parts.append("### Classification ROC-AUC (test)\n" + _fmt(tab))

    # ---- Static-LP (arms only; no trivial degree-LP baseline here) ----
    if slp is not None:
        s = slp[(slp.split == "test") & (slp.model.isin(ARMS))]
        if not s.empty:
            parts.append("### Static-LP ROC-AUC (test)\n"
                         + _fmt(s.pivot_table(index="model", columns="dataset", values="roc_auc").reindex(ARMS)))

    # ---- T2 / T3 (40k) ----
    ab = tload("ablation_2x2_40k")
    if ab is not None and not ab.empty:
        feat = ab[ab.task.isin(["reg", "pl"])]
        parts.append("### T2 — 2x2 retained (feature tasks)\n" + _fmt(
            feat.pivot_table(index="arm", columns="condition", values="retained", aggfunc="mean").reindex(
                columns=["random_feat", "rewired_edge", "both"])))
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
