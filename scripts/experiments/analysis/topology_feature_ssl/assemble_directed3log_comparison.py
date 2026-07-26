#!/usr/bin/env python3
"""Assemble the directed3_log (input-scaling-fix) vs original comparison for
topology_feature_ssl, into RESULTS_directed3log.md.

Baselines are the CLEAN matched-40k rows (model=<arm>_40k), which reproduce the
published FINDINGS exactly (the bare model=<arm> rows in the CSVs are a contaminated
aggregate over unrelated evals and must NOT be used). _log arms were evaluated fresh.
Pretext val_roc_auc @40k comes from wandb. Run after D3LOG_EVAL_DONE."""
from __future__ import annotations
import os
import sys

try:
    import pandas as pd
except Exception as e:  # pragma: no cover
    print(f"pandas unavailable: {e}", file=sys.stderr)
    sys.exit(1)

PLOT = "scripts/experiments/analysis"
OUT = "scripts/experiments/analysis/topology_feature_ssl/RESULTS_directed3log.md"
REG = f"{PLOT}/node_regression/data/node_regression.csv"
CLS = f"{PLOT}/node_classification/data/node_classification.csv"
SLP = f"{PLOT}/static_link_prediction/data/static_link_prediction.csv"
PROBE_LOG = f"{PLOT}/topology_feature_ssl/data/capability_probes_directed3log.csv"
PROBE_ORIG = f"{PLOT}/topology_feature_ssl/data/capability_probes_40k.csv"

ARMS = ["E1", "E2", "E2b", "E4", "E4r"]
LOG_RUNIDS = {"E1_log": "5fioegpt", "E2_log": "rhvnfqh5", "E2b_log": "erux4ah7",
              "E4_log": "g3oioeez", "E4r_log": "rqmbtbgm"}
# (display, pretext_key, csv_model, probe_arm, kind)
ROWS = [("B0", "B0", "B0_40k", "B0_40k", "orig")]
for a in ARMS:
    ROWS.append((a, a, f"{a}_40k", f"{a}_40k", "orig"))
    ROWS.append((f"{a}_log", f"{a}_log", f"{a}_log", f"{a}_log", "log"))

REG_TARGETS = ["followers_count", "friends_count", "statuses_count",
               "favourites_count", "listed_count", "account_age_days"]
REG_CORE = ["followers_count", "statuses_count", "account_age_days"]  # present for all arms
PROBE_RULES = ["count_threshold", "in_degree", "out_degree", "existence", "conjunction"]


def load(f):
    if not os.path.exists(f):
        return pd.DataFrame()
    df = pd.read_csv(f)
    if "split" in df.columns:
        df = df[df.split == "test"].copy()
    return df


def mean_of(df, model, col, targets=None, **filt):
    if df.empty or "model" not in df.columns:
        return None
    m = df[df.model == model]
    for k, v in filt.items():
        m = m[m[k] == v]
    if targets is not None:
        m = m[m.target.isin(targets)]
    s = pd.to_numeric(m[col], errors="coerce").dropna()
    return float(s.mean()) if len(s) else None


def fmt(x, d=3):
    return f"{x:.{d}f}" if isinstance(x, (int, float)) and x == x else "—"


def delta(a, b):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and a == a and b == b:
        return f"{b - a:+.3f}"
    return "—"


def pretext_from_wandb():
    pre = {}
    try:
        import wandb
        api = wandb.Api()
        proj = "eibl-usc/graph-clip"

        def hist(rid):
            d = {}
            try:
                for r in api.run(f"{proj}/{rid}").history(keys=["val_roc_auc"], pandas=False):
                    s, v = r.get("_step"), r.get("val_roc_auc")
                    if s is not None and v is not None:
                        d[int(s)] = v
            except Exception:
                pass
            return d

        def pick(tag, exclude=None):
            best, n = None, -1
            for r in api.runs(proj, filters={"tags": tag}):
                if exclude and exclude in (r.name or ""):
                    continue
                h = hist(r.id)
                if len(h) > n:
                    best, n = r.id, len(h)
            return best

        def at(h, s=40000):
            if not h:
                return None
            if s in h:
                return h[s]
            near = [k for k in h if abs(k - s) <= 1500]
            return h[min(near, key=lambda k: abs(k - s))] if near else None

        pre["B0"] = at(hist(pick("B0")))
        for a in ARMS:
            pre[a] = at(hist(pick(a, exclude=f"{a}_log")))
            pre[f"{a}_log"] = at(hist(LOG_RUNIDS[f"{a}_log"]))
    except Exception as e:  # pragma: no cover
        print(f"[assemble] wandb pretext skipped: {e}", file=sys.stderr)
    return pre


def main():
    reg, cls, slp = load(REG), load(CLS), load(SLP)
    pre = pretext_from_wandb()
    plog = pd.read_csv(PROBE_LOG) if os.path.exists(PROBE_LOG) else pd.DataFrame()
    porig = pd.read_csv(PROBE_ORIG) if os.path.exists(PROBE_ORIG) else pd.DataFrame()
    out = []
    out.append("# directed3_log (input-scaling fix) vs original — topology_feature_ssl\n")
    out.append("`directed3_log` = log1p the raw in/out degree counts before z-scoring (fixes the "
               "~1322σ heavy-tailed input that suppressed E1). Each `_log` arm changed ONLY that "
               "vs its original. **Baselines are the clean matched-40k rows (`*_40k`), which "
               "reproduce the published FINDINGS exactly.** `_log` arms evaluated fresh with the "
               "identical harness. Single seed. Downstream = mean over eval datasets (test split); "
               "reg headline mean is over the 3 targets present for all arms "
               "(followers/statuses/account_age).\n")

    # ---- headline ----
    out.append("## Headline — pretext + downstream\n")
    out.append("| arm | pretext val_auc | reg ρ (core3) | reg account_age | cls AUC | static-LP AUC | min(cls,slp) |")
    out.append("|---|---|---|---|---|---|---|")

    def hrow(disp, pre_key, csv_model):
        p = pre.get(pre_key)
        rm = mean_of(reg, csv_model, "spearman", targets=REG_CORE)
        ra = mean_of(reg, csv_model, "spearman", target="account_age_days")
        cm = mean_of(cls, csv_model, "roc_auc")
        sm = mean_of(slp, csv_model, "roc_auc")
        j = min(cm, sm) if (isinstance(cm, float) and isinstance(sm, float)) else None
        return f"| {disp} | {fmt(p)} | {fmt(rm)} | {fmt(ra)} | {fmt(cm)} | {fmt(sm)} | {fmt(j)} |"

    for disp, pk, cm, _pa, _k in ROWS:
        out.append(hrow(disp, pk, cm))
    out.append("")
    out.append("**Δ (\\_log − original\\_40k):**\n")
    out.append("| arm | Δ pretext | Δ reg core3 | Δ reg account_age | Δ cls | Δ static-LP |")
    out.append("|---|---|---|---|---|---|")
    for a in ARMS:
        o, l = f"{a}_40k", f"{a}_log"
        out.append(f"| {a}→{a}_log | {delta(pre.get(a), pre.get(f'{a}_log'))} "
                   f"| {delta(mean_of(reg,o,'spearman',targets=REG_CORE), mean_of(reg,l,'spearman',targets=REG_CORE))} "
                   f"| {delta(mean_of(reg,o,'spearman',target='account_age_days'), mean_of(reg,l,'spearman',target='account_age_days'))} "
                   f"| {delta(mean_of(cls,o,'roc_auc'), mean_of(cls,l,'roc_auc'))} "
                   f"| {delta(mean_of(slp,o,'roc_auc'), mean_of(slp,l,'roc_auc'))} |")
    out.append("")

    # ---- regression per target ----
    out.append("## Regression — Spearman per target (mean over datasets)\n")
    out.append("| arm | " + " | ".join(t.replace("_count", "").replace("_days", "") for t in REG_TARGETS) + " |")
    out.append("|" + "---|" * (len(REG_TARGETS) + 1))
    for disp, _pk, cm, _pa, _k in ROWS:
        cells = [fmt(mean_of(reg, cm, "spearman", target=t)) for t in REG_TARGETS]
        out.append(f"| {disp} | " + " | ".join(cells) + " |")
    out.append("")

    # ---- static-LP per dataset ----
    if not slp.empty:
        out.append("## Static link prediction — ROC-AUC per dataset\n")
        dsets = sorted(slp[slp.model.isin([r[2] for r in ROWS])].dataset.dropna().unique())
        out.append("| arm | " + " | ".join(dsets) + " | mean |")
        out.append("|" + "---|" * (len(dsets) + 2))
        for disp, _pk, cm, _pa, _k in ROWS:
            cells = [fmt(mean_of(slp, cm, "roc_auc", dataset=d)) for d in dsets]
            out.append(f"| {disp} | " + " | ".join(cells) + f" | {fmt(mean_of(slp, cm, 'roc_auc'))} |")
        out.append("")

    # ---- capability probes ----
    if not plog.empty or not porig.empty:
        out.append("## Capability probes — linear-probe AUC (chance 0.50)\n")
        out.append("| arm | " + " | ".join(r.replace("_threshold", "_thr").replace("_degree", "_deg") for r in PROBE_RULES) + " |")
        out.append("|" + "---|" * (len(PROBE_RULES) + 1))

        def probe_val(df, arm, rule):
            if df.empty:
                return None
            m = df[(df.arm == arm) & (df.rule == rule)]
            return float(m.roc_auc.iloc[0]) if len(m) else None

        for disp, _pk, _cm, pa, kind in ROWS:
            src = plog if kind == "log" else porig
            cells = [fmt(probe_val(src, pa, r)) for r in PROBE_RULES]
            out.append(f"| {disp} | " + " | ".join(cells) + " |")
        out.append("")

    # ---- row-count sanity ----
    out.append("## Row-count sanity (test rows found per _log arm)\n")
    out.append("| arm | reg | cls | slp |")
    out.append("|---|---|---|---|")
    for a in ARMS:
        l = f"{a}_log"
        rc = 0 if reg.empty else int((reg.model == l).sum())
        cc = 0 if cls.empty else int((cls.model == l).sum())
        sc = 0 if slp.empty else int((slp.model == l).sum())
        out.append(f"| {l} | {rc} | {cc} | {sc} |")
    out.append("")

    text = "\n".join(out)
    with open(OUT, "w") as f:
        f.write(text)
    print(text)
    print(f"\n[assemble] wrote {OUT}")


if __name__ == "__main__":
    main()
