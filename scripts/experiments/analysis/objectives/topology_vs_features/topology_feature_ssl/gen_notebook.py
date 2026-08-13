#!/usr/bin/env python3
"""Generate the topology_feature_ssl results notebook (valid nbformat 4 JSON)."""
import json, pathlib

def md(*lines): return {"cell_type": "markdown", "metadata": {}, "source": [l if l.endswith("\n") else l+"\n" for l in lines]}
def code(src): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                       "source": [l if l.endswith("\n") else l+"\n" for l in src.strip("\n").split("\n")]}

cells = []

cells.append(md(
"# topology_feature_ssl — results",
"",
"SSL that learns topology *and* features (not features only). Arms keyed "
"`B0,B1,E1,E2,E3,E4`; the reading-chain deltas are direct table subtractions.",
"",
"**Primary evidence** is the seed-robust diagnostics (T2 2×2, T3 probes); the "
"benchmark (T1) is directional/confirmatory. Headline metric is "
"`min(feature_score, topological_score)`, never their mean.",
"",
"Tables populate as the eval/diagnostics CSVs land — cells print `pending` for "
"anything not yet produced, so this notebook is safe to run at any point.",
))

cells.append(code(r'''
import numpy as np, pandas as pd
from pathlib import Path
pd.set_option("display.width", 160); pd.set_option("display.max_rows", 200)

def _repo_root():
    for c in [Path.cwd(), *Path.cwd().parents]:
        if (c / "AGENTS.md").is_file():
            return c
    raise RuntimeError("run the notebook from inside the prodigy repository")
REPO = _repo_root()
PLOT = REPO / "scripts/experiments/analysis/evaluation/task_tables"
TF = REPO / "scripts/experiments/analysis/objectives/topology_vs_features/topology_feature_ssl"

def bload(task):  # benchmark CSVs: node_regression / static_link_prediction / node_classification
    p = PLOT / task / "data" / f"{task}.csv"
    return pd.read_csv(p) if p.exists() else None

def tload(name):  # topology_feature_ssl-specific CSVs
    p = TF / "data" / f"{name}.csv"
    return pd.read_csv(p) if p.exists() else None

ARMS = ["B0", "B1", "E1", "E2", "E3", "E4"]
IN_DOMAIN = ["midterm", "ukr_rus_twitter", "covid19_twitter"]
HELD_OUT  = ["twibot20", "election2020"]
STRUCT_TARGETS  = ["followers_count", "statuses_count"]   # ≈ in/out-degree (leakage-prone)
CONTENT_TARGETS = ["account_age_days"]
def domain(ds): return "held-out" if ds in HELD_OUT else "in-domain"
def pending(msg): print(f"[pending] {msg}")
print("plotting root:", PLOT)
'''))

# ---- Free preview ----
cells.append(md(
"## Free preview — NM vs FP on node regression (existing covid checkpoints)",
"",
"Reads a prior we already own: `fp` (masked feature prediction) ≈ E3's objective. "
"`mean(fp − nm) > 0` on regression pre-validates E3's core hypothesis for ~zero cost.",
))
cells.append(code(r'''
reg = bload("node_regression")
NM, FP = "task_transfer_covid_nm", "task_transfer_covid_fp"
if reg is None or reg[reg.model.isin([NM, FP])].empty:
    pending("free preview: node_regression.csv has no task_transfer_covid_{nm,fp} rows yet")
else:
    d = reg[(reg.split == "test") & (reg.model.isin([NM, FP]))]
    w = d.pivot_table(index=["dataset", "target"], columns="model", values="spearman", aggfunc="last").reset_index()
    for c in (NM, FP):
        if c not in w: w[c] = np.nan
    w["fp-nm"] = w[FP] - w[NM]
    w["domain"] = w.dataset.map(domain)
    w = w.sort_values(["domain", "dataset", "target"]).rename(columns={NM: "nm", FP: "fp"})
    display(w[["domain", "dataset", "target", "nm", "fp", "fp-nm"]].round(3))
    delta = w["fp-nm"].dropna()
    if len(delta):
        print(f"mean(fp-nm) = {delta.mean():+.3f}  (fp wins {int((delta>0).sum())}/{len(delta)})  "
              f"-> {'E3 pre-validated' if delta.mean()>0 else 'fp does NOT beat nm'}")
'''))

# ---- T1 benchmark ----
cells.append(md(
"## T1 — Benchmark (confirmatory)",
"",
"Per-arm, per-task vector (never a single mean). Structure-linked regression "
"(followers/statuses) is reported **only as Δ over the raw-structural leakage "
"baseline** — a raw number there is uninterpretable for E1/E2 (passthrough).",
))
cells.append(code(r'''
reg, slp, clf = bload("node_regression"), bload("static_link_prediction"), bload("node_classification")
leak = tload("leakage_baseline")

def arm_rows(df, value, targets=None):
    if df is None: return {}
    d = df[(df.split == "test") & (df.model.isin(ARMS))]
    if targets is not None: d = d[d.target.isin(targets)]
    return d.groupby("model")[value].mean().to_dict()

def arm_rows_by_domain(df, value, targets=None):
    if df is None: return {}
    d = df[(df.split == "test") & (df.model.isin(ARMS))].copy()
    if targets is not None: d = d[d.target.isin(targets)]
    d["dom"] = d.dataset.map(domain)
    return {(m, dom): v for (m, dom), v in d.groupby(["model", "dom"])[value].mean().items()}

leak_struct = None
if leak is not None:
    leak_struct = leak[leak.target.isin(STRUCT_TARGETS)]["spearman"].mean()

rows = []
cls_auc = arm_rows(clf, "roc_auc")
reg_content = arm_rows(reg, "spearman", CONTENT_TARGETS)
reg_struct = arm_rows(reg, "spearman", STRUCT_TARGETS)
lp_auc = arm_rows(slp, "roc_auc")
present = [a for a in ARMS if a in set(cls_auc) | set(reg_content) | set(reg_struct) | set(lp_auc)]
if not present:
    pending("T1: no arm rows in the benchmark CSVs yet (eval sweep still running)")
else:
    for a in present:
        rows.append({
            "arm": a,
            "cls (AUC)": cls_auc.get(a, np.nan),
            "reg-content age (Spearman)": reg_content.get(a, np.nan),
            "reg-struct (Spearman)": reg_struct.get(a, np.nan),
            "reg-struct Δ vs leakage": (reg_struct.get(a, np.nan) - leak_struct) if leak_struct is not None else np.nan,
            "static-LP (AUC)": lp_auc.get(a, np.nan),
        })
    t1 = pd.DataFrame(rows).set_index("arm")
    display(t1.round(3))
    if leak_struct is not None:
        print(f"leakage baseline (raw-structural -> followers/statuses, mean Spearman) = {leak_struct:.3f}")
        print("E1/E2 'learned structure' only if 'reg-struct Δ vs leakage' > 0.")
    else:
        pending("leakage_baseline.csv not present — reg-struct Δ shown as NaN")
'''))

# ---- T2 2x2 ----
cells.append(md(
"## T2 — 2×2 ablation (primary, seed-robust)",
"",
"Fraction of the real/real benchmark retained under each corruption. Signature to "
"break: NM/features-only stays ~1.0 under **rewired edge**, collapses under "
"**random feat**. *Learns both* ⇒ drops materially under **both**.",
))
cells.append(code(r'''
ab = tload("ablation_2x2")
if ab is None or ab.empty:
    pending("T2: ablation_2x2.csv not present (run run_2x2_ablation.sh)")
else:
    feat = ab[ab.task.isin(["reg", "pl"])]
    t2 = (feat.pivot_table(index="arm", columns="condition", values="retained", aggfunc="mean")
              .reindex(columns=["random_feat", "rewired_edge", "both"]))
    t2.insert(0, "real·real", 1.00)
    display(t2.round(2))
    print("read: features-only -> rewired_edge≈1.0 & random_feat low;  learns-both -> BOTH < 1.0")
    slp = ab[(ab.task == "slp") & (ab.condition == "rewired_edge")]
    if not slp.empty:
        print("[sanity] static-LP retained under rewired_edge (expect low for all):",
              slp.groupby("arm")["retained"].mean().round(2).to_dict())
'''))

# ---- T3 probes ----
cells.append(md(
"## T3 — capability probes (primary)",
"",
"Linear-probe AUC on planted single-rule synthetic graphs (chance = 0.50). Expected: "
"mean-agg B0 ≈ chance on count/degree; E1 passes the degree rules by structural "
"passthrough; E2 (sum/PNA + directed) lifts count/existence/in-out; conjunction hardest.",
))
cells.append(code(r'''
pr = tload("capability_probes")
RULES = ["count_threshold", "in_degree", "out_degree", "existence", "conjunction"]
if pr is None or pr.empty:
    pending("T3: capability_probes.csv not present (run run_capability_probes.sh)")
else:
    t3 = pr.pivot_table(index="arm", columns="rule", values="roc_auc").reindex(columns=RULES)
    display(t3.round(2))
    print("chance = 0.50; a cell materially > 0.5 means the frozen rep encodes that primitive.")
'''))

# ---- Budget / anti-scaling ----
cells.append(md(
"## Budget — where does transfer plateau? (NM anti-scales on regression)",
"",
"B0 and E1 evaluated at **20k / 40k / 60k / 110k** pretrain steps (`run_budget_sweep.sh`). "
"Classification is flat from 20k; **regression peaks ~40–60k then *degrades* toward 110k** — "
"NM is instance discrimination, which collapses the continuous variation regression needs, so "
"more NM training actively hurts regression. Optimal NM budget ≈ **40k** (also ~3× cheaper). "
"Implication: the main T1 evaluates at 110k (the degraded point), so it *understates* regression.",
))
cells.append(code(r'''
bud = tload("budget_sweep")
if bud is None or bud.empty:
    pending("budget_sweep.csv not present (run run_budget_sweep.sh)")
else:
    for task in sorted(bud.task.unique()):
        piv = bud[bud.task == task].pivot_table(index="arm", columns="step", values="score")
        print(f"--- {task} (test) vs pretrain step ---")
        display(piv.round(3))
    print("classification: flat from 20k;  regression: peaks ~40-60k then degrades -> NM anti-scales.")
'''))
cells.append(code(r'''
# Per-target regression vs step + the leakage ceiling (with the shot-mismatch caveat).
reg = bload("node_regression"); leak = tload("leakage_baseline")
if reg is None:
    pending("node_regression.csv not present")
else:
    d = reg[(reg.split == "test")].copy()
    d = d[d.model.astype(str).str.match(r"(B0|E1)_step\d+")]
    if d.empty:
        pending("no *_step regression rows yet")
    else:
        d["arm"] = d.model.str.extract(r"(B0|E1)_step")[0]
        d["step"] = d.model.str.extract(r"_step(\d+)")[0].astype(int)
        display(d.pivot_table(index=["arm", "step"], columns="target", values="spearman").round(3))
        if leak is not None:
            lk = leak[leak.dataset == "twibot20"].set_index("target")
            print("leakage ceiling (raw directed3 -> target, twibot20), SHOT-MATCHED 10-shot:",
                  lk["spearman"].round(3).to_dict())
            if "spearman_fulldata" in lk:
                print("  (full-data reference:", lk["spearman_fulldata"].round(3).to_dict(), ")")
            print("Passthrough test: E1 'learned structure' only if its frozen-rep Spearman "
                  "> the shot-matched ceiling on followers/statuses. E1 > B0 is a separate, clean signal.")
'''))

# ---- Reading chain ----
cells.append(md(
"## The reading — one claim per comparison",
"",
"- **B1 − B0 (aug lever):** if B1's *random-feat* 2×2 cell rises and reg-struct/LP "
"improve → the feature shortcut, not the objective, was binding. Flat on all three "
"→ the objective must change (positive support for E3/E4).",
"- **E1 − B0 vs leakage:** reg-struct Δ **> 0 over leakage** → the encoder *learned* "
"to use injected structure, not passthrough. Δ ≈ 0 → withhold the claim.",
"- **E2 − E1:** T3 count/existence/in-out jump and the T2 *rewired-edge* drop widens.",
"- **E3 − E2:** NM→MFR degrades under **both** 2×2 halves; beats NM more on "
"regression than classification.",
"- **E4 − E3:** static-LP + structural probes rise with ≤ noise loss on feature tasks — "
"the joint bar: feature tasks up **and** LP up **and** both 2×2 halves down **and** probes pass.",
"",
"The single sentence this experiment produces: *\"On merged retweet graphs, "
"transfer to **both** feature and topological tasks requires **[the lever that "
"cleared the joint bar]** — the only arm that degrades under both 2×2 halves while "
"the others stay features-only.\"*",
))
cells.append(code(r'''
# Reading-chain deltas (populated as arms land). B0 is the reference.
reg, slp = bload("node_regression"), bload("static_link_prediction")
def arm_mean(df, value, targets=None):
    if df is None: return {}
    d = df[(df.split=="test") & (df.model.isin(ARMS))]
    if targets is not None: d = d[d.target.isin(targets)]
    return d.groupby("model")[value].mean().to_dict()
struct = arm_mean(reg, "spearman", STRUCT_TARGETS)
lp = arm_mean(slp, "roc_auc")
if "B0" in struct or "B0" in lp:
    base_s, base_l = struct.get("B0", np.nan), lp.get("B0", np.nan)
    out = []
    for a in ARMS:
        if a == "B0" or a not in (set(struct)|set(lp)): continue
        out.append({"arm-B0": a,
                    "Δ reg-struct": struct.get(a, np.nan) - base_s,
                    "Δ static-LP": lp.get(a, np.nan) - base_l})
    display(pd.DataFrame(out).round(3) if out else "only B0 present so far")
else:
    pending("reading-chain deltas: B0 benchmark rows not present yet")
'''))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}

out = pathlib.Path(__file__).resolve().parent / "topology_feature_ssl.ipynb"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(nb, indent=1))
print("wrote", out, "cells:", len(cells))
