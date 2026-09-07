#!/usr/bin/env python3
"""Generate ten exploratory cross-experiment figures from committed evidence.

Figures 1, 2, 3, 7, 8, and 9 are empirical. Figure 10 is an explicitly labelled
diagnostic proxy. Figures 4--6 are data-readiness panels, not scientific results.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FIG = HERE / "figures"
DATA = HERE / "data"
LADDER = ROOT / "transfer/ladders/prodigy_nm/baseline/nm_ladder/data"
ORDER = ROOT / "transfer/ladders/prodigy_nm/robustness/nm_ladder_order_robustness/data"
SIM = ROOT / "graphs/transfer_prediction/similarity_vs_transfer_v2/data"
FINAL = ROOT / "transfer/matrices/cross_model/final_core/data"
PATH_FEATURE = ROOT / "graphs/structure_features/path_feature_coupling/data"

GRAPH_LABEL = {
    "ukr_rus_twitter": "Ukraine", "covid19_twitter": "COVID-19",
    "midterm": "Midterm", "covid_political": "COVID-political",
    "election2020": "Election 2020", "ukr_rus_suspended": "Ukraine suspended",
    "twibot20": "TwiBot-20", "cp_hk_twitter": "Hong Kong",
    "facebook_page_reference": "Facebook Pages",
}
PALETTE = ["#2676b8", "#e07a2d", "#42a36b", "#a05eb5", "#d34e4e", "#7b6d62", "#d69bc4", "#6879c9"]


def save(fig: plt.Figure, stem: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def ladder() -> tuple[pd.DataFrame, list[str]]:
    d = pd.read_csv(LADDER / "nm_ladder_full.csv")
    graphs = [c for c in d.columns if c in GRAPH_LABEL]
    return d, graphs


def plot_01() -> None:
    d, graphs = ladder()
    rows = []
    for i in range(1, len(d)):
        added = d.loc[i, "added"]
        old = list(d.loc[:i - 1, "added"])
        held = [g for g in graphs if g not in set(d.loc[:i, "added"])]
        rows.append({"added": added, "acquisition": d.loc[i, added] - d.loc[i - 1, added],
                     "retention": (d.loc[i, old] - d.loc[i - 1, old]).mean(),
                     "ood": (d.loc[i, held] - d.loc[i - 1, held]).mean() if held else np.nan})
    out = pd.DataFrame(rows); DATA.mkdir(parents=True, exist_ok=True)
    out.to_csv(DATA / "forgetting_acquisition.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    sizes = 100 + 1300 * out.ood.abs().fillna(0)
    ax.scatter(out.acquisition, out.retention, s=sizes, c=PALETTE[:len(out)], edgecolor="white", linewidth=1)
    for _, r in out.iterrows(): ax.annotate(GRAPH_LABEL[r.added], (r.acquisition, r.retention), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.axhline(0, c="0.6", lw=1); ax.axvline(0, c="0.6", lw=1)
    ax.set(xlabel="Acquisition on newly added graph (Δ AUC)", ylabel="Retention on earlier graphs (mean Δ AUC)", title="Acquisition versus forgetting at each ladder entry")
    ax.text(.02, .02, "Area ∝ |mean change on still-held-out graphs|", transform=ax.transAxes, fontsize=8, color="0.35")
    save(fig, "01_forgetting_acquisition")


def plot_02() -> None:
    d = pd.read_csv(LADDER / "nm_ladder_plus_single_source.csv")
    graphs = [c for c in d.columns if c in GRAPH_LABEL]
    pts = []
    for _, r in d.iterrows():
        native = r.diag_or_added
        home = r[native]
        breadth = r[[g for g in graphs if g != native]].mean()
        pts.append((r.block, r.row, home, breadth))
    p = pd.DataFrame(pts, columns=["type", "model", "home", "foreign"])
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    for kind, marker, color in [("single_source", "o", "#777777"), ("ladder", "s", "#2676b8")]:
        q = p[p.type == kind]; ax.scatter(q.home, q.foreign, marker=marker, c=color, s=55, label=kind.replace("_", " "))
    # Empirical upper envelope in home-performance order.
    q = p.sort_values("home"); frontier = q[q.foreign.eq(q.foreign[::-1].cummax()[::-1])]
    ax.step(frontier.home, frontier.foreign, where="post", c="#d34e4e", lw=1.5, label="non-dominated frontier")
    ax.set(xlabel="Native / entry-graph AUC", ylabel="Mean foreign-graph AUC", title="Specialization–breadth Pareto map")
    ax.legend(frameon=False); save(fig, "02_specialist_generalist_pareto")


def plot_03() -> None:
    d = pd.read_csv(SIM / "extended/model_predictions_logo.csv")
    # One predeclared operational model: pair+source, ridge when available.
    candidates = [("source_plus_pair", "ridge"), ("handpicked", "ridge"), ("all", "ridge")]
    q = pd.DataFrame()
    for fs, alg in candidates:
        q = d[(d.feature_set == fs) & (d.algorithm == alg)]
        if len(q): break
    if q.empty: q = d.groupby(["held_graph", "source"], as_index=False).first()
    q = q.copy(); q["residual"] = q.truth - q.prediction
    mat = q.pivot_table(index="source", columns="held_graph", values="residual")
    fig, ax = plt.subplots(figsize=(8.2, 6.2)); vmax = np.nanmax(np.abs(mat.to_numpy()))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(mat.columns)), [GRAPH_LABEL.get(x, x) for x in mat.columns], rotation=40, ha="right")
    ax.set_yticks(range(len(mat.index)), [GRAPH_LABEL.get(x, x) for x in mat.index])
    ax.set(title=f"Transfer residuals: observed − graph-holdout prediction\n{fs} / {alg}", xlabel="Held-out target", ylabel="Source")
    fig.colorbar(im, ax=ax, label="AUC residual"); save(fig, "03_transfer_residual_heatmap")


def readiness(stem: str, title: str, required: list[str], available: list[str], next_step: str) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.8)); ax.axis("off")
    ax.set_title(title, fontsize=14, pad=18)
    y = .82
    for item in required:
        ok = item in available
        ax.text(.08, y, "✓" if ok else "○", fontsize=18, color="#27864b" if ok else "#c44e52", va="center")
        ax.text(.14, y, item, fontsize=11, va="center"); y -= .13
    ax.text(.08, .08, "Not a result panel — required evidence is not committed.\nNext: " + next_step,
            fontsize=9, color="0.35", va="bottom")
    save(fig, stem)


def plot_04_06() -> None:
    readiness("04_layerwise_representation_shift_readiness", "Layerwise representation shift — input readiness",
              ["raw feature graph-domain distances", "post-S embeddings", "post-U embeddings", "post-M embeddings"],
              ["raw feature graph-domain distances"], "export the same sampled node/episode IDs after S, U, and M from one fixed checkpoint.")
    readiness("05_node_rescue_map_readiness", "Node-level rescue map — input readiness",
              ["stable node IDs", "random-init probabilities", "pretrained probabilities", "labels and profile metadata"],
              [], "export paired prediction JSONL from random initialization and the selected pretrained checkpoint.")
    readiness("06_topology_feature_disagreement_readiness", "Topology–feature disagreement — input readiness",
              ["raw-feature prediction per node", "topology-only prediction per node", "full-model prediction per node", "stable node IDs and labels"],
              [], "run paired prediction export for raw-feature, ablated-feature, and full-model arms on identical nodes.")


def plot_07() -> None:
    d, graphs = ladder(); rows = []
    for i in range(1, len(d)):
        added = d.loc[i, "added"]; held = [g for g in graphs if g not in set(d.loc[:i, "added"])]
        rows.append((added, (d.loc[i, held] - d.loc[i-1, held]).mean() if held else np.nan,
                     (d.loc[i, graphs] - d.loc[i-1, graphs]).mean()))
    q = pd.DataFrame(rows, columns=["source", "held_out_gain", "all_graph_gain"])
    fig, ax = plt.subplots(figsize=(7.8, 4.7)); x = np.arange(len(q)); w=.36
    ax.bar(x-w/2, q.held_out_gain, w, label="still-held-out graphs", color="#2676b8")
    ax.bar(x+w/2, q.all_graph_gain, w, label="all graphs", color="#e07a2d")
    ax.axhline(0, c="0.5", lw=1); ax.set_xticks(x, [GRAPH_LABEL[s] for s in q.source], rotation=35, ha="right")
    ax.set(ylabel="Mean incremental AUC per added source", title="Marginal transfer value of each ladder entry")
    ax.legend(frameon=False); save(fig, "07_marginal_source_value")


def plot_08() -> None:
    p = pd.read_csv(FINAL / "prodigy_2500_cls_transfer_matrix_with_mixture.csv")
    p = p[p.model_source.str.startswith("ss_")].copy(); p["source"] = p.model_source.str.removeprefix("ss_")
    targets = [c for c in p.columns if c in GRAPH_LABEL]
    pl = p.melt(id_vars="source", value_vars=targets, var_name="target", value_name="prodigy")
    s = pd.read_csv(FINAL / "samgpt_downstream_cls/cells.csv")
    s = s[(s.n_sources == 1) & s.model_id.str.startswith("ss_")].copy(); s["source"] = s.model_id.str.removeprefix("ss_")
    sl = s.groupby(["source", "target"], as_index=False).roc_auc_mean.mean().rename(columns={"roc_auc_mean":"samgpt"})
    q = pl.merge(sl, on=["source", "target"])
    fig, ax = plt.subplots(figsize=(6.3, 5.7)); ax.scatter(q.prodigy, q.samgpt, c="#6b62a8", alpha=.72, s=35)
    lo=min(q.prodigy.min(),q.samgpt.min()); hi=max(q.prodigy.max(),q.samgpt.max()); ax.plot([lo,hi],[lo,hi], c="0.5", ls="--")
    r=np.corrcoef(q.prodigy,q.samgpt)[0,1]
    ax.set(xlabel="PRODIGY/NM transfer AUC", ylabel="SAMGPT/GraphCL transfer AUC", title=f"Cross-model pairwise transfer agreement (r = {r:.2f})")
    ax.text(.02,.02,f"n = {len(q)} matched source–target cells",transform=ax.transAxes,fontsize=9,color="0.35")
    save(fig, "08_cross_model_transfer_agreement")


def plot_09() -> None:
    seed = pd.read_csv(SIM / "final_core_auc/specialist_cells_summary.csv")
    seed_col = "roc_auc_ovr_macro_std"
    seed_sd = seed[seed_col].dropna().to_numpy()
    order = pd.read_csv(ORDER / "nm_ladder_order_robustness_long.csv")
    order_sd = order.groupby(["rung", "test_graph"]).auc.std().dropna().to_numpy()
    fig, ax = plt.subplots(figsize=(7,4.8)); ax.boxplot([seed_sd, order_sd], tick_labels=["training-seed spread\n(specialists)", "ladder-order spread\n(matched rung/target)"], showfliers=False)
    ax.set(ylabel="Cell-level AUC standard deviation", title="Observed robustness components (descriptive, not additive)")
    ax.text(.02,.02,"Evaluation-episode SD is excluded: it is not a training-robustness estimate.",transform=ax.transAxes,fontsize=8,color="0.35")
    save(fig, "09_uncertainty_components")


def plot_10() -> None:
    d = pd.read_csv(PATH_FEATURE / "graph_identity_per_dimension.csv")
    d = d[d.scope == "organic_twitter"].copy() if (d.scope == "organic_twitter").any() else d[d.scope == d.scope.iloc[0]].copy()
    d = d.nlargest(20, "test_mean_oriented_ovr_auc").sort_values("test_mean_oriented_ovr_auc")
    fig, ax = plt.subplots(figsize=(7.4,5.4)); ax.barh(d.dimension.astype(str), d.test_mean_oriented_ovr_auc, color="#42a36b")
    ax.axvline(.5,c="0.5",ls="--",lw=1); ax.set(xlabel="Mean one-vs-rest graph-identity AUC", ylabel="Embedding dimension", title="Dimensions prioritized for permutation intervention")
    ax.text(.02,.02,"Diagnostic ranking only; no causal permutation results are available.",transform=ax.transAxes,fontsize=8,color="0.35")
    save(fig, "10_dimension_intervention_priority_proxy")


def main() -> None:
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    DATA.mkdir(parents=True, exist_ok=True)
    plot_01(); plot_02(); plot_03(); plot_04_06(); plot_07(); plot_08(); plot_09(); plot_10()
    manifest = {"empirical": [1,2,3,7,8,9], "diagnostic_proxy": [10], "readiness_only": [4,5,6]}
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__": main()
