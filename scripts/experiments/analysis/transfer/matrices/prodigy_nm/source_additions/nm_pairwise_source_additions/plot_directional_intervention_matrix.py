from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import pandas as pd
import seaborn as sns


HERE = Path(__file__).resolve().parent
BASELINE = (
    HERE.parents[3]
    / "matrices/cross_model/final_core/data/prodigy_final_core/auc/summary/"
    "single_source_metrics_long.tsv"
)
PAIRS = HERE / "data/raw/pair_metrics_long.tsv"

SOURCES = [
    "ukr_rus",
    "covid",
    "midterm",
    "covid_political",
    "election2020",
    "ukr_rus_suspended",
    "twibot20",
    "cp_hk",
    "facebook_page_reference",
]
DISPLAY = {
    "ukr_rus": "Ukraine–Russia",
    "covid": "COVID",
    "midterm": "Midterm",
    "covid_political": "COVID Political",
    "election2020": "Election 2020",
    "ukr_rus_suspended": "Ukraine–Russia Susp.",
    "twibot20": "TwiBot-20",
    "cp_hk": "CP/HK",
    "facebook_page_reference": "Facebook Pages",
}


specialists = pd.read_csv(BASELINE, sep="\t")
specialists = specialists[specialists.seed == 0].copy()
specialists["sources"] = specialists["source"]
pairs = pd.read_csv(PAIRS, sep="\t")
pairs["sources"] = pairs["source_left"] + "," + pairs["source_right"]
if len(specialists) != 81 or len(pairs) != 324:
    raise ValueError(
        f"expected 81 specialist and 324 pair cells, got "
        f"{len(specialists)} and {len(pairs)}"
    )

baseline = specialists.set_index(["sources", "target"])["roc_auc_ovr_macro"]
baseline_means = specialists.groupby("sources")["roc_auc_ovr_macro"].mean()

records: list[dict[str, object]] = []
for pair_model, pair_group in pairs.groupby("model_id", sort=False):
    left, right = pair_group["sources"].iloc[0].split(",")
    pair_auc = pair_group.set_index("target")["roc_auc_ovr_macro"]
    for base_source, added_source in [(left, right), (right, left)]:
        for target in SOURCES:
            base_auc = baseline.loc[(base_source, target)]
            current_pair_auc = pair_auc.loc[target]
            records.append(
                {
                    "base_source": base_source,
                    "added_source": added_source,
                    "pair_model_id": pair_model,
                    "target": target,
                    "base_specialist_auc": base_auc,
                    "pair_auc": current_pair_auc,
                    "delta_auc": current_pair_auc - base_auc,
                    "target_is_base": int(target == base_source),
                    "target_is_added": int(target == added_source),
                    "target_heldout_from_pair": int(target not in {base_source, added_source}),
                }
            )

long = pd.DataFrame.from_records(records)
if len(long) != 648:
    raise ValueError(f"expected 648 directional intervention cells, got {len(long)}")
if long[["base_source", "added_source", "target"]].duplicated().any():
    raise ValueError("directional intervention keys must be unique")

summary = (
    long.groupby(["base_source", "added_source", "pair_model_id"], as_index=False)
    .apply(
        lambda group: pd.Series(
            {
                "base_all_target_mean_auc": group["base_specialist_auc"].mean(),
                "heldout_mean_delta_auc": group.loc[
                    group.target_heldout_from_pair == 1, "delta_auc"
                ].mean(),
                "all_target_mean_delta_auc": group["delta_auc"].mean(),
            }
        ),
        include_groups=False,
    )
    .reset_index()
)

# Stronger specialists first; within each block, best genuinely held-out intervention first.
base_order = (
    baseline_means.sort_values(ascending=False).index.tolist()
)
summary["base_order"] = summary["base_source"].map(
    {source: index for index, source in enumerate(base_order)}
)
summary = summary.sort_values(
    ["base_order", "heldout_mean_delta_auc", "added_source"],
    ascending=[True, False, True],
).reset_index(drop=True)
summary["row_order"] = np.arange(len(summary))

# Rank evaluation-target columns by the average effect of adding any dataset to
# the Ukraine–Russia specialist. Summary columns remain fixed at the right.
TARGET_ORDER = (
    long.loc[long.base_source == "ukr_rus"]
    .groupby("target")["delta_auc"]
    .mean()
    .sort_values(ascending=True)
    .index.tolist()
)

long = long.merge(
    summary[
        [
            "base_source",
            "added_source",
            "base_all_target_mean_auc",
            "heldout_mean_delta_auc",
            "all_target_mean_delta_auc",
            "row_order",
        ]
    ],
    on=["base_source", "added_source"],
    validate="many_to_one",
)
long["target_order"] = long["target"].map(
    {source: index for index, source in enumerate(TARGET_ORDER)}
)
long = long.sort_values(["row_order", "target_order"]).reset_index(drop=True)

long_output = HERE / "data/nm_directional_intervention_deltas_all_targets_long.csv"
long.drop(columns=["row_order", "target_order"]).to_csv(long_output, index=False)

keys = list(zip(summary.base_source, summary.added_source))
delta_matrix = (
    long.pivot(
        index=["base_source", "added_source"],
        columns="target",
        values="delta_auc",
    )
    .loc[keys, TARGET_ORDER]
    * 100
)

wide = summary[
    [
        "base_source",
        "added_source",
        "pair_model_id",
        "base_all_target_mean_auc",
        "heldout_mean_delta_auc",
        "all_target_mean_delta_auc",
    ]
].copy()
for column in [
    "base_all_target_mean_auc",
    "heldout_mean_delta_auc",
    "all_target_mean_delta_auc",
]:
    wide[column] *= 100
for target in TARGET_ORDER:
    wide[f"delta_to_{target}"] = delta_matrix[target].to_numpy()
wide_output = HERE / "data/nm_directional_intervention_deltas_ranked_wide.csv"
wide.to_csv(wide_output, index=False)

delta_plot = delta_matrix.copy()
delta_plot.columns = [DISPLAY[source] for source in TARGET_ORDER]
delta_plot["Held-out mean Δ"] = summary["heldout_mean_delta_auc"].to_numpy() * 100
delta_plot["All-target mean Δ"] = summary["all_target_mean_delta_auc"].to_numpy() * 100
row_labels = [f"+ {DISPLAY[added]}" for added in summary.added_source]
delta_plot.index = row_labels

base_plot = pd.DataFrame(
    {"Base mean": summary["base_all_target_mean_auc"].to_numpy() * 100},
    index=row_labels,
)

sns.set_theme(style="white", context="talk")
fig = plt.figure(figsize=(19.5, 32))
layout = gridspec.GridSpec(
    1, 2, width_ratios=[1.15, 11], wspace=0.035,
    left=0.19, right=0.94, bottom=0.065, top=0.925,
)
ax_base = fig.add_subplot(layout[0, 0])
ax_delta = fig.add_subplot(layout[0, 1], sharey=ax_base)

sns.heatmap(
    base_plot,
    cmap=sns.light_palette("#167a45", as_cmap=True),
    vmin=68,
    vmax=88,
    annot=True,
    fmt=".1f",
    annot_kws={"fontsize": 8.3, "fontweight": "semibold"},
    linewidths=0.8,
    linecolor="white",
    cbar=False,
    ax=ax_base,
)

delta_limit = max(12.0, float(np.ceil(np.nanmax(np.abs(delta_plot.to_numpy())))))
sns.heatmap(
    delta_plot,
    cmap=sns.diverging_palette(12, 135, s=82, l=44, center="light", as_cmap=True),
    center=0,
    vmin=-delta_limit,
    vmax=delta_limit,
    annot=True,
    fmt="+.1f",
    annot_kws={"fontsize": 7.5, "fontweight": "semibold"},
    linewidths=0.8,
    linecolor="white",
    cbar_kws={"label": "Change in macro ROC–AUC (percentage points)", "shrink": 0.48, "pad": 0.02},
    ax=ax_delta,
)

# Solid = the base specialist's own target; dashed = the newly added source's target.
for row_index, row in summary.iterrows():
    base_column = TARGET_ORDER.index(row.base_source)
    added_column = TARGET_ORDER.index(row.added_source)
    ax_delta.add_patch(
        plt.Rectangle(
            (base_column, row_index), 1, 1, fill=False,
            edgecolor="#222222", linewidth=1.7,
        )
    )
    ax_delta.add_patch(
        plt.Rectangle(
            (added_column, row_index), 1, 1, fill=False,
            edgecolor="#222222", linewidth=1.7, linestyle="--",
        )
    )

block_starts = []
cursor = 0
for base_source in base_order:
    block_starts.append((cursor, base_source))
    cursor += int((summary.base_source == base_source).sum())
for boundary, _ in block_starts[1:]:
    ax_base.axhline(boundary, color="#202020", linewidth=2.8)
    ax_delta.axhline(boundary, color="#202020", linewidth=2.8)

for start, base_source in block_starts:
    fig.text(
        0.012,
        0.925 - ((start + 4.0) / len(summary)) * 0.86,
        f"Base: {DISPLAY[base_source]}",
        ha="left",
        va="center",
        fontsize=11.5,
        fontweight="bold",
        rotation=90,
        color="#303030",
    )

ax_base.set_xlabel("")
ax_base.set_ylabel("Added training dataset", fontsize=15, labelpad=14)
ax_base.set_xticklabels(ax_base.get_xticklabels(), rotation=38, ha="right", fontsize=10)
ax_base.set_yticklabels(ax_base.get_yticklabels(), rotation=0, fontsize=8.7)
ax_delta.set_xlabel("Evaluation target and intervention summaries", fontsize=15, labelpad=14)
ax_delta.set_ylabel("")
ax_delta.tick_params(axis="y", left=False, labelleft=False)
ax_delta.set_xticklabels(
    ax_delta.get_xticklabels(), rotation=39, ha="right", rotation_mode="anchor", fontsize=10.2
)

fig.suptitle(
    "Effect of adding each dataset to every NM specialist",
    fontsize=25,
    fontweight="bold",
    y=0.977,
)
fig.text(
    0.565,
    0.947,
    "Each cell is pair AUC minus the corresponding base-specialist AUC; "
    "target columns rank by mean Δ for the Ukraine–Russia base; "
    "additions rank by held-out mean Δ",
    ha="center",
    fontsize=13.5,
    color="#505050",
)
fig.text(
    0.5,
    0.018,
    "Green = improvement, red = degradation • solid outline = base-source target • "
    "dashed outline = newly added-source target • seed 0, step 2,500",
    ha="center",
    fontsize=11,
    color="#555555",
)

figure_output = HERE / "figures/nm_directional_intervention_deltas_all_targets.png"
fig.savefig(figure_output, dpi=300, bbox_inches="tight", facecolor="white")

print(long_output)
print(wide_output)
print(figure_output)
print(
    {
        "directional_rows": len(summary),
        "delta_cells": len(long),
        "base_blocks": len(base_order),
        "delta_limit_pp": delta_limit,
        "target_order": TARGET_ORDER,
    }
)
