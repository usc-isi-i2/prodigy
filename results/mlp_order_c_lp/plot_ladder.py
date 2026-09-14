import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "data/ladder_results.csv"
OUTPUT = ROOT / "figures/mlp_order_c_lp_ladder.png"

ORDER = [
    "election2020", "covid_political", "cp_hk_twitter", "ukr_rus_suspended",
    "midterm", "twibot20", "ukr_rus_twitter", "covid19_twitter",
    "facebook_page_reference",
]
LABEL = {
    "election2020": "Election",
    "covid_political": "COVID Pol.",
    "cp_hk_twitter": "CP/HK",
    "ukr_rus_suspended": "Suspended",
    "midterm": "Midterm",
    "twibot20": "TwiBot",
    "ukr_rus_twitter": "UKR/RUS",
    "covid19_twitter": "COVID-19",
    "facebook_page_reference": "Facebook",
}

rows = []
with INPUT.open(newline="") as handle:
    for row in csv.DictReader(handle):
        row["n_sources"] = int(row["n_sources"])
        row["roc_auc"] = float(row["roc_auc"])
        row["target_in_pretraining"] = row["target_in_pretraining"] == "True"
        rows.append(row)

assert len(rows) == 81
by_target = defaultdict(dict)
for row in rows:
    by_target[row["target"]][row["n_sources"]] = row["roc_auc"]
assert set(by_target) == set(ORDER)
assert all(set(values) == set(range(1, 10)) for values in by_target.values())

x = np.arange(1, 10)
palette = plt.get_cmap("tab10").colors
fig, (ax, mean_ax) = plt.subplots(
    2, 1, figsize=(12.5, 9), gridspec_kw={"height_ratios": [2.25, 1]},
    constrained_layout=True,
)

for i, target in enumerate(ORDER):
    y = np.array([by_target[target][r] for r in x])
    entry = ORDER.index(target) + 1
    ax.plot(x, y, marker="o", ms=4, lw=1.8, color=palette[i % 10], label=LABEL[target])
    ax.scatter([entry], [y[entry - 1]], s=80, facecolor="white",
               edgecolor=palette[i % 10], linewidth=2.2, zorder=5)

ax.set_title("MLP Order C link-prediction ladder", loc="left", fontsize=15, weight="bold", pad=12)
ax.set_ylabel("Target ROC AUC")
ax.grid(axis="y", alpha=0.25)
ax.set_xlim(0.8, 9.2)
ax.legend(ncol=3, fontsize=8.5, frameon=False, loc="upper left",
          title="Open circle = target enters pretraining", title_fontsize=8.5)

all_means, heldout_means, seen_means = [], [], []
for rung in x:
    rung_rows = [row for row in rows if row["n_sources"] == rung]
    all_means.append(np.mean([row["roc_auc"] for row in rung_rows]))
    heldout = [row["roc_auc"] for row in rung_rows if not row["target_in_pretraining"]]
    seen = [row["roc_auc"] for row in rung_rows if row["target_in_pretraining"]]
    heldout_means.append(np.mean(heldout) if heldout else np.nan)
    seen_means.append(np.mean(seen) if seen else np.nan)

mean_ax.plot(x, all_means, color="black", marker="o", lw=2.5, label="All 9 targets")
mean_ax.plot(x, heldout_means, color="#2a6fbb", marker="o", lw=2.2,
             label="Targets not yet included")
mean_ax.plot(x, seen_means, color="#d65f5f", marker="o", lw=2.2,
             label="Included targets")
mean_ax.set_ylabel("Mean ROC AUC")
mean_ax.set_xlabel("Number of pretraining graphs and graph added at each rung")
mean_ax.grid(axis="y", alpha=0.25)
mean_ax.set_xlim(0.8, 9.2)
mean_ax.legend(ncol=3, fontsize=9, frameon=False, loc="best")

ticks = [f"{i}\n{LABEL[name]}" for i, name in enumerate(ORDER, 1)]
for axis in (ax, mean_ax):
    axis.set_xticks(x, ticks, fontsize=8)
    axis.spines[["top", "right"]].set_visible(False)

fig.savefig(OUTPUT, dpi=220, bbox_inches="tight", facecolor="white")
print(OUTPUT)
print("all_mean", [round(v, 4) for v in all_means])
print("heldout_mean", [None if np.isnan(v) else round(v, 4) for v in heldout_means])
print("seen_mean", [None if np.isnan(v) else round(v, 4) for v in seen_means])
