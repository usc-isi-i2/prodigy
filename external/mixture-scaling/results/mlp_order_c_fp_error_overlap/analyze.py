import csv
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


ROOT = Path("/Users/philipp/Documents/Codex/2026-09-13/did")
SCORES = ROOT / "work/fp_error_scores"
SUMMARY = ROOT / "outputs/mlp_order_c_fp_error_overlap_by_target.tsv"
PAIRWISE = ROOT / "outputs/mlp_order_c_fp_error_overlap_pairwise.tsv"
FIGURE = ROOT / "outputs/mlp_order_c_fp_error_overlap.png"

summary_rows, pair_rows = [], []
for path in sorted(SCORES.glob("*.npz")):
    z = np.load(path)
    target = str(z["target"])
    model_ids = z["model_ids"].astype(str).tolist()
    errors = z["errors"].astype(float)
    n_models, n_nodes = errors.shape
    if n_models != 9 or len(np.unique(z["node_ids"])) != n_nodes:
        raise ValueError(f"unexpected alignment for {target}")

    hard = np.zeros_like(errors, dtype=bool)
    k = int(np.ceil(0.20 * n_nodes))
    for i in range(n_models):
        hard[i, np.argpartition(errors[i], -k)[-k:]] = True

    target_pairs = []
    for a, b in itertools.combinations(range(n_models), 2):
        inter = int((hard[a] & hard[b]).sum())
        union = int((hard[a] | hard[b]).sum())
        rho = float(spearmanr(errors[a], errors[b]).statistic)
        row = {
            "target": target,
            "model_a": model_ids[a],
            "model_b": model_ids[b],
            "error_spearman": rho,
            "hardest_20pct_jaccard": inter / union,
            "shared_hard_fraction": inter / n_nodes,
            "independent_expected_shared_hard_fraction": (k / n_nodes) ** 2,
            "shared_hard_lift": (inter / n_nodes) / ((k / n_nodes) ** 2),
        }
        pair_rows.append(row)
        target_pairs.append(row)

    hard_count = hard.sum(axis=0)
    mean_error_by_model = errors.mean(axis=1)
    oracle_error = errors.min(axis=0).mean()
    best_mean_error = mean_error_by_model.min()
    summary_rows.append({
        "target": target,
        "n_validation_nodes": n_nodes,
        "mask_replicates": int(z["replicates"]),
        "mean_pairwise_error_spearman": np.mean([r["error_spearman"] for r in target_pairs]),
        "mean_pairwise_hardest_20pct_jaccard": np.mean([r["hardest_20pct_jaccard"] for r in target_pairs]),
        "mean_shared_hard_lift": np.mean([r["shared_hard_lift"] for r in target_pairs]),
        "fraction_hard_for_all_9": np.mean(hard_count == 9),
        "fraction_hard_for_majority": np.mean(hard_count >= 5),
        "fraction_hard_for_none": np.mean(hard_count == 0),
        "mean_model_error": mean_error_by_model.mean(),
        "best_single_model_mean_error": best_mean_error,
        "oracle_per_node_mean_error": oracle_error,
        "oracle_error_reduction_vs_best_single": 1 - oracle_error / best_mean_error,
    })

for output, rows in [(SUMMARY, summary_rows), (PAIRWISE, pair_rows)]:
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

labels = {
    "covid19_twitter": "COVID-19", "covid_political": "COVID Pol.",
    "cp_hk_twitter": "CP/HK", "election2020": "Election",
    "facebook_page_reference": "Facebook", "midterm": "Midterm",
    "twibot20": "TwiBot", "ukr_rus_suspended": "Suspended",
    "ukr_rus_twitter": "UKR/RUS",
}
plot_rows = sorted(summary_rows, key=lambda row: row["mean_pairwise_hardest_20pct_jaccard"], reverse=True)
y = np.arange(len(plot_rows))
fig, axes = plt.subplots(1, 2, figsize=(12, 5.8), constrained_layout=True)
axes[0].barh(y, [r["mean_pairwise_error_spearman"] for r in plot_rows], color="#4477AA")
axes[0].set_yticks(y, [labels[r["target"]] for r in plot_rows])
axes[0].invert_yaxis()
axes[0].set_xlabel("Mean pairwise Spearman correlation")
axes[0].set_title("Agreement on per-node FP error rank", loc="left", weight="bold")
axes[0].grid(axis="x", alpha=.25)

axes[1].barh(y, [r["mean_pairwise_hardest_20pct_jaccard"] for r in plot_rows], color="#CC6677")
axes[1].set_yticks(y, [labels[r["target"]] for r in plot_rows])
axes[1].invert_yaxis()
axes[1].set_xlabel("Mean Jaccard of hardest-20% node sets")
axes[1].set_title("Agreement on which nodes are hardest", loc="left", weight="bold")
axes[1].grid(axis="x", alpha=.25)
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("Order C node-only MLP: FP error overlap across source models", fontsize=15, weight="bold")
fig.savefig(FIGURE, dpi=220, bbox_inches="tight", facecolor="white")

print(SUMMARY)
for row in summary_rows:
    print(row)
print(FIGURE)
