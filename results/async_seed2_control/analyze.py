"""Analyze the existing-seed-2 Ukraine-only control without changing selections.

The previous KD checkpoint choices are frozen. This script reports the declared
exposure/update controls and the separate source-selected recipe comparison.
It neither chooses models from transfer scores nor substitutes unavailable arms.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SOURCES = ("ukr_rus_twitter", "facebook_page_reference")
SOURCE_NAMES = ("Ukraine", "Facebook")
TARGETS = ("covid19_twitter", "covid_political", "cp_hk_twitter", "midterm", "twibot20", "ukr_rus_suspended")
TARGET_NAMES = ("COVID", "COVID political", "Hong Kong", "Midterm", "TwiBot-20", "Ukraine suspended")
CONTRASTS = (
    ("fixed_equal_ukraine_exposure", "kd_w010_fixed", "ukraine_exposure_fixed",
     "Fixed KD; same added Ukraine exposure", "Ukraine input exposure; unequal total updates/compute"),
    ("fixed_equal_optimizer_updates", "kd_w010_fixed", "ukraine_updates_fixed",
     "Fixed KD; same added optimizer updates", "Optimizer updates; unequal Ukraine exposure and teacher compute"),
    ("selected_equal_ukraine_exposure", "kd_w010_selected", "ukraine_exposure_selected",
     "Frozen-selected KD; same Ukraine exposure", "Ukraine input exposure; unequal total updates/compute"),
    ("selected_equal_optimizer_updates", "kd_w010_selected", "ukraine_updates_selected",
     "Frozen-selected KD; same optimizer updates", "Optimizer updates; unequal Ukraine exposure and teacher compute"),
    ("source_selected_recipes", "kd_w010_selected", "ukraine_only_selected",
     "Separate source-selected recipes", "Same source-only selection rule; durations may differ"),
)
COLORS = {"kd_w010": "#087c80", "ukraine_only": "#b87922"}
LABELS = {"kd_w010": "Ukraine BCE + Facebook KD (weight 0.1)", "ukraine_only": "Ukraine BCE only"}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def history_row(data: dict, entry: dict) -> dict | None:
    rows = data.get("history", {}).get(entry.get("arm"), [])
    if entry.get("additional_step") is None:
        return None
    matches = [row for row in rows if row["additional_step"] == entry["additional_step"]]
    if len(matches) > 1:
        raise ValueError(f"Multiple history rows match {entry['run_id']}")
    if not matches:
        return None
    row = matches[0]
    for key in ("counts", "validation"):
        if entry.get(key) is not None and entry[key] != row[key]:
            raise ValueError(f"Manifest/history mismatch for {entry['run_id']}: {key}")
    return row


def source_tables(data: dict, output: Path) -> pd.DataFrame:
    thresholds = data["manifest"]["singleton_source_auc"]
    records = []
    for entry in data["manifest"].get("models", []):
        validation, counts = entry.get("validation"), entry.get("counts", {})
        if validation is None:
            continue
        matched = history_row(data, entry)
        for i, source in enumerate(SOURCES):
            if isinstance(validation, dict):
                if source not in validation:
                    continue
                val, count = validation[source], counts.get(source, {})
                probe = None
            else:
                val, count = validation[i], counts[i]
                probe = matched["training_probe"][i]["bce"] if matched else None
            records.append({
                "seed": 2, "run_id": entry["run_id"], "arm": entry.get("arm"), "source": source,
                "selection": entry.get("selection"), "provenance": json.dumps(entry.get("provenance")),
                "additional_step": entry.get("additional_step"), "validation_auc_pct": val["auc"] * 100,
                "validation_bce": val["bce"], "hard_label_training_probe_bce": probe,
                "own_singleton_validation_auc_pct": thresholds[source] * 100,
                "delta_own_singleton_validation_pp": (val["auc"] - thresholds[source]) * 100,
                "retains_own_singleton_validation": val["auc"] >= thresholds[source], **count,
            })
    checkpoints = pd.DataFrame(records)
    checkpoints.to_csv(output / "source_checkpoint_metrics.csv", index=False)
    trajectory = []
    for arm, rows in data.get("history", {}).items():
        if not rows:
            continue
        initial_ukraine = rows[0]["counts"][0]["supervised_updates"]
        for row in rows:
            retains_both = all(row["validation"][i]["auc"] >= thresholds[source] for i, source in enumerate(SOURCES))
            for i, source in enumerate(SOURCES):
                trajectory.append({"seed": 2, "arm": arm, "source": source,
                                   "additional_step": row["additional_step"],
                                   "added_ukraine_supervised_updates": row["counts"][0]["supervised_updates"] - initial_ukraine,
                                   "validation_auc_pct": row["validation"][i]["auc"] * 100,
                                   "validation_bce": row["validation"][i]["bce"],
                                   "hard_label_training_probe_bce": row["training_probe"][i]["bce"],
                                   "retains_both_source_validation": retains_both,
                                   "checkpoint": row.get("checkpoint"), **row["counts"][i]})
    pd.DataFrame(trajectory).to_csv(output / "source_trajectory.csv", index=False)
    differences = []
    if not checkpoints.empty:
        for label, treatment, control, _, budget in CONTRASTS:
            for source in SOURCES:
                left = checkpoints[(checkpoints.run_id == treatment) & (checkpoints.source == source)]
                right = checkpoints[(checkpoints.run_id == control) & (checkpoints.source == source)]
                if left.empty or right.empty:
                    continue
                a, b = left.iloc[0], right.iloc[0]
                differences.append({"seed": 2, "comparison": label, "source": source, "budget_relationship": budget,
                                    "kd_run_id": treatment, "control_run_id": control,
                                    "kd_additional_steps": a.additional_step, "control_additional_steps": b.additional_step,
                                    "kd_cumulative_ukraine_updates": checkpoints[(checkpoints.run_id == treatment) & (checkpoints.source == SOURCES[0])].iloc[0].supervised_updates,
                                    "control_cumulative_ukraine_updates": checkpoints[(checkpoints.run_id == control) & (checkpoints.source == SOURCES[0])].iloc[0].supervised_updates,
                                    "kd_minus_control_validation_auc_pp": a.validation_auc_pct - b.validation_auc_pct,
                                    "kd_minus_control_validation_bce": a.validation_bce - b.validation_bce})
    pd.DataFrame(differences).to_csv(output / "source_validation_comparisons.csv", index=False)
    return checkpoints


def save(fig, directory: Path, name: str) -> None:
    for suffix in ("png", "pdf"):
        fig.savefig(directory / f"{name}.{suffix}", dpi=180)
    plt.close(fig)


def source_figure(data: dict, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.7, 8.2), layout="constrained")
    for i, source in enumerate(SOURCES):
        for arm, rows in data.get("history", {}).items():
            if not rows:
                continue
            start = rows[0]["counts"][0]["supervised_updates"]
            xs = [(r["counts"][0]["supervised_updates"] - start) / 1000 for r in rows]
            for j, ys in enumerate(([r["validation"][i]["auc"] * 100 for r in rows],
                                    [r["training_probe"][i]["bce"] for r in rows])):
                axes[j, i].plot(xs, ys, color=COLORS.get(arm), lw=1.7, label=LABELS.get(arm, arm))
        axes[0, i].axhline(data["manifest"]["singleton_source_auc"][source] * 100, color="#62686e", ls="--", lw=1,
                          label="Seed-2 selected own-source singleton")
        axes[0, i].set(title=SOURCE_NAMES[i], ylabel="Source-validation AUC (%)")
        axes[1, i].set(ylabel="Fixed-probe hard-label training BCE")
        for ax in axes[:, i]:
            ax.set_xlabel("Added Ukraine supervised updates (thousands)")
            ax.grid(alpha=.2)
            ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False, fontsize=9)
    fig.suptitle("Existing seed 2: continued KD versus Ukraine-only training\n"
                 "Same parent checkpoint. Exposure is Ukraine supervision, not total compute.\n"
                 "Facebook training BCE is a hard-label diagnostic even when its optimized loss is KD or absent.", fontsize=11)
    save(fig, output, "source_trajectories")


def transfer_tables(data: dict, matrix: pd.DataFrame, output: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if matrix.duplicated(["arm", "target"]).any():
        raise ValueError("Duplicate model/target evaluations; do not silently aggregate")
    pivot = matrix.pivot(index="target", columns="arm", values="roc_auc")
    models = {entry["run_id"]: entry for entry in data["manifest"].get("models", [])}
    means = []
    for arm in pivot:
        values = pivot[arm].reindex(TARGETS)
        complete = values.notna().all()
        means.append({"seed": 2, "arm": arm, "available_targets": int(values.notna().sum()),
                      "expected_targets": len(TARGETS), "complete": complete,
                      "mean_transfer_auc_pct": values.mean() * 100 if complete else np.nan,
                      "selection": models.get(arm, {}).get("selection"),
                      "additional_step": models.get(arm, {}).get("additional_step")})
    pd.DataFrame(means).to_csv(output / "transfer_means.csv", index=False)
    comparisons, target_rows, test_rows = [], [], []
    for label, treatment, control, description, budget in CONTRASTS:
        if treatment not in pivot or control not in pivot:
            comparisons.append({"seed": 2, "comparison": label, "description": description,
                                "budget_relationship": budget, "complete": False,
                                "status": "unavailable or pending; no substitute comparison"})
            continue
        deltas = (pivot[treatment] - pivot[control]).reindex(TARGETS) * 100
        complete = deltas.notna().all()
        comparisons.append({"seed": 2, "comparison": label, "description": description,
                            "kd_run_id": treatment, "control_run_id": control, "budget_relationship": budget,
                            "kd_additional_steps": models.get(treatment, {}).get("additional_step"),
                            "control_additional_steps": models.get(control, {}).get("additional_step"),
                            "available_targets": int(deltas.notna().sum()), "expected_targets": len(TARGETS),
                            "complete": complete, "status": "complete" if complete else "partial evaluation",
                            "mean_kd_minus_control_pp": deltas.mean() if complete else np.nan,
                            "positive_target_deltas": int((deltas > 0).sum()) if complete else None})
        for target, value in deltas.items():
            target_rows.append({"seed": 2, "comparison": label, "target": target, "kd_minus_control_auc_pp": value})
        for source in SOURCES:
            if source in pivot.index:
                test_rows.append({"seed": 2, "comparison": label, "source": source,
                                  "kd_test_auc_pct": pivot.loc[source, treatment] * 100,
                                  "control_test_auc_pct": pivot.loc[source, control] * 100,
                                  "kd_minus_control_test_auc_pp": (pivot.loc[source, treatment] - pivot.loc[source, control]) * 100})
    source_retention = []
    for source, singleton in zip(SOURCES, ("singleton_ukraine", "singleton_facebook")):
        if singleton not in pivot or source not in pivot.index:
            continue
        own = pivot.loc[source, singleton]
        for arm in pivot:
            if arm.startswith("singleton_"):
                continue
            value = pivot.loc[source, arm]
            source_retention.append({"seed": 2, "arm": arm, "source": source,
                                     "source_test_auc_pct": value * 100, "own_singleton_test_auc_pct": own * 100,
                                     "delta_own_singleton_test_pp": (value - own) * 100,
                                     "observed_nonnegative_delta": bool(value >= own) if pd.notna(value) and pd.notna(own) else None})
    compare_df, target_df = pd.DataFrame(comparisons), pd.DataFrame(target_rows)
    compare_df.to_csv(output / "transfer_comparisons.csv", index=False)
    target_df.to_csv(output / "target_deltas.csv", index=False)
    pd.DataFrame(test_rows).to_csv(output / "source_test_comparisons.csv", index=False)
    pd.DataFrame(source_retention).to_csv(output / "source_test_retention.csv", index=False)
    print("Declared seed-2 comparisons; positive delta means KD is higher:")
    print(compare_df.to_string(index=False))
    return compare_df, target_df


def comparison_figure(comparisons: pd.DataFrame, targets: pd.DataFrame, output: Path) -> None:
    if comparisons.empty or targets.empty:
        return
    eligible = comparisons[comparisons.complete].copy()
    if eligible.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, max(4.8, len(eligible) * .8)), layout="constrained",
                             gridspec_kw={"width_ratios": [1, 1.45]})
    vals = eligible.mean_kd_minus_control_pp.to_numpy()
    axes[0].barh(range(len(eligible)), vals, color=["#087c80" if v >= 0 else "#b85f48" for v in vals])
    axes[0].set_yticks(range(len(eligible)), eligible.description, fontsize=9)
    axes[0].invert_yaxis()
    axes[0].axvline(0, color="#555555", lw=.8)
    axes[0].set_xlabel("Mean transfer AUC: KD − Ukraine-only (pp)")
    axes[0].grid(axis="x", alpha=.2)
    axes[0].spines[["top", "right"]].set_visible(False)
    heat = targets.pivot(index="comparison", columns="target", values="kd_minus_control_auc_pp").reindex(index=eligible.comparison, columns=TARGETS)
    values = heat.to_numpy()
    limit = max(float(np.nanmax(np.abs(values))), .01)
    im = axes[1].imshow(values, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
    axes[1].set_yticks(range(len(eligible)), [""] * len(eligible))
    axes[1].set_xticks(range(len(TARGETS)), TARGET_NAMES, rotation=35, ha="right", fontsize=9)
    for i in range(len(eligible)):
        for j in range(len(TARGETS)):
            axes[1].text(j, i, f"{values[i,j]:+.2f}", ha="center", va="center", fontsize=9,
                         color="white" if abs(values[i,j]) > .6 * limit else "#222222")
    fig.colorbar(im, ax=axes[1], label="KD − Ukraine-only test AUC (pp)", shrink=.8)
    fig.suptitle("Existing seed 2: declared control comparisons on six non-source graphs\n"
                 "Exposure matching, update matching, and source-selected recipes are separate contrasts.\n"
                 "One existing seed; no confidence interval or independent replication claim.", fontsize=11)
    save(fig, output, "matched_control_comparisons")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    args = parser.parse_args()
    aggregate_path = args.data_dir / "aggregate.json"
    if not aggregate_path.exists():
        print(f"Waiting for measured existing-seed-2 control data: {aggregate_path}")
        return
    data = read_json(aggregate_path)
    if data.get("replication_seed", 2) != 2:
        raise ValueError("This analysis is prescribed for existing seed 2 only")
    output, plots = args.output_dir / "data", args.output_dir / "figures"
    output.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)
    source_tables(data, output)
    if any(data.get("history", {}).values()):
        source_figure(data, plots)
    for item in data["manifest"].get("unavailable", []):
        print("Declared unavailable:", item)
    matrix_path = args.data_dir / "matrix.csv"
    if matrix_path.exists():
        matrix = pd.read_csv(matrix_path)
        if not matrix.empty:
            comparisons, targets = transfer_tables(data, matrix, output)
            comparison_figure(comparisons, targets, plots)
    else:
        print(f"Transfer test evaluations pending: {matrix_path}")
    print("Source validation and source-test retention are separate outcomes.\n"
          "KD selections are frozen; separately source-selected recipes may use different durations.\n"
          "These are controls for the existing seed-2 run, not new training-seed replications.")


if __name__ == "__main__":
    main()
