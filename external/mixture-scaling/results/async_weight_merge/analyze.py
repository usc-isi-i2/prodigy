"""Analyze source-selected interpolation of two existing seed-2 models.

No optimization or new seeds are involved. A merge is test-evaluated only if it
meets both declared source-validation floors; test metrics never select alpha.
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
NAMES = ("Ukraine", "Facebook")
TARGETS = ("covid19_twitter", "covid_political", "cp_hk_twitter", "midterm", "twibot20", "ukr_rus_suspended")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def source_table(data: dict, output: Path) -> pd.DataFrame:
    floors = data["metadata"]["singleton_source_auc"]
    rows = sorted(data["rows"], key=lambda row: row["alpha"])
    if len(rows) != 11 or not np.allclose([r["alpha"] for r in rows], np.linspace(0, 1, 11), atol=1e-12, rtol=0):
        raise ValueError("Expected all 11 declared alpha values from 0 through 1 in increments of 0.1")
    records = []
    for row in rows:
        record = {"alpha": row["alpha"], "run_id": row["run_id"], "checkpoint": row.get("checkpoint")}
        margins = []
        for i, source in enumerate(SOURCES):
            name = NAMES[i].lower()
            val, probe = row["validation"][i], row.get("training_probe", [{}, {}])[i]
            margin = (val["auc"] - floors[source]) * 100
            margins.append(margin)
            record.update({f"{name}_validation_auc_pct": val["auc"] * 100,
                           f"{name}_validation_bce": val["bce"],
                           f"{name}_floor_auc_pct": floors[source] * 100,
                           f"{name}_delta_floor_pp": margin,
                           f"{name}_hard_label_training_bce": probe.get("bce")})
        record["minimum_margin_pp"] = min(margins)
        record["qualifies_both_source_floors"] = all(margin >= 0 for margin in margins)
        records.append(record)
    table = pd.DataFrame(records)
    table.to_csv(output / "all_alpha_source_metrics.csv", index=False)
    feasible = [row for row, record in zip(rows, records) if record["qualifies_both_source_floors"]]
    selection = data["selection"]
    expected_alphas = [row["alpha"] for row in feasible]
    declared_alphas = sorted(selection.get("qualifying_alphas", []))
    if len(expected_alphas) != len(declared_alphas) or not np.allclose(expected_alphas, declared_alphas, atol=1e-12, rtol=0):
        raise ValueError("Declared qualifying alphas disagree with source-validation floors")
    if not feasible:
        if selection["status"] != "no_feasible_merge" or selection.get("selected_alpha") is not None or selection.get("run_id") is not None:
            raise ValueError("No merge passes both floors, but the selection manifest claims a selected merge")
        print("No feasible merge: none of the 11 declared alphas meets both source-validation floors.")
        print("There is no selected merge and no authorized downstream test evaluation.")
    else:
        expected = max(feasible, key=lambda row: (row["validation"][0]["auc"], row["validation"][1]["auc"], -row["alpha"]))
        if (selection["status"] != "selected" or selection.get("run_id") != "merge_selected"
                or selection.get("candidate_run_id") != expected["run_id"]
                or not np.isclose(selection.get("selected_alpha"), expected["alpha"], atol=1e-12, rtol=0)):
            raise ValueError("Frozen selection does not follow the declared source-only tie-breaking rule")
        print(f"Source-selected alpha: {expected['alpha']:g}; qualifying alphas: {expected_alphas}")
    print(table[["alpha", "ukraine_validation_auc_pct", "facebook_validation_auc_pct",
                 "ukraine_delta_floor_pp", "facebook_delta_floor_pp", "minimum_margin_pp",
                 "qualifies_both_source_floors"]].to_string(index=False))
    return table


def source_figure(data: dict, table: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), layout="constrained")
    chosen = data["selection"].get("selected_alpha")
    for i, name in enumerate(NAMES):
        key = name.lower()
        ax = axes[i]
        ax.plot(table.alpha, table[f"{key}_validation_auc_pct"], marker="o", lw=1.5,
                color="#567c9f", markersize=4, label="Evaluated alpha")
        ax.axhline(table[f"{key}_floor_auc_pct"].iloc[0], color="#555555", ls="--", lw=1,
                   label="Selected own-source singleton floor")
        eligible = table[table.qualifies_both_source_floors]
        if not eligible.empty:
            ax.scatter(eligible.alpha, eligible[f"{key}_validation_auc_pct"], color="#087c80", s=50,
                       zorder=6, label="Passes both source floors")
        if chosen is not None:
            selected = table[np.isclose(table.alpha, chosen)]
            ax.scatter(selected.alpha, selected[f"{key}_validation_auc_pct"], marker="*", color="#087c80",
                       edgecolor="black", linewidth=.6, s=175, zorder=8, label="Source-selected merge")
        ax.set(title=name, xlabel="Alpha (0 = KD endpoint; 1 = Ukraine-only endpoint)", ylabel="Source-validation AUC (%)")
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.grid(alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False, fontsize=9)
    outcome = "No alpha satisfies both source-validation floors" if chosen is None else f"Alpha {chosen:g} selected using source validation only"
    fig.suptitle("Existing seed 2: parameter interpolation, with no additional training\n"
                 f"{outcome}\nPassing validation floors does not establish source-test retention.", fontsize=11)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"source_validation_by_alpha.{suffix}", dpi=180)
    plt.close(fig)


def transfer_tables(data: dict, matrix_path: Path, previous_dir: Path, output: Path) -> None:
    selected_id = data["selection"].get("run_id")
    if not matrix_path.exists():
        if selected_id is not None:
            print("Selected merge test evaluation is pending.")
        return
    matrix = pd.read_csv(matrix_path)
    if matrix.empty:
        print("No merge test measurements are present.")
        return
    if selected_id is None:
        raise ValueError("Unexpected test matrix despite no feasible merge; refusing to interpret unselected test results")
    if set(matrix.arm) != {selected_id} or matrix.duplicated(["arm", "target"]).any():
        raise ValueError("Test matrix must contain only the one frozen selected merge, with unique targets")
    merge = matrix.set_index("target").roc_auc * 100
    comparison = merge.rename(selected_id).to_frame()
    previous_path = previous_dir / "matrix.csv"
    if previous_path.exists():
        previous = pd.read_csv(previous_path).pivot(index="target", columns="arm", values="roc_auc") * 100
        for arm in ("kd_w010_fixed", "ukraine_updates_fixed", "singleton_ukraine", "singleton_facebook"):
            if arm in previous:
                comparison[arm] = previous[arm]
    records = []
    for arm in comparison:
        values = comparison[arm].reindex(TARGETS)
        complete = values.notna().all()
        records.append({"arm": arm, "available_targets": int(values.notna().sum()), "expected_targets": len(TARGETS),
                        "complete": complete, "mean_transfer_auc_pct": values.mean() if complete else np.nan})
    means = pd.DataFrame(records)
    means.to_csv(output / "transfer_means.csv", index=False)
    for arm in list(comparison):
        if arm != selected_id:
            comparison[f"merge_minus_{arm}_pp"] = comparison[selected_id] - comparison[arm]
    comparison["graph_role"] = ["training graph test" if target in SOURCES else "transfer graph test" for target in comparison.index]
    comparison.to_csv(output / "transfer_comparison.csv")
    source_rows = []
    for source, singleton in zip(SOURCES, ("singleton_ukraine", "singleton_facebook")):
        if source not in comparison.index or singleton not in comparison:
            continue
        source_rows.append({"source": source, "merge_test_auc_pct": comparison.loc[source, selected_id],
                            "own_singleton_test_auc_pct": comparison.loc[source, singleton],
                            "merge_minus_own_singleton_test_pp": comparison.loc[source, selected_id] - comparison.loc[source, singleton],
                            "merge_minus_kd_endpoint_test_pp": comparison.loc[source, selected_id] - comparison.loc[source, "kd_w010_fixed"] if "kd_w010_fixed" in comparison else np.nan,
                            "merge_minus_ukraine_only_endpoint_test_pp": comparison.loc[source, selected_id] - comparison.loc[source, "ukraine_updates_fixed"] if "ukraine_updates_fixed" in comparison else np.nan})
    pd.DataFrame(source_rows).to_csv(output / "source_test_retention.csv", index=False)
    print("Frozen selected merge: six non-source target test means:")
    print(means.to_string(index=False))
    if source_rows:
        print("Source-test retention, separate from source-validation feasibility:")
        print(pd.DataFrame(source_rows).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--previous-data-dir", type=Path, default=ROOT.parent / "async_seed2_control" / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    args = parser.parse_args()
    aggregate = args.data_dir / "aggregate.json"
    if not aggregate.exists():
        print(f"Waiting for measured merge source-validation data: {aggregate}")
        return
    data = read_json(aggregate)
    output, figures = args.output_dir / "data", args.output_dir / "figures"
    output.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    table = source_table(data, output)
    source_figure(data, table, figures)
    transfer_tables(data, args.data_dir / "matrix.csv", args.previous_data_dir, output)
    print("One existing seed; interpolation adds no training and provides no confidence interval.\n"
          "Test metrics do not select alpha. No-feasible outcomes remain no-feasible outcomes.")


if __name__ == "__main__":
    main()
