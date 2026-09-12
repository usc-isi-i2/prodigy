"""Analyze the declared Facebook KD-weight study using measured exports only.

The manifest, frozen using source validation, supplies selected checkpoints and
the chosen weight. Transfer metrics are reported after that selection, never used
to choose a checkpoint or weight here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
SOURCES = ("ukr_rus_twitter", "facebook_page_reference")
NAMES = ("Ukraine", "Facebook")
COLORS = {0.1: "#3166ad", 0.3: "#087c80", 0.5: "#a97224", 1.0: "#965c99"}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def weights(data: dict) -> dict:
    result = {}
    for arm in data["history"]:
        value = data.get("summary", {}).get(arm, {}).get("protocol", {}).get("kd_weight")
        if value is None:
            declared = {m["weight"] for m in data.get("manifest", {}).get("models", [])
                        if m.get("arm") == arm and "weight" in m}
            if len(declared) != 1:
                raise ValueError(f"No unique declared KD weight for {arm}")
            value = declared.pop()
        result[arm] = float(value)
    return result


def baselines(data: dict) -> tuple[list[float], dict]:
    selected = {}
    for source in SOURCES:
        reference = data["references"][source]
        step = reference["summary"]["historical_best_step"]
        selected[source] = next(row for row in reference["history"] if row["step"] == step)
    aucs = [selected[source]["validation"]["auc"] for source in SOURCES]
    declared = data.get("manifest", {}).get("singleton_source_auc")
    if declared and any(abs(declared[source] - auc) > 1e-12 for source, auc in zip(SOURCES, aucs)):
        raise ValueError("Manifest singleton thresholds disagree with historical selected checkpoints")
    return aucs, selected


def record(arm: str, row: dict, weight: float, baseline: list[float]) -> dict:
    result = {"arm": arm, "weight": weight, "additional_step": row.get("additional_step"),
              "logical_step": row.get("logical_step"), "checkpoint": row.get("checkpoint")}
    for i, name in enumerate(("ukraine", "facebook")):
        val, probe, count = row["validation"][i], row["training_probe"][i], row["counts"][i]
        result.update({f"{name}_validation_auc_pct": val["auc"] * 100,
                       f"{name}_validation_bce": val["bce"],
                       f"{name}_delta_singleton_pp": (val["auc"] - baseline[i]) * 100,
                       f"{name}_hard_label_training_bce": probe["bce"],
                       f"retains_{name}": val["auc"] >= baseline[i]})
        result.update({f"{name}_{key}": value for key, value in count.items()})
    result["retains_both"] = result["retains_ukraine"] and result["retains_facebook"]
    return result


def manifest_rows(data: dict) -> list[tuple[dict, dict]]:
    result = []
    for entry in data.get("manifest", {}).get("models", []):
        rows = [r for r in data["history"].get(entry["arm"], [])
                if r.get("checkpoint") == entry["original_checkpoint"]]
        if len(rows) != 1:
            raise ValueError(f"Expected one logged checkpoint for manifest run {entry['run_id']}; found {len(rows)}")
        row = rows[0]
        for key in ("additional_step", "counts", "validation"):
            if key in entry and entry[key] != row[key]:
                raise ValueError(f"Manifest/history mismatch for {entry['run_id']}: {key}")
        result.append((entry, row))
    return result


def source_tables(data: dict, arm_weights: dict, baseline: list[float], output: Path) -> None:
    all_rows = [record(arm, row, arm_weights[arm], baseline)
                for arm, rows in data["history"].items() for row in rows]
    history = pd.DataFrame(all_rows)
    history.to_csv(output / "source_history.csv", index=False)
    history[history.retains_both].to_csv(output / "both_preserved_candidates.csv", index=False)
    history.groupby(["arm", "weight"]).agg(
        logged_checkpoints=("retains_both", "size"),
        both_preserved_checkpoints=("retains_both", "sum"),
        maximum_ukraine_delta_pp=("ukraine_delta_singleton_pp", "max"),
        maximum_facebook_delta_pp=("facebook_delta_singleton_pp", "max"),
    ).to_csv(output / "retention_counts.csv")
    chosen_id = data.get("manifest", {}).get("chosen_run_id")
    checkpoints = []
    for entry, row in manifest_rows(data):
        item = record(entry["arm"], row, arm_weights[entry["arm"]], baseline)
        item.update(run_id=entry["run_id"], selection=entry["selection"],
                    start_fallback=entry.get("start_fallback", False),
                    chosen_by_source_validation=entry["run_id"] == chosen_id)
        checkpoints.append(item)
    if checkpoints:
        declared = pd.DataFrame(checkpoints)
        declared.to_csv(output / "source_checkpoints.csv", index=False)
        print("Declared fixed and source-selected checkpoints:")
        print(declared[["run_id", "weight", "ukraine_supervised_updates", "ukraine_validation_auc_pct",
                        "facebook_validation_auc_pct", "retains_both", "chosen_by_source_validation"]].to_string(index=False))
    else:
        print("No frozen checkpoint manifest yet; source histories are available, selections are pending.")


def save_figure(fig, directory: Path, name: str) -> None:
    for suffix in ("png", "pdf"):
        fig.savefig(directory / f"{name}.{suffix}", dpi=180)
    plt.close(fig)


def style(ax) -> None:
    ax.grid(alpha=.2)
    ax.spines[["top", "right"]].set_visible(False)


def source_figures(data: dict, arm_weights: dict, baseline: list[float], selected: dict, output: Path) -> None:
    ordered = sorted(data["history"], key=lambda arm: arm_weights[arm])
    declared = manifest_rows(data)
    chosen = data.get("manifest", {}).get("chosen_run_id")
    fig, ax = plt.subplots(figsize=(8.8, 6.3), layout="constrained")
    for arm in ordered:
        rows = data["history"][arm]
        weight = arm_weights[arm]
        ax.plot([r["validation"][0]["auc"] * 100 for r in rows],
                [r["validation"][1]["auc"] * 100 for r in rows],
                color=COLORS.get(weight), lw=1.2, alpha=.65, label=f"Facebook KD weight {weight:g}")
        ax.scatter([rows[-1]["validation"][0]["auc"] * 100],
                   [rows[-1]["validation"][1]["auc"] * 100],
                   marker="s", s=55, color=COLORS.get(weight), zorder=7)
    for entry, row in declared:
        is_chosen = entry["run_id"] == chosen
        ax.scatter([row["validation"][0]["auc"] * 100], [row["validation"][1]["auc"] * 100],
                   color=COLORS.get(arm_weights[entry["arm"]]), marker="*" if is_chosen else "D",
                   s=180 if is_chosen else 35, edgecolor="black" if is_chosen else "white",
                   linewidth=.7, zorder=10 if is_chosen else 8)
    ax.axvline(baseline[0] * 100, color="#555555", ls="--", lw=1)
    ax.axhline(baseline[1] * 100, color="#555555", ls="--", lw=1)
    ax.set(xlabel="Ukraine source-validation AUC (%)", ylabel="Facebook source-validation AUC (%)",
           title="Facebook KD weight: source-retention tradeoff\n"
                 "Dashed lines: historical selected singleton thresholds (two different models).\n"
                 "Squares: final checkpoints. Diamonds: declared evaluations. Star: source-selected weight.")
    ax.legend(frameon=False, fontsize=9, loc="best")
    style(ax)
    save_figure(fig, output, "source_retention_tradeoff")

    fig, ax = plt.subplots(figsize=(9, 5.3), layout="constrained")
    for arm in ordered:
        rows, weight = data["history"][arm], arm_weights[arm]
        ax.plot([r["counts"][0]["supervised_updates"] / 1000 for r in rows],
                [r["validation"][0]["auc"] * 100 for r in rows],
                color=COLORS.get(weight), label=f"Facebook KD weight {weight:g}", lw=1.7)
    singleton = [r for r in data["references"][SOURCES[0]]["history"] if r["step"] >= 20000]
    ax.plot([r["counts"]["supervised_updates"] / 1000 for r in singleton],
            [r["validation"]["auc"] * 100 for r in singleton], color="#6d7278", ls=":", label="Ukraine singleton trajectory")
    ax.axhline(baseline[0] * 100, color="#6d7278", ls="--", lw=1, label="Selected Ukraine singleton")
    ax.set(xlabel="Cumulative Ukraine supervised updates (thousands)", ylabel="Ukraine source-validation AUC (%)",
           title="Ukraine learning at matched source exposure\nAll weights continue the same original KD checkpoint; source validation only.")
    ax.legend(frameon=False, fontsize=8)
    style(ax)
    save_figure(fig, output, "ukraine_exposure")

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.4), layout="constrained")
    for i, (source, name) in enumerate(zip(SOURCES, NAMES)):
        ax = axes[i]
        for arm in ordered:
            rows, weight = data["history"][arm], arm_weights[arm]
            ax.plot([r["training_probe"][i]["bce"] for r in rows],
                    [r["validation"][i]["auc"] * 100 for r in rows],
                    color=COLORS.get(weight), label=f"KD weight {weight:g}", lw=1.5)
        baseline_row = selected[source]
        ax.scatter([baseline_row["training_probe"]["bce"]], [baseline[i] * 100],
                   marker="D", color="#6d7278", s=45, label="Selected own-source singleton")
        ax.set(title=name, xlabel="Fixed-probe hard-label training BCE (lower to right)", ylabel="Source-validation AUC (%)")
        ax.invert_xaxis()
        style(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle("Training-loss diagnostics\nSame hard-label probe for every weight, including after loss switching.\n"
                 "Trajectories stay in time order; equal scalar loss does not establish a causal comparison.", fontsize=11)
    save_figure(fig, output, "source_training_loss")


def transfer_tables(data: dict, data_dir: Path, previous: Path, original: Path, output: Path) -> None:
    path = data_dir / "matrix.csv"
    if not path.exists():
        print(f"Transfer evaluations pending: {path}")
        return
    matrix = pd.read_csv(path)
    if matrix.empty:
        print("Transfer matrix is empty; no transfer summary generated.")
        return
    key = next((k for k in ("arm", "run_id", "model") if k in matrix), None)
    metric = next((k for k in ("roc_auc", "auc") if k in matrix), None)
    if key is None or metric is None:
        raise ValueError(f"Unrecognized transfer matrix columns: {list(matrix)}")
    comparison = matrix.pivot(index="target", columns=key, values=metric) * 100
    current = list(comparison)
    previous_matrix = previous / "matrix.csv"
    expected = set()
    if previous_matrix.exists():
        prior = pd.read_csv(previous_matrix).pivot(index="target", columns="arm", values="roc_auc") * 100
        expected = set(prior.index)
        for arm in ("kd_extended_selected", "kd_fixed", "ukraine_only_selected",
                    "ukraine_exposure", "ukraine_updates"):
            if arm in prior:
                comparison[f"previous_{arm}"] = prior[arm]
    if not expected:
        expected = set(comparison.index)
    targets = sorted(expected - set(SOURCES) - {"election2020"})
    baseline_path = original / "singleton_baselines.csv"
    if baseline_path.exists():
        singletons = pd.read_csv(baseline_path)
        comparison["Ukraine singleton"] = singletons[singletons.source == SOURCES[0]].set_index("target").auc * 100
    means = []
    chosen = data.get("manifest", {}).get("chosen_run_id")
    for model in comparison:
        values = comparison[model].reindex(targets)
        complete = values.notna().all()
        means.append({"model": model, "available_targets": int(values.notna().sum()), "expected_targets": len(targets),
                      "complete": complete, "mean_transfer_auc_pct": values.mean() if complete else float("nan"),
                      "chosen_by_source_validation": model == chosen})
    means = pd.DataFrame(means)
    means.to_csv(output / "transfer_means.csv", index=False)
    for model in current:
        for baseline in ("previous_kd_extended_selected", "previous_ukraine_only_selected",
                         "previous_ukraine_exposure", "previous_ukraine_updates", "Ukraine singleton"):
            if baseline in comparison:
                comparison[f"{model}_minus_{baseline}_pp"] = comparison[model] - comparison[baseline]
    comparison["graph_role"] = ["training graph test" if t in SOURCES else "excluded target" if t == "election2020"
                                else "transfer graph test" for t in comparison.index]
    comparison.to_csv(output / "transfer_comparison.csv")
    print("Transfer test means; chosen weight is copied from source-only manifest:")
    print(means.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--previous-data-dir", type=Path, default=ROOT.parent / "async_extension" / "data")
    parser.add_argument("--original-data-dir", type=Path, default=ROOT.parent / "async_convergence" / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    args = parser.parse_args()
    aggregate = args.data_dir / "aggregate.json"
    if not aggregate.exists():
        print(f"Waiting for measured KD-weight data: {aggregate}")
        return
    data = read_json(aggregate)
    if not any(data.get("history", {}).values()):
        print("No measured weight-study checkpoints yet.")
        return
    data["history"] = {arm: rows for arm, rows in data["history"].items() if rows}
    if not data.get("references"):
        data["references"] = read_json(args.original_data_dir / "aggregate.json")["references"]
    arm_weights = weights(data)
    baseline, selected = baselines(data)
    output, figures = args.output_dir / "data", args.output_dir / "figures"
    output.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    source_tables(data, arm_weights, baseline, output)
    source_figures(data, arm_weights, baseline, selected, figures)
    transfer_tables(data, args.data_dir, args.previous_data_dir, args.original_data_dir, output)
    print("One seed; exploratory development targets. Retention refers to these source-validation pairs, not a guarantee.\n"
          "Same sampled-input schedule across KD weights does not imply equivalence to a supervision-only budget.")


if __name__ == "__main__":
    main()
