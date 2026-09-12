"""Analyze the prespecified KD extension without selecting on transfer scores.

Run with the Homebrew Python used for local scientific plots. Missing aggregate
or transfer exports are reported explicitly; no placeholder measurements are made.
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
NAMES = {SOURCES[0]: "Ukraine", SOURCES[1]: "Facebook"}
LABELS = {
    "kd_extended": "Continue Ukraine BCE + Facebook KD",
    "ukraine_only": "Continue Ukraine BCE only",
    "async_kd": "Original KD trajectory",
    "extended_bce": "Original long BCE control",
    "singleton": "Own-source singleton",
}
COLORS = {
    "kd_extended": "#087c80", "ukraine_only": "#ab7423",
    "async_kd": "#579bc5", "extended_bce": "#8a8f95",
    "singleton": "#c54b52",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def selected_references(refs: dict) -> dict:
    selected = {}
    for source in SOURCES:
        reference = refs[source]
        step = reference["summary"]["historical_best_step"]
        selected[source] = next(r for r in reference["history"] if r["step"] == step)
    return selected


def observation(arm: str, row: dict, source_index: int, baseline: dict,
                role: str = "logged checkpoint") -> dict:
    source = SOURCES[source_index]
    val = row["validation"][source_index]
    train = row["training_probe"][source_index]
    counts = row["counts"][source_index]
    return {
        "arm": arm, "source": source, "checkpoint_role": role,
        "additional_step": row.get("additional_step"),
        "logical_step": row.get("logical_step"),
        "checkpoint": row.get("checkpoint"),
        "validation_auc_pct": val["auc"] * 100,
        "validation_auc_minus_singleton_pp": (val["auc"] - baseline["validation"]["auc"]) * 100,
        "validation_bce": val["bce"], "hard_label_training_probe_bce": train["bce"],
        "singleton_reference_step": baseline["step"],
        "singleton_reference_auc_pct": baseline["validation"]["auc"] * 100,
        **counts,
    }


def tables(data: dict, prior: dict, refs: dict, output: Path, tolerance: float) -> None:
    baselines = selected_references(refs)
    endpoints, observations, candidates, retention = [], [], [], []
    for arm, rows in data["history"].items():
        if not rows:
            continue
        # Keep logged time order; sorting by training loss would change trajectories.
        for row in rows:
            for i, source in enumerate(SOURCES):
                observations.append(observation(arm, row, i, baselines[source]))
            deltas = [(row["validation"][i]["auc"] - baselines[source]["validation"]["auc"]) * 100
                      for i, source in enumerate(SOURCES)]
            record = {
                "arm": arm, "additional_step": row.get("additional_step"),
                "logical_step": row.get("logical_step"), "checkpoint": row.get("checkpoint"),
                "ukraine_validation_auc_pct": row["validation"][0]["auc"] * 100,
                "facebook_validation_auc_pct": row["validation"][1]["auc"] * 100,
                "ukraine_delta_pp": deltas[0], "facebook_delta_pp": deltas[1],
                "auc_tolerance_pp": tolerance,
                "retains_ukraine": deltas[0] >= -tolerance,
                "retains_facebook": deltas[1] >= -tolerance,
                "retains_both": all(delta >= -tolerance for delta in deltas),
            }
            retention.append(record)
            if record["retains_both"]:
                candidates.append(record)
        for i, source in enumerate(SOURCES):
            endpoints.append(observation(arm, rows[-1], i, baselines[source], "last logged checkpoint"))

    # Report the frozen evaluation selections separately from each arm's final
    # logged point (e.g. Ukraine-only's matched-exposure checkpoint is earlier).
    for entry in data.get("manifest", {}).get("models", []):
        if "arm" not in entry or "original_checkpoint" not in entry:
            continue
        matches = [row for row in data["history"].get(entry["arm"], [])
                   if row.get("checkpoint") == entry["original_checkpoint"]]
        if len(matches) != 1:
            raise ValueError(f"Manifest selection {entry['run_id']} must match exactly one logged checkpoint; "
                             f"found {len(matches)} for {entry['original_checkpoint']}")
        row = matches[0]
        for key in ("additional_step", "counts", "validation"):
            if key in entry and entry[key] != row[key]:
                raise ValueError(f"Manifest selection {entry['run_id']} disagrees with history on {key}")
        for i, source in enumerate(SOURCES):
            endpoint = observation(entry["run_id"], row, i, baselines[source], entry["selection"])
            endpoint.update(training_arm=entry["arm"], start_fallback=entry.get("start_fallback", False))
            endpoints.append(endpoint)

    # Original KD selected endpoint may precede its final logged patience tail.
    original = prior.get("summary", {}).get("async_kd")
    if original:
        row = {**original["endpoint"], "counts": original["surviving_counts"],
               "logical_step": original["logical_step"]}
        for i, source in enumerate(SOURCES):
            endpoints.append(observation("async_kd", row, i, baselines[source], "original selected endpoint"))
    pd.DataFrame(observations).to_csv(output / "source_history.csv", index=False)
    pd.DataFrame(endpoints).to_csv(output / "source_endpoints.csv", index=False)
    retention_df = pd.DataFrame(retention)
    retention_df.to_csv(output / "source_retention.csv", index=False)
    # Preserve headers even when no checkpoint satisfies both source thresholds.
    pd.DataFrame(candidates, columns=retention_df.columns).to_csv(output / "both_preserved_candidates.csv", index=False)
    if not retention_df.empty:
        totals = retention_df.groupby("arm").agg(
            logged_checkpoints=("retains_both", "size"),
            both_preserved_checkpoints=("retains_both", "sum"),
            maximum_ukraine_delta_pp=("ukraine_delta_pp", "max"),
            maximum_facebook_delta_pp=("facebook_delta_pp", "max"),
        )
        totals.to_csv(output / "retention_counts.csv")
        print("Source-validation candidates; no transfer-based selection:")
        print(totals.to_string())


def curves(data: dict, prior: dict, refs: dict, figures: Path) -> None:
    baselines = selected_references(refs)
    panels = [
        ("source_auc_by_exposure", "Source positive examples on retained path (millions)", "exposure"),
        ("source_auc_by_training_loss", "Fixed-probe hard-label training BCE (lower to right)", "loss"),
    ]
    for filename, xlabel, kind in panels:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.3), layout="constrained")
        for i, source in enumerate(SOURCES):
            ax = axes[i]
            for arm, records in [(name, prior.get("history", {}).get(name, []))
                                 for name in ("extended_bce", "async_kd")] + list(data["history"].items()):
                records = [r for r in records if r.get("logical_step", 1) > 0]
                if not records:
                    continue
                xs = [r["counts"][i]["positive_examples"] / 1e6 if kind == "exposure"
                      else r["training_probe"][i]["bce"] for r in records]
                ys = [r["validation"][i]["auc"] * 100 for r in records]
                is_old = arm in ("extended_bce", "async_kd")
                ax.plot(xs, ys, label=LABELS.get(arm, arm), color=COLORS.get(arm),
                        lw=1.3 if is_old else 2, alpha=.55 if is_old else .95,
                        marker=None if is_old else ".", markersize=4)
                if not is_old:
                    ax.scatter([xs[-1]], [ys[-1]], s=95, marker="*", color=COLORS.get(arm), zorder=8)
            records = [r for r in refs[source]["history"] if r["step"] > 0]
            xs = [r["counts"]["positive_examples"] / 1e6 if kind == "exposure"
                  else r["training_probe"]["bce"] for r in records]
            ax.plot(xs, [r["validation"]["auc"] * 100 for r in records],
                    label=LABELS["singleton"], color=COLORS["singleton"], ls="--", lw=1.6)
            baseline = baselines[source]
            baseline_x = (baseline["counts"]["positive_examples"] / 1e6 if kind == "exposure"
                          else baseline["training_probe"]["bce"])
            ax.scatter([baseline_x], [baseline["validation"]["auc"] * 100], marker="D", s=45,
                       color=COLORS["singleton"], zorder=8)
            ax.axhline(baseline["validation"]["auc"] * 100, color=COLORS["singleton"], ls=":", lw=.9)
            ax.set(title=NAMES[source], xlabel=xlabel, ylabel="Source validation AUC (%)")
            ax.grid(alpha=.2)
            ax.spines[["top", "right"]].set_visible(False)
            if kind == "loss":
                ax.set_xscale("log")
                ax.invert_xaxis()
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False, fontsize=9)
        explanation = ("Input exposure includes KD batches; it is not equal supervision or compute."
                       if kind == "exposure" else
                       "The same hard-label probe is measured after switching; equal scalar loss is descriptive.")
        fig.suptitle("KD extension: source retention and learning progress\n" + explanation +
                     "\nStars: last logged extension checkpoint. Diamonds/dotted lines: historical selected singletons.\n"
                     "Original trajectories include discarded patience tails; rewinds can move curves left.", fontsize=10)
        for suffix in ("png", "pdf"):
            fig.savefig(figures / f"{filename}.{suffix}", dpi=180)
        plt.close(fig)


def transfer_tables(matrix_path: Path, prior_data: Path, output: Path) -> None:
    if not matrix_path.exists():
        print(f"Transfer evaluation not available yet: {matrix_path}")
        return
    matrix = pd.read_csv(matrix_path)
    if matrix.empty:
        print("Transfer matrix is empty; no transfer means generated.")
        return
    arm_column = next((c for c in ("arm", "run_id", "model") if c in matrix), None)
    metric = next((c for c in ("roc_auc", "auc") if c in matrix), None)
    if arm_column is None or metric is None:
        raise ValueError(f"Expected arm/run_id/model and roc_auc/auc columns, got {list(matrix)}")
    transfer = matrix.pivot(index="target", columns=arm_column, values=metric) * 100
    current_arms = list(transfer.columns)
    prior_path = prior_data / "matrix.csv"
    if prior_path.exists():
        old = pd.read_csv(prior_path).pivot(index="target", columns="arm", values="roc_auc") * 100
        for arm in ("async_kd", "extended_bce", "extended_bce_matched"):
            if arm in old:
                transfer[f"original_{arm}"] = old[arm]
        targets = sorted(set(old.index) - set(SOURCES) - {"election2020"})
    else:
        targets = sorted(set(transfer.index) - set(SOURCES) - {"election2020"})
    singleton_path = prior_data / "singleton_baselines.csv"
    if singleton_path.exists():
        singletons = pd.read_csv(singleton_path)
        for source in SOURCES:
            transfer[f"singleton_{source}"] = singletons[singletons.source == source].set_index("target").auc * 100
    comparison = transfer.copy()
    for arm in current_arms:
        for baseline in ("original_async_kd", "original_extended_bce", f"singleton_{SOURCES[0]}"):
            if baseline in transfer:
                comparison[f"{arm}_minus_{baseline}_pp"] = transfer[arm] - transfer[baseline]
    comparison["graph_role"] = ["training graph test" if t in SOURCES else
                                "excluded target" if t == "election2020" else "transfer graph test"
                                for t in comparison.index]
    comparison.to_csv(output / "transfer_comparison.csv")
    records = []
    for arm in transfer:
        values = transfer[arm].reindex(targets)
        complete = values.notna().all()
        records.append({"model": arm, "available_targets": int(values.notna().sum()),
                        "expected_targets": len(targets), "complete": complete,
                        "mean_transfer_auc_pct": values.mean() if complete else float("nan")})
    means = pd.DataFrame(records)
    means.to_csv(output / "transfer_means.csv", index=False)
    print("Transfer test means (source graphs excluded; incomplete models have no mean):")
    print(means.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--prior-data-dir", type=Path, default=ROOT.parent / "async_convergence" / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    parser.add_argument("--auc-tolerance-pp", type=float, default=0.,
                        help="Source-validation retention tolerance in percentage points; default exact historical threshold.")
    args = parser.parse_args()
    if args.auc_tolerance_pp < 0:
        parser.error("--auc-tolerance-pp must be nonnegative")
    aggregate = args.data_dir / "aggregate.json"
    if not aggregate.exists():
        print(f"Waiting for measured extension data: {aggregate}")
        return
    data = read_json(aggregate)
    prior_path = args.prior_data_dir / "aggregate.json"
    prior = read_json(prior_path) if prior_path.exists() else {}
    refs = data.get("references") or prior.get("references")
    if not refs:
        raise ValueError("Historical selected singleton references are required for retention analysis.")
    if not any(data.get("history", {}).values()):
        print("No logged extension checkpoints yet; no tables or plots generated.")
        return
    output = args.output_dir / "data"
    figures = args.output_dir / "figures"
    output.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    tables(data, prior, refs, output, args.auc_tolerance_pp)
    curves(data, prior, refs, figures)
    transfer_tables(args.data_dir / "matrix.csv", args.prior_data_dir, output)
    print("Single-seed exploratory analysis. Source-validation candidate retention is not a guarantee,\n"
          "and repeatedly inspected transfer targets are not untouched confirmation data.")


if __name__ == "__main__":
    main()
