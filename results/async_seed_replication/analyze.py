"""Report locked-weight replications separately from seed-0 development.

All model/checkpoint choices come from the experiment's source-only manifests.
The stronger constituent singleton is a retrospective comparator, determined
from each seed's mean transfer test AUC; it is not a selection signal for KD.
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
TARGETS = ("covid19_twitter", "covid_political", "cp_hk_twitter", "midterm", "twibot20", "ukr_rus_suspended")
TARGET_NAMES = ("COVID", "COVID political", "Hong Kong", "Midterm", "TwiBot-20", "Ukraine suspended")
ARMS = ("kd_w010_selected", "kd_w100_selected", "kd_w010_fixed", "kd_w100_fixed")
SINGLETONS = {SOURCES[0]: "singleton_ukraine", SOURCES[1]: "singleton_facebook"}
COMPARISONS = (
    ("selected_vs_stronger_singleton", "kd_w010_selected", "stronger_singleton"),
    ("selected_vs_ukraine_singleton", "kd_w010_selected", "singleton_ukraine"),
    ("selected_vs_facebook_singleton", "kd_w010_selected", "singleton_facebook"),
    ("selected_vs_weight_one", "kd_w010_selected", "kd_w100_selected"),
    ("fixed_vs_stronger_singleton", "kd_w010_fixed", "stronger_singleton"),
    ("fixed_vs_ukraine_singleton", "kd_w010_fixed", "singleton_ukraine"),
    ("fixed_vs_facebook_singleton", "kd_w010_fixed", "singleton_facebook"),
    ("fixed_vs_weight_one", "kd_w010_fixed", "kd_w100_fixed"),
)
COMPARE_NAMES = {"selected_vs_stronger_singleton": "0.1 selected − stronger singleton",
                 "selected_vs_weight_one": "0.1 selected − 1.0 selected"}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def phase(seed: int) -> str:
    return "development" if seed == 0 else "replication"


def measured_matrix(data_dir: Path, development_dir: Path, singleton_dir: Path) -> pd.DataFrame:
    pieces = []
    current_path = data_dir / "matrix.csv"
    if current_path.exists():
        frame = pd.read_csv(current_path)
        if "seed" not in frame:
            raise ValueError("Replication matrix must identify each training seed explicitly")
        pieces.append(frame)
    development_path = development_dir / "matrix.csv"
    if development_path.exists():
        frame = pd.read_csv(development_path)
        frame = frame[frame.arm.isin(ARMS)].copy()
        frame["seed"] = 0
        pieces.append(frame)
    singleton_path = singleton_dir / "singleton_baselines.csv"
    if singleton_path.exists():
        frame = pd.read_csv(singleton_path)
        frame = frame[frame.source.isin(SOURCES)].copy()
        frame["arm"] = frame.source.map(SINGLETONS)
        frame["seed"] = 0
        frame = frame.rename(columns={"auc": "roc_auc"})
        pieces.append(frame)
    if not pieces:
        return pd.DataFrame()
    matrix = pd.concat(pieces, ignore_index=True)
    matrix["seed"] = matrix.seed.astype(int)
    required = ["seed", "arm", "target", "roc_auc"]
    if matrix.duplicated(required[:3]).any():
        raise ValueError("Duplicate seed/model/target evaluations; resolve provenance before analysis")
    matrix["phase"] = matrix.seed.map(phase)
    return matrix


def validation_table(aggregate: dict, output: Path) -> None:
    records = []
    for seed, trial in aggregate.get("seeds", {}).items():
        manifest = trial.get("manifest", {})
        locked_weight = manifest.get("locked_weight")
        if locked_weight is not None and float(locked_weight) != .1:
            raise ValueError(f"Seed {seed} manifest does not retain the locked weight 0.1")
        baselines = manifest.get("singleton_source_auc", {})
        for model in manifest.get("models", []):
            if "validation" not in model:
                continue
            validation = model["validation"]
            counts = model.get("counts", {})
            if isinstance(validation, dict):
                # Singleton exports contain only their own source, keyed by name.
                # Never invent a singleton's validation on the other source.
                measurements = [(source, validation[source], counts.get(source, {}))
                                for source in SOURCES if source in validation]
            else:
                measurements = [(source, validation[i], counts[i] if isinstance(counts, list)
                                 else counts.get(source, {}))
                                for i, source in enumerate(SOURCES) if i < len(validation)]
            for source, val, count in measurements:
                threshold = baselines.get(source)
                records.append({
                    "seed": int(seed), "phase": phase(int(seed)), "run_id": model["run_id"],
                    "arm": model.get("arm"), "weight": model.get("weight"),
                    "selection": model.get("selection"), "locked_weight": locked_weight,
                    "additional_step": model.get("additional_step"), "source": source,
                    "validation_auc_pct": val["auc"] * 100, "validation_bce": val["bce"],
                    "own_singleton_validation_auc_pct": threshold * 100 if threshold is not None else None,
                    "delta_own_singleton_pp": (val["auc"] - threshold) * 100 if threshold is not None else None,
                    **count,
                })
    if records:
        pd.DataFrame(records).to_csv(output / "source_validation_checkpoints.csv", index=False)


def comparison_tables(matrix: pd.DataFrame, output: Path, replication_seeds: list[int]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mean_rows, comparisons, target_deltas, source_rows = [], [], [], []
    for seed, seed_rows in matrix.groupby("seed", sort=True):
        pivot = seed_rows.pivot(index="target", columns="arm", values="roc_auc")
        means = {}
        for arm in pivot:
            values = pivot[arm].reindex(TARGETS)
            complete = values.notna().all()
            means[arm] = float(values.mean()) if complete else np.nan
            mean_rows.append({"seed": seed, "phase": phase(seed), "arm": arm,
                              "available_targets": int(values.notna().sum()), "expected_targets": len(TARGETS),
                              "complete": complete, "mean_transfer_auc_pct": means[arm] * 100})
        # Do not pick a comparator from incomplete evaluation coverage.
        stronger = None
        if all(np.isfinite(means.get(arm, np.nan)) for arm in SINGLETONS.values()):
            stronger = max(SINGLETONS.values(), key=lambda arm: means[arm])
        for label, arm, requested_baseline in COMPARISONS:
            baseline = stronger if requested_baseline == "stronger_singleton" else requested_baseline
            if baseline is None or arm not in pivot or baseline not in pivot:
                continue
            deltas = (pivot[arm] - pivot[baseline]).reindex(TARGETS) * 100
            complete = deltas.notna().all()
            comparisons.append({"seed": seed, "phase": phase(seed), "comparison": label,
                                "arm": arm, "baseline": baseline,
                                "baseline_selection": ("retrospective per-seed mean transfer AUC" if requested_baseline == "stronger_singleton"
                                                       else "prespecified same-seed constituent singleton" if baseline in SINGLETONS.values()
                                                       else "prespecified weight-1 control"),
                                "available_targets": int(deltas.notna().sum()), "expected_targets": len(TARGETS),
                                "complete": complete, "mean_delta_pp": deltas.mean() if complete else np.nan,
                                "positive_target_deltas": int((deltas > 0).sum()) if complete else None})
            for target, value in deltas.items():
                target_deltas.append({"seed": seed, "phase": phase(seed), "comparison": label,
                                      "baseline": baseline, "target": target, "delta_auc_pp": value})
        for source, singleton in SINGLETONS.items():
            if source not in pivot.index or singleton not in pivot:
                continue
            own = pivot.loc[source, singleton]
            for arm in ARMS:
                if arm not in pivot:
                    continue
                value = pivot.loc[source, arm]
                source_rows.append({"seed": seed, "phase": phase(seed), "arm": arm, "source": source,
                                    "source_test_auc_pct": value * 100, "own_singleton_test_auc_pct": own * 100,
                                    "delta_own_singleton_test_pp": (value - own) * 100,
                                    "observed_nonnegative_delta": bool(value >= own) if pd.notna(value) and pd.notna(own) else None})
    means_df, comparisons_df, targets_df = pd.DataFrame(mean_rows), pd.DataFrame(comparisons), pd.DataFrame(target_deltas)
    means_df.to_csv(output / "per_seed_transfer_means.csv", index=False)
    comparisons_df.to_csv(output / "per_seed_comparisons.csv", index=False)
    targets_df.to_csv(output / "target_deltas.csv", index=False)
    pd.DataFrame(source_rows).to_csv(output / "source_test_retention.csv", index=False)
    if not comparisons_df.empty:
        replication = comparisons_df[comparisons_df.seed.isin(replication_seeds) & comparisons_df.complete]
        summaries = []
        for label, rows in replication.groupby("comparison"):
            complete = set(rows.seed) == set(replication_seeds)
            values = rows.mean_delta_pp
            summaries.append({"comparison": label, "observed_training_seeds": len(rows),
                              "expected_training_seeds": len(replication_seeds), "complete": complete,
                              "mean_seed_delta_pp": values.mean() if complete else np.nan,
                              "minimum_observed_seed_delta_pp": values.min(), "maximum_observed_seed_delta_pp": values.max(),
                              "positive_seed_deltas": int((values > 0).sum()),
                              "interpretation": "descriptive across training seeds; no confidence interval or significance claim"})
        pd.DataFrame(summaries).to_csv(output / "replication_summary.csv", index=False)
        print("Per-seed transfer comparisons; seed 0 is development:")
        print(comparisons_df[["seed", "phase", "comparison", "baseline", "mean_delta_pp", "complete"]].to_string(index=False))
        if summaries:
            print("Replication-only descriptive summary (targets are not independent seeds):")
            print(pd.DataFrame(summaries).drop(columns="interpretation").to_string(index=False))
    return means_df, comparisons_df, targets_df


def save(fig, figures: Path, name: str) -> None:
    for suffix in ("png", "pdf"):
        fig.savefig(figures / f"{name}.{suffix}", dpi=180)
    plt.close(fig)


def figures(means: pd.DataFrame, comparisons: pd.DataFrame, targets: pd.DataFrame, output: Path) -> None:
    if means.empty or comparisons.empty:
        return
    seeds = sorted(means.seed.unique())
    fig, axes = plt.subplots(1, 2, figsize=(12.1, 5.4), layout="constrained")
    styles = (("kd_w010_selected", "Locked weight 0.1", "#087c80", -.2),
              ("kd_w100_selected", "Weight 1 control", "#965c99", 0),
              ("stronger_singleton", "Stronger constituent singleton", "#646b73", .2))
    for arm, label, color, offset in styles:
        xs, ys = [], []
        for index, seed in enumerate(seeds):
            actual_arm = arm
            if arm == "stronger_singleton":
                rows = comparisons[(comparisons.seed == seed) & (comparisons.comparison == "selected_vs_stronger_singleton")]
                if rows.empty:
                    continue
                actual_arm = rows.iloc[0].baseline
            rows = means[(means.seed == seed) & (means.arm == actual_arm) & means.complete]
            if rows.empty:
                continue
            xs.append(index + offset)
            ys.append(rows.iloc[0].mean_transfer_auc_pct)
        axes[0].scatter(xs, ys, color=color, s=58, label=label, zorder=5)
    for offset, label, color in [(-.18, "selected_vs_stronger_singleton", "#087c80"),
                                  (.18, "selected_vs_weight_one", "#965c99")]:
        for index, seed in enumerate(seeds):
            rows = comparisons[(comparisons.seed == seed) & (comparisons.comparison == label) & comparisons.complete]
            if rows.empty:
                continue
            axes[1].bar(index + offset, rows.iloc[0].mean_delta_pp, width=.32, color=color,
                        label=COMPARE_NAMES[label] if index == 0 else None, hatch="//" if seed == 0 else None)
    for ax in axes:
        ax.set_xticks(range(len(seeds)), [f"Seed {s}\n{'development' if s == 0 else 'replication'}" for s in seeds])
        if 0 in seeds:
            index = seeds.index(0)
            ax.axvspan(index - .46, index + .46, color="#e5e7e9", alpha=.5, zorder=0)
        ax.grid(axis="y", alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
    axes[0].set(title="Paired results for each training seed", ylabel="Mean test AUC on six transfer graphs (%)")
    axes[1].axhline(0, color="#555555", lw=.8)
    axes[1].set(title="Within-seed mean differences", ylabel="AUC difference (percentage points)")
    fig.suptitle("Locked Facebook KD weight 0.1: development and replication kept separate\n"
                 "Checkpoints selected by source validation. Stronger singleton is a retrospective comparator.\n"
                 "Summaries use training seeds; six graph targets are not six seeds.", fontsize=11)
    save(fig, output, "paired_seed_comparison")

    labels = list(COMPARE_NAMES)
    rows, row_labels = [], []
    for seed in seeds:
        for label in labels:
            subset = targets[(targets.seed == seed) & (targets.comparison == label)]
            if subset.empty:
                continue
            rows.append(subset.set_index("target").delta_auc_pp.reindex(TARGETS).to_numpy())
            row_labels.append(f"Seed {seed} ({'development' if seed == 0 else 'replication'})\n{COMPARE_NAMES[label]}")
    if not rows:
        return
    values = np.asarray(rows)
    limit = max(float(np.nanmax(np.abs(values))) if np.isfinite(values).any() else .01, .01)
    cmap = plt.get_cmap("RdBu").copy()
    cmap.set_bad("#dddddd")
    fig, ax = plt.subplots(figsize=(12.3, max(4.5, len(rows) * .78)), layout="constrained")
    im = ax.imshow(values, cmap=cmap, vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(len(TARGETS)), TARGET_NAMES, rotation=25, ha="right")
    ax.set_yticks(range(len(rows)), row_labels, fontsize=9)
    for i in range(len(rows)):
        for j in range(len(TARGETS)):
            value = values[i, j]
            ax.text(j, i, f"{value:+.2f}" if np.isfinite(value) else "pending", ha="center", va="center",
                    color="white" if np.isfinite(value) and abs(value) > .58 * limit else "#222222", fontsize=10)
    fig.colorbar(im, ax=ax, label="Test AUC difference (percentage points)", shrink=.85)
    ax.set_title("Per-target transfer differences within each training seed\nBlue: weight 0.1 higher. Red: comparator higher. Source graphs excluded.", fontsize=11)
    save(fig, output, "target_delta_heatmap")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--development-data-dir", type=Path, default=ROOT.parent / "async_kd_weights" / "data")
    parser.add_argument("--singleton-data-dir", type=Path, default=ROOT.parent / "async_convergence" / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    parser.add_argument("--replication-seeds", nargs="+", type=int, default=[1, 2])
    args = parser.parse_args()
    if 0 in args.replication_seeds or len(set(args.replication_seeds)) != len(args.replication_seeds):
        parser.error("Replication seeds must be unique and exclude development seed 0")
    aggregate_path = args.data_dir / "aggregate.json"
    if not aggregate_path.exists() and not (args.data_dir / "matrix.csv").exists():
        print(f"Waiting for measured seed-replication exports in {args.data_dir}")
        return
    aggregate = read_json(aggregate_path) if aggregate_path.exists() else {}
    output, plots = args.output_dir / "data", args.output_dir / "figures"
    output.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)
    validation_table(aggregate, output)
    matrix = measured_matrix(args.data_dir, args.development_data_dir, args.singleton_data_dir)
    if matrix.empty:
        print("Source-validation export available; transfer evaluations are pending.")
        return
    means, comparisons, targets = comparison_tables(matrix, output, args.replication_seeds)
    figures(means, comparisons, targets, plots)
    required_arms = set(ARMS) | set(SINGLETONS.values())
    observed = {seed for seed in args.replication_seeds
                if required_arms <= set(means[(means.seed == seed) & means.complete].arm)}
    pending = sorted(set(args.replication_seeds) - observed)
    if pending:
        print(f"Replication seeds with incomplete or unavailable prescribed evaluations: {pending}")
    print("Source validation and source test retention are separate tables.\n"
          "Seed 0 is development only; replication summaries exclude it and do not treat targets as seeds.\n"
          "Checkpoint selection is source-only; comparisons to stronger singletons are labelled retrospective.")


if __name__ == "__main__":
    main()
