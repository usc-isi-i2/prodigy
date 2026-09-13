#!/usr/bin/env python3
"""Assemble, validate, summarize, and plot the matched three-seed flagship ladders."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]
SEED0_NM = (
    ROOT
    / "scripts/experiments/analysis/transfer/ablations/prodigy_nm/"
    "nm_interventions_overnight/data/nm_results.csv"
)
ARMS = ("baseline", "objective", "exposure", "schedule", "composition")
CAPACITY_ARMS = ("baseline", "capacity")
RUNGS = tuple(range(1, 9))
SEEDS = (0, 1, 2)
NM_TARGETS = (
    "ukr_rus", "covid", "midterm", "covid_political", "election2020",
    "ukr_rus_suspended", "cp_hk", "facebook_page_reference", "twibot20",
)
SOURCE_ORDER = NM_TARGETS[:-1]
CLS_TARGETS = (
    "covid_political", "election2020", "facebook_page_reference", "twibot20",
    "ukr_rus_suspended",
)
MODEL_RE = re.compile(
    r"^nmi_(?P<arm>baseline|objective|exposure|schedule|composition|capacity)_"
    r"r(?P<rung>[1-8])_s(?P<seed>[0-2])$"
)
DISPLAY = {
    "baseline": "Balanced / interleaved / graph-local",
    "objective": "+ auxiliary reconstruction",
    "exposure": "Size-proportional exposure",
    "schedule": "Blocked schedule",
    "composition": "Cross-graph episodes",
}
COLORS = {
    "baseline": "#006D77",
    "objective": "#E76F51",
    "exposure": "#3A86FF",
    "schedule": "#8338EC",
    "composition": "#A6761D",
}
PRACTICAL_DELTA = 0.001


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed0-nm", type=Path, default=SEED0_NM)
    parser.add_argument("--replicate-nm-root", type=Path, required=True)
    parser.add_argument("--classification", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=HERE)
    return parser.parse_args()


def annotate_models(frame: pd.DataFrame) -> pd.DataFrame:
    parts = frame["model_id"].str.extract(MODEL_RE)
    if parts.isna().any().any():
        bad = frame.loc[parts.isna().any(axis=1), "model_id"].unique().tolist()
        raise ValueError(f"unrecognized flagship model ids: {bad[:10]}")
    result = frame.copy()
    parsed_arm = parts["arm"]
    parsed_rung = parts["rung"].astype(int)
    parsed_seed = parts["seed"].astype(int)
    declared = {
        "arm": parsed_arm,
        "rung": parsed_rung,
        "training_seed": parsed_seed,
        "seed": parsed_seed,
    }
    for column, expected in declared.items():
        if column not in result:
            continue
        observed = result[column]
        if column in {"rung", "training_seed", "seed"}:
            observed = pd.to_numeric(observed, errors="coerce")
        else:
            observed = observed.astype(str)
        mismatch = observed.ne(expected)
        if mismatch.any():
            bad = result.loc[mismatch, ["model_id", column]].head(10).to_dict("records")
            raise ValueError(f"{column} disagrees with model id: {bad}")
    result["arm"] = parsed_arm
    result["rung"] = parsed_rung
    result["training_seed"] = parsed_seed
    return result


def source_tuple(value: object) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(str(item).strip() for item in value)
    return tuple(part.strip() for part in str(value).split(",") if part.strip())


def load_nm(seed0_path: Path, replicate_root: Path) -> pd.DataFrame:
    seed0 = pd.read_csv(seed0_path)
    seed0 = seed0[seed0["model_id"].str.fullmatch(MODEL_RE)].copy()
    rows = []
    for path in sorted(replicate_root.glob("*/*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if MODEL_RE.fullmatch(str(row.get("model_id", ""))):
            rows.append(row)
    replicas = pd.DataFrame(rows)
    if replicas.empty:
        raise ValueError(f"no flagship NM cells found beneath {replicate_root}")
    frame = pd.concat([seed0, replicas], ignore_index=True, sort=False)
    frame = annotate_models(frame)
    frame["roc_auc"] = pd.to_numeric(frame["roc_auc"])
    frame["sources"] = frame["sources"].map(
        lambda value: ",".join(value) if isinstance(value, list) else str(value)
    )
    return frame


def validate_grid(
    frame: pd.DataFrame,
    targets: tuple[str, ...],
    task: str,
    arms: tuple[str, ...] = ARMS,
    *,
    expected_episodes: int,
    expected_protocol: str | None = None,
) -> None:
    required = {
        "model_id", "arm", "rung", "training_seed", "target", "roc_auc",
        "fingerprint", "episodes", "sources", "checkpoint", "checkpoint_step",
        "checkpoint_sha256", "training_revision",
    }
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        raise ValueError(f"missing {task} provenance columns: {missing_columns}")
    keys = list(zip(frame.arm, frame.rung, frame.training_seed, frame.target))
    expected = {
        (arm, rung, seed, target)
        for arm in arms for rung in RUNGS for seed in SEEDS for target in targets
    }
    observed = set(keys)
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        raise ValueError(f"duplicate {task} cells: {duplicates[:10]}")
    if observed != expected:
        raise ValueError(
            f"{task} coverage mismatch: missing={sorted(expected-observed)[:10]} "
            f"extra={sorted(observed-expected)[:10]}"
        )
    if not np.isfinite(frame.roc_auc).all() or not frame.roc_auc.between(0, 1).all():
        raise ValueError(f"invalid {task} ROC-AUC values")
    expected_sources = frame.rung.map(
        lambda rung: SOURCE_ORDER[: int(rung)]
    )
    observed_sources = frame.sources.map(source_tuple)
    source_mismatch = observed_sources.ne(expected_sources)
    if source_mismatch.any():
        bad = frame.loc[source_mismatch, ["model_id", "sources"]].head(10).to_dict("records")
        raise ValueError(f"{task} source-set mismatch: {bad}")
    if not frame.checkpoint.astype(str).str.strip().ne("").all():
        raise ValueError(f"missing {task} checkpoint paths")
    if not frame.checkpoint_sha256.astype(str).str.fullmatch(r"[0-9a-f]{64}").all():
        raise ValueError(f"invalid {task} checkpoint hashes")
    if not frame.training_revision.astype(str).str.fullmatch(r"[0-9a-f]{7,40}").all():
        raise ValueError(f"invalid {task} training revisions")
    checkpoint_steps = pd.to_numeric(frame.checkpoint_step, errors="coerce")
    if not np.isfinite(checkpoint_steps).all() or not (checkpoint_steps > 0).all():
        raise ValueError(f"invalid {task} checkpoint steps")
    if expected_protocol is not None:
        if "protocol" not in frame:
            raise ValueError(f"missing {task} protocol column")
        protocols = sorted(frame.protocol.astype(str).unique().tolist())
        if protocols != [expected_protocol]:
            raise ValueError(
                f"{task} protocol drift: expected={expected_protocol} observed={protocols}"
            )
    drift = frame.groupby("target").fingerprint.nunique()
    if not (drift == 1).all():
        raise ValueError(f"{task} episode fingerprint drift: {drift[drift != 1].to_dict()}")
    if not (pd.to_numeric(frame.episodes) == expected_episodes).all():
        observed_episodes = sorted(pd.to_numeric(frame.episodes).unique().tolist())
        raise ValueError(
            f"{task} episode count drift: expected={expected_episodes} "
            f"observed={observed_episodes}"
        )


def validate_cross_task_models(nm: pd.DataFrame, cls: pd.DataFrame) -> pd.DataFrame:
    """Require NM and classification to evaluate the same frozen model states."""
    fields = (
        "checkpoint_sha256", "checkpoint_step", "training_revision", "sources"
    )

    def one_row_per_model(frame: pd.DataFrame, task: str) -> pd.DataFrame:
        for field in fields:
            drift = frame.groupby("model_id")[field].nunique()
            if not (drift == 1).all():
                raise ValueError(
                    f"{task} per-model {field} drift: "
                    f"{drift[drift != 1].head(10).to_dict()}"
                )
        result = frame[["model_id", *fields]].drop_duplicates("model_id").copy()
        result["sources"] = result.sources.map(lambda value: ",".join(source_tuple(value)))
        return result

    nm_models = one_row_per_model(nm, "NM")
    cls_models = one_row_per_model(cls, "CLS")
    expected_models = len(ARMS) * len(RUNGS) * len(SEEDS)
    if len(nm_models) != expected_models or len(cls_models) != expected_models:
        raise ValueError(
            "cross-task physical-model coverage mismatch: "
            f"NM={len(nm_models)} CLS={len(cls_models)} expected={expected_models}"
        )
    paired = nm_models.merge(
        cls_models,
        on="model_id",
        suffixes=("_nm", "_cls"),
        validate="one_to_one",
    )
    if len(paired) != expected_models:
        missing_nm = sorted(set(cls_models.model_id) - set(nm_models.model_id))
        missing_cls = sorted(set(nm_models.model_id) - set(cls_models.model_id))
        raise ValueError(
            f"cross-task model-id mismatch: missing_nm={missing_nm[:10]} "
            f"missing_cls={missing_cls[:10]}"
        )
    for field in fields:
        mismatch = paired[f"{field}_nm"].astype(str).ne(
            paired[f"{field}_cls"].astype(str)
        )
        if mismatch.any():
            bad = paired.loc[
                mismatch,
                ["model_id", f"{field}_nm", f"{field}_cls"],
            ].head(10).to_dict("records")
            raise ValueError(f"cross-task {field} mismatch: {bad}")
    return paired


def per_seed_metrics(nm: pd.DataFrame, cls: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (arm, rung, seed), group in nm.groupby(["arm", "rung", "training_seed"]):
        sources = set(group.iloc[0].sources.split(","))
        source_views = group[group.target.isin(sources)]
        future_views = group[(group.target != "twibot20") & ~group.target.isin(sources)]
        row = {
            "arm": arm,
            "rung": int(rung),
            "training_seed": int(seed),
            "nm_fixed_panel": group.roc_auc.mean(),
            "nm_included": source_views.roc_auc.mean(),
            "nm_future_sources": future_views.roc_auc.mean() if len(future_views) else np.nan,
            "nm_heldout_twibot": group.loc[group.target == "twibot20", "roc_auc"].item(),
        }
        cls_group = cls[
            (cls.arm == arm) & (cls.rung == rung) & (cls.training_seed == seed)
        ]
        row["cls_fixed_panel"] = cls_group.roc_auc.mean()
        rows.append(row)
    result = pd.DataFrame(rows)
    if len(result) != len(ARMS) * len(RUNGS) * len(SEEDS):
        raise ValueError("per-seed metric coverage mismatch")
    return result


def seed_summary(per_seed: pd.DataFrame) -> pd.DataFrame:
    value_columns = [column for column in per_seed if column.startswith(("nm_", "cls_"))]
    rows = []
    for (arm, rung), group in per_seed.groupby(["arm", "rung"], sort=False):
        for metric in value_columns:
            values = group[metric].dropna().to_numpy(float)
            rows.append(
                {
                    "arm": arm,
                    "rung": int(rung),
                    "metric": metric,
                    "seed_count": len(values),
                    "mean": float(values.mean()) if len(values) else np.nan,
                    "min": float(values.min()) if len(values) else np.nan,
                    "max": float(values.max()) if len(values) else np.nan,
                    "sd": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def decision_table(per_seed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in ("nm_fixed_panel", "cls_fixed_panel", "nm_heldout_twibot"):
        for rung in RUNGS:
            wide = per_seed[per_seed.rung == rung].pivot(
                index="training_seed", columns="arm", values=metric
            )
            means = wide.mean()
            winner = means.idxmax()
            for arm in ARMS:
                delta = wide[arm] - wide["baseline"]
                rows.append(
                    {
                        "metric": metric,
                        "rung": rung,
                        "arm": arm,
                        "mean": means[arm],
                        "delta_vs_balanced_mean": delta.mean(),
                        "seed_wins_vs_balanced": int((delta > 0).sum()),
                        "winner_by_mean": arm == winner,
                    }
                )
    return pd.DataFrame(rows)


def ladder_area_per_seed(per_seed: pd.DataFrame) -> pd.DataFrame:
    """Summarize the entire eight-rung curve without pooling training seeds."""
    metrics = (
        "nm_fixed_panel", "cls_fixed_panel", "nm_included",
        "nm_future_sources", "nm_heldout_twibot",
    )
    rows = []
    for (arm, seed), group in per_seed.groupby(["arm", "training_seed"]):
        ordered = group.sort_values("rung")
        if tuple(ordered.rung) != RUNGS:
            raise ValueError(f"incomplete ladder for area: arm={arm} seed={seed}")
        for metric in metrics:
            values = ordered[metric].to_numpy(float)
            finite = np.isfinite(values)
            if not finite.all():
                # The future-source set is empty at the final rung by definition.
                values = values[finite]
            if len(values) < 2:
                raise ValueError(f"insufficient finite ladder points for {arm}/{seed}/{metric}")
            normalized_area = float((values[0] / 2 + values[1:-1].sum() + values[-1] / 2) / (len(values) - 1))
            rows.append(
                {
                    "arm": arm,
                    "training_seed": int(seed),
                    "metric": metric,
                    "normalized_ladder_area": normalized_area,
                    "rungs_in_area": int(len(values)),
                }
            )
    result = pd.DataFrame(rows)
    expected = len(ARMS) * len(SEEDS) * len(metrics)
    if len(result) != expected:
        raise ValueError(f"ladder-area coverage mismatch: {len(result)} != {expected}")
    return result


def ladder_area_summary(per_seed_area: pd.DataFrame) -> pd.DataFrame:
    baseline = per_seed_area[per_seed_area.arm.eq("baseline")][
        ["training_seed", "metric", "normalized_ladder_area"]
    ].rename(columns={"normalized_ladder_area": "baseline_area"})
    paired = per_seed_area.merge(
        baseline, on=["training_seed", "metric"], validate="many_to_one"
    )
    paired["delta_vs_baseline"] = paired.normalized_ladder_area - paired.baseline_area
    return (
        paired.groupby(["metric", "arm"], as_index=False)
        .agg(
            training_seeds=("training_seed", "nunique"),
            mean=("normalized_ladder_area", "mean"),
            min=("normalized_ladder_area", "min"),
            max=("normalized_ladder_area", "max"),
            sd=("normalized_ladder_area", "std"),
            delta_vs_baseline_mean=("delta_vs_baseline", "mean"),
            seed_wins_vs_baseline=("delta_vs_baseline", lambda values: int((values > 0).sum())),
        )
    )


def target_diagnostics(
    nm: pd.DataFrame, cls: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Keep target-level paired effects visible instead of hiding them in macros."""
    rows = []
    for task, frame in (("NM", nm), ("CLS", cls)):
        baseline = frame[frame.arm.eq("baseline")][
            ["rung", "training_seed", "target", "roc_auc"]
        ].rename(columns={"roc_auc": "baseline_roc_auc"})
        rung_one = frame[frame.rung.eq(1)][
            ["arm", "training_seed", "target", "roc_auc"]
        ].rename(columns={"roc_auc": "rung1_roc_auc"})
        paired = frame.merge(
            baseline,
            on=["rung", "training_seed", "target"],
            validate="many_to_one",
        ).merge(
            rung_one,
            on=["arm", "training_seed", "target"],
            validate="many_to_one",
        )
        paired["task"] = task
        paired["delta_vs_baseline"] = paired.roc_auc - paired.baseline_roc_auc
        paired["delta_vs_rung1"] = paired.roc_auc - paired.rung1_roc_auc
        rows.append(
            paired[
                [
                    "task", "arm", "rung", "training_seed", "target", "roc_auc",
                    "baseline_roc_auc", "rung1_roc_auc", "delta_vs_baseline",
                    "delta_vs_rung1",
                ]
            ]
        )
    per_seed = pd.concat(rows, ignore_index=True)
    per_target = (
        per_seed.groupby(["task", "arm", "rung", "target"], as_index=False)
        .agg(
            training_seeds=("training_seed", "nunique"),
            delta_vs_baseline_mean=("delta_vs_baseline", "mean"),
            delta_vs_baseline_min=("delta_vs_baseline", "min"),
            delta_vs_baseline_max=("delta_vs_baseline", "max"),
            delta_vs_rung1_mean=("delta_vs_rung1", "mean"),
            delta_vs_rung1_min=("delta_vs_rung1", "min"),
            delta_vs_rung1_max=("delta_vs_rung1", "max"),
        )
    )
    endpoint_rows = []
    endpoint = per_target[per_target.rung.eq(max(RUNGS))]
    for (task, arm), group in endpoint.groupby(["task", "arm"]):
        worst_baseline = group.loc[group.delta_vs_baseline_mean.idxmin()]
        worst_rung1 = group.loc[group.delta_vs_rung1_mean.idxmin()]
        endpoint_rows.append(
            {
                "task": task,
                "arm": arm,
                "targets": len(group),
                "targets_improved_vs_baseline": int(
                    (group.delta_vs_baseline_mean > 0).sum()
                ),
                "fraction_targets_improved_vs_baseline": float(
                    (group.delta_vs_baseline_mean > 0).mean()
                ),
                "worst_target_vs_baseline": worst_baseline.target,
                "worst_target_delta_vs_baseline_mean": float(
                    worst_baseline.delta_vs_baseline_mean
                ),
                "targets_materially_regressed_vs_baseline": int(
                    (group.delta_vs_baseline_mean < -PRACTICAL_DELTA).sum()
                ),
                "targets_improved_from_rung1": int(
                    (group.delta_vs_rung1_mean > 0).sum()
                ),
                "fraction_targets_improved_from_rung1": float(
                    (group.delta_vs_rung1_mean > 0).mean()
                ),
                "worst_target_from_rung1": worst_rung1.target,
                "worst_target_delta_from_rung1_mean": float(
                    worst_rung1.delta_vs_rung1_mean
                ),
            }
        )
    return per_seed, per_target, pd.DataFrame(endpoint_rows)


def scientific_decision(
    per_seed: pd.DataFrame,
    area_summary: pd.DataFrame,
    endpoint_targets: pd.DataFrame | None = None,
) -> dict:
    fixed_metrics = ("nm_fixed_panel", "cls_fixed_panel")
    endpoint = per_seed[per_seed.rung.eq(max(RUNGS))]
    endpoint_winners = {}
    area_winners = {}
    endpoint_winner_margins = {}
    area_winner_margins = {}
    for metric in fixed_metrics:
        endpoint_wide = endpoint.pivot(index="training_seed", columns="arm", values=metric)
        endpoint_means = endpoint_wide.mean().sort_values(ascending=False)
        endpoint_winners[metric] = endpoint_means.index[0]
        endpoint_winner_margins[metric] = float(endpoint_means.iloc[0] - endpoint_means.iloc[1])
        metric_area = area_summary[area_summary.metric.eq(metric)]
        area_means = metric_area.set_index("arm")["mean"].sort_values(ascending=False)
        area_winners[metric] = area_means.index[0]
        area_winner_margins[metric] = float(area_means.iloc[0] - area_means.iloc[1])

    all_winners = list(endpoint_winners.values()) + list(area_winners.values())
    common_winner = all_winners[0] if len(set(all_winners)) == 1 else None
    practical_margins_pass = all(
        margin >= PRACTICAL_DELTA
        for margin in (*endpoint_winner_margins.values(), *area_winner_margins.values())
    )
    alternatives_sweep_winner = {}
    target_safety = {}
    target_safety_pass = endpoint_targets is None
    if common_winner is not None:
        for metric in fixed_metrics:
            endpoint_wide = endpoint.pivot(index="training_seed", columns="arm", values=metric)
            alternatives_sweep_winner[metric] = sorted(
                arm for arm in ARMS
                if arm != common_winner
                and ((endpoint_wide[arm] - endpoint_wide[common_winner]) >= PRACTICAL_DELTA).all()
            )
        if endpoint_targets is not None:
            target_safety_pass = True
            for metric, task in (("nm_fixed_panel", "NM"), ("cls_fixed_panel", "CLS")):
                matched = endpoint_targets[
                    endpoint_targets.task.eq(task)
                    & endpoint_targets.arm.eq(common_winner)
                ]
                if len(matched) != 1:
                    raise ValueError(
                        f"missing endpoint target diagnostics for {common_winner}/{task}"
                    )
                row = matched.iloc[0]
                worst = float(row.worst_target_delta_vs_baseline_mean)
                target_safety[metric] = {
                    "worst_target": row.worst_target_vs_baseline,
                    "worst_target_delta_vs_baseline_mean": worst,
                    "materially_regressed_targets": int(
                        row.targets_materially_regressed_vs_baseline
                    ),
                }
                target_safety_pass &= worst >= -PRACTICAL_DELTA
    universal_winner = (
        common_winner
        if common_winner is not None
        and practical_margins_pass
        and target_safety_pass
        and not any(alternatives_sweep_winner.values())
        else None
    )
    return {
        "headline": f"{universal_winner}_universal_winner"
        if universal_winner is not None else "target_dependent_or_pareto_tradeoff",
        "universal_winner": universal_winner,
        "endpoint_winners": endpoint_winners,
        "endpoint_winner_margins": endpoint_winner_margins,
        "whole_ladder_area_winners": area_winners,
        "whole_ladder_area_winner_margins": area_winner_margins,
        "practical_delta": PRACTICAL_DELTA,
        "practical_margins_passed": practical_margins_pass,
        "target_safety": target_safety,
        "target_safety_passed": target_safety_pass,
        "alternatives_beating_common_winner_all_three_seeds_at_endpoint": alternatives_sweep_winner,
        "universal_winner_gate_passed": universal_winner is not None,
        "whole_ladder_estimand": (
            "Normalized trapezoidal area over rungs 1-8, computed within each training seed; "
            "future-source area uses its seven finite rungs."
        ),
    }


def plot_flagship(summary: pd.DataFrame, output: Path) -> None:
    plt.rcParams.update({"font.size": 8.5, "axes.titleweight": "bold"})
    fig, axes = plt.subplots(1, 2, figsize=(7.35, 3.05), sharex=True)
    panels = (
        ("nm_fixed_panel", "a  Structural transfer", "Mean ROC–AUC on 9 fixed NM receivers"),
        ("cls_fixed_panel", "b  Downstream transfer", "Mean ROC–AUC on 5 fixed labeled targets"),
    )
    for ax, (metric, title, ylabel) in zip(axes, panels):
        for arm in ARMS:
            curve = summary[(summary.arm == arm) & (summary.metric == metric)].sort_values("rung")
            x = curve.rung.to_numpy(float)
            mean = curve["mean"].to_numpy(float)
            lo = curve["min"].to_numpy(float)
            hi = curve["max"].to_numpy(float)
            ax.fill_between(x, lo, hi, color=COLORS[arm], alpha=0.12, linewidth=0)
            ax.plot(
                x, mean, color=COLORS[arm], marker="o", markersize=3.0,
                linewidth=2.2 if arm == "baseline" else 1.5, label=DISPLAY[arm],
            )
        ax.set_title(title, loc="left")
        ax.set_xlabel("Number of pretraining graphs")
        ax.set_ylabel(ylabel)
        ax.set_xticks(RUNGS)
        ax.grid(alpha=0.22, linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=3, frameon=False)
    fig.text(0.5, -0.015, "Lines are means across three matched training seeds; bands show the seed range.", ha="center")
    fig.tight_layout(rect=(0, 0.025, 1, 0.93), w_pad=2.0)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def capacity_per_seed(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (arm, rung, seed), group in frame.groupby(["arm", "rung", "training_seed"]):
        rows.append(
            {
                "arm": arm,
                "rung": int(rung),
                "training_seed": int(seed),
                "nm_fixed_panel": group.roc_auc.mean(),
            }
        )
    result = pd.DataFrame(rows)
    expected = len(CAPACITY_ARMS) * len(RUNGS) * len(SEEDS)
    if len(result) != expected:
        raise ValueError(f"capacity per-seed coverage mismatch: {len(result)} != {expected}")
    return result


def plot_capacity(summary: pd.DataFrame, output: Path) -> None:
    colors = {"baseline": COLORS["baseline"], "capacity": "#D62828"}
    labels = {"baseline": DISPLAY["baseline"], "capacity": "Wide encoder"}
    fig, ax = plt.subplots(figsize=(3.8, 3.0))
    for arm in CAPACITY_ARMS:
        curve = summary[(summary.arm == arm) & (summary.metric == "nm_fixed_panel")].sort_values("rung")
        x = curve.rung.to_numpy(float)
        ax.fill_between(
            x, curve["min"].to_numpy(float), curve["max"].to_numpy(float),
            color=colors[arm], alpha=0.13, linewidth=0,
        )
        ax.plot(x, curve["mean"], color=colors[arm], marker="o", markersize=3,
                linewidth=2, label=labels[arm])
    ax.set(
        xlabel="Number of pretraining graphs",
        ylabel="Mean ROC–AUC on 9 fixed NM receivers",
        title="Capacity diagnostic",
        xticks=RUNGS,
    )
    ax.grid(alpha=0.22, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    all_nm = load_nm(args.seed0_nm, args.replicate_nm_root)
    nm = all_nm[all_nm.arm.isin(ARMS)].copy()
    capacity_nm = all_nm[all_nm.arm.isin(CAPACITY_ARMS)].copy()
    validate_grid(
        nm,
        NM_TARGETS,
        "NM",
        expected_episodes=512,
        expected_protocol="nmi_fixed_nm_v1",
    )
    validate_grid(
        capacity_nm,
        NM_TARGETS,
        "capacity NM",
        CAPACITY_ARMS,
        expected_episodes=512,
        expected_protocol="nmi_fixed_nm_v1",
    )
    cls = pd.read_csv(args.classification, sep="\t")
    if "target" not in cls:
        cls = cls.rename(columns={"dataset": "target", "episode_fingerprint": "fingerprint"})
    cls = annotate_models(cls)
    validate_grid(cls, CLS_TARGETS, "CLS", expected_episodes=128)
    cross_task_models = validate_cross_task_models(nm, cls)
    metrics = per_seed_metrics(nm, cls)
    summary = seed_summary(metrics)
    decisions = decision_table(metrics)
    area_per_seed = ladder_area_per_seed(metrics)
    area_summary = ladder_area_summary(area_per_seed)
    target_per_seed, target_summary, endpoint_targets = target_diagnostics(nm, cls)
    scientific_conclusion = scientific_decision(
        metrics, area_summary, endpoint_targets
    )
    capacity_metrics = capacity_per_seed(capacity_nm)
    capacity_summary = seed_summary(capacity_metrics)

    data_dir = args.output_root / "data"
    figure_dir = args.output_root / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    nm.to_csv(data_dir / "nm_cells.csv", index=False)
    capacity_nm.to_csv(data_dir / "capacity_nm_cells.csv", index=False)
    cls.to_csv(data_dir / "classification_cells.csv", index=False)
    cross_task_models.to_csv(data_dir / "cross_task_model_provenance.csv", index=False)
    metrics.to_csv(data_dir / "ladder_per_seed.csv", index=False)
    summary.to_csv(data_dir / "ladder_seed_summary.csv", index=False)
    decisions.to_csv(data_dir / "design_decisions.csv", index=False)
    area_per_seed.to_csv(data_dir / "ladder_area_per_seed.csv", index=False)
    area_summary.to_csv(data_dir / "ladder_area_summary.csv", index=False)
    target_per_seed.to_csv(data_dir / "target_effects_per_seed.csv", index=False)
    target_summary.to_csv(data_dir / "target_effects_summary.csv", index=False)
    endpoint_targets.to_csv(data_dir / "endpoint_target_robustness.csv", index=False)
    capacity_metrics.to_csv(data_dir / "capacity_per_seed.csv", index=False)
    capacity_summary.to_csv(data_dir / "capacity_seed_summary.csv", index=False)
    plot_flagship(summary, figure_dir / "flagship_ladders.png")
    plot_capacity(capacity_summary, figure_dir / "capacity_ladder.png")

    payload = {
        "status": "complete",
        "nm_cells": len(nm),
        "classification_cells": len(cls),
        "capacity_nm_cells": len(capacity_nm),
        "cross_task_physical_models": len(cross_task_models),
        "target_effect_cells": len(target_per_seed),
        "endpoint_target_rows": len(endpoint_targets),
        "training_seeds": list(SEEDS),
        "scientific_decision": scientific_conclusion,
        "claim_rule": (
            "Treat any design as a universal winner only if the same arm leads the mean endpoint "
            "and whole-ladder area on both fixed panels by at least 0.001 ROC-AUC and no "
            "alternative beats it by that margin in all three seeds at either endpoint; "
            "its worst target must also avoid a mean regression greater than 0.001 against "
            "the matched baseline. Otherwise report a target-dependent or Pareto tradeoff."
        ),
    }
    (data_dir / "audit.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
