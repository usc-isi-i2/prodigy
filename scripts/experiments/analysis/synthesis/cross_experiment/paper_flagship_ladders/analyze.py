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
    result["arm"] = parts["arm"]
    result["rung"] = parts["rung"].astype(int)
    result["training_seed"] = parts["seed"].astype(int)
    return result


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
) -> None:
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
    drift = frame.groupby("target").fingerprint.nunique()
    if not (drift == 1).all():
        raise ValueError(f"{task} episode fingerprint drift: {drift[drift != 1].to_dict()}")
    if not (pd.to_numeric(frame.episodes) == expected_episodes).all():
        observed_episodes = sorted(pd.to_numeric(frame.episodes).unique().tolist())
        raise ValueError(
            f"{task} episode count drift: expected={expected_episodes} "
            f"observed={observed_episodes}"
        )


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


def scientific_decision(per_seed: pd.DataFrame, area_summary: pd.DataFrame) -> dict:
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
    if common_winner is not None:
        for metric in fixed_metrics:
            endpoint_wide = endpoint.pivot(index="training_seed", columns="arm", values=metric)
            alternatives_sweep_winner[metric] = sorted(
                arm for arm in ARMS
                if arm != common_winner
                and ((endpoint_wide[arm] - endpoint_wide[common_winner]) >= PRACTICAL_DELTA).all()
            )
    universal_winner = (
        common_winner
        if common_winner is not None
        and practical_margins_pass
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
    validate_grid(nm, NM_TARGETS, "NM", expected_episodes=512)
    validate_grid(
        capacity_nm,
        NM_TARGETS,
        "capacity NM",
        CAPACITY_ARMS,
        expected_episodes=512,
    )
    cls = pd.read_csv(args.classification, sep="\t")
    if "target" not in cls:
        cls = cls.rename(columns={"dataset": "target", "episode_fingerprint": "fingerprint"})
    validate_grid(cls, CLS_TARGETS, "CLS", expected_episodes=128)
    metrics = per_seed_metrics(nm, cls)
    summary = seed_summary(metrics)
    decisions = decision_table(metrics)
    area_per_seed = ladder_area_per_seed(metrics)
    area_summary = ladder_area_summary(area_per_seed)
    scientific_conclusion = scientific_decision(metrics, area_summary)
    capacity_metrics = capacity_per_seed(capacity_nm)
    capacity_summary = seed_summary(capacity_metrics)

    data_dir = args.output_root / "data"
    figure_dir = args.output_root / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    nm.to_csv(data_dir / "nm_cells.csv", index=False)
    capacity_nm.to_csv(data_dir / "capacity_nm_cells.csv", index=False)
    cls.to_csv(data_dir / "classification_cells.csv", index=False)
    metrics.to_csv(data_dir / "ladder_per_seed.csv", index=False)
    summary.to_csv(data_dir / "ladder_seed_summary.csv", index=False)
    decisions.to_csv(data_dir / "design_decisions.csv", index=False)
    area_per_seed.to_csv(data_dir / "ladder_area_per_seed.csv", index=False)
    area_summary.to_csv(data_dir / "ladder_area_summary.csv", index=False)
    capacity_metrics.to_csv(data_dir / "capacity_per_seed.csv", index=False)
    capacity_summary.to_csv(data_dir / "capacity_seed_summary.csv", index=False)
    plot_flagship(summary, figure_dir / "flagship_ladders.png")
    plot_capacity(capacity_summary, figure_dir / "capacity_ladder.png")

    payload = {
        "status": "complete",
        "nm_cells": len(nm),
        "classification_cells": len(cls),
        "capacity_nm_cells": len(capacity_nm),
        "training_seeds": list(SEEDS),
        "scientific_decision": scientific_conclusion,
        "claim_rule": (
            "Treat any design as a universal winner only if the same arm leads the mean endpoint "
            "and whole-ladder area on both fixed panels by at least 0.001 ROC-AUC and no "
            "alternative beats it by that margin in all three seeds at either endpoint; "
            "otherwise report a target-dependent or Pareto tradeoff."
        ),
    }
    (data_dir / "audit.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
