#!/usr/bin/env python3
"""Assemble, validate, summarize, and plot the matched three-seed flagship ladders."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

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
    r"^nmi_(?P<arm>baseline|objective|exposure|schedule|composition)_"
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


def validate_grid(frame: pd.DataFrame, targets: tuple[str, ...], task: str) -> None:
    keys = list(zip(frame.arm, frame.rung, frame.training_seed, frame.target))
    expected = {
        (arm, rung, seed, target)
        for arm in ARMS for rung in RUNGS for seed in SEEDS for target in targets
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
    expected_episodes = 512 if task == "NM" else 128
    if not (pd.to_numeric(frame.episodes) == expected_episodes).all():
        raise ValueError(f"{task} episode count drift")


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


def main() -> None:
    args = parse_args()
    nm = load_nm(args.seed0_nm, args.replicate_nm_root)
    validate_grid(nm, NM_TARGETS, "NM")
    cls = pd.read_csv(args.classification, sep="\t")
    if "target" not in cls:
        cls = cls.rename(columns={"dataset": "target", "episode_fingerprint": "fingerprint"})
    validate_grid(cls, CLS_TARGETS, "CLS")
    metrics = per_seed_metrics(nm, cls)
    summary = seed_summary(metrics)
    decisions = decision_table(metrics)

    data_dir = args.output_root / "data"
    figure_dir = args.output_root / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    nm.to_csv(data_dir / "nm_cells.csv", index=False)
    cls.to_csv(data_dir / "classification_cells.csv", index=False)
    metrics.to_csv(data_dir / "ladder_per_seed.csv", index=False)
    summary.to_csv(data_dir / "ladder_seed_summary.csv", index=False)
    decisions.to_csv(data_dir / "design_decisions.csv", index=False)
    plot_flagship(summary, figure_dir / "flagship_ladders.png")

    endpoint = decisions[decisions.rung == 8]
    winners = {
        metric: group.loc[group.winner_by_mean, "arm"].item()
        for metric, group in endpoint.groupby("metric")
    }
    payload = {
        "status": "complete",
        "nm_cells": len(nm),
        "classification_cells": len(cls),
        "training_seeds": list(SEEDS),
        "winners_at_rung8": winners,
        "claim_rule": (
            "Treat the balanced/interleaved/graph-local design as a universal winner only if it "
            "has the highest mean on both fixed panels and no alternative wins all three seeds "
            "on either panel; otherwise report a target-dependent or Pareto tradeoff."
        ),
    }
    (data_dir / "audit.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
