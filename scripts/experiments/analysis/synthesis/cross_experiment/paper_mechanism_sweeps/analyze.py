#!/usr/bin/env python3
"""Audit and analyze the matched ratio, data-scale, and capacity campaign."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ARMS = {
    "mix_p000": (0.00, 256),
    "mix_p010": (0.10, 256),
    "mix_p025": (0.25, 256),
    "mix_p050": (0.50, 256),
    "mix_p100": (1.00, 256),
    "wide_p000": (0.00, 512),
}
STEPS = (2_000, 4_000, 6_000, 8_000, 10_000)
SEEDS = (0, 1, 2)
NM_TARGETS = (
    "ukr_rus", "covid", "midterm", "covid_political", "election2020",
    "ukr_rus_suspended", "twibot20", "cp_hk", "facebook_page_reference",
)
CLS_TARGETS = (
    "covid_political", "election2020", "facebook_page_reference", "twibot20",
    "ukr_rus_suspended",
)
MODEL_RE = re.compile(
    r"^paper_mech_(?P<arm>mix_p000|mix_p010|mix_p025|mix_p050|mix_p100|wide_p000)_"
    r"step(?P<step>2000|4000|6000|8000|10000)_s(?P<seed>[0-2])$"
)
MARGIN = 0.001
EXPECTED_SOURCES = (
    "ukr_rus,covid,midterm,covid_political,election2020,ukr_rus_suspended,"
    "cp_hk,facebook_page_reference"
)


def annotate(frame: pd.DataFrame, task: str) -> pd.DataFrame:
    rows = []
    for row in frame.to_dict("records"):
        match = MODEL_RE.fullmatch(str(row.get("model_id", "")))
        if match is None:
            raise ValueError(f"unexpected mechanism model id: {row.get('model_id')}")
        arm = match["arm"]
        step = int(match["step"])
        seed = int(match["seed"])
        if int(row.get("checkpoint_step", -1)) != step:
            raise ValueError(f"checkpoint declaration mismatch for {row['model_id']}")
        declared_seed = row.get("seed", row.get("training_seed"))
        if int(declared_seed) != seed:
            raise ValueError(f"training seed declaration mismatch for {row['model_id']}")
        sources = row.get("sources")
        if isinstance(sources, list):
            row["sources"] = ",".join(str(source) for source in sources)
        elif not isinstance(sources, str):
            raise ValueError(f"invalid source declaration for {row['model_id']}")
        probability, emb_dim = ARMS[arm]
        row.update(task=task, arm=arm, checkpoint_step=step, training_seed=seed,
                   cross_graph_prob=probability, emb_dim=emb_dim)
        rows.append(row)
    return pd.DataFrame(rows)


def load_nm(root: Path) -> pd.DataFrame:
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(root.glob("cells/*/*.json"))]
    frame = annotate(pd.DataFrame(rows), "nm")
    expected = len(ARMS) * len(STEPS) * len(SEEDS) * len(NM_TARGETS)
    if len(frame) != expected:
        raise ValueError(f"NM coverage mismatch: {len(frame)} != {expected}")
    return frame.rename(columns={"target": "dataset"}) if "target" in frame else frame


def load_cls(path: Path) -> pd.DataFrame:
    frame = annotate(pd.read_csv(path, sep="\t"), "classification")
    expected = 5 * len(STEPS) * len(SEEDS) * len(CLS_TARGETS)
    if len(frame) != expected:
        raise ValueError(f"classification coverage mismatch: {len(frame)} != {expected}")
    return frame


def validate_grid(frame: pd.DataFrame, task: str, targets: tuple[str, ...], arms: tuple[str, ...]) -> None:
    required = {"model_id", "arm", "training_seed", "checkpoint_step", "dataset", "roc_auc",
                "checkpoint", "checkpoint_sha256", "training_revision", "sources"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{task} missing columns: {sorted(missing)}")
    keys = frame[["arm", "checkpoint_step", "training_seed", "dataset"]].apply(tuple, axis=1)
    if keys.duplicated().any():
        raise ValueError(f"duplicate {task} cells")
    expected = {
        (arm, step, seed, target)
        for arm in arms for step in STEPS for seed in SEEDS for target in targets
    }
    observed = set(keys)
    if observed != expected:
        raise ValueError(f"{task} grid drift: missing={len(expected-observed)} extra={len(observed-expected)}")
    metrics = pd.to_numeric(frame.roc_auc, errors="coerce")
    if not np.isfinite(metrics).all() or not metrics.between(0, 1).all():
        raise ValueError(f"invalid {task} ROC-AUC")
    if set(frame.sources) != {EXPECTED_SOURCES}:
        raise ValueError(f"{task} source-set drift: {sorted(set(frame.sources))}")
    if frame.training_revision.nunique() != 1:
        raise ValueError(f"{task} mixes training revisions")
    normalized_task = task.lower()
    for target, part in frame.groupby("dataset"):
        fingerprint = "fingerprint" if normalized_task == "nm" else "episode_fingerprint"
        if fingerprint not in part:
            raise ValueError(f"{task} missing episode fingerprint column: {fingerprint}")
        if part[fingerprint].nunique() != 1:
            raise ValueError(f"{task} episode drift for {target}")


def provenance_join(nm: pd.DataFrame, cls: pd.DataFrame) -> pd.DataFrame:
    fields = ["model_id", "checkpoint", "checkpoint_sha256", "checkpoint_step",
              "training_revision", "sources", "training_seed"]
    def collapse(frame: pd.DataFrame, label: str) -> pd.DataFrame:
        for field in fields[1:]:
            counts = frame.groupby("model_id")[field].nunique(dropna=False)
            bad = counts[counts.ne(1)]
            if not bad.empty:
                raise ValueError(f"{label} per-model {field} drift: {bad.index.tolist()[:5]}")
        return frame[fields].drop_duplicates("model_id")

    left = collapse(nm[nm.emb_dim.eq(256)], "NM")
    right = collapse(cls, "classification")
    if len(left) != 75 or len(right) != 75:
        raise ValueError(f"cross-task model coverage mismatch: NM={len(left)} CLS={len(right)}")
    joined = left.merge(right, on="model_id", suffixes=("_nm", "_cls"), validate="one_to_one")
    for field in fields[1:]:
        if not joined[f"{field}_nm"].astype(str).equals(joined[f"{field}_cls"].astype(str)):
            raise ValueError(f"cross-task provenance drift: {field}")
    return joined


def macro_tables(combined: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_seed = (
        combined.groupby(["task", "arm", "cross_graph_prob", "emb_dim", "checkpoint_step", "training_seed"],
                         as_index=False).roc_auc.mean()
    )
    summary = (
        per_seed.groupby(["task", "arm", "cross_graph_prob", "emb_dim", "checkpoint_step"])
        .roc_auc.agg(["mean", "min", "max"]).reset_index()
    )
    return per_seed, summary


def paired_target_effects(combined: pd.DataFrame, challenger: str, reference: str,
                          *, step: int = 10_000) -> pd.DataFrame:
    subset = combined[combined.checkpoint_step.eq(step)]
    index = ["task", "training_seed", "dataset"]
    left = subset[subset.arm.eq(challenger)][index + ["roc_auc"]]
    right = subset[subset.arm.eq(reference)][index + ["roc_auc"]]
    joined = left.merge(right, on=index, suffixes=("_challenger", "_reference"), validate="one_to_one")
    joined["challenger"] = challenger
    joined["reference"] = reference
    joined["delta"] = joined.roc_auc_challenger - joined.roc_auc_reference
    return joined


def decisions(combined: pd.DataFrame, per_seed: pd.DataFrame) -> dict:
    endpoint = per_seed[(per_seed.emb_dim.eq(256)) & (per_seed.checkpoint_step.eq(10_000))]
    winners = {}
    ratio_details = {}
    for task, part in endpoint.groupby("task"):
        ranked = part.groupby("arm").roc_auc.mean().sort_values(ascending=False)
        winner, runner = ranked.index[:2]
        effects = paired_target_effects(combined[combined.task.eq(task)], winner, "mix_p000")
        target_means = effects.groupby("dataset").delta.mean()
        winners[task] = winner
        ratio_details[task] = {
            "winner": winner,
            "winner_probability": ARMS[winner][0],
            "winner_mean_auc": float(ranked.iloc[0]),
            "margin_over_runner_up": float(ranked.iloc[0] - ranked.iloc[1]),
            "worst_target_delta_vs_p0": float(target_means.min()),
            "worst_target": str(target_means.idxmin()),
        }
    common = len(set(winners.values())) == 1
    winner = next(iter(winners.values())) if common else None
    universal = bool(
        common
        and all(row["margin_over_runner_up"] >= MARGIN for row in ratio_details.values())
        and all(row["worst_target_delta_vs_p0"] >= -MARGIN for row in ratio_details.values())
    )
    scaling = {}
    for task in ("nm", "classification"):
        part = combined[(combined.task.eq(task)) & (combined.arm.eq("mix_p000"))]
        wide = part.pivot(index=["training_seed", "dataset"], columns="checkpoint_step", values="roc_auc")
        delta = wide[10_000] - wide[2_000]
        seed_delta = delta.groupby(level="training_seed").mean()
        target_delta = delta.groupby(level="dataset").mean()
        scaling[task] = {
            "mean_10k_minus_2k": float(delta.mean()),
            "positive_seed_count": int((seed_delta > 0).sum()),
            "worst_target_delta": float(target_delta.min()),
            "worst_target": str(target_delta.idxmin()),
            "positive": bool(delta.mean() >= MARGIN and (seed_delta > 0).sum() >= 2 and target_delta.min() >= -MARGIN),
        }
    cap = combined[(combined.task.eq("nm")) & combined.arm.isin(["mix_p000", "wide_p000"])
                   & combined.checkpoint_step.eq(10_000)]
    base = cap[cap.arm.eq("mix_p000")][["training_seed", "dataset", "roc_auc"]]
    wide = cap[cap.arm.eq("wide_p000")][["training_seed", "dataset", "roc_auc"]]
    cap_join = wide.merge(base, on=["training_seed", "dataset"], suffixes=("_wide", "_base"), validate="one_to_one")
    cap_join["delta"] = cap_join.roc_auc_wide - cap_join.roc_auc_base
    cap_seed = cap_join.groupby("training_seed").delta.mean()
    cap_target = cap_join.groupby("dataset").delta.mean()
    capacity = {
        "mean_wide_minus_base": float(cap_join.delta.mean()),
        "positive_seed_count": int((cap_seed > 0).sum()),
        "worst_target_delta": float(cap_target.min()),
        "worst_target": str(cap_target.idxmin()),
        "positive": bool(cap_join.delta.mean() >= MARGIN and (cap_seed > 0).sum() >= 2 and cap_target.min() >= -MARGIN),
    }
    return {
        "practical_margin": MARGIN,
        "ratio": {"universal_winner": universal, "winner": winner, "tasks": ratio_details},
        "data_scale": scaling,
        "capacity_nm": capacity,
    }


def plot(summary: pd.DataFrame, output: Path) -> None:
    colors = {"mix_p000": "#1B4F72", "mix_p010": "#2E86AB", "mix_p025": "#4FA3C7",
              "mix_p050": "#F28E2B", "mix_p100": "#D1495B", "wide_p000": "#6A4C93"}
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.8), constrained_layout=True)
    for ax, task, title in zip(axes[0], ("nm", "classification"),
                               ("a  Cross-graph ratio · NM", "b  Cross-graph ratio · classification")):
        part = summary[(summary.task.eq(task)) & summary.emb_dim.eq(256) & summary.checkpoint_step.eq(10_000)]
        part = part.sort_values("cross_graph_prob")
        ax.fill_between(part.cross_graph_prob, part["min"], part["max"], color="#2E86AB", alpha=.16)
        ax.plot(part.cross_graph_prob, part["mean"], marker="o", color="#1B4F72", lw=2)
        ax.set(xlabel="Probability an episode spans graphs", ylabel="ROC-AUC", title=title)
        ax.grid(alpha=.22, linewidth=.6)
    specs = [
        (axes[1, 0], "nm", ("mix_p000", "mix_p100", "wide_p000"), "c  Fixed-corpus data scale · NM"),
        (axes[1, 1], "classification", ("mix_p000", "mix_p100"), "d  Fixed-corpus data scale · classification"),
    ]
    labels = {"mix_p000": "Graph-local · 256d", "mix_p100": "Cross-graph · 256d",
              "wide_p000": "Graph-local · 512d"}
    for ax, task, arms, title in specs:
        for arm in arms:
            part = summary[(summary.task.eq(task)) & summary.arm.eq(arm)].sort_values("checkpoint_step")
            x = part.checkpoint_step / 1000
            ax.fill_between(x, part["min"], part["max"], color=colors[arm], alpha=.12)
            ax.plot(x, part["mean"], marker="o", color=colors[arm], lw=1.8, label=labels[arm])
        ax.set(xlabel="Pretraining updates (thousands)", ylabel="ROC-AUC", title=title)
        ax.grid(alpha=.22, linewidth=.6)
        ax.legend(frameon=False, fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=220)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nm-root", type=Path, required=True)
    parser.add_argument("--classification", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    nm = load_nm(args.nm_root.resolve())
    cls = load_cls(args.classification.resolve())
    validate_grid(nm, "NM", NM_TARGETS, tuple(ARMS))
    validate_grid(cls, "classification", CLS_TARGETS, tuple(name for name, (_, dim) in ARMS.items() if dim == 256))
    provenance = provenance_join(nm, cls)
    combined = pd.concat([nm, cls], ignore_index=True, sort=False)
    per_seed, summary = macro_tables(combined)
    effect_rows = [
        paired_target_effects(combined, arm, "mix_p000")
        for arm in ("mix_p010", "mix_p025", "mix_p050", "mix_p100", "wide_p000")
    ]
    effects = pd.concat(effect_rows, ignore_index=True)
    verdict = decisions(combined, per_seed)
    data_dir = args.output / "data"
    figure_dir = args.output / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    nm.to_csv(data_dir / "nm_cells.csv", index=False)
    cls.to_csv(data_dir / "classification_cells.csv", index=False)
    per_seed.to_csv(data_dir / "macro_per_seed.csv", index=False)
    summary.to_csv(data_dir / "macro_summary.csv", index=False)
    effects.to_csv(data_dir / "target_effects_per_seed.csv", index=False)
    provenance.to_csv(data_dir / "cross_task_model_provenance.csv", index=False)
    (data_dir / "decision.json").write_text(json.dumps(verdict, indent=2) + "\n", encoding="utf-8")
    plot(summary, figure_dir / "mechanism_sweeps")
    audit = {
        "status": "complete", "nm_cells": len(nm), "classification_cells": len(cls),
        "physical_checkpoint_models": combined.model_id.nunique(),
        "cross_task_models": len(provenance), "decision": verdict,
    }
    (data_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
