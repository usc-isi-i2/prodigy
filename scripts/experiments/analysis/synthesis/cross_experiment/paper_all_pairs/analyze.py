#!/usr/bin/env python3
"""Audit and analyze the preregistered complete source-pair panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(ROOT))
from scripts.experiments.setup.final_core.core_plan import ORDERS
from scripts.experiments.setup.nm_interventions_overnight.plan import TARGETS
from scripts.experiments.setup.paper_all_pairs.plan import SEEDS, SOURCES, build_pairs


PROTOCOL = "paper_all_pairs_fixed_nm_v1"
EXPECTED_CELLS = 36 * 3 * 9
SPECIALISTS = (
    ROOT / "scripts/experiments/analysis/transfer/matrices/cross_model/final_core/"
    "data/results_full_graphwide.tsv"
)
PREDICTORS = ("best_included", "specialist_mean", "global_donor_selected")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pearson(left: pd.Series, right: pd.Series) -> float | None:
    x, y = left.to_numpy(float), right.to_numpy(float)
    if len(x) < 2 or np.std(x) <= 1e-15 or np.std(y) <= 1e-15:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def load_pair_cells(root: Path) -> pd.DataFrame:
    rows = [json.loads(path.read_text(encoding="utf-8"))
            for path in sorted((root / "cells").glob("*/*.json"))]
    if len(rows) != EXPECTED_CELLS:
        raise ValueError(f"expected {EXPECTED_CELLS} pair cells, got {len(rows)}")
    frame = pd.DataFrame(rows)
    required = {
        "protocol", "model_id", "arm", "seed", "target", "sources", "episodes",
        "checkpoint", "checkpoint_step", "checkpoint_sha256", "training_revision",
        "evaluation_revision", "fingerprint", "roc_auc", "accuracy", "loss",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"pair cells lack required fields: {missing}")
    expected = {(pair.arm, seed, target)
                for pair in build_pairs() for seed in SEEDS for target in TARGETS}
    observed = set(zip(frame.arm, frame.seed.astype(int), frame.target))
    if observed != expected or len(frame) != len(observed):
        raise ValueError(f"pair Cartesian product mismatch: missing={len(expected-observed)} extra={len(observed-expected)}")
    if set(frame.protocol) != {PROTOCOL} or set(frame.episodes.astype(int)) != {512}:
        raise ValueError("pair evaluation protocol drift")
    if set(frame.checkpoint_step.astype(int)) != {2500}:
        raise ValueError("pair checkpoint step drift")
    for metric in ("roc_auc", "accuracy", "loss"):
        if not np.isfinite(frame[metric].astype(float)).all():
            raise ValueError(f"non-finite pair metric: {metric}")
    for target, part in frame.groupby("target"):
        if part.fingerprint.nunique() != 1:
            raise ValueError(f"episode fingerprint drift for {target}")
    registry = {pair.arm: pair.sources for pair in build_pairs()}
    for row in frame.itertuples(index=False):
        if tuple(row.sources) != registry[row.arm]:
            raise ValueError(f"source registry mismatch for {row.model_id}")
    frame["seed"] = frame.seed.astype(int)
    frame["roc_auc"] = frame.roc_auc.astype(float)
    return frame


def load_specialists(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, sep="\t", low_memory=False)
    frame = raw[
        raw.architecture.eq("PRODIGY")
        & raw.component.eq("matrix")
        & raw.result_status.eq("observed")
        & raw.train_graph_count.eq(1)
        & raw.training_seed.isin(SEEDS)
        & raw.test_graph.isin(TARGETS)
    ].copy()
    frame["source"] = frame.train_graphs.map(lambda value: json.loads(value)[0])
    frame = frame[["training_seed", "source", "test_graph", "nm_roc_auc_ovr_macro"]].rename(
        columns={"training_seed": "seed", "test_graph": "target",
                 "nm_roc_auc_ovr_macro": "specialist_score"}
    )
    frame["seed"] = frame.seed.astype(int)
    frame["specialist_score"] = frame.specialist_score.astype(float)
    expected = {(seed, source, target) for seed in SEEDS for source in SOURCES for target in TARGETS}
    observed = set(zip(frame.seed, frame.source, frame.target))
    if observed != expected or len(frame) != len(observed):
        raise ValueError(f"specialist Cartesian product mismatch: {len(frame)} rows")
    if not np.isfinite(frame.specialist_score).all():
        raise ValueError("non-finite specialist score")
    return frame


def attach_predictions(cells: pd.DataFrame, specialists: pd.DataFrame) -> pd.DataFrame:
    lookup = specialists.set_index(["seed", "source", "target"]).specialist_score.to_dict()
    quality = {}
    for seed in SEEDS:
        for source in SOURCES:
            for held_out_target in TARGETS:
                values = [lookup[(seed, source, target)] for target in TARGETS if target != held_out_target]
                quality[(seed, source, held_out_target)] = float(np.mean(values))
    nested_pairs = {frozenset(order[:2]) for order in ORDERS.values()}
    rows = []
    for raw in cells.itertuples(index=False):
        sources = tuple(raw.sources)
        scores = {source: lookup[(raw.seed, source, raw.target)] for source in sources}
        selected = max(sources, key=lambda source: (quality[(raw.seed, source, raw.target)], -SOURCES.index(source)))
        row = raw._asdict()
        row.update({
            "source_left": sources[0],
            "source_right": sources[1],
            "target_in_pair": raw.target in sources,
            "nested_rung2_pair": frozenset(sources) in nested_pairs,
            "best_included": max(scores.values()),
            "specialist_mean": float(np.mean(list(scores.values()))),
            "global_donor_selected": scores[selected],
            "global_donor_source": selected,
        })
        for predictor in PREDICTORS:
            row[f"residual_{predictor}"] = raw.roc_auc - row[predictor]
        rows.append(row)
    return pd.DataFrame(rows)


def scope_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "all": pd.Series(True, index=frame.index),
        "target_in_pair": frame.target_in_pair,
        "target_held_out": ~frame.target_in_pair,
        "nonnested_target_held_out": (~frame.target_in_pair) & (~frame.nested_rung2_pair),
    }


def predictor_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope, mask in scope_masks(frame).items():
        data = frame[mask].copy()
        for seed_label, part in [("all", data)] + [(str(seed), data[data.seed.eq(seed)]) for seed in SEEDS]:
            if part.empty:
                continue
            part = part.copy()
            part["observed_demeaned"] = part.roc_auc - part.groupby(["seed", "target"]).roc_auc.transform("mean")
            for predictor in PREDICTORS:
                prediction = part[predictor]
                prediction_demeaned = prediction - part.groupby(["seed", "target"])[predictor].transform("mean")
                residual = part.roc_auc - prediction
                rows.append({
                    "scope": scope,
                    "seed": seed_label,
                    "predictor": predictor,
                    "cells": len(part),
                    "mae": float(residual.abs().mean()),
                    "bias_observed_minus_prediction": float(residual.mean()),
                    "pearson": pearson(part.roc_auc, prediction),
                    "target_demeaned_pearson": pearson(part.observed_demeaned, prediction_demeaned),
                    "fraction_observed_above_prediction": float((residual > 0).mean()),
                })
    return pd.DataFrame(rows)


def group_summaries(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pair = frame.groupby(["arm", "source_left", "source_right"], as_index=False).agg(
        mean_roc_auc=("roc_auc", "mean"),
        mean_residual_best=("residual_best_included", "mean"),
        worst_residual_best=("residual_best_included", "min"),
        fraction_above_best=("residual_best_included", lambda values: float((values > 0).mean())),
        cells=("roc_auc", "size"),
    )
    target = frame.groupby("target", as_index=False).agg(
        mean_roc_auc=("roc_auc", "mean"),
        mean_residual_best=("residual_best_included", "mean"),
        mae_best=("residual_best_included", lambda values: float(values.abs().mean())),
        cells=("roc_auc", "size"),
    )
    return pair, target


def decision(summary: pd.DataFrame) -> dict:
    scope = summary[(summary.scope == "nonnested_target_held_out") & (summary.seed != "all")]
    by = {(int(row.seed), row.predictor): row for row in scope.itertuples(index=False)}
    seed_checks = {}
    for seed in SEEDS:
        best = by[(seed, "best_included")]
        demeaned = float(best.target_demeaned_pearson) if best.target_demeaned_pearson is not None else math.nan
        seed_checks[str(seed)] = {
            "best_mae": best.mae,
            "mean_mae": by[(seed, "specialist_mean")].mae,
            "global_donor_mae": by[(seed, "global_donor_selected")].mae,
            "best_target_demeaned_pearson": None if not math.isfinite(demeaned) else demeaned,
            "best_beats_both_alternatives": bool(
                best.mae < by[(seed, "specialist_mean")].mae
                and best.mae < by[(seed, "global_donor_selected")].mae
            ),
            "positive_target_demeaned_association": bool(math.isfinite(demeaned) and demeaned > 0),
        }
    replicated = all(
        row["best_beats_both_alternatives"] and row["positive_target_demeaned_association"]
        for row in seed_checks.values()
    )
    return {
        "rule": "best included specialist must beat mean and LOO global-donor selection in MAE, with positive target-demeaned correlation, in every seed on non-nested held-out cells",
        "best_specialist_envelope_replicated": replicated,
        "seed_checks": seed_checks,
    }


def plot_results(frame: pd.DataFrame, summary: pd.DataFrame, output: Path) -> None:
    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))

    held = frame[(~frame.target_in_pair) & (~frame.nested_rung2_pair)]
    colors = dict(zip(TARGETS, plt.cm.tab10(np.linspace(0, 1, len(TARGETS)))))
    for target, part in held.groupby("target"):
        axes[0].scatter(part.best_included, part.roc_auc, s=13, alpha=.55,
                        color=colors[target], label=target.replace("_", " "))
    low = min(held.best_included.min(), held.roc_auc.min())
    high = max(held.best_included.max(), held.roc_auc.max())
    axes[0].plot([low, high], [low, high], color="#333333", linewidth=1)
    axes[0].set(xlabel="Best included specialist ROC-AUC", ylabel="Pair ROC-AUC",
                title="Non-nested held-out targets")

    seed_rows = summary[(summary.scope == "nonnested_target_held_out") & (summary.seed != "all")]
    labels = ["Best included", "Specialist mean", "Global donor"]
    for index, predictor in enumerate(PREDICTORS):
        values = seed_rows[seed_rows.predictor.eq(predictor)].mae.to_numpy(float)
        axes[1].bar(index, values.mean(), color=("#0072B2", "#999999", "#D55E00")[index])
        axes[1].vlines(index, values.min(), values.max(), color="#222222", linewidth=1.5)
        axes[1].scatter(np.repeat(index, len(values)), values, color="#222222", s=18, zorder=3)
    axes[1].set_xticks(range(3), labels, rotation=18, ha="right")
    axes[1].set(ylabel="MAE", title="Predictor error across seeds")

    matrix = np.full((len(SOURCES), len(SOURCES)), np.nan)
    means = frame.groupby(["source_left", "source_right"]).residual_best_included.mean()
    for (left, right), value in means.items():
        i, j = SOURCES.index(left), SOURCES.index(right)
        matrix[i, j] = matrix[j, i] = value
    vmax = max(abs(np.nanmin(matrix)), abs(np.nanmax(matrix)))
    image = axes[2].imshow(matrix, cmap="RdBu", vmin=-vmax, vmax=vmax)
    short = [source.replace("_political", " pol.").replace("_page_reference", "")
             .replace("ukr_rus_suspended", "ukr. susp.").replace("ukr_rus", "ukraine")
             .replace("election2020", "election").replace("twibot20", "twibot")
             .replace("cp_hk", "hong kong") for source in SOURCES]
    axes[2].set_xticks(range(len(SOURCES)), short, rotation=55, ha="right")
    axes[2].set_yticks(range(len(SOURCES)), short)
    axes[2].set_title("Pair minus best-specialist residual")
    fig.colorbar(image, ax=axes[2], fraction=.046, pad=.04)

    for axis in axes[:2]:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#dddddd", linewidth=.6)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, .12, 1, 1))
    fig.savefig(figure_dir / "all_pairs_composition.png", dpi=220, bbox_inches="tight")
    fig.savefig(figure_dir / "all_pairs_composition.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-eval", type=Path, required=True)
    parser.add_argument("--specialists", type=Path, default=SPECIALISTS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"refusing non-empty output without --overwrite: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    status = json.loads((args.pair_eval / "status.json").read_text(encoding="utf-8"))
    if status.get("status") != "complete" or status.get("audit", {}).get("cells") != EXPECTED_CELLS:
        raise ValueError("pair evaluator has not passed its 972-cell audit")
    cells = attach_predictions(load_pair_cells(args.pair_eval), load_specialists(args.specialists))
    summary = predictor_summary(cells)
    pairs, targets = group_summaries(cells)
    result = {
        "status": "complete",
        "protocol": PROTOCOL,
        "cells": len(cells),
        "pairs": cells.arm.nunique(),
        "training_seeds": sorted(cells.seed.unique().tolist()),
        "targets": sorted(cells.target.unique().tolist()),
        "pair_evaluation_audit_sha256": sha256(args.pair_eval / "audit.json"),
        "specialist_input_sha256": sha256(args.specialists),
        "decision": decision(summary),
    }
    cells.to_csv(args.output / "pair_cells_with_predictions.csv", index=False)
    summary.to_csv(args.output / "predictor_summary.csv", index=False)
    pairs.to_csv(args.output / "pair_summary.csv", index=False)
    targets.to_csv(args.output / "target_summary.csv", index=False)
    plot_results(cells, summary, args.output)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
