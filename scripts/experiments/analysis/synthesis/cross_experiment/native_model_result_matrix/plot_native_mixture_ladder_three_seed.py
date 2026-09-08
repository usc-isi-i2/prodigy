#!/usr/bin/env python3
"""Plot three-seed native-objective mixture ladders for three model families."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[5]
sys.path.insert(0, str(REPO))

from scripts.experiments.setup.final_core.core_plan import ORDERS, build_models


TARGETS = (
    "covid_political",
    "election2020",
    "facebook_page_reference",
    "twibot20",
    "ukr_rus_suspended",
)
MODEL_RUNGS = {
    "PRODIGY": tuple(range(1, 10)),
    "VISION": (1, 3, 5, 7, 9),
    "SAMGPT": tuple(range(1, 10)),
}
MODEL_DETAILS = {
    "PRODIGY": "Neighbor matching · 2,500 updates",
    "VISION": "Feature similarity · 2,500 updates",
    "SAMGPT": "GraphCL · 500 updates",
}
COLORS = {"A": "#4477AA", "B": "#CC6677", "C": "#228833"}


def model_for(order: str, rung: int) -> str:
    wanted = frozenset(ORDERS[order][:rung])
    matches = [model.model_id for model in build_models() if frozenset(model.sources) == wanted]
    if len(matches) != 1:
        raise ValueError(f"{order}{rung}: expected one final-core model, got {matches}")
    return matches[0]


def load_prodigy(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, sep="\t")
    raw = raw[raw.dataset.isin(TARGETS) & raw.checkpoint_step.eq(2500)].copy()
    rows = []
    for order in ORDERS:
        for rung in MODEL_RUNGS["PRODIGY"]:
            part = raw[raw.model_id.eq(model_for(order, rung))]
            if len(part) != 3 * len(TARGETS) or part.training_seed.nunique() != 3:
                raise ValueError(f"PRODIGY {order}{rung}: incomplete three-seed grid")
            for row in part.itertuples(index=False):
                rows.append(
                    {
                        "model": "PRODIGY", "order": order, "rung": rung,
                        "training_seed": int(row.training_seed), "target": row.dataset,
                        "roc_auc": float(row.roc_auc), "fingerprint": row.episode_fingerprint,
                    }
                )
    return pd.DataFrame(rows)


def load_vision(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    raw = raw[raw.checkpoint_step.eq(2500) & raw.dataset.isin(TARGETS)].copy()
    raw = raw.rename(columns={"dataset": "target", "episode_fingerprint": "fingerprint"})
    raw["model"] = "VISION"
    expected = {
        (order, rung, seed, target)
        for order in ORDERS for rung in MODEL_RUNGS["VISION"]
        for seed in (0, 1, 2) for target in TARGETS
    }
    observed = set(raw[["order", "rung", "training_seed", "target"]].itertuples(index=False, name=None))
    if observed != expected or len(raw) != len(expected):
        raise ValueError(
            f"VISION three-seed grid mismatch: missing={len(expected-observed)} extra={len(observed-expected)}"
        )
    return raw[["model", "order", "rung", "training_seed", "target", "roc_auc", "fingerprint"]]


def load_samgpt(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    raw["target"] = raw.target.replace(
        {"facebook_page_category_top30": "facebook_page_reference"}
    )
    raw = raw[raw.target.isin(TARGETS) & raw.checkpoint_update.eq(500)].copy()
    rows = []
    for order in ORDERS:
        for rung in MODEL_RUNGS["SAMGPT"]:
            model_id = {
                "ss_ukr_rus": "ss_ukr_rus_twitter",
                "ss_covid": "ss_covid19_twitter",
                "ss_cp_hk": "ss_cp_hk_twitter",
            }.get(model_for(order, rung), model_for(order, rung))
            part = raw[raw.model_id.eq(model_id)]
            if len(part) != 3 * len(TARGETS) or part.seed.nunique() != 3:
                raise ValueError(f"SAMGPT {order}{rung}: incomplete three-seed grid")
            for row in part.itertuples(index=False):
                rows.append(
                    {
                        "model": "SAMGPT", "order": order, "rung": rung,
                        "training_seed": int(row.seed), "target": row.target,
                        "roc_auc": float(row.roc_auc_mean), "fingerprint": row.episode_fingerprint,
                    }
                )
    return pd.DataFrame(rows)


def validate_cells(frame: pd.DataFrame) -> None:
    expected_count = sum(
        len(rungs) * len(ORDERS) * 3 * len(TARGETS)
        for rungs in MODEL_RUNGS.values()
    )
    keys = ["model", "order", "rung", "training_seed", "target"]
    if len(frame) != expected_count or frame.duplicated(keys).any():
        raise ValueError(f"expected {expected_count} unique cross-family cells, got {len(frame)}")
    if set(frame.model) != set(MODEL_RUNGS):
        raise ValueError("cross-family model registry changed")
    if not np.isfinite(frame.roc_auc).all() or not frame.roc_auc.between(0, 1).all():
        raise ValueError("invalid cross-family ROC-AUC")
    for model, rungs in MODEL_RUNGS.items():
        part = frame[frame.model.eq(model)]
        if set(part.rung) != set(rungs):
            raise ValueError(f"{model} rung registry changed")
        if part.training_seed.nunique() != 3:
            raise ValueError(f"{model} does not contain three training seeds")
        drift = part.groupby("target").fingerprint.nunique()
        if not drift.eq(1).all():
            raise ValueError(f"{model} episode fingerprint drift: {drift.to_dict()}")


def summarize(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_seed = (
        frame.groupby(["model", "order", "rung", "training_seed"], as_index=False)
        .agg(roc_auc=("roc_auc", "mean"), targets=("target", "nunique"))
    )
    if not per_seed.targets.eq(len(TARGETS)).all():
        raise ValueError("a cross-family ladder point does not contain all fixed targets")
    summary = (
        per_seed.groupby(["model", "order", "rung"], as_index=False)
        .agg(
            training_seeds=("training_seed", "nunique"),
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_min=("roc_auc", "min"),
            roc_auc_max=("roc_auc", "max"),
            roc_auc_sample_std=("roc_auc", "std"),
        )
    )
    if not summary.training_seeds.eq(3).all():
        raise ValueError("a cross-family summary point does not contain all three seeds")
    return per_seed, summary


def plot(summary: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.35), sharey=True)
    for axis, model in zip(axes, MODEL_RUNGS, strict=True):
        part = summary[summary.model.eq(model)]
        for order in ORDERS:
            curve = part[part.order.eq(order)].sort_values("rung")
            x = curve.rung.to_numpy(float)
            mean = curve.roc_auc_mean.to_numpy(float)
            low = curve.roc_auc_min.to_numpy(float)
            high = curve.roc_auc_max.to_numpy(float)
            axis.fill_between(x, low, high, color=COLORS[order], alpha=0.14, linewidth=0)
            axis.plot(x, mean, marker="o", markersize=3.5, linewidth=2,
                      color=COLORS[order], label=f"Order {order}")
        axis.set_title(f"{model}\n{MODEL_DETAILS[model]}", fontsize=10)
        axis.set_xlabel("Number of pretraining graphs")
        axis.set_xticks(MODEL_RUNGS[model])
        axis.grid(alpha=0.22, linewidth=0.6)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
    axes[0].set_ylabel("Mean ROC-AUC on five fixed labeled targets")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.08))
    fig.text(
        0.5, -0.01,
        "Each panel uses its model's native objective and family-specific fixed compute. "
        "Lines are three-seed means; bands are observed seed ranges.",
        ha="center", fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94), w_pad=1.2)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    base = REPO / "scripts/experiments/analysis/transfer/matrices/cross_model/final_core/data"
    parser.add_argument(
        "--prodigy", type=Path,
        default=base / "classification_ladder/classification_long.tsv",
    )
    parser.add_argument(
        "--vision", type=Path,
        default=HERE / "data/vision_native_mixture_three_seed_per_target.csv",
    )
    parser.add_argument(
        "--samgpt", type=Path,
        default=base / "samgpt_downstream_cls/cells.csv",
    )
    parser.add_argument("--output-root", type=Path, default=HERE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.concat(
        [load_prodigy(args.prodigy), load_vision(args.vision), load_samgpt(args.samgpt)],
        ignore_index=True,
    )
    validate_cells(frame)
    per_seed, summary = summarize(frame)
    data = args.output_root / "data"
    figures = args.output_root / "figures"
    data.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    frame.to_csv(data / "native_mixture_ladder_three_seed_cells.csv", index=False)
    per_seed.to_csv(data / "native_mixture_ladder_three_seed_per_seed.csv", index=False)
    summary.to_csv(data / "native_mixture_ladder_three_seed_summary.csv", index=False)
    plot(summary, figures / "native_mixture_ladder_three_seed")
    print(
        "NATIVE_MIXTURE_LADDER_THREE_SEED_OK "
        f"cells={len(frame)} models={frame.model.nunique()} targets={frame.target.nunique()}"
    )


if __name__ == "__main__":
    main()
