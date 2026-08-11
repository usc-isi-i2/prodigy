#!/usr/bin/env python3
"""Analyze the one-seed, common-CLS PRODIGY/VISION/GILT matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd


ARCHITECTURES = ("prodigy", "vision", "gilt")
TARGETS = ("covid_political", "election2020", "ukr_rus_suspended", "twibot20")
DISPLAY = {
    "prodigy": "PRODIGY",
    "vision": "VISION",
    "gilt": "GILT",
    "covid_political": "Covid Political",
    "election2020": "Election 2020",
    "ukr_rus_suspended": "Ukraine Suspended",
    "twibot20": "TwiBot-20",
}
COLORS = {"prodigy": "#2878B5", "vision": "#E07A3F", "gilt": "#4E9F6D"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="aggregate classification_long.csv")
    parser.add_argument("--output-root", default=str(Path(__file__).parent))
    return parser.parse_args()


def _pearson(left: pd.Series, right: pd.Series) -> float:
    return float(np.corrcoef(left.to_numpy(float), right.to_numpy(float))[0, 1])


def validate(frame: pd.DataFrame) -> None:
    expected = 3 * 31 * 4
    if len(frame) != expected:
        raise ValueError(f"expected {expected} rows, found {len(frame)}")
    keys = ["architecture", "model_id", "dataset"]
    if frame.duplicated(keys).any():
        raise ValueError("duplicate architecture/model/target cells")
    if set(frame.architecture) != set(ARCHITECTURES):
        raise ValueError(f"architecture mismatch: {sorted(set(frame.architecture))}")
    if set(frame.dataset) != set(TARGETS):
        raise ValueError(f"target mismatch: {sorted(set(frame.dataset))}")
    if set(frame.seed.astype(int)) != {0} or set(frame.checkpoint_step.astype(int)) != {100}:
        raise ValueError("registered analysis requires seed 0 at checkpoint 100")
    for target, target_frame in frame.groupby("dataset"):
        if target_frame.episode_fingerprint.nunique() != 1:
            raise ValueError(f"episode fingerprint drift for {target}")
    counts = frame.groupby("architecture").model_id.nunique().to_dict()
    if counts != {architecture: 31 for architecture in ARCHITECTURES}:
        raise ValueError(f"model-count mismatch: {counts}")


def add_max_rule(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["source_list"] = frame.sources.fillna("").str.split(",")
    specialists = frame[frame.model_id.str.startswith("ss_")].copy()
    specialist_lookup = {
        (row.architecture, row.source_list[0], row.dataset): float(row.roc_auc)
        for row in specialists.itertuples()
    }
    mixtures = frame[~frame.model_id.str.startswith("ss_")].copy()

    def predict(row):
        values = [specialist_lookup[(row.architecture, source, row.dataset)] for source in row.source_list]
        return max(values)

    mixtures["best_included_specialist"] = mixtures.apply(predict, axis=1)
    mixtures["residual"] = mixtures.roc_auc - mixtures.best_included_specialist
    mixtures["absolute_error"] = mixtures.residual.abs()
    mixtures["target_in_sources"] = mixtures.apply(
        lambda row: row.dataset in row.source_list, axis=1
    )
    return mixtures


def architecture_summary(frame: pd.DataFrame, mixtures: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for architecture in ARCHITECTURES:
        all_cells = frame[frame.architecture == architecture]
        specialist_cells = all_cells[all_cells.model_id.str.startswith("ss_")]
        mixture_cells = mixtures[mixtures.architecture == architecture].copy()
        mixture_cells["observed_centered"] = mixture_cells.roc_auc - mixture_cells.groupby(
            "dataset"
        ).roc_auc.transform("mean")
        mixture_cells["predicted_centered"] = (
            mixture_cells.best_included_specialist
            - mixture_cells.groupby("dataset").best_included_specialist.transform("mean")
        )
        in_mix = mixture_cells[mixture_cells.target_in_sources]
        held_out = mixture_cells[~mixture_cells.target_in_sources]
        rows.append(
            {
                "architecture": architecture,
                "mean_roc_auc_all_cells": all_cells.roc_auc.mean(),
                "mean_roc_auc_specialists": specialist_cells.roc_auc.mean(),
                "mean_roc_auc_mixtures": mixture_cells.roc_auc.mean(),
                "best_specialist_mae": mixture_cells.absolute_error.mean(),
                "best_specialist_pearson": _pearson(
                    mixture_cells.best_included_specialist, mixture_cells.roc_auc
                ),
                "best_specialist_within_target_pearson": _pearson(
                    mixture_cells.predicted_centered, mixture_cells.observed_centered
                ),
                "mean_mixture_residual": mixture_cells.residual.mean(),
                "fraction_below_best_specialist": (mixture_cells.residual < 0).mean(),
                "in_mixture_mae": in_mix.absolute_error.mean(),
                "in_mixture_mean_residual": in_mix.residual.mean(),
                "held_out_mae": held_out.absolute_error.mean(),
                "held_out_mean_residual": held_out.residual.mean(),
                "mixture_cells": len(mixture_cells),
                "in_mixture_cells": len(in_mix),
                "held_out_cells": len(held_out),
            }
        )
    return pd.DataFrame(rows)


def paired_architecture_cells(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    wide = frame.pivot(index=["model_id", "dataset"], columns="architecture", values="roc_auc")
    rows = []
    for left, right in (("vision", "prodigy"), ("gilt", "prodigy"), ("gilt", "vision")):
        delta = wide[left] - wide[right]
        rows.append(
            {
                "contrast": f"{left}_minus_{right}",
                "mean_delta": delta.mean(),
                "median_delta": delta.median(),
                "fraction_positive": (delta > 0).mean(),
                "pearson": _pearson(wide[left], wide[right]),
                "spearman": float(wide[left].corr(wide[right], method="spearman")),
                "cells": len(delta),
            }
        )
        wide[f"{left}_minus_{right}"] = delta
    return wide.reset_index(), pd.DataFrame(rows)


def make_figure(
    frame: pd.DataFrame,
    mixtures: pd.DataFrame,
    paired_cells: pd.DataFrame,
    output_root: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.55))

    target_summary = (
        frame.groupby(["dataset", "architecture"], as_index=False).roc_auc.mean()
    )
    x = np.arange(len(TARGETS))
    width = 0.24
    for index, architecture in enumerate(ARCHITECTURES):
        values = [
            target_summary[
                (target_summary.dataset == target) & (target_summary.architecture == architecture)
            ].roc_auc.item()
            for target in TARGETS
        ]
        axes[0].bar(
            x + (index - 1) * width,
            values,
            width,
            color=COLORS[architecture],
            label=DISPLAY[architecture],
        )
    axes[0].axhline(0.5, color="#777777", lw=0.8, ls="--")
    axes[0].set_xticks(x, [DISPLAY[target] for target in TARGETS], rotation=28, ha="right")
    axes[0].set_ylabel("ROC-AUC")
    axes[0].set_title("a  Same-task downstream means", loc="left", fontweight="bold")
    axes[0].legend(frameon=False, fontsize=8)

    limits = [
        min(mixtures.best_included_specialist.min(), mixtures.roc_auc.min()) - 0.015,
        max(mixtures.best_included_specialist.max(), mixtures.roc_auc.max()) + 0.015,
    ]
    for architecture in ARCHITECTURES:
        part = mixtures[mixtures.architecture == architecture]
        axes[1].scatter(
            part.best_included_specialist,
            part.roc_auc,
            s=12,
            alpha=0.58,
            color=COLORS[architecture],
            label=DISPLAY[architecture],
        )
    axes[1].plot(limits, limits, color="#333333", lw=0.9, ls="--")
    axes[1].set_xlim(limits)
    axes[1].set_ylim(limits)
    axes[1].set_xlabel("Best included specialist ROC-AUC")
    axes[1].set_ylabel("Mixture ROC-AUC")
    axes[1].set_title("b  Composition envelope", loc="left", fontweight="bold")

    contrast_columns = [
        "vision_minus_prodigy",
        "gilt_minus_prodigy",
        "gilt_minus_vision",
    ]
    box = axes[2].boxplot(
        [paired_cells[column].dropna() for column in contrast_columns],
        patch_artist=True,
        showfliers=False,
        widths=0.58,
        medianprops={"color": "black", "linewidth": 1.1},
    )
    for patch, color in zip(box["boxes"], (COLORS["vision"], COLORS["gilt"], "#7A6FA3")):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
    axes[2].axhline(0, color="#333333", lw=0.9, ls="--")
    axes[2].set_xticks(
        [1, 2, 3],
        ["VISION\n− PRODIGY", "GILT\n− PRODIGY", "GILT\n− VISION"],
    )
    axes[2].set_ylabel("Paired ROC-AUC difference")
    axes[2].set_title("c  Architecture sensitivity", loc="left", fontweight="bold")

    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#dddddd", lw=0.5, alpha=0.6)
    fig.tight_layout()
    figure_root = output_root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_root / "architecture_comparison.pdf", bbox_inches="tight")
    fig.savefig(figure_root / "architecture_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_root = Path(args.output_root)
    data_root = output_root / "data"
    data_root.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(input_path)
    validate(frame)
    mixtures = add_max_rule(frame)
    architecture = architecture_summary(frame, mixtures)
    paired_cells, paired_summary = paired_architecture_cells(frame)
    target_summary = frame.groupby(["architecture", "dataset"], as_index=False).agg(
        mean_roc_auc=("roc_auc", "mean"),
        median_roc_auc=("roc_auc", "median"),
        min_roc_auc=("roc_auc", "min"),
        max_roc_auc=("roc_auc", "max"),
        cells=("roc_auc", "size"),
    )

    architecture.to_csv(data_root / "architecture_summary.csv", index=False)
    mixtures.drop(columns="source_list").to_csv(data_root / "max_rule_cells.csv", index=False)
    paired_cells.to_csv(data_root / "paired_architecture_cells.csv", index=False)
    paired_summary.to_csv(data_root / "paired_architecture_summary.csv", index=False)
    target_summary.to_csv(data_root / "target_summary.csv", index=False)
    make_figure(frame, mixtures, paired_cells, output_root)

    summary = {
        "input": str(input_path),
        "input_rows": len(frame),
        "seed": 0,
        "checkpoint_step": 100,
        "architectures": architecture.to_dict(orient="records"),
        "paired_architecture_contrasts": paired_summary.to_dict(orient="records"),
        "episode_fingerprints": {
            target: frame[frame.dataset == target].episode_fingerprint.iloc[0]
            for target in TARGETS
        },
        "claim_boundary": (
            "One training seed and a matched 100-update budget; use for qualitative "
            "architecture sensitivity and composition replication, not final ranking."
        ),
    }
    (data_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
