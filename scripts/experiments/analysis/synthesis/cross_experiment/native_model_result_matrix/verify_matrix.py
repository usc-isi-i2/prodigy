#!/usr/bin/env python3
"""Verify the local evidence counts and exclusion boundary behind the coverage table."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[5]))
ANALYSIS = ROOT.parents[2]
EXPERIMENTS = ANALYSIS.parent
EXPECTED_MODELS = {
    "PRODIGY",
    "VISION",
    "GILT",
    "SAMGPT",
    "GraphSAGE",
    "MLP",
    "Logistic regression",
}
VALID_STATUS = {"complete", "partial", "pending", "missing", "N/A"}


def main() -> int:
    from scripts.experiments.setup.vision_native_mixture_finalcore.mixture_plan import (
        build_mixture_models,
    )

    with (ROOT / "data" / "coverage.csv").open(newline="", encoding="utf-8") as handle:
        coverage = list(csv.DictReader(handle))
    if {row["model"] for row in coverage} != EXPECTED_MODELS:
        raise ValueError("coverage model registry mismatch")
    status_columns = (
        "ssl_to_cls_saturation",
        "cross_ssl_matrix",
        "downstream_cls_matrix",
        "mixture_diversity_to_cls",
        "adaptation_efficiency",
    )
    for row in coverage:
        for column in status_columns:
            if row[column] not in VALID_STATUS:
                raise ValueError(f"invalid status {row[column]!r} for {row['model']}/{column}")
    gilt = next(row for row in coverage if row["model"] == "GILT")
    if gilt["native_pretext"] != "GraphCL (social-graph port/checkpoint absent)":
        raise ValueError("GILT upstream-native GraphCL registry marker missing")
    if any(gilt[column] != "missing" for column in status_columns):
        raise ValueError("GILT cannot be credited without a native social checkpoint")

    graphsage_trajectory = pd.read_csv(
        ROOT / "data" / "graphsage_pilot_v1_trajectory_manifest.csv"
    )
    if graphsage_trajectory.step.tolist() != [0, 20, 60, 100, 300, 900, 2000]:
        raise ValueError("GraphSAGE reconstructed trajectory steps changed")
    terminal_state = graphsage_trajectory.loc[
        graphsage_trajectory.step == 2000, "state_sha256"
    ].item()
    if terminal_state != "cbca0b2ab6bf9eb0707f90ef2bf4073caf89da14460e7466cf326068f672f72f":
        raise ValueError("GraphSAGE reconstructed terminal state hash changed")

    vision_mixture = build_mixture_models()
    if len(vision_mixture) != 13 or sum(model.model_id == "all9" for model in vision_mixture) != 1:
        raise ValueError("VISION native mixture plan changed")

    final_core = pd.read_csv(
        ANALYSIS / "transfer/matrices/cross_model/final_core/data/results_full_long.tsv",
        sep="\t",
    )
    if len(final_core) != 1944 or set(final_core.result_status) != {"observed"}:
        raise ValueError("final-core family-native table is incomplete")
    expected_architecture_rows = {"PRODIGY": 972, "SAMGPT": 972}
    if final_core.architecture.value_counts().to_dict() != expected_architecture_rows:
        raise ValueError("final-core architecture counts changed")

    downstream = pd.read_csv(
        ANALYSIS
        / "transfer/matrices/cross_model/final_core/data/samgpt_downstream_cls/three_seed_mean.csv"
    )
    if (
        len(downstream) != 279
        or downstream.model_id.nunique() != 31
        or downstream.target.nunique() != 9
        or set(downstream.training_seeds) != {3}
    ):
        raise ValueError("SAMGPT downstream CLS registry is incomplete")

    native = pd.read_csv(
        ANALYSIS
        / "transfer/matrices/cross_architecture/icl_arch_matrix/data/native_source_900_seed0/classification_all.tsv",
        sep="\t",
    )
    if len(native) != 390 or native.architecture.value_counts().to_dict() != {
        "prodigy": 130,
        "vision": 130,
        "gilt": 130,
    }:
        raise ValueError("native-source classification export counts changed")

    trainer = (
        EXPERIMENTS / "setup/icl_arch_matrix/train_native_source_model.py"
    ).read_text(encoding="utf-8")
    if "vision_native_feature_similarity_pseudo_episodes" not in trainer:
        raise ValueError("VISION native pretext marker missing")
    if "gilt_native_source_classification_episodes" not in trainer:
        raise ValueError("GILT supervised exclusion marker missing")

    for name in (
        "coverage.png",
        "coverage.pdf",
        "graphsage_pilot_v1_twibot_cls_saturation.png",
        "graphsage_pilot_v1_twibot_cls_saturation.pdf",
    ):
        if not (ROOT / "figures" / name).is_file():
            raise FileNotFoundError(name)

    graphsage_cls = json.loads(
        (
            ROOT
            / "data"
            / "graphsage_pilot_v1_twibot_cls_trajectory_raw"
            / "results.json"
        ).read_text(encoding="utf-8")
    )
    if len(graphsage_cls["models"]) != 8 or graphsage_cls["task"] != "twibot20_bot_classification":
        raise ValueError("GraphSAGE narrow CLS trajectory changed")

    raw_vision = ROOT / "data" / "vision_all9_saturation_raw"
    vision_cells = 0
    if raw_vision.is_dir():
        from analyze_vision_saturation import load_rows

        vision_cells = len(load_rows(raw_vision))

    raw_samgpt = ROOT / "data" / "samgpt_all9_saturation_raw"
    samgpt_saturation_cells = 0
    if raw_samgpt.is_dir():
        from analyze_samgpt_saturation import load_cells

        samgpt_saturation_cells = len(load_cells(raw_samgpt))

    print(
        "NATIVE_MODEL_MATRIX_OK "
        f"coverage_models={len(coverage)} final_core={len(final_core)} "
        f"samgpt_downstream={len(downstream)} native_source={len(native)} "
        f"vision_trajectory={vision_cells} samgpt_trajectory={samgpt_saturation_cells}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
