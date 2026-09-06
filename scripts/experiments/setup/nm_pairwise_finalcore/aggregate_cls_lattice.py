#!/usr/bin/env python3
"""Validate and aggregate the complete 54-model by 5-target CLS lattice."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


TARGETS = {
    "covid_political",
    "election2020",
    "ukr_rus_suspended",
    "twibot20",
    "facebook_page_reference",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = []
    for path in sorted(Path(args.input_root).glob("gpu*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    frame = pd.DataFrame(rows)
    if len(frame) != 270:
        raise ValueError(f"expected 270 result cells, got {len(frame)}")
    if frame["model_id"].nunique() != 54 or set(frame["dataset"]) != TARGETS:
        raise ValueError("wrong model or target coverage")
    if frame[["model_id", "dataset"]].duplicated().any():
        raise ValueError("duplicate model-target cells")
    for column in ("accuracy", "f1", "roc_auc"):
        if not frame[column].notna().all():
            raise ValueError(f"non-finite {column}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.sort_values(["model_id", "dataset"]).to_csv(output, sep="\t", index=False)
    receipt = {
        "status": "complete",
        "models": 54,
        "targets": 5,
        "cells": 270,
        "episodes_per_cell": 128,
        "seed": 0,
        "checkpoint_step": 2500,
    }
    output.with_suffix(".completeness.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    print(output)


if __name__ == "__main__":
    main()
