"""Aggregate cached four-source NM pre-metagraph/final predictions."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


SOURCES = ("ukr_rus", "covid", "midterm", "cp_hk")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def metrics(part):
    pre = part.pre_correct.astype(bool)
    final = part.final_correct.astype(bool)
    return {
        "rows": int(len(part)),
        "unique_queries": int(part["query"].nunique()),
        "pre_accuracy": float(pre.mean()),
        "final_accuracy": float(final.mean()),
        "net_readout_accuracy": float(final.mean() - pre.mean()),
        "correct_to_wrong": int((pre & ~final).sum()),
        "wrong_to_correct": int((~pre & final).sum()),
        "both_wrong": int((~pre & ~final).sum()),
        "both_correct": int((pre & final).sum()),
        "mean_pre_rank": float(part.pre_rank.mean()),
        "mean_final_rank": float(part.final_rank.mean()),
        "mean_rank_improvement": float((part.pre_rank - part.final_rank).mean()),
    }


def paired_cohorts(frame, native, foreign):
    key = ["episode", "sample", "query", "anchor"]
    columns = ["pre_correct", "final_correct", "pre_rank", "final_rank"]
    wide = frame[frame.model.isin([native, foreign])].pivot(
        index=key, columns="model", values=columns
    )
    wide.columns = [f"{metric}_{model}" for metric, model in wide.columns]
    wide = wide.reset_index()
    native_final = wide[f"final_correct_{native}"].astype(bool)
    foreign_final = wide[f"final_correct_{foreign}"].astype(bool)
    result = {}
    masks = {
        "native_only": native_final & ~foreign_final,
        "foreign_only": ~native_final & foreign_final,
        "both_correct": native_final & foreign_final,
        "both_wrong": ~native_final & ~foreign_final,
    }
    for name, mask in masks.items():
        part = wide[mask]
        result[name] = {
            "rows": int(len(part)),
            "native_pre_correct_fraction": float(part[f"pre_correct_{native}"].mean()),
            "foreign_pre_correct_fraction": float(part[f"pre_correct_{foreign}"].mean()),
            "native_mean_pre_rank": float(part[f"pre_rank_{native}"].mean()),
            "native_mean_final_rank": float(part[f"final_rank_{native}"].mean()),
            "foreign_mean_pre_rank": float(part[f"pre_rank_{foreign}"].mean()),
            "foreign_mean_final_rank": float(part[f"final_rank_{foreign}"].mean()),
            "native_better_pre_rank_fraction": float(
                (part[f"pre_rank_{native}"] < part[f"pre_rank_{foreign}"]).mean()
            ),
            "equal_pre_rank_fraction": float(
                (part[f"pre_rank_{native}"] == part[f"pre_rank_{foreign}"]).mean()
            ),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_text())
    if not receipt.get("complete") or not receipt.get("model_states_unchanged"):
        raise ValueError("incomplete or mutated source receipt")
    report = {
        "protocol": "nm_broad_stage_aggregate_v1",
        "source_protocol": receipt["protocol"],
        "randomized_published_fixed_test": True,
        "seed": 0,
        "receipt_sha256": digest(args.receipt),
        "targets": {},
    }
    for target in SOURCES:
        path = args.input_root / target / "stage_predictions_private.csv"
        frame = pd.read_csv(path)
        if len(frame) != 245760 or frame.duplicated(
            ["episode", "sample", "model"]
        ).any():
            raise ValueError(f"unexpected rows for {target}")
        cells = {model: metrics(frame[frame.model.eq(model)]) for model in SOURCES}
        native = target
        foreign = {}
        for model in SOURCES:
            if model != native:
                foreign[model] = paired_cohorts(frame, native, model)
        report["targets"][target] = {
            "native_model": native,
            "csv_sha256": digest(path),
            "cells": cells,
            "native_vs_foreign_cohorts": foreign,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
