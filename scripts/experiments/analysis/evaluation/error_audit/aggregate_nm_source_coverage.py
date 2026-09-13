"""Aggregate replicated NM source-manifold audit receipts."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


METRICS = ("target_similarity", "other_similarity", "coverage_gap")
AUC_TYPES = ("correctness_auc", "node_balanced_correctness_auc")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def distribution(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(values.mean()),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "values": values.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipts", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    receipts = [json.loads(path.read_text()) for path in args.receipts]
    replicates = [int(item["reference_replicate"]) for item in receipts]
    if len(receipts) < 2 or len(set(replicates)) != len(receipts):
        raise ValueError("need at least two distinct reference replicates")
    invariant_keys = ("protocol", "reference_size", "top_k", "chunk_size", "input_hashes")
    for key in invariant_keys:
        canonical = receipts[0][key]
        if any(item[key] != canonical for item in receipts[1:]):
            raise ValueError(f"replicate mismatch for {key}")

    report = {
        "complete": True,
        "protocol": receipts[0]["protocol"],
        "replicates": replicates,
        "reference_size": receipts[0]["reference_size"],
        "top_k": receipts[0]["top_k"],
        "geometry_uses_outcomes": receipts[0]["geometry_uses_outcomes"],
        "identity_exclusion": receipts[0]["identity_exclusion"],
        "reference_selection": receipts[0]["reference_selection"],
        "input_hashes": receipts[0]["input_hashes"],
        "receipt_hashes": {
            str(path): digest(path) for path in args.receipts
        },
        "targets": {},
    }
    for target in receipts[0]["targets"]:
        report["targets"][target] = {}
        for model in receipts[0]["targets"][target]["models"]:
            model_out = {"auc": {}, "cohorts": {}}
            for auc_type in AUC_TYPES:
                model_out["auc"][auc_type] = {
                    metric: distribution([
                        item["targets"][target]["models"][model][auc_type][metric]
                        for item in receipts
                    ])
                    for metric in METRICS
                }
            cohorts = receipts[0]["targets"][target]["models"][model]["cohorts"]
            for cohort in cohorts:
                cohort_out = {
                    "rows": cohorts[cohort]["rows"],
                    "unique_queries": cohorts[cohort]["unique_queries"],
                }
                for metric in METRICS:
                    cohort_out[metric] = {
                        weighting: distribution([
                            item["targets"][target]["models"][model]["cohorts"]
                            [cohort][metric][weighting]
                            for item in receipts
                        ])
                        for weighting in ("occurrence_mean", "node_weighted_mean")
                    }
                cohort_out["rows"] = cohorts[cohort]["rows"]
                cohort_out["unique_queries"] = cohorts[cohort]["unique_queries"]
                model_out["cohorts"][cohort] = cohort_out
            report["targets"][target][model] = model_out

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
