#!/usr/bin/env python3
"""Combine corrected all-target LP evaluations into one long-form TSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


SOURCES = (
    "ukr_rus_twitter", "covid19_twitter", "midterm", "covid_political",
    "election2020", "ukr_rus_suspended", "twibot20", "cp_hk_twitter",
    "facebook_page_reference",
)
ALIASES = {
    "ukr_rus": "ukr_rus_twitter", "covid": "covid19_twitter",
    "midterm": "midterm", "covid_political": "covid_political",
    "election2020": "election2020", "ukr_rus_suspended": "ukr_rus_suspended",
    "twibot20": "twibot20", "cp_hk": "cp_hk_twitter",
    "facebook_page_reference": "facebook_page_reference",
}
FIELDS = (
    "architecture", "pretraining_objective", "model_id", "n_sources", "sources",
    "target", "auc", "average_precision", "hits_at_50", "orientation",
    "n_test_pairs", "holdout_leakage_edges", "endpoint_sensitivity",
    "endpoint_permutation_auc", "checkpoint_step", "seed", "protocol",
)


def prodigy_sources(model_id: str) -> tuple[str, ...]:
    if model_id.startswith("nmpair_"):
        names = model_id.removeprefix("nmpair_").split("__")
        return tuple(ALIASES[name] for name in names)
    if model_id.startswith("nmloo_without_"):
        held_out = ALIASES[model_id.removeprefix("nmloo_without_")]
        return tuple(name for name in SOURCES if name != held_out)
    if model_id.startswith("finalcore_ss_") and model_id.endswith("_s0"):
        alias = model_id.removeprefix("finalcore_ss_").removesuffix("_s0")
        return (ALIASES[alias],)
    raise ValueError(f"unrecognized PRODIGY model id: {model_id}")


def read_rows(path: Path, delimiter: str = ","):
    with path.open(newline="") as handle:
        yield from csv.DictReader(handle, delimiter=delimiter)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prodigy-dir", required=True, type=Path)
    parser.add_argument("--sage-lp", required=True, type=Path)
    parser.add_argument("--sage-graphmae", required=True, type=Path)
    parser.add_argument("--samgpt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output: list[dict[str, object]] = []

    for path in sorted(args.prodigy_dir.glob("*__pair_lp.csv")):
        for row in read_rows(path):
            if row["model"] == "__floor__" or row["negative_kind"] != "degree_matched":
                continue
            sources = prodigy_sources(row["model"])
            output.append({
                "architecture": "prodigy", "pretraining_objective": "nm",
                "model_id": row["model"], "n_sources": len(sources),
                "sources": ",".join(sources), "target": row["dataset"],
                "auc": row["auc"], "average_precision": row["average_precision"],
                "hits_at_50": row["hits_at_50"], "orientation": row["orientation"],
                "n_test_pairs": row["n_pairs"],
                "holdout_leakage_edges": row["leakage_edges"],
                "endpoint_sensitivity": row["endpoint_sensitivity"],
                "endpoint_permutation_auc": row["endpoint_permutation_auc"],
                "checkpoint_step": 2500, "seed": 0,
                "protocol": "prodigy_frozen_static_lp_degree_matched_cached_v2",
            })

    for objective, path in (("lp", args.sage_lp), ("graphmae", args.sage_graphmae)):
        for row in read_rows(path, "\t"):
            output.append({
                "architecture": "graphsage", "pretraining_objective": objective,
                "model_id": row["run_id"], "n_sources": row["mixture_size"],
                "sources": row["sources"], "target": row["target"], "auc": row["auc"],
                "average_precision": "", "hits_at_50": "", "orientation": "",
                "n_test_pairs": 2800, "holdout_leakage_edges": 0,
                "endpoint_sensitivity": "", "endpoint_permutation_auc": "",
                "checkpoint_step": row["checkpoint_step"], "seed": 0,
                "protocol": "graphsage_frozen_static_lp_degree_matched_v1",
            })

    for row in read_rows(args.samgpt):
        output.append({
            "architecture": "samgpt", "pretraining_objective": "samgpt",
            "model_id": row["model_id"], "n_sources": row["n_sources"],
            "sources": row["sources"], "target": row["target"], "auc": row["auc"],
            "average_precision": row["average_precision"],
            "hits_at_50": row["hits_at_50"], "orientation": row["orientation"],
            "n_test_pairs": row["n_test_pairs"],
            "holdout_leakage_edges": row["holdout_leakage_edges"],
            "endpoint_sensitivity": row["endpoint_sensitivity"],
            "endpoint_permutation_auc": row["endpoint_permutation_auc"],
            "checkpoint_step": row["checkpoint_update"], "seed": row["seed"],
            "protocol": "samgpt_frozen_static_lp_degree_matched_v1",
        })

    output.sort(key=lambda row: (
        str(row["architecture"]), str(row["pretraining_objective"]),
        int(row["n_sources"]), str(row["model_id"]), str(row["target"]),
    ))
    keys = [(r["architecture"], r["pretraining_objective"], r["model_id"], r["target"]) for r in output]
    if len(output) != 1944 or len(set(keys)) != 1944:
        raise RuntimeError(f"expected 1,944 unique rows, found {len(output)} rows/{len(set(keys))} keys")
    groups: dict[tuple[object, object], int] = {}
    for row in output:
        key = (row["architecture"], row["pretraining_objective"])
        groups[key] = groups.get(key, 0) + 1
        if row["target"] not in SOURCES or not (0.0 <= float(row["auc"]) <= 1.0):
            raise ValueError(f"invalid row: {row}")
    if set(groups.values()) != {486}:
        raise RuntimeError(f"incomplete architecture/objective blocks: {groups}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(output)
    print(f"wrote {len(output)} rows to {args.output}")
    print(groups)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
