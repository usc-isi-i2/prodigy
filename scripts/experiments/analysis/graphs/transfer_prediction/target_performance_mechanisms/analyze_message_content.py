"""Validate and summarize complete frozen message-content controls."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean


def summarize(rows):
    baseline = {(r["target"], r["stream"], r["model_id"]): r for r in rows if r["condition"] == "intact"}
    paired = []
    for r in rows:
        b = baseline[(r["target"], r["stream"], r["model_id"])]
        paired.append({**r, **{"delta_"+k: r[k]-b[k] for k in ("roc_auc", "accuracy", "f1", "nll")}})
    groups = {}
    for r in paired:
        groups.setdefault((r["target"], r["source"], r["condition"], r["role"]), []).append(r)
    output = []
    for key, group in sorted(groups.items()):
        row = dict(zip(("target", "source", "condition", "role"), key))
        row["seed_streams"] = len(group)
        for metric in ("roc_auc", "accuracy", "nll"):
            v = [r["delta_"+metric] for r in group]
            row.update({metric+"_mean": mean(v), metric+"_min": min(v), metric+"_max": max(v),
                        metric+"_positive": sum(x > 0 for x in v), metric+"_negative": sum(x < 0 for x in v)})
        output.append(row)
    return paired, output


def write_csv(path, rows):
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    read = lambda name: json.loads((args.input / f"{name}.json").read_text())
    protocol, done, rows, receipts, audits = [read(n) for n in ("protocol", "DONE", "metrics", "receipts", "aggregation_audit")]
    if done["smoke_only"] or protocol["smoke_only"] or not done["all_baseline_and_zero_metrics_match"] or not done["all_weights_and_operators_restored"]:
        raise ValueError("complete full-input replay required")
    panel = {(t, s, m) for t in protocol["targets"] for s in ("original", "fresh") for m in protocol["models"]}
    keys = [(r["target"], r["stream"], r["model_id"], r["condition"], r["role"]) for r in rows]
    expected = {(*cell, *condition) for cell in panel for condition in protocol["conditions"]}
    if len(keys) != len(set(keys)) or set(keys) != expected or len(keys) != done["cells"]:
        raise ValueError("incomplete or duplicated result grid")
    for records in (receipts, audits):
        if len(records) != len(panel) or {(r["target"], r["stream"], r["model_id"]) for r in records} != panel:
            raise ValueError("incomplete audit panel")
    for r in receipts:
        if not r["baseline_and_zero_match"] or not r["state_and_operator_restored"] or len(r["role_checks"]) != 10:
            raise ValueError("incomplete restoration/role checks")
        if max(x["max_error"] for x in r["role_checks"]) > 1e-5:
            raise ValueError("role factorization failed")
    for r in audits:
        if r["aggregation"] != [{"layer": 0, "configured_aggr": "mean", "aggregation_module": "SumAggregation", "unit_message_probe": [1., 2., 0.]}]:
            raise ValueError("actual aggregation contract differs")
    if not all(math.isfinite(r[k]) for r in rows for k in ("roc_auc", "accuracy", "f1", "nll")):
        raise ValueError("nonfinite metric")
    paired, summary = summarize(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "cells.csv", paired)
    write_csv(args.output / "source_summary.csv", summary)
    validation = {"cells": len(rows), "model_target_streams": len(panel), "runtime_revision": protocol["revision"],
                  "sha256": {n: hashlib.sha256((args.input / f"{n}.json").read_bytes()).hexdigest()
                             for n in ("protocol", "DONE", "metrics", "receipts", "aggregation_audit")},
                  "uncertainty": "Observed seed/stream ranges, not confidence intervals; no new training."}
    (args.output / "validation.json").write_text(json.dumps(validation, indent=2)+"\n")
    print(json.dumps({"validation": validation, "support_controls": [r for r in summary if r["role"] == "support"]}, indent=2))


if __name__ == "__main__":
    main()
