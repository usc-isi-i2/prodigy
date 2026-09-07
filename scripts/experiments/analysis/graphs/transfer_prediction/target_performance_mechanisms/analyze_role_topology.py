"""Summarize a complete, provenance-checked role x topology replay."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean


def read(root, name):
    return json.loads((root / f"{name}.json").read_text())


def validate(root):
    protocol, done, rows, receipts = [read(root, n) for n in ("protocol", "DONE", "metrics", "receipts")]
    if protocol["smoke_only"] or done["smoke_only"] or not done["baseline_metrics_match"] or not done["all_weights_unchanged"]:
        raise ValueError("complete full-input replay required, not smoke")
    panel = {(target, stream, model) for target in protocol["targets"] for stream in ("original", "fresh") for model in protocol["models"]}
    conditions = {tuple(c) for c in protocol["conditions"]}
    expected = {(*cell, *condition) for cell in panel for condition in conditions}
    keys = [(r["target"], r["stream"], r["model_id"], r["query_condition"], r["support_condition"], r["draw"]) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected or done["cells"] != len(keys):
        raise ValueError("missing or duplicate result cells")
    if len(receipts) != len(panel) or {(r["target"], r["stream"], r["model_id"]) for r in receipts} != panel:
        raise ValueError("missing model/input receipts")
    for r in receipts:
        if not r["weights_unchanged"] or not r["baseline_metrics_match"]:
            raise ValueError("baseline or frozen checkpoint failed")
    for r in rows:
        if not all(math.isfinite(r[k]) for k in ("roc_auc", "accuracy", "f1", "nll")):
            raise ValueError("nonfinite metrics")
    topology = read(root, "topology")
    inputs = read(root, "input_geometry")
    input_keys = [(r["target"], r["stream"], r["batch"], r["subgraph"]) for r in inputs]
    if len(input_keys) != len(set(input_keys)):
        raise ValueError("duplicate original subgraph descriptors")
    if {(r["target"], r["stream"], r["batch"]) for r in inputs} != {(t, s, b) for t in protocol["targets"] for s in ("original", "fresh") for b in range(32)}:
        raise ValueError("missing input batches")
    topology_keys = [(r["target"], r["stream"], r["batch"], r["subgraph"], r["draw"]) for r in topology]
    if len(topology_keys) != len(set(topology_keys)) or set(topology_keys) != {(*k, d) for k in input_keys for d in range(protocol["rewire_draws"])}:
        raise ValueError("missing or duplicate rewiring receipts")
    if not all(r["degree_exact"] and 0 <= r["changed_edges"] <= r["edges"] and r["retained_edges"] == r["edges"] for r in topology):
        raise ValueError("invalid degree-exact null receipts")
    return protocol, rows, topology


def conditional_effects(rows):
    groups = {}
    for row in rows:
        key = row["target"], row["stream"], row["model_id"]
        groups.setdefault(key, {})[(row["query_condition"], row["support_condition"], row["draw"])] = row
    effects = []
    for (target, stream, model), cells in sorted(groups.items()):
        draws = sorted({key[2] for key in cells})
        base = cells[("intact", "intact", 0)]
        for draw in draws:
            def value(q, s, metric):
                return cells[(q, s, draw if "rewired" in (q, s) else 0)][metric]
            for change in ("removed", "rewired"):
                if change == "removed" and draw != 0:
                    continue
                effect = {"target": target, "stream": stream, "model_id": model,
                          "source": base["source"], "seed": base["seed"], "change": change, "draw": draw}
                for metric in ("roc_auc", "accuracy", "nll"):
                    b = value("intact", "intact", metric)
                    q = value(change, "intact", metric)
                    s = value("intact", change, metric)
                    both = value(change, change, metric)
                    effect.update({f"{metric}_baseline": b, f"{metric}_query_only": q-b,
                                   f"{metric}_support_only": s-b, f"{metric}_both": both-b,
                                   f"{metric}_query_given_support_changed": both-s,
                                   f"{metric}_support_given_query_changed": both-q,
                                   f"{metric}_interaction": both-q-s+b})
                effects.append(effect)
    return effects


def write_csv(path, rows):
    if not rows:
        raise ValueError("empty export")
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    protocol, rows, topology = validate(args.input)
    effects = conditional_effects(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "cells.csv", rows)
    write_csv(args.output / "conditional_effects.csv", effects)
    topology_summary = []
    for target in protocol["targets"]:
        for stream in ("original", "fresh"):
            for query in (False, True):
                subset = [r for r in topology if r["target"] == target and r["stream"] == stream and r["query"] == query]
                total = sum(r["edges"] for r in subset)
                topology_summary.append({"target": target, "stream": stream, "query": query,
                    "subgraph_draws": len(subset), "nonempty_subgraph_draws": sum(r["edges"] > 0 for r in subset),
                    "changed_subgraph_draws": sum(r["changed_edges"] > 0 for r in subset),
                    "edge_turnover": sum(r["changed_edges"] for r in subset) / total if total else None,
                    "successful_swaps": sum(r["accepted_swaps"] for r in subset)})
    write_csv(args.output / "topology_summary.csv", topology_summary)
    summary = []
    for target in protocol["targets"]:
        for source in ("cp_hk", "ukr_rus"):
            for change in ("removed", "rewired"):
                subset = [r for r in effects if (r["target"], r["source"], r["change"]) == (target, source, change)]
                result = {"target": target, "source": source, "change": change,
                          "seed_stream_draw_cells": len(subset)}
                for contrast in ("query_only", "support_only", "both", "query_given_support_changed", "interaction"):
                    values = [r[f"roc_auc_{contrast}"] for r in subset]
                    result.update({contrast + "_mean": mean(values), contrast + "_min": min(values), contrast + "_max": max(values)})
                summary.append(result)
    write_csv(args.output / "source_summary.csv", summary)
    checks = {"cells": len(rows), "contrasts": len(effects), "runtime_revision": protocol["revision"],
              "input_sha256": {name: hashlib.sha256((args.input / f"{name}.json").read_bytes()).hexdigest()
                               for name in ("protocol", "DONE", "metrics", "receipts", "topology")},
              "scope": "Ranges across seed, stream and rewiring draws; not confidence intervals. Rewiring draws are not training replications."}
    (args.output / "validation.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps({"validation": checks, "source_summary": summary}, indent=2))


if __name__ == "__main__":
    main()
