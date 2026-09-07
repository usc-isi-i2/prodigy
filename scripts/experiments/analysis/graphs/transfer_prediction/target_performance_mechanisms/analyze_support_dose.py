"""Validate and summarize the complete saved-step/dose replay, all targets."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean

import numpy as np


STEPS = (0, 100, 300, 900, 2500)
TARGETS = {"covid_political", "election2020", "facebook_page_reference", "twibot20", "ukr_rus_suspended"}
METRICS = ("roc_auc", "accuracy", "f1", "nll")


def summarize(rows):
    base = {(r["target"], r["stream"], r["model_id"], r["step"]): r for r in rows if r["suppression_percent"] == 0}
    paired = []
    for r in rows:
        b = base[r["target"], r["stream"], r["model_id"], r["step"]]
        paired.append({**r, **{"delta_"+m: r[m]-b[m] for m in METRICS}})
    step_groups, dose_groups = {}, {}
    for r in paired:
        if r["suppression_percent"] == 100:
            step_groups.setdefault((r["target"], r["source"], r["step"]), []).append(r)
        if r["step"] == 2500:
            dose_groups.setdefault((r["target"], r["stream"], r["model_id"], r["source"], r["seed"], r["suppression_percent"]), []).append(r)
    step_summary = []
    for key, group in sorted(step_groups.items()):
        r = dict(zip(("target", "source", "step"), key), seed_streams=len(group))
        for metric in METRICS:
            values = [v["delta_"+metric] for v in group]
            r.update({metric+"_mean": mean(values), metric+"_min": min(values), metric+"_max": max(values),
                      metric+"_positive": sum(v > 0 for v in values), metric+"_negative": sum(v < 0 for v in values)})
        step_summary.append(r)
    dose_cells = []
    for key, group in sorted(dose_groups.items()):
        r = dict(zip(("target", "stream", "model_id", "source", "seed", "suppression_percent"), key), draws=len(group))
        for metric in METRICS:
            r[metric] = mean(v[metric] for v in group)
            r["delta_"+metric] = mean(v["delta_"+metric] for v in group)
        dose_cells.append(r)
    curves = {}
    for r in dose_cells:
        curves.setdefault((r["target"], r["stream"], r["model_id"], r["source"], r["seed"]), {})[r["suppression_percent"]] = r
    monotonicity = []
    for key, curve in sorted(curves.items()):
        doses = sorted(curve)
        values = np.array([curve[d]["roc_auc"] for d in doses])
        diffs = np.diff(values)
        r = dict(zip(("target", "stream", "model_id", "source", "seed"), key))
        r.update({"auc_nondecreasing": bool((diffs >= -1e-12).all()), "auc_nonincreasing": bool((diffs <= 1e-12).all()),
                  "auc_endpoint_gain": float(values[-1]-values[0]),
                  "largest_auc_up_step": float(diffs.max()), "largest_auc_down_step": float(diffs.min()),
                  **{f"auc_delta_at_{d}": curve[d]["delta_roc_auc"] for d in doses}})
        monotonicity.append(r)
    return paired, step_summary, dose_cells, monotonicity


def write_csv(path, rows):
    with path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    read = lambda n: json.loads((args.input / f"{n}.json").read_text())
    protocol, done, rows, receipts, inventory, masks = [read(n) for n in ("protocol", "DONE", "metrics", "receipts", "checkpoint_inventory", "mask_receipts")]
    if protocol["smoke_only"] or done["smoke_only"] or not done["endpoint_reference_matches"] or not done["weights_unchanged"]:
        raise ValueError("complete full-batch replay required")
    if protocol["steps"] != list(STEPS) or set(protocol["targets"]) != TARGETS or set(protocol["seeds"]) != {0, 1, 2} or protocol["draws"] != 3:
        raise ValueError("complete declared steps/targets/seeds/draws required")
    for step in STEPS:
        required = {(0, 0), (100, 0)} | ({(dose, draw) for dose in (25, 50, 75) for draw in range(3)} if step == 2500 else set())
        if set(map(tuple, protocol["conditions_by_step"][str(step)])) != required:
            raise ValueError("incorrect dose conditions for saved step")
    model_ids = set(protocol["models"])
    if len(model_ids) != 6:
        raise ValueError("six source/seed controls required")
    panel = {(t, s, m, step) for t in TARGETS for s in ("original", "fresh") for m in model_ids for step in STEPS}
    expected = {(*p, dose, draw) for p in panel for dose, draw in protocol["conditions_by_step"][str(p[-1])]}
    keys = [(r["target"], r["stream"], r["model_id"], r["step"], r["suppression_percent"], r["draw"]) for r in rows]
    if len(rows) != 1140 or len(keys) != len(set(keys)) or set(keys) != expected or done["cells"] != 1140:
        raise ValueError("incomplete or duplicate metric panel")
    if len(receipts) != 300 or {(r["target"], r["stream"], r["model_id"], r["step"]) for r in receipts} != panel:
        raise ValueError("incomplete frozen-input receipts")
    for r in receipts:
        count = 32*(len(protocol["conditions_by_step"][str(r["step"])])-1)
        if not r["weights_unchanged"] or not r["completed_steps_verified"] or r["query_pre_exact_checks"] != count or r["query_post_exact_checks"] != count or r["direct_suffix_max_error"] > 1e-5:
            raise ValueError("frozen checkpoint, query role or direct suffix check failed")
        if r["step"] == 2500 and not r["endpoint_reference_match"]:
            raise ValueError("historical endpoint does not match")
    if len(inventory) != 30 or {(r["model_id"], r["step"]) for r in inventory} != {(m, step) for m in model_ids for step in STEPS}:
        raise ValueError("incomplete checkpoint inventory")
    for r in inventory:
        if r["available_steps"] != list(STEPS) or not r["completed_steps_verified"]:
            raise ValueError("not all saved steps evaluated")
    if {(r["source"], r["seed"]) for r in inventory} != {(s, n) for s in ("cp_hk", "ukr_rus") for n in range(3)}:
        raise ValueError("incorrect source/seed coverage")
    # Source models share actual saved initializations within a seed. Verify
    # both their weights and every initial-condition metric, not just averages.
    for seed in range(3):
        if len({r["weights_sha256"] for r in inventory if r["step"] == 0 and r["seed"] == seed}) != 1:
            raise ValueError("initial weights differ between source arms")
    initial = {}
    for r in rows:
        if not all(np.isfinite(r[m]) for m in METRICS):
            raise ValueError("nonfinite metric")
        if r["step"] == 0:
            k = r["target"], r["stream"], r["seed"], r["suppression_percent"]
            if k in initial:
                np.testing.assert_allclose([r[m] for m in METRICS], initial[k], atol=1e-12, rtol=0)
            initial[k] = [r[m] for m in METRICS]
    mask_keys = [(r["target"], r["stream"], r["batch"], r["draw"], r["suppression_percent"]) for r in masks]
    mask_expected = {(t, s, b, d, f) for t in TARGETS for s in ("original", "fresh") for b in range(32) for d in range(3) for f in (0, 25, 50, 75, 100)}
    if len(mask_keys) != len(set(mask_keys)) or set(mask_keys) != mask_expected:
        raise ValueError("incomplete mask receipts")
    groups = {}
    for r in masks:
        if not 0 <= r["retained_support_edges"] <= r["support_edges"]:
            raise ValueError("invalid edge counts")
        if r["suppression_percent"] in (0, 100) and r["retained_support_edges"] != (r["support_edges"] if r["suppression_percent"] == 0 else 0):
            raise ValueError("incorrect mask endpoint")
        groups.setdefault((r["target"], r["stream"], r["suppression_percent"], r["draw"]), []).append(r)
    fractions = []
    for key, group in sorted(groups.items()):
        total = sum(r["support_edges"] for r in group)
        kept = sum(r["retained_support_edges"] for r in group)
        fractions.append({**dict(zip(("target", "stream", "suppression_percent", "draw"), key)),
                          "support_edges": total, "retained_support_edges": kept,
                          "actual_suppressed_fraction": 1-kept/total if total else 0,
                          "emptied_support_subgraphs": sum(r["emptied_support_subgraphs"] for r in group),
                          "nonempty_support_subgraphs": sum(r["nonempty_support_subgraphs"] for r in group)})
    paired, step_summary, dose_cells, monotonicity = summarize(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, values in (("cells", paired), ("step_summary", step_summary), ("dose_cells", dose_cells), ("monotonicity", monotonicity), ("actual_fractions", fractions)):
        write_csv(args.output / f"{name}.csv", values)
    validation = {"cells": len(rows), "saved_checkpoints": len(inventory), "input_model_step_receipts": len(receipts),
                  "query_pre_exact_checks": sum(r["query_pre_exact_checks"] for r in receipts),
                  "query_post_exact_checks": sum(r["query_post_exact_checks"] for r in receipts),
                  "initial_source_parity": True, "runtime_revision": protocol["revision"],
                  "sha256": {n: hashlib.sha256((args.input / f"{n}.json").read_bytes()).hexdigest() for n in ("protocol", "DONE", "metrics", "receipts", "checkpoint_inventory", "mask_receipts")},
                  "uncertainty": "Observed seed/stream ranges, not confidence intervals. Dose means average the three draws within each seed/stream; shared initialization is not two independent source models."}
    (args.output / "validation.json").write_text(json.dumps(validation, indent=2)+"\n")
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
