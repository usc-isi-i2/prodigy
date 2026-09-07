"""Independently regenerate gradient-audit summaries from retained full tensors."""
import argparse
import csv
import json
from pathlib import Path

import torch

from .replay import batch_hash
from .support_identity_gradients import (VARIANTS, MODES, support_plan, verify_interventions,
    probe_summary, gradient_comparison, tensor_receipt)


def compare(a, b, where="root"):
    if isinstance(a, dict):
        if not isinstance(b, dict) or a.keys() != b.keys():
            raise ValueError(f"different fields at {where}")
        for k in a:
            compare(a[k], b[k], where+"."+k)
    elif isinstance(a, list):
        if not isinstance(b, list) or len(a) != len(b):
            raise ValueError(f"different lists at {where}")
        for i, (x, y) in enumerate(zip(a, b)):
            compare(x, y, where+f"[{i}]")
    elif a != b:
        # Same version/threads, deterministic saved tensor reductions: no
        # approximate aggregate tolerance is needed for this receipt check.
        raise ValueError(f"different value at {where}: {a!r} versus {b!r}")


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists() or torch.cuda.is_available():
        raise ValueError("new output and no visible GPU required")
    torch.set_num_threads(4)
    done = json.loads((args.run/"DONE.json").read_text())
    protocol = json.loads((args.run/"protocol.json").read_text())
    if not done["complete"] or done["phase"] != "probe" or not done["no_optimizer_updates"]:
        raise ValueError("incomplete probe campaign")
    smoke = done["smoke"]
    caches = {}
    for root in sorted(args.inputs.glob("job_*")):
        receipt = json.loads((root/"DONE.json").read_text())
        caches[receipt["arm"]["model_id"]] = (root, receipt)
    cells, gradients, bindings, groups, inventory = [], [], [], [], []
    keys, repeats = set(), 0
    for root in sorted(args.run.glob("job_*")):
        receipt = json.loads((root/"DONE.json").read_text())
        model_id = receipt["model_id"]
        cache_root, cache = caches[model_id]
        arm = cache["arm"]
        rows = [json.loads(s) for s in (root/"cells.jsonl").read_text().splitlines()]
        by_key = {(r["input_step"], r["checkpoint_step"], r["mode"], r["variant"]): r for r in rows}
        expected = {(i, ck, mode, v) for i in range(1, 2 if smoke else 5) for ck in (0, 2500) for mode in MODES for v in VARIANTS}
        if len(rows) != len(expected) or set(by_key) != expected or not receipt["all_baseline_repeats_bit_exact"]:
            raise ValueError("missing or duplicate model cell")
        for input_step in range(1, 2 if smoke else 5):
            batch = torch.load(cache_root/f"batch_{input_step:03d}.pt", map_location="cpu", weights_only=False)
            digest = batch_hash(batch)
            if digest != cache["inputs"][input_step-1]["batch_sha256"]:
                raise ValueError("retained input hash mismatch")
            plan = support_plan(batch, 99101+arm["seed"]*100+input_step)
            intervention = verify_interventions(batch, plan)
            for checkpoint in (0, 2500):
                expected_weights = arm["initial_sha256" if checkpoint == 0 else "final_sha256"]
                for mode in MODES:
                    path = root/f"input{input_step}_ckpt{checkpoint}_{mode}.pt"
                    payload = torch.load(path, map_location="cpu", weights_only=False)
                    if payload["input_sha256"] != digest or payload["weights_sha256"] != expected_weights:
                        raise ValueError("tensor provenance mismatch")
                    saved_plan = payload["plan"]
                    for k in ("ids", "query", "tasks", "classes"):
                        torch.testing.assert_close(saved_plan[k], plan[k], rtol=0, atol=0)
                    for k in ("groups", "null_groups"):
                        compare([v.tolist() for v in saved_plan[k]], [v.tolist() for v in plan[k]])
                    tensors = payload["conditions"]
                    if set(tensors) != set(VARIANTS):
                        raise ValueError("missing retained condition")
                    baseline = tensors["baseline"]
                    truth = batch[2][plan["query"]]
                    if checkpoint == 0 and input_step == 1 and mode == "training":
                        compare(tensor_receipt(baseline), cache["reference_probe"], "actual_trainer_reference")
                    repeats += 1
                    meta = {"model_id": model_id, "source": arm["source"], "seed": arm["seed"],
                        "input_step": input_step, "checkpoint_step": checkpoint, "mode": mode}
                    for variant, tensor in tensors.items():
                        torch.testing.assert_close(tensor["truth"], truth, rtol=0, atol=0)
                        record = by_key[(input_step, checkpoint, mode, variant)]
                        if (record["model_id"], record["source"], record["seed"], record["input_sha256"], record["weights_sha256"]) != (
                            model_id, arm["source"], arm["seed"], digest, expected_weights):
                            raise ValueError("journal provenance mismatch")
                        summary = probe_summary(tensor, baseline, plan)
                        compare({k: record[k] for k in summary}, summary, "recomputed_summary")
                        compare(record["intervention_audit"], intervention, "intervention")
                        if mode == "meta_frozen" and summary["post_query_max_difference"] != 0:
                            raise ValueError("frozen-BN query changed")
                        key = (model_id, input_step, checkpoint, mode, variant)
                        if key in keys:
                            raise ValueError("duplicate overall cell")
                        keys.add(key)
                        row = {**meta, "variant": variant, "input_sha256": digest, "weights_sha256": expected_weights,
                            **{k: summary[k] for k in ("pre_bit_exact", "post_query_max_difference", "post_label_max_difference")},
                            **intervention}
                        for cohort, values in summary["cohorts"].items():
                            row.update({cohort+"_"+k: v for k, v in values.items()})
                        cells.append(row)
                        gradients.extend({**meta, "variant": variant, "block": block, **value}
                            for block, value in summary["gradient_comparison"].items())
                        for kind in ("identity_group_gradients", "null_group_gradients"):
                            groups.extend({**meta, "variant": variant, "kind": kind, "group": i,
                                **{k: v for k, v in value.items() if k != "positions"}}
                                for i, value in enumerate(summary[kind]))
                    binding = gradient_comparison(tensors["identity_soft"]["gradients"], tensors["permuted_soft"]["gradients"])
                    compare(binding, json.loads((root/f"input{input_step}_ckpt{checkpoint}_{mode}_actual_vs_null.json").read_text()))
                    bindings.extend({**meta, "block": block, **value} for block, value in binding.items())
                    inventory.append({**meta, "tensor_path": str(path), "tensor_bytes": path.stat().st_size,
                        "input_sha256": digest, "weights_sha256": expected_weights})
                    del tensors, payload, baseline
            del batch
        print(f"Verified full retained tensors: {model_id}", flush=True)
    expected_models = {"readouttrain_cp_hk_free_s0"} if smoke else {
        f"readouttrain_{s}_free_s{i}" for s in ("ukr_rus", "cp_hk", "covid") for i in range(3)}
    if {c["model_id"] for c in cells} != expected_models or len(cells) != (12 if smoke else 432):
        raise ValueError("incomplete overall campaign grid")
    args.output.mkdir(parents=True)
    for name, rows in (("cells", cells), ("gradients", gradients), ("binding_gradients", bindings), ("groups", groups)):
        with (args.output/f"{name}.csv").open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (args.output/"inventory.json").write_text(json.dumps(inventory, indent=2))
    result = {"complete": True, "smoke": smoke, "cells": len(cells), "models": len(expected_models),
        "input_batches": len(expected_models)*(1 if smoke else 4), "checkpoint_steps": [0, 2500],
        "every_reported_tensor_digest_and_gradient_statistic_recomputed": True,
        "every_input_full_hash_reverified": True, "observed_deterministic_baseline_repeats": repeats,
        "pre_metagraph_identical_under_label_interventions": True, "frozen_meta_bn_query_vectors_bit_exact": True,
        "no_target_outcomes": True, "no_optimizer_updates": True, "runtime_revision": protocol["revision"],
        "probe_root": str(args.run), "inputs_root": str(args.inputs)}
    (args.output/"DONE.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
