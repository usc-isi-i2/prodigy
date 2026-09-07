"""Audit a completed new comparator against matched existing native training."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from scripts.experiments.setup.trace_schedule_scaling.verify_training import find_audit, read_audit, payload_digest


def grouped(rows):
    result = {}
    for row in rows:
        source = tuple(sorted(set(row["source_ids"])))
        if len(source) != 1:
            raise ValueError("Expected source-private episodes")
        result.setdefault(source[0], []).append(row)
    return result


def compare_payloads(current, reference):
    a, b = grouped(current), grouped(reference)
    if a.keys() != b.keys():
        raise ValueError("Source inventories differ")
    for source, rows in a.items():
        if len(b[source]) < len(rows) or payload_digest(rows) != payload_digest(b[source][:len(rows)]):
            raise ValueError(f"Consumed payload differs for source {source}")
    return {str(k): len(v) for k, v in a.items()}


def verify(root, reference):
    done = json.loads((root / "DONE.json").read_text())
    plan = json.loads((root / "plan.json").read_text())
    previous = json.loads((reference / "plan.json").read_text())
    old_done = json.loads((reference / "DONE.json").read_text())
    if done["arms"] != 2 or done["steps"] not in (20, 2500) or len(plan) != 2:
        raise ValueError("Expected both completed nominated arms")
    if old_done["arms"] != 8 or old_done["steps"] != 2500:
        raise ValueError("Expected completed original eight-arm comparison")
    if {e["arm"]["schedule"] for e in plan} != {"blocked", "interleaved"}:
        raise ValueError("Schedule inventory differs")
    reports = []
    for entry in plan:
        if entry["mode"] != "ridge_centered_scaled" or entry["arm"]["seed"] != 0:
            raise ValueError("Unexpected objective or seed")
        candidates = [e for e in previous if e["mode"] == "native" and e["arm"]["schedule"] == entry["arm"]["schedule"]]
        if len(candidates) != 1:
            raise ValueError("Ambiguous reference arm")
        old = candidates[0]
        name, old_name = entry["name"] + "_isolation_v1", old["name"] + "_isolation_v1"
        directory = root / "state" / name / "checkpoint"
        old_directory = reference / "state" / old_name / "checkpoint"
        init_path, old_path = directory / "state_dict_0.ckpt", old_directory / "state_dict_0.ckpt"
        initial, initial_ref = [torch.load(p, map_location="cpu", weights_only=True)["model"] for p in (init_path, old_path)]
        if initial.keys() != initial_ref.keys() or any(initial[k].dtype != initial_ref[k].dtype or not torch.equal(initial[k], initial_ref[k]) for k in initial):
            raise ValueError("Initial model states differ")
        training = torch.load(directory / f"training_state_{done['steps']}.ckpt", map_location="cpu", weights_only=False)
        meta = training["_training_checkpoint"]
        if meta["completed_steps"] != done["steps"] or meta["parameter_contract"]["encoder_solver_objective"] != entry["mode"]:
            raise ValueError("Training metadata differs")
        rows = read_audit(find_audit(root / "native_log", name))
        if len(rows) != done["steps"] or [r["step"] for r in rows] != list(range(1, done["steps"] + 1)):
            raise ValueError("Incomplete consumed-episode audit")
        counts = compare_payloads(rows, read_audit(find_audit(reference / "native_log", old_name)))
        final_path = directory / f"state_dict_{done['steps']}.ckpt"
        final = torch.load(final_path, map_location="cpu", weights_only=True)["model"]
        scale = final["logit_scale"].exp()
        if not torch.isfinite(scale) or scale <= 0 or torch.equal(final["logit_scale"], initial["logit_scale"]):
            raise ValueError("Training scale is invalid or did not update")
        reports.append(dict(model=name, steps=done["steps"], initialization_exact=True,
            source_payload_prefix_exact=True, source_counts=counts, positive_scale=float(scale),
            checkpoint_sha256=hashlib.sha256(final_path.read_bytes()).hexdigest()))
    return dict(arms=reports, scope="Payload audit covers IDs, roles and sampled topology, not every feature byte")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.root, args.reference), indent=2))
