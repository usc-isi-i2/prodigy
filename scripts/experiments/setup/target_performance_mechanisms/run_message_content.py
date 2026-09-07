"""Discriminate degree scaling, affine messages and neighbor-specific content."""
import argparse
import json
from pathlib import Path
import subprocess
import time

import torch
import torch_geometric

from experiments.run_shared_graph import write_json
from .analyze_mixture_predictions import input_labels, evaluate_logits, METRICS
from .episode_cardinality import cached_meta_forward
from .message_content import aggregation_audit, message_control
from .replay import batch_hash
from .role_context import query_mask
from .role_topology import encode
from .run_episode_cardinality import make_model
from .verify_member_training import model_digest


CHANGES = ("actual_mean", "no_message_bias", "bias_only", "mean_message", "zero_messages")
CONDITIONS = [("intact", "both")] + [(change, role) for change in CHANGES for role in ("query", "support", "both")]


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--reference-replay", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--targets", nargs="+")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--max-batches", type=int, default=32)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if args.output.exists() or torch.cuda.is_available() or not 1 <= args.threads <= 8 or not 1 <= args.max_batches <= 32:
        raise ValueError("new output, hidden GPUs and bounded replay required")
    torch.set_num_threads(args.threads)
    prior = json.loads((args.reference_replay / "protocol.json").read_text())
    done = json.loads((args.reference_replay / "DONE.json").read_text())
    if done["smoke_only"] or not done["baseline_metrics_match"] or not done["all_weights_unchanged"]:
        raise ValueError("reference must be the completed frozen-model factorial")
    targets = args.targets or prior["targets"]
    if len(set(targets)) != len(targets) or not set(targets) <= set(prior["targets"]):
        raise ValueError("duplicate or unknown target")
    if len(set(args.seeds)) != len(args.seeds) or not set(args.seeds) <= {0, 1, 2}:
        raise ValueError("duplicate or unknown training seed")
    arms = [a for a in json.loads(Path(prior["arms"]).read_text()) if a["policy"] == "lowest_sorted" and a["seed"] in args.seeds]
    if {(a["source"], a["seed"]) for a in arms} != {(s, n) for s in ("cp_hk", "ukr_rus") for n in args.seeds} or len(arms) != 2*len(args.seeds):
        raise ValueError("complete requested source/seed panel required")
    arms.sort(key=lambda a: (a["source"], a["seed"]))
    old_rows = json.loads((args.reference_replay / "metrics.json").read_text())
    reference = {(r["target"], r["stream"], r["model_id"], r["query_condition"], r["support_condition"]): r
                 for r in old_rows if r["draw"] == 0 and "rewired" not in (r["query_condition"], r["support_condition"])}
    receipt = {(r["target"], r["stream"], r["model_id"]): r for r in json.loads((args.reference_replay / "receipts.json").read_text())}
    protocol = {"revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "reference_replay": str(args.reference_replay), "reference_revision": prior["revision"],
                "targets": targets, "models": [a["model_id"] for a in arms], "seeds": args.seeds,
                "conditions": CONDITIONS, "batches": args.max_batches, "threads": args.threads,
                "smoke_only": args.max_batches != 32, "new_training": False,
                "torch": torch.__version__, "torch_geometric": torch_geometric.__version__,
                "motivation": "After the completed degree-preserving null showed tiny effects, inspect actual mean versus sum aggregation and affine versus feature-dependent messages. This is an exploratory follow-up, not a preregistered confirmation.",
                "controls": {"actual_mean": "Use real MeanAggregation for the fixed checkpoint, leaving all weights and connections fixed.",
                    "no_message_bias": "Subtract only the message projection's affine bias before aggregation, retaining its feature term.",
                    "bias_only": "Remove the feature-dependent message term, retaining the same learned affine bias per edge.",
                    "mean_message": "Every sender emits the mean projected feature over its sampled subgraph's real nodes; retain original receiver degrees and all node-specific self projections.",
                    "zero_messages": "Zero every background message; must reproduce existing edge-removal metrics for each role."},
                "reporting": "All targets, seeds and role conditions; no target-selected rule. Query labels enter only metric evaluation. Different randomization streams are not training replications."}
    if args.dry_run:
        print(json.dumps({**protocol, "planned_cells": len(CONDITIONS)*len(arms)*len(targets)*2}, indent=2))
        return
    args.output.mkdir(parents=True)
    write_json(args.output / "protocol.json", protocol)
    rows, receipts, audits = [], [], []
    started = time.monotonic()
    with torch.no_grad():
        for target in targets:
            for stream in ("original", "fresh"):
                root = Path(prior["cache_roots"][f"{target}/{stream}"])
                encoder_protocol = json.loads((root / "protocol.json").read_text())
                if encoder_protocol["eval_episode_seed_offset"] != (0 if stream == "original" else 100003):
                    raise ValueError("wrong cached stream")
                directory = root / target
                labels = input_labels(directory, target)
                first = torch.load(directory / "batches/batch_000.pt", map_location="cpu", weights_only=False)
                models, initial_aggregation = {}, {}
                for arm in arms:
                    mid = arm["model_id"]
                    saved = receipt[(target, stream, mid)]
                    if saved["checkpoint"] != arm["checkpoint"] or saved["weights_sha256"] != arm["final_sha256"] or saved["episode_fingerprint"] != labels["cache"]["episode_fingerprint"]:
                        raise ValueError("reference checkpoint/input identity differs")
                    state = torch.load(arm["checkpoint"], map_location="cpu", weights_only=True)["model"]
                    if model_digest(state) != arm["final_sha256"]:
                        raise ValueError("checkpoint changed")
                    model = make_model(encoder_protocol, target, labels["cache"]["graph_path"], first[0].x.shape[1], state)
                    audit = aggregation_audit(model)
                    if len(audit) != 1 or audit[0]["configured_aggr"] != "mean" or audit[0]["aggregation_module"] != "SumAggregation" or audit[0]["unit_message_probe"] != [1., 2., 0.]:
                        raise ValueError("expected sum/mean runtime mismatch not observed")
                    audits.append({"target": target, "stream": stream, "model_id": mid, "aggregation": audit})
                    models[mid] = model
                    initial_aggregation[mid] = audit
                collected = {a["model_id"]: {c: [] for c in CONDITIONS} for a in arms}
                checks = {a["model_id"]: [] for a in arms}
                for bi in range(args.max_batches):
                    batch = torch.load(directory / "batches" / f"batch_{bi:03d}.pt", map_location="cpu", weights_only=False)
                    fingerprint = batch_hash(batch)
                    if fingerprint != labels["cache"]["batch_sha256"][bi]:
                        raise ValueError("cached input changed")
                    q = query_mask(batch)
                    for arm in arms:
                        mid, model = arm["model_id"], models[arm["model_id"]]
                        original, baseline = encode(model, batch)
                        collected[mid][("intact", "both")].append(baseline)
                        for condition in CHANGES:
                            with message_control(model, batch[0], condition):
                                changed, direct_both = encode(model, batch)
                            for role in ("query", "support", "both"):
                                selected = q if role == "query" else ~q if role == "support" else torch.ones_like(q)
                                mixed = torch.where(selected[:, None], changed, original)
                                _, prediction = cached_meta_forward(model, batch, mixed)
                                if not torch.isfinite(prediction).all():
                                    raise ValueError("nonfinite prediction")
                                collected[mid][(condition, role)].append(prediction)
                                if role == "both":
                                    torch.testing.assert_close(prediction, direct_both, rtol=0, atol=1e-5)
                                elif bi == 0:
                                    with message_control(model, batch[0], condition, node_mask=selected[batch[0].batch]):
                                        direct_pre, direct = encode(model, batch)
                                    torch.testing.assert_close(direct_pre[~selected], original[~selected], rtol=0, atol=0)
                                    torch.testing.assert_close(prediction, direct, rtol=0, atol=1e-5)
                                    checks[mid].append({"condition": condition, "role": role, "max_error": float((prediction-direct).abs().max())})
                        if model_digest(model.state_dict()) != arm["final_sha256"] or aggregation_audit(model) != initial_aggregation[mid]:
                            raise ValueError("model state or aggregation operator changed after restoration")
                    if batch_hash(batch) != fingerprint:
                        raise ValueError("input mutated")
                    print(json.dumps({"target": target, "stream": stream, "batch": bi, "models": len(arms), "elapsed_seconds": round(time.monotonic()-started, 1)}), flush=True)
                eval_labels = dict(labels)
                if args.max_batches < 32:
                    count = sum(labels["batch_counts"][:args.max_batches])
                    for key in ("local_y", "mapping", "episode_ids"):
                        eval_labels[key] = labels[key][:count]
                for arm in arms:
                    mid = arm["model_id"]
                    predictions = {c: torch.cat(v) for c, v in collected[mid].items()}
                    for (condition, role), prediction in predictions.items():
                        scores = evaluate_logits(prediction, eval_labels)
                        if condition in ("intact", "zero_messages") and args.max_batches == 32:
                            qc = "removed" if condition == "zero_messages" and role in ("query", "both") else "intact"
                            sc = "removed" if condition == "zero_messages" and role in ("support", "both") else "intact"
                            ref = reference[(target, stream, mid, qc, sc)]
                            if max(abs(scores[k]-ref[k]) for k in METRICS) > 1e-6:
                                raise ValueError("baseline or zero-message/removal parity failed")
                        rows.append({"target": target, "stream": stream, "model_id": mid, "source": arm["source"], "seed": arm["seed"],
                                     "condition": condition, "role": role, **scores})
                    out = args.output / "predictions" / target / stream
                    out.mkdir(parents=True, exist_ok=True)
                    torch.save({"logits": predictions, "labels": eval_labels}, out / f"{mid}.pt")
                    receipts.append({**receipt[(target, stream, mid)], "baseline_and_zero_match": args.max_batches == 32,
                                     "state_and_operator_restored": True, "role_checks": checks[mid]})
                for name, records in (("metrics", rows), ("receipts", receipts), ("aggregation_audit", audits)):
                    write_json(args.output / f"{name}.json", records)
    expected = len(CONDITIONS)*len(arms)*len(targets)*2
    if len(rows) != expected or len(receipts) != len(arms)*len(targets)*2:
        raise ValueError("incomplete requested grid")
    write_json(args.output / "DONE.json", {"cells": len(rows), "model_target_streams": len(receipts),
               "smoke_only": args.max_batches != 32, "all_baseline_and_zero_metrics_match": args.max_batches == 32,
               "all_weights_and_operators_restored": True, "elapsed_seconds": time.monotonic()-started})


if __name__ == "__main__":
    main()
