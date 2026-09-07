"""Matched support-suppression dose and saved-checkpoint sensitivity panel."""
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
from .message_content import aggregation_audit
from .replay import batch_hash
from .role_context import query_mask
from .role_topology import encode, stable_seed
from .run_episode_cardinality import make_model
from .support_dose import dose_masks, masked_batch
from .verify_member_training import model_digest


STEPS = (0, 100, 300, 900, 2500)


def conditions(step, draws):
    return [(0, 0), (100, 0)] + ([(dose, draw) for draw in range(draws) for dose in (25, 50, 75)] if step == 2500 else [])


def inventory_checkpoints(arms, steps):
    records, states = [], {}
    for arm in arms:
        directory = Path(arm["checkpoint"]).parent
        available = sorted(int(p.stem.split("_")[-1]) for p in directory.glob("state_dict_*.ckpt"))
        if not set(steps) <= set(available):
            raise ValueError(f"missing saved steps for {arm['model_id']}: {available}")
        for step in steps:
            path = directory / f"state_dict_{step}.ckpt"
            state = torch.load(path, map_location="cpu", weights_only=True)["model"]
            if not all(torch.isfinite(v).all() for v in state.values()):
                raise ValueError("nonfinite checkpoint")
            sha = model_digest(state)
            sidecar = torch.load(directory / f"training_state_{step}.ckpt", map_location="cpu", weights_only=False)
            if sidecar["_training_checkpoint"]["completed_steps"] != step or model_digest(sidecar["model"]) != sha:
                raise ValueError("saved completed-step counter or sidecar weights disagree")
            if step in (0, 2500) and sha != arm["initial_sha256" if step == 0 else "final_sha256"]:
                raise ValueError("verified initial/final identity changed")
            key = (arm["model_id"], step)
            states[key] = state
            records.append({"model_id": arm["model_id"], "source": arm["source"], "seed": arm["seed"],
                            "step": step, "checkpoint": str(path), "weights_sha256": sha,
                            "available_steps": available, "completed_steps_verified": True})
            del sidecar
    return records, states


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--reference-replay", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--targets", nargs="+")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--steps", type=int, nargs="+", default=list(STEPS))
    p.add_argument("--draws", type=int, default=3)
    p.add_argument("--max-batches", type=int, default=32)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if args.output.exists() or torch.cuda.is_available() or not 1 <= args.threads <= 8 or not 1 <= args.max_batches <= 32 or not 1 <= args.draws <= 8:
        raise ValueError("new output, hidden GPUs and bounded CPU replay required")
    torch.set_num_threads(args.threads)
    prior = json.loads((args.reference_replay / "protocol.json").read_text())
    done = json.loads((args.reference_replay / "DONE.json").read_text())
    if done["smoke_only"] or not done["baseline_metrics_match"] or not done["all_weights_unchanged"]:
        raise ValueError("completed reference replay required")
    targets = args.targets or prior["targets"]
    if len(set(targets)) != len(targets) or not set(targets) <= set(prior["targets"]):
        raise ValueError("unknown or duplicate target")
    if len(set(args.seeds)) != len(args.seeds) or not set(args.seeds) <= {0, 1, 2}:
        raise ValueError("unknown or duplicate seed")
    if len(set(args.steps)) != len(args.steps) or not set(args.steps) <= set(STEPS) or args.steps != sorted(args.steps):
        raise ValueError("unknown, duplicate or unordered saved steps")
    arms = [a for a in json.loads(Path(prior["arms"]).read_text()) if a["policy"] == "lowest_sorted" and a["seed"] in args.seeds]
    arms.sort(key=lambda a: (a["source"], a["seed"]))
    if len(arms) != 2*len(args.seeds) or {(a["source"], a["seed"]) for a in arms} != {(s, n) for s in ("cp_hk", "ukr_rus") for n in args.seeds}:
        raise ValueError("complete source/seed panel required")
    inventory, states = inventory_checkpoints(arms, args.steps)
    checkpoint = {(r["model_id"], r["step"]): r for r in inventory}
    old = json.loads((args.reference_replay / "metrics.json").read_text())
    reference = {(r["target"], r["stream"], r["model_id"], 0 if r["support_condition"] == "intact" else 100): r
                 for r in old if r["draw"] == 0 and r["query_condition"] == "intact" and r["support_condition"] in ("intact", "removed")}
    old_receipts = {(r["target"], r["stream"], r["model_id"]): r
                    for r in json.loads((args.reference_replay / "receipts.json").read_text())}
    planned = sum(len(conditions(step, args.draws)) for step in args.steps)*len(arms)*len(targets)*2
    protocol = {"revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "reference_replay": str(args.reference_replay), "reference_revision": prior["revision"],
                "targets": targets, "models": [a["model_id"] for a in arms], "seeds": args.seeds,
                "steps": args.steps, "conditions_by_step": {str(s): conditions(s, args.draws) for s in args.steps},
                "batches": args.max_batches, "draws": args.draws, "threads": args.threads,
                "smoke_only": args.max_batches != 32, "new_training": False, "planned_cells": planned,
                "torch": torch.__version__, "torch_geometric": torch_geometric.__version__,
                "cache_roots": prior["cache_roots"],
                "design": "At every saved step evaluate intact and support-only complete removal. At step 2500 add 25/50/75 percent suppression in three nested draws. Fixed queries, original node membership/features/pooling. Within each subgraph, uniform random permutation of directed edges or reciprocal pairs, self loops as singleton units, round-half-up deletion counts. All masks shared across checkpoints; report actual edge fractions and unchanged empty subgraphs.",
                "reporting": "Exploratory follow-up prompted by the review; no query-selected dose, checkpoint or target. Initialization, streams and mask draws are distinguished. Curves do not imply significance or an automatic deployment rule."}
    if args.dry_run:
        print(json.dumps({**protocol, "inventory": inventory}, indent=2))
        return
    args.output.mkdir(parents=True)
    write_json(args.output / "protocol.json", protocol)
    write_json(args.output / "checkpoint_inventory.json", inventory)
    rows, receipts, mask_records = [], [], []
    start = time.monotonic()
    with torch.no_grad():
        for target in targets:
            for stream in ("original", "fresh"):
                root = Path(prior["cache_roots"][f"{target}/{stream}"])
                ep = json.loads((root / "protocol.json").read_text())
                if ep["eval_episode_seed_offset"] != (0 if stream == "original" else 100003):
                    raise ValueError("wrong cached stream")
                directory = root / target
                labels = input_labels(directory, target)
                first = torch.load(directory / "batches/batch_000.pt", map_location="cpu", weights_only=False)
                models = {}
                for key, state in states.items():
                    mid, step = key
                    ref = old_receipts[target, stream, mid]
                    if ref["episode_fingerprint"] != labels["cache"]["episode_fingerprint"]:
                        raise ValueError("reference inputs changed")
                    if step == 2500 and (ref["checkpoint"] != checkpoint[key]["checkpoint"] or ref["weights_sha256"] != checkpoint[key]["weights_sha256"]):
                        raise ValueError("reference checkpoint identity changed")
                    models[key] = make_model(ep, target, labels["cache"]["graph_path"], first[0].x.shape[1], state)
                    if aggregation_audit(models[key]) != [{"layer": 0, "configured_aggr": "mean", "aggregation_module": "SumAggregation", "unit_message_probe": [1., 2., 0.]}]:
                        raise ValueError("unexpected aggregation runtime")
                collected = {key: {c: [] for c in conditions(key[1], args.draws)} for key in models}
                checks = {key: {"query_pre_exact_checks": 0, "query_post_exact_checks": 0, "direct_suffix_max_error": 0.} for key in models}
                for bi in range(args.max_batches):
                    batch = torch.load(directory / "batches" / f"batch_{bi:03d}.pt", map_location="cpu", weights_only=False)
                    fingerprint = batch_hash(batch)
                    if fingerprint != labels["cache"]["batch_sha256"][bi]:
                        raise ValueError("cached input fingerprint changed")
                    q = query_mask(batch)
                    all_masks = {}
                    for draw in range(args.draws):
                        masks, stats = dose_masks(batch, seed=stable_seed("support_dose_v1", target, stream, bi, draw))
                        for dose, mask in masks.items():
                            if dose in (0, 100) and draw:
                                torch.testing.assert_close(mask, all_masks[dose, 0], rtol=0, atol=0)
                            else:
                                all_masks[dose, draw] = mask
                            mask_records.append({"target": target, "stream": stream, "batch": bi, "draw": draw, **stats[dose]})
                    for key, model in models.items():
                        original, baseline = encode(model, batch)
                        original_post, reproduced = cached_meta_forward(model, batch, original)
                        torch.testing.assert_close(baseline, reproduced, rtol=0, atol=1e-5)
                        collected[key][0, 0].append(baseline)
                        for dose, draw in conditions(key[1], args.draws)[1:]:
                            altered = masked_batch(batch, all_masks[dose, draw])
                            pre, logits = encode(model, altered)
                            torch.testing.assert_close(pre[q], original[q], rtol=0, atol=0)
                            checks[key]["query_pre_exact_checks"] += 1
                            post, suffix = cached_meta_forward(model, batch, pre)
                            torch.testing.assert_close(post[q], original_post[q], rtol=0, atol=0)
                            torch.testing.assert_close(logits, suffix, rtol=0, atol=1e-5)
                            checks[key]["query_post_exact_checks"] += 1
                            checks[key]["direct_suffix_max_error"] = max(checks[key]["direct_suffix_max_error"], float((logits-suffix).abs().max()))
                            if not torch.isfinite(logits).all():
                                raise ValueError("nonfinite logits")
                            collected[key][dose, draw].append(logits)
                    if fingerprint != batch_hash(batch):
                        raise ValueError("input mutated")
                    print(json.dumps({"target": target, "stream": stream, "batch": bi, "elapsed_seconds": round(time.monotonic()-start, 1)}), flush=True)
                eval_labels = dict(labels)
                if args.max_batches != 32:
                    count = sum(labels["batch_counts"][:args.max_batches])
                    for k in ("local_y", "mapping", "episode_ids"):
                        eval_labels[k] = labels[k][:count]
                for key, model in models.items():
                    mid, step = key
                    rec = checkpoint[key]
                    if model_digest(model.state_dict()) != rec["weights_sha256"]:
                        raise ValueError("checkpoint weights/buffers changed")
                    predictions = {c: torch.cat(v) for c, v in collected[key].items()}
                    for (dose, draw), logits in predictions.items():
                        metrics = evaluate_logits(logits, eval_labels)
                        if step == 2500 and dose in (0, 100) and args.max_batches == 32:
                            ref = reference[target, stream, mid, dose]
                            if max(abs(metrics[k]-ref[k]) for k in METRICS) > 1e-6:
                                raise ValueError("historical endpoint metric parity failed")
                        rows.append({"target": target, "stream": stream, "model_id": mid, "source": rec["source"], "seed": rec["seed"],
                                     "step": step, "suppression_percent": dose, "draw": draw, **metrics})
                    out = args.output / "predictions" / target / stream
                    out.mkdir(parents=True, exist_ok=True)
                    torch.save({"logits": predictions, "labels": eval_labels}, out / f"{mid}__step{step}.pt")
                    receipts.append({**rec, "target": target, "stream": stream, "episode_fingerprint": labels["cache"]["episode_fingerprint"],
                                     "weights_unchanged": True, "endpoint_reference_match": args.max_batches == 32 if step == 2500 else None,
                                     **checks[key]})
                for name, values in (("metrics", rows), ("receipts", receipts), ("mask_receipts", mask_records)):
                    write_json(args.output / f"{name}.json", values)
    if len(rows) != planned:
        raise ValueError("incomplete requested grid")
    write_json(args.output / "DONE.json", {"cells": len(rows), "model_step_target_streams": len(receipts),
               "smoke_only": args.max_batches != 32, "weights_unchanged": True,
               "endpoint_reference_matches": args.max_batches == 32 and 2500 in args.steps,
               "elapsed_seconds": time.monotonic()-start})


if __name__ == "__main__":
    main()
