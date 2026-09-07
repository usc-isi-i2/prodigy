"""Fixed-checkpoint role x topology follow-up, with paired degree-preserving nulls."""
import argparse
import json
from pathlib import Path
import subprocess
import time

import pandas as pd
import torch

from experiments.run_shared_graph import write_json
from .analyze_mixture_predictions import input_labels, evaluate_logits, METRICS
from .replay import batch_hash, clone_batch
from .role_context import query_mask
from .role_topology import stable_seed, topology_batch, encode, factorial_predictions, input_geometry, assert_background_attributes_unused
from .run_episode_cardinality import make_model
from .verify_member_training import model_digest


def cache_root(roots, target, offset):
    for root in roots:
        if (root / target / "cache.json").is_file():
            protocol = json.loads((root / "protocol.json").read_text())
            if protocol["eval_episode_seed_offset"] != offset:
                raise ValueError("wrong cached episode stream")
            return root, protocol
    raise ValueError(f"missing cached inputs for {target}")


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--arms", type=Path, required=True)
    p.add_argument("--references", type=Path, required=True)
    p.add_argument("--original-roots", type=Path, nargs="+", required=True)
    p.add_argument("--fresh-roots", type=Path, nargs="+", required=True)
    p.add_argument("--targets", nargs="+", default=["covid_political", "election2020", "twibot20"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--rewire-draws", type=int, default=3)
    p.add_argument("--swaps-per-edge", type=int, default=5)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--max-batches", type=int, default=32)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if args.output.exists() or torch.cuda.is_available() or not 1 <= args.threads <= 8:
        raise ValueError("new output, hidden GPUs and bounded CPU threads required")
    if not 1 <= args.max_batches <= 32 or not 1 <= args.rewire_draws <= 5 or args.swaps_per_edge < 1:
        raise ValueError("invalid replay or randomization budget")
    if len(set(args.targets)) != len(args.targets) or not set(args.targets) <= {"covid_political", "election2020", "twibot20"}:
        raise ValueError("use the declared nonredundant target panel")
    if len(set(args.seeds)) != len(args.seeds) or not set(args.seeds) <= {0, 1, 2}:
        raise ValueError("unknown/duplicate training seed")
    torch.set_num_threads(args.threads)
    arms = [a for a in json.loads(args.arms.read_text()) if a["policy"] == "lowest_sorted" and a["seed"] in args.seeds]
    if len(arms) != 2 * len(args.seeds) or {(a["source"], a["seed"]) for a in arms} != {(s, n) for s in ("cp_hk", "ukr_rus") for n in args.seeds}:
        raise ValueError("complete requested source x training-seed panel required")
    arms.sort(key=lambda a: (a["source"], a["seed"]))
    refs = pd.read_csv(args.references)
    refs = refs[refs.policy.eq("lowest_sorted") & refs.decoder.eq("full_model")]
    roots = {(target, stream): cache_root(getattr(args, stream + "_roots"), target, offset)
             for target in args.targets for stream, offset in (("original", 0), ("fresh", 100003))}
    conditions = [(q, s, draw) for draw in range(args.rewire_draws)
                  for q in ("intact", "removed", "rewired") for s in ("intact", "removed", "rewired")
                  if draw == 0 or "rewired" in (q, s)]
    manifest = {"revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "targets": args.targets, "models": [a["model_id"] for a in arms], "seeds": args.seeds,
                "conditions": conditions, "batches": args.max_batches, "rewire_draws": args.rewire_draws,
                "swaps_per_edge": args.swaps_per_edge, "threads": args.threads,
                "arms": str(args.arms), "references": str(args.references),
                "cache_roots": {f"{t}/{s}": str(r[0]) for (t, s), r in roots.items()},
                "smoke_only": args.max_batches != 32, "checkpoint_step": 2500, "new_training": False,
                "design": "Within each cached subgraph, simple degree-preserving double-edge swaps; reciprocal graphs preserve reciprocal pairs, loops remain fixed. Five attempted-success targets per movable edge, at most twenty attempts per requested swap. Report accepted swaps and final edge turnover, including rigid graphs. Not uniform sampling or a pure homophily intervention.",
                "primary": "All four intact/removed role conditions and all nine role x intact/removed/rewired combinations, every selected seed/target/stream. Quantify conditional role effects; no outcome-selected null draw.",
                "interpretation": "AUC interaction is an interaction on the reported performance scale, not proof of nonadditive logits. Report accuracy and NLL too. Target selection uses earlier exploratory results; no untouched-target or significance claim."}
    if args.dry_run:
        print(json.dumps({**manifest, "planned_cells": len(conditions)*len(arms)*len(roots)}, indent=2))
        return
    args.output.mkdir(parents=True)
    write_json(args.output / "protocol.json", manifest)
    rows, receipts, topology, inputs, geometry = [], [], [], [], []
    start_time = time.monotonic()
    with torch.no_grad():
        for (target, stream), (root, protocol) in roots.items():
            directory = root / target
            labels = input_labels(directory, target)
            first = torch.load(directory / "batches/batch_000.pt", map_location="cpu", weights_only=False)
            models, ref_rows = {}, {}
            for arm in arms:
                ref = refs[refs.dataset.eq(target) & refs.stream.eq(stream) & refs.model_id.eq(arm["model_id"])]
                if len(ref) != 1:
                    raise ValueError("unique prior reference required")
                ref = ref.iloc[0]
                if ref.checkpoint != arm["checkpoint"] or ref.episode_fingerprint != labels["cache"]["episode_fingerprint"]:
                    raise ValueError("reference checkpoint/input identity differs")
                state = torch.load(arm["checkpoint"], map_location="cpu", weights_only=True)["model"]
                if model_digest(state) != arm["final_sha256"] or ref.weights_sha256 != arm["final_sha256"]:
                    raise ValueError("checkpoint differs from saved receipt")
                models[arm["model_id"]] = make_model(protocol, target, labels["cache"]["graph_path"], first[0].x.shape[1], state)
                assert_background_attributes_unused(models[arm["model_id"]])
                ref_rows[arm["model_id"]] = ref
            collected = {a["model_id"]: {c: [] for c in conditions} for a in arms}
            for bi in range(args.max_batches):
                batch = torch.load(directory / "batches" / f"batch_{bi:03d}.pt", map_location="cpu", weights_only=False)
                fingerprint = batch_hash(batch)
                if fingerprint != labels["cache"]["batch_sha256"][bi]:
                    raise ValueError("cached inputs changed")
                common = {"target": target, "stream": stream, "batch": bi}
                inputs.extend({**common, **r} for r in input_geometry(batch))
                removed, _ = topology_batch(batch, "removed", seed=0, edge_attributes_unused=True)
                rewired = []
                for draw in range(args.rewire_draws):
                    altered, audit = topology_batch(batch, "rewired", seed=stable_seed(target, stream, bi, draw), swaps_per_edge=args.swaps_per_edge, edge_attributes_unused=True)
                    rewired.append(altered)
                    topology.extend({**common, "draw": draw, **r} for r in audit)
                for arm in arms:
                    mid = arm["model_id"]
                    model = models[mid]
                    intact_pre, base = encode(model, batch)
                    if bi == 0 and batch[0].edge_attr is not None:
                        attr_control = clone_batch(batch)
                        attr_control[0].edge_attr.zero_()
                        attr_pre, attr_logits = encode(model, attr_control)
                        torch.testing.assert_close(attr_pre, intact_pre, rtol=0, atol=0)
                        torch.testing.assert_close(attr_logits, base, rtol=0, atol=0)
                    removed_pre, _ = encode(model, removed)
                    qmask = query_mask(batch)
                    for draw, altered in enumerate(rewired):
                        rewire_pre, _ = encode(model, altered)
                        values, descriptors = factorial_predictions(model, batch, {"intact": intact_pre, "removed": removed_pre, "rewired": rewire_pre})
                        torch.testing.assert_close(values[("intact", "intact")], base, rtol=0, atol=0)
                        for (q, s), logits in values.items():
                            if draw == 0 or "rewired" in (q, s):
                                collected[mid][(q, s, draw)].append(logits)
                        geometry.extend({**common, "model_id": mid, "source": arm["source"], "seed": arm["seed"], "draw": draw, **r}
                                        for r in descriptors if draw == 0 or r["condition"] == "rewired")
                        if bi == 0:
                            for role in ("support", "query"):
                                direct_batch, _ = topology_batch(batch, "rewired", seed=stable_seed(target, stream, bi, draw), role=role, swaps_per_edge=args.swaps_per_edge, edge_attributes_unused=True)
                                direct_pre, direct_logits = encode(model, direct_batch)
                                key = ("intact", "rewired") if role == "support" else ("rewired", "intact")
                                untouched = qmask if role == "support" else ~qmask
                                torch.testing.assert_close(direct_pre[untouched], intact_pre[untouched], rtol=0, atol=0)
                                torch.testing.assert_close(direct_logits, values[key], rtol=0, atol=1e-5)
                    if model_digest(model.state_dict()) != arm["final_sha256"]:
                        raise ValueError("model weights or normalization buffers changed")
                if batch_hash(batch) != fingerprint:
                    raise ValueError("input mutated")
                print(json.dumps({**common, "models": len(arms), "elapsed_seconds": round(time.monotonic()-start_time, 1)}), flush=True)
            eval_labels = dict(labels)
            if args.max_batches < 32:
                count = sum(labels["batch_counts"][:args.max_batches])
                for name in ("local_y", "mapping", "episode_ids"):
                    eval_labels[name] = labels[name][:count]
            for arm in arms:
                mid = arm["model_id"]
                predictions = {c: torch.cat(v) for c, v in collected[mid].items()}
                for (q, s, draw), logits in predictions.items():
                    scores = evaluate_logits(logits, eval_labels)
                    if (q, s, draw) == ("intact", "intact", 0) and args.max_batches == 32:
                        if max(abs(scores[k] - ref_rows[mid][k]) for k in METRICS) > 1e-6:
                            raise ValueError("full baseline metrics differ from existing evaluation")
                    rows.append({"target": target, "stream": stream, "model_id": mid, "source": arm["source"], "seed": arm["seed"],
                                 "query_condition": q, "support_condition": s, "draw": draw, **scores})
                out = args.output / "predictions" / target / stream
                out.mkdir(parents=True, exist_ok=True)
                torch.save({"logits": predictions, "labels": eval_labels}, out / f"{mid}.pt")
                receipts.append({"target": target, "stream": stream, "model_id": mid, "weights_sha256": arm["final_sha256"],
                                 "checkpoint": arm["checkpoint"], "episode_fingerprint": labels["cache"]["episode_fingerprint"],
                                 "baseline_metrics_match": args.max_batches == 32, "weights_unchanged": True})
            for name, records in (("metrics", rows), ("receipts", receipts), ("topology", topology), ("input_geometry", inputs), ("support_geometry", geometry)):
                write_json(args.output / f"{name}.json", records)
    expected = len(conditions)*len(arms)*len(roots)
    if len(rows) != expected or len(receipts) != len(arms)*len(roots):
        raise ValueError("incomplete result grid")
    write_json(args.output / "DONE.json", {"cells": len(rows), "model_target_streams": len(receipts), "smoke_only": args.max_batches != 32,
               "baseline_metrics_match": args.max_batches == 32, "all_weights_unchanged": True,
               "elapsed_seconds": time.monotonic()-start_time})


if __name__ == "__main__":
    main()
