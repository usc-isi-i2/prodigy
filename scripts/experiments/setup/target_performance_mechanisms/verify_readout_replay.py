"""Read-only tensor audit and saved-logit cue check for exact readout swaps."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from .build_readout_interventions import MODES, READOUT_KEYS
from .verify_member_training import model_digest
from .analyze_twibot_cue_alignment import episode_alignment
from .finish_member_pipeline import compare_cached_inputs, TARGETS
from scripts.experiments.analysis.graphs.transfer_prediction.target_performance_mechanisms.analyze_trajectories import DECODERS

UNCHANGED = {f"{stage}/{method}" for stage in ("raw_center", "raw_context", "raw_joint", "S0_conv_center", "S0_pool")
             for method in ("prototype", "ridge")}


def check_saved_hybrid(row):
    initial = torch.load(row["initial_checkpoint"], map_location="cpu", weights_only=True)["model"]
    terminal = torch.load(row["terminal_checkpoint"], map_location="cpu", weights_only=True)["model"]
    hybrid = torch.load(row["checkpoint"], map_location="cpu", weights_only=True)["model"]
    for state, key in ((initial, "initial_sha256"), (terminal, "terminal_sha256"), (hybrid, "weights_sha256")):
        if model_digest(state) != row[key]:
            raise ValueError("persisted model no longer matches manifest digest")
    if row["intervention"] not in MODES or set(initial) != set(terminal) or set(hybrid) != set(terminal):
        raise ValueError("invalid state layout or intervention")
    background, donor = (terminal, initial) if row["intervention"] == MODES[0] else (initial, terminal)
    changed = []
    for key, value in hybrid.items():
        expected = donor[key] if key in READOUT_KEYS else background[key]
        if value.dtype != expected.dtype or value.shape != expected.shape or not torch.equal(value, expected):
            raise ValueError(f"wrong donor tensor: {key}")
        if not torch.equal(value, background[key]):
            changed.append(key)
    if not changed or set(changed) != set(row["changed_keys"]) or not set(changed) <= READOUT_KEYS:
        raise ValueError("changed tensor inventory differs")
    return dict(model_id=row["model_id"], weights_sha256=row["weights_sha256"],
                tensor_count=len(hybrid), changed_keys=sorted(changed), exact_donor_tensors=True)


def check_predictions(current, reference, hashes):
    if len(current) != 32 or len(reference) != 32 or len(hashes) != 32:
        raise ValueError("incomplete prediction cache")
    count = 0
    for index, (a, b) in enumerate(zip(current, reference)):
        if a["batch"] != index or b["batch"] != index or a["batch_sha256"] != hashes[index] or b["batch_sha256"] != hashes[index]:
            raise ValueError("prediction input/order mismatch")
        if set(a["logits"]) != DECODERS or set(b["logits"]) != DECODERS:
            raise ValueError("incomplete stage decoder inventory")
        for decoder, value in a["logits"].items():
            ref = b["logits"][decoder]
            if value.shape != ref.shape or value.dtype != ref.dtype or not torch.isfinite(value).all() or not torch.isfinite(ref).all():
                raise ValueError("invalid prediction tensors")
            if decoder in UNCHANGED:
                if not torch.equal(value, ref):
                    raise ValueError(f"upstream probe output changed: {decoder}")
                count += 1
    return count


def cue_rows(predictions, cues, cache, metadata, stream):
    if len(predictions) != 32 or len(cues) != 32:
        raise ValueError("incomplete TwiBot cue inputs")
    result = []
    for decoder in ("S0_conv_center/ridge", "S0_pool/ridge", "U1_pre_meta/ridge", "full_model"):
        comparisons = {key: [] for key in ("center_indegree", "raw_center", "raw_context")}
        for index, (model, cue) in enumerate(zip(predictions, cues)):
            if model["batch"] != index or cue["batch"] != index or model["batch_sha256"] != cue["batch_sha256"] or cue["batch_sha256"] != cache["batch_sha256"][index]:
                raise ValueError("cue input/order mismatch")
            logits = model["logits"][decoder]
            if logits.shape != (96, 2):
                raise ValueError("expected four 24-query TwiBot episodes")
            refs = {"center_indegree": cue["logits"]["scalar_center_indegree"],
                    "raw_center": model["logits"]["raw_center/ridge"],
                    "raw_context": model["logits"]["raw_context/ridge"]}
            for name, ref in refs.items():
                comparisons[name].extend(episode_alignment(logits.numpy(), ref.numpy()))
        for cue, values in comparisons.items():
            valid = [v for v in values if v is not None]
            result.append({"stream": stream, "model_id": metadata["model_id"], "decoder": decoder, "cue": cue,
                "weights_sha256": metadata["weights_sha256"], "episode_fingerprint": cache["episode_fingerprint"],
                "valid_episodes": len(valid), "total_episodes": len(values),
                **{f"mean_within_episode_{key}": float(np.mean([v[key] for v in valid])) if valid else None
                   for key in ("spearman", "pearson", "decision_agreement")}})
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--interventions", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--fresh", type=Path, required=True)
    parser.add_argument("--terminal-root", type=Path, default=Path("/dataMeR1/phil/gfm/prodigy-mechanisms-audit/log/target_mechanisms/member_evaluation_20260906"))
    parser.add_argument("--initial-root", type=Path, default=Path("/dataMeR1/phil/gfm/prodigy-mechanisms-followup/log/target_mechanisms"))
    parser.add_argument("--reference-root", type=Path, default=Path("/dataMeR1/phil/gfm/prodigy-mechanisms/log/target_mechanisms"))
    parser.add_argument("--original-scalars", type=Path, default=Path("/dataMeR1/phil/gfm/prodigy-mechanisms-audit/log/target_mechanisms/input_scalars_20260906"))
    parser.add_argument("--fresh-scalars", type=Path, default=Path("/dataMeR1/phil/gfm/prodigy-mechanisms/log/target_mechanisms/fresh_input_scalars_20260906"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("output already exists")
    torch.set_num_threads(4)
    manifest = json.loads((args.interventions / "manifest.json").read_text())
    done = json.loads((args.interventions / "DONE.json").read_text())
    rows = manifest["models"]
    if len(rows) != 48 or len({r["model_id"] for r in rows}) != 48 or set(manifest["readout_keys"]) != READOUT_KEYS:
        raise ValueError("incomplete intervention manifest")
    if done.get("models") != 48 or not all(done.get(k) is True for k in ("no_training", "exact_donor_tensors_verified", "only_declared_readout_keys_changed")):
        raise ValueError("construction receipt failed")
    inputs = compare_cached_inputs({s: getattr(args, s) for s in ("original", "fresh")}, args.reference_root)
    weights = [check_saved_hybrid(row) for row in rows]
    output_cells, alignment = [], []
    for stream in ("original", "fresh"):
        output = getattr(args, stream)
        initial_root = args.initial_root / f"initial_reference_{stream}_20260906"
        terminal_root = args.terminal_root / stream
        scalars = getattr(args, f"{stream}_scalars")
        if not all((p / "DONE").is_file() for p in (initial_root, terminal_root, scalars)):
            raise ValueError("incomplete baseline/cue reference")
        cues = torch.load(scalars / "twibot20.pt", map_location="cpu", weights_only=False)
        for target in TARGETS:
            cache = json.loads((output / target / "cache.json").read_text())
            baseline_predictions, baseline_metadata = {}, {}
            for root in (initial_root, terminal_root):
                prior_cache = json.loads((root / target / "cache.json").read_text())
                if any(cache[k] != prior_cache[k] for k in ("batch_sha256", "episode_fingerprint", "episodes", "graph_path")):
                    raise ValueError("baseline cached inputs differ")
                for line in (root / target / "metrics.jsonl").read_text().splitlines():
                    metadata = json.loads(line)
                    if metadata["decoder"] == "full_model":
                        model_id = metadata["model_id"]
                        if model_id in baseline_metadata:
                            raise ValueError("duplicate baseline model")
                        baseline_metadata[model_id] = metadata
                        baseline_predictions[model_id] = torch.load(root / target / f"{model_id}__baseline.pt", map_location="cpu", weights_only=False)
            for row in rows:
                background_id = row["parent_model_id"] if row["intervention"] == MODES[0] else f"memberinit_s{row['seed']}"
                meta = baseline_metadata[background_id]
                expected_hash = row["terminal_sha256"] if row["intervention"] == MODES[0] else row["initial_sha256"]
                if meta["weights_sha256"] != expected_hash:
                    raise ValueError("baseline prediction weights differ from hybrid donor")
                current = torch.load(output / target / f"{row['model_id']}__baseline.pt", map_location="cpu", weights_only=False)
                count = check_predictions(current, baseline_predictions[background_id], cache["batch_sha256"])
                output_cells.append(dict(stream=stream, target=target, model_id=row["model_id"],
                    background_model_id=background_id, background_weights_sha256=expected_hash,
                    weights_sha256=row["weights_sha256"], episode_fingerprint=cache["episode_fingerprint"],
                    unchanged_prediction_tensors=count))
                if target == "twibot20":
                    alignment.extend(cue_rows(current, cues, cache, row, stream))
            if target == "twibot20":
                for model_id, prediction in baseline_predictions.items():
                    alignment.extend(cue_rows(prediction, cues, cache, baseline_metadata[model_id], stream))
    if len(output_cells) != 480 or sum(r["unchanged_prediction_tensors"] for r in output_cells) != 153600 or len(alignment) != 1800:
        raise ValueError("incomplete output audit")
    args.output.mkdir(parents=True)
    (args.output / "input_validation.json").write_text(json.dumps(inputs, indent=2) + "\n")
    (args.output / "verification.json").write_text(json.dumps({"models": 48, "stream_target_model_cells": 480,
        "exact_donor_tensors_verified": True, "unchanged_upstream_prediction_tensors": 153600,
        "weights": weights, "cells": output_cells, "verification_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip()}, indent=2) + "\n")
    (args.output / "cue_alignment.json").write_text(json.dumps(alignment, indent=2) + "\n")
    (args.output / "DONE").write_text("Complete exact-weight/input/upstream-output audit and saved-logit cue comparisons.\n")
    print(json.dumps({"models": 48, "exact_upstream_tensor_comparisons": 153600, "cue_rows": 1800, "query_labels_fitted": 0}))


if __name__ == "__main__":
    main()
