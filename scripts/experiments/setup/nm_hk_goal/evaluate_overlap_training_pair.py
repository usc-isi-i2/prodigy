"""Evaluate fixed checkpoints from the matched HK training pair on canonical inputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
import pandas as pd
import torch

from scripts.experiments.setup.nm_complete_input_audit.run import update_hash
from scripts.experiments.setup.nm_hk_mechanism.run import build_model
from scripts.experiments.setup.nm_support_resampling.run import capture, digest_state


DEFAULT_STEPS = (0, 100, 300, 900, 2500)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def checkpoint(run, step):
    matches = list(run.glob(f"state/**/checkpoint/state_dict_{step}.ckpt"))
    if len(matches) != 1:
        raise ValueError(f"expected one step-{step} checkpoint under {run}, found {len(matches)}")
    return matches[0]


def restore_hk(packed, features, offset):
    batch = packed["batch"]
    ids = batch[0].global_node_ids
    real = ids >= 0
    local = (ids - offset).clamp_min(0)
    if not ((local[real] >= 0) & (local[real] < len(features))).all():
        raise ValueError("cached HK node ID falls outside the standalone feature artifact")
    batch[0].x = features[local].clone()
    batch[0].x[~real] = 0
    return batch


def summarize(frame, prediction, baseline=None):
    row = np.arange(len(frame))
    truth = frame.truth.to_numpy(dtype=int)
    positives = np.stack(frame.positives)
    assigned = prediction == truth
    multi = positives[row, prediction]
    unique = frame.valid_count.to_numpy() == 1
    by_node_assigned = pd.DataFrame({"query": frame.query, "ok": assigned}).groupby("query").ok.mean()
    by_node_multi = pd.DataFrame({"query": frame.query, "ok": multi}).groupby("query").ok.mean()
    result = {
        "rows": len(frame),
        "assigned_accuracy": float(assigned.mean()),
        "multi_positive_accuracy": float(multi.mean()),
        "unique_rows": int(unique.sum()),
        "unique_accuracy": float(assigned[unique].mean()),
        "node_weighted_assigned_accuracy": float(by_node_assigned.mean()),
        "node_weighted_multi_positive_accuracy": float(by_node_multi.mean()),
    }
    if baseline is not None:
        base_assigned = baseline == truth
        base_multi = positives[row, baseline]
        result.update(
            assigned_recovered=int((~base_assigned & assigned).sum()),
            assigned_lost=int((base_assigned & ~assigned).sum()),
            multi_positive_recovered=int((~base_multi & multi).sum()),
            multi_positive_lost=int((base_multi & ~multi).sum()),
            unique_recovered=int((~base_assigned & assigned & unique).sum()),
            unique_lost=int((base_assigned & ~assigned & unique).sum()),
        )
        episode = pd.DataFrame({"episode": frame.episode, "delta": multi.astype(int) - base_multi.astype(int)})
        deltas = episode.groupby("episode").delta.mean()
        result["episode_multi_positive_delta"] = {
            "mean": float(deltas.mean()),
            "median": float(deltas.median()),
            "improved": int((deltas > 0).sum()),
            "tied": int((deltas == 0).sum()),
            "worsened": int((deltas < 0).sum()),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, default=Path("/dataMeR1/phil/gfm/error_audit/nm_complete_inputs_20260908_v2"))
    parser.add_argument("--audit", type=Path, default=Path("/dataMeR1/phil/gfm/error_audit/nm_canonical_split_20260908"))
    parser.add_argument("--references", type=Path, default=Path("/dataMeR1/phil/gfm/error_audit/nm_canonical_split_bios_20260908/cp_hk_twitter/paired_cluster_queries_private.tsv"))
    parser.add_argument("--views", type=Path, default=Path("/dataMeR1/phil/gfm/error_audit/nm_hk_support_extremes_20260908/hk_canonical_views_private.pt"))
    parser.add_argument("--features", type=Path, default=Path("/dataMeR1/phil/data/cp_hk_twitter/graphs/retweet_graph.pt"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--steps", default=",".join(map(str, DEFAULT_STEPS)))
    args = parser.parse_args()
    if not 1 <= args.threads <= 2:
        raise ValueError("use at most two CPU threads")
    args.out.mkdir(parents=True, exist_ok=False)
    steps = tuple(int(value) for value in args.steps.split(","))
    if not steps or len(set(steps)) != len(steps) or any(step < 0 for step in steps):
        raise ValueError("steps must be distinct nonnegative integers")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    started = time.time()

    receipt = json.loads((args.inputs / "receipt.json").read_text())
    info = receipt["targets"]["cp_hk"]
    if not receipt["complete"]:
        raise ValueError("complete-input cache is incomplete")
    params = json.loads((args.audit / "effective_config.json").read_text())
    params["device"] = args.device
    raw = torch.load(args.features, map_location="cpu", weights_only=False)
    features = raw["x"]
    refs = pd.read_csv(args.references, sep="\t")
    refs = refs[refs.split.eq("test")].set_index(["episode", "sample"])
    packed_views = torch.load(args.views, map_location="cpu", weights_only=False)
    test_edges = {tuple(sorted(map(int, edge))) for edge in packed_views["views"]["test"].T.tolist()}

    metadata = []
    for batch_index in range(16):
        for episode_index in range(32):
            episode = f"{batch_index}:{episode_index}"
            query_slots = [slot for slot in range(210) if slot % 7 >= 3]
            index = pd.MultiIndex.from_tuples([(episode, episode_index * 210 + slot) for slot in query_slots])
            rows = refs.loc[index]
            truth = np.asarray(query_slots) // 7
            classes = rows.assign(class_slot=truth).drop_duplicates("class_slot").sort_values("class_slot")
            anchors = classes.anchor.to_numpy(dtype=int)
            queries = rows["query"].to_numpy(dtype=int)
            positives = np.asarray([
                [tuple(sorted((int(query), int(anchor)))) in test_edges for anchor in anchors]
                for query in queries
            ], dtype=bool)
            if not positives[np.arange(120), truth].all():
                raise ValueError(f"assigned anchor absent from test view in {episode}")
            metadata.extend(dict(episode=episode, sample=key[1], query=int(query), truth=int(label),
                                 positives=positive, valid_count=int(positive.sum()))
                            for key, query, label, positive in zip(index, queries, truth, positives))
    frame = pd.DataFrame(metadata)
    if len(frame) != 61440 or frame.duplicated(["episode", "sample"]).any():
        raise ValueError("canonical row identity mismatch")

    predictions = {}
    states = {}
    input_hashes = []
    for condition in ("baseline", "treatment"):
        run = args.run / condition
        for step in steps:
            path = checkpoint(run, step)
            model = build_model(params, path, args.device)
            states[f"{condition}:{step}"] = digest_state(model)
            chunks = []
            input_hash = hashlib.sha256()
            with torch.inference_mode():
                for batch_index, item in enumerate(info["files"]):
                    path = args.inputs / "cp_hk" / item["file"]
                    if digest(path) != item["sha256"]:
                        raise ValueError(f"cached input digest changed: {path}")
                    packed = torch.load(path, map_location="cpu", weights_only=False)
                    batch = restore_hk(packed, features, info["source_node_offset"])
                    update_hash(input_hash, batch)
                    _, logits = capture(model, batch, args.device)
                    chunks.append(logits.reshape(-1, 30).argmax(1).cpu().numpy())
            if input_hash.hexdigest() != info["expected_hash"]:
                raise ValueError("restored complete-input hash mismatch")
            input_hashes.append(input_hash.hexdigest())
            predictions[f"{condition}:{step}"] = np.concatenate(chunks)
            print(condition, step, "seconds", round(time.time() - started, 2), flush=True)

    if states["baseline:0"] != states["treatment:0"]:
        raise ValueError("step-zero model states differ")
    if not np.array_equal(predictions["baseline:0"], predictions["treatment:0"]):
        raise ValueError("step-zero predictions differ")
    report = {
        "complete": True,
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "seconds": time.time() - started,
        "device": args.device,
        "threads": args.threads,
        "rows": len(frame),
        "steps": list(steps),
        "exact_complete_input_hash": input_hashes[0],
        "all_input_hashes_equal": len(set(input_hashes)) == 1,
        "step_zero_state_sha256": states["baseline:0"],
        "model_state_sha256": states,
        "results": {},
        "selection": "All checkpoints were fixed before training; no test-performance selection.",
    }
    for step in steps:
        baseline = predictions[f"baseline:{step}"]
        treatment = predictions[f"treatment:{step}"]
        report["results"][str(step)] = {
            "baseline": summarize(frame, baseline),
            "treatment": summarize(frame, treatment, baseline),
        }
    np.savez_compressed(args.out / "predictions_private.npz", **predictions)
    report["predictions_sha256"] = digest(args.out / "predictions_private.npz")
    (args.out / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
