"""Export a matched two-specialist, one-target example-level contrast.

This consumes the verified target-performance replay. It does not run a model or
change checkpoints. The output keeps exact query order, labels, representations,
and logits needed for exploratory localization on one stream and validation on
the other.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import torch

from experiments.run_shared_graph import write_json
from scripts.experiments.setup.target_performance_mechanisms.replay import batch_hash


STAGES = (
    "raw_center",
    "raw_context",
    "raw_joint",
    "S0_conv_center",
    "S0_pool",
    "U1_pre_meta",
    "M2_post_meta",
    "final_input",
)


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def same_labels(a, b):
    for key in ("target", "use_global", "batch_counts", "cache"):
        if a[key] != b[key]:
            raise ValueError(f"label metadata differs at {key}")
    for key in ("local_y", "mapping", "episode_ids"):
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)


def load_stream(role_root, stream, target, model_ids, inventory):
    item = [r for r in inventory if r["stream"] == stream and r["target"] == target]
    if len(item) != 1:
        raise ValueError(f"expected one input inventory row for {stream}/{target}")
    item = item[0]
    cache_root = Path(item["root"])
    role_records = {}
    baseline_records = {}
    for model_id in model_ids:
        role_path = role_root / "predictions" / stream / target / f"{model_id}.pt"
        base_path = cache_root / f"{model_id}__baseline.pt"
        if not role_path.is_file() or not base_path.is_file():
            raise FileNotFoundError(f"missing verified replay for {stream}/{target}/{model_id}")
        role_records[model_id] = torch.load(role_path, map_location="cpu", weights_only=False)
        baseline_records[model_id] = torch.load(base_path, map_location="cpu", weights_only=False)

    first = role_records[model_ids[0]]
    for model_id in model_ids[1:]:
        same_labels(first["labels"], role_records[model_id]["labels"])
        for key in first["query_metadata"]:
            torch.testing.assert_close(
                first["query_metadata"][key], role_records[model_id]["query_metadata"][key],
                rtol=0, atol=0,
            )

    batches = []
    for batch_index, expected in enumerate(item["batch_sha256"]):
        path = cache_root / "batches" / f"batch_{batch_index:03d}.pt"
        batch = torch.load(path, map_location="cpu", weights_only=False)
        if batch_hash(batch) != expected:
            raise ValueError(f"cached input hash mismatch at {stream}/{batch_index}")
        batches.append(batch)

    output = {
        "target": target,
        "stream": stream,
        "labels": first["labels"],
        "query_metadata": first["query_metadata"],
        "models": {},
        "input": {"embeddings": {}},
        "receipts": {
            "episode_fingerprint": item["episode_fingerprint"],
            "batch_sha256": item["batch_sha256"],
            "cache_root": str(cache_root),
            "graph_path": item["graph_path"],
        },
    }
    query_count = len(first["labels"]["local_y"])
    for model_id in model_ids:
        role = role_records[model_id]
        saved = baseline_records[model_id]
        if len(saved) != len(batches):
            raise ValueError("baseline batch count mismatch")
        for batch_index, record in enumerate(saved):
            if record["batch"] != batch_index or record["batch_sha256"] != item["batch_sha256"][batch_index]:
                raise ValueError("baseline record order/hash mismatch")
        baseline = torch.cat([record["logits"]["full_model"] for record in saved])
        torch.testing.assert_close(baseline, role["logits"]["baseline"], rtol=0, atol=0)
        model = {
            "weights_sha256": role["weights_sha256"],
            "logits": {
                name: torch.cat([record["logits"][name] for record in saved])
                for name in saved[0]["logits"]
            },
            "embeddings": {},
        }
        for stage in STAGES:
            values = []
            for batch, record in zip(batches, saved):
                n_way = int(batch[2].shape[1])
                query = batch[5].reshape(-1, n_way)[:, 0].bool()
                values.append(record["embeddings"][stage][query])
            model["embeddings"][stage] = torch.cat(values)
            if len(model["embeddings"][stage]) != query_count:
                raise ValueError(f"query count mismatch for {model_id}/{stage}")
        output["models"][model_id] = model

    # Raw stages are input properties and must be identical across checkpoints.
    first_model = output["models"][model_ids[0]]
    for stage in ("raw_center", "raw_context", "raw_joint"):
        reference = first_model["embeddings"][stage]
        for model_id in model_ids[1:]:
            torch.testing.assert_close(
                reference, output["models"][model_id]["embeddings"][stage], rtol=0, atol=0
            )
        output["input"]["embeddings"][stage] = first_model["embeddings"].pop(stage)
        for model_id in model_ids[1:]:
            output["models"][model_id]["embeddings"].pop(stage)
    return output


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--role-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", default="facebook_page_reference")
    parser.add_argument("--model-b", default="ss_twibot20")
    parser.add_argument("--model-c", default="ss_election2020")
    parser.add_argument("--models", help="comma-separated model IDs; overrides --model-b/--model-c")
    parser.add_argument("--streams", default="original,fresh")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("choose a new output directory")
    done = json.loads((args.role_root / "DONE.json").read_text())
    if done.get("cells") != 720 or not done.get("all_weights_unchanged"):
        raise ValueError("complete verified role-context replay required")
    inventory = json.loads((args.role_root / "input_inventory.json").read_text())
    streams = tuple(part.strip() for part in args.streams.split(",") if part.strip())
    if set(streams) != {"original", "fresh"}:
        raise ValueError("original and fresh streams are required")
    model_ids = tuple(part.strip() for part in args.models.split(",") if part.strip()) if args.models else (
        args.model_b, args.model_c,
    )
    if len(model_ids) < 2 or len(set(model_ids)) != len(model_ids):
        raise ValueError("at least two distinct model IDs are required")
    args.output.mkdir(parents=True)
    outputs = []
    for stream in streams:
        result = load_stream(args.role_root, stream, args.target, model_ids, inventory)
        destination = args.output / f"{stream}.pt"
        torch.save(result, destination)
        outputs.append({"stream": stream, "path": str(destination), "sha256": file_sha256(destination),
                        "queries": len(result["labels"]["local_y"])})
        print(json.dumps(outputs[-1]), flush=True)
    write_json(args.output / "protocol.json", {
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_replay": str(args.role_root),
        "target": args.target,
        "model_ids": list(model_ids),
        "streams": list(streams),
        "selection_basis": "three-seed final-core transfer matrix before example-level inspection",
        "discovery_stream": "original",
        "validation_stream": "fresh",
        "new_training": False,
        "model_weights_changed": False,
        "raw_input_identity_verified": True,
        "baseline_logits_bit_exact": True,
        "purpose": "test complementarity and localize matched-query transfer differences",
    })
    write_json(args.output / "DONE.json", {"complete": True, "files": outputs})


if __name__ == "__main__":
    main()
