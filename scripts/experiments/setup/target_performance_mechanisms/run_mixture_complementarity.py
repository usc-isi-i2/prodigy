"""Verify historical weights, replay both complete mixture grids, preserve inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

import pandas as pd
import torch
import math

from experiments.run_shared_graph import write_json
from .finish_member_pipeline import compare_cached_inputs
from .prepare_mixture_complementarity import TARGETS, validate_lattice
from .verify_member_training import model_digest
from scripts.experiments.analysis.graphs.transfer_prediction.target_performance_mechanisms.analyze_trajectories import DECODERS


def verify_replay_tables(outputs, inventory):
    expected = {(r["model_id"], d) for r in inventory if r["source_count"] > 1 for d in DECODERS}
    lookup = {r["model_id"]: r for r in inventory}
    count, full_count = 0, 0
    for stream, output in outputs.items():
        if not (output / "DONE").is_file():
            raise ValueError("mixture replay incomplete")
        for target in TARGETS:
            cache = json.loads((output / target / "cache.json").read_text())
            rows = [json.loads(line) for line in (output / target / "metrics.jsonl").read_text().splitlines()]
            if len(rows) != 765 or {(r["model_id"], r["decoder"]) for r in rows} != expected:
                raise ValueError("mixture replay grid incomplete or duplicated")
            for row in rows:
                model = lookup[row["model_id"]]
                if row["dataset"] != target or row["variant"] != "baseline" or row["episodes"] != 128:
                    raise ValueError("mixture replay protocol differs")
                if row["checkpoint"] != model["checkpoint"] or row["weights_sha256"] != model["weights_sha256"] or sorted(row["sources"]) != model["sources"]:
                    raise ValueError("replay differs from verified weights or source set")
                if row["episode_fingerprint"] != cache["episode_fingerprint"] or len(cache["batch_sha256"]) != 32:
                    raise ValueError("replay differs from cached inputs")
                if any(not math.isfinite(row[k]) for k in ("roc_auc", "accuracy", "f1", "nll")):
                    raise ValueError("nonfinite replay metric")
                if row["decoder"] == "full_model":
                    full_count += 1
                    if stream == "original" and (row.get("official_metric_max_abs_error", math.inf) > 1e-5 or row.get("official_decision_metric_max_abs_error", math.inf) > 1e-6):
                        raise ValueError("missing or failed original official parity")
            count += len(rows)
    if set(outputs) != {"original", "fresh"} or count != 7650 or full_count != 450:
        raise ValueError("complete original and fresh mixture grid required")
    return {"rows": count, "full_model_cells": full_count, "original_official_parity_cells": 225}


def verify_inventory(inputs):
    manifest = json.loads((inputs / "manifest.json").read_text())
    for name, key in (("classification_long.tsv", "snapshot_metrics_sha256"), ("model_list.tsv", "snapshot_models_sha256")):
        if hashlib.sha256((inputs / name).read_bytes()).hexdigest() != manifest[key]:
            raise ValueError("frozen input snapshot changed")
    models = validate_lattice(pd.read_csv(inputs / "classification_long.tsv", sep="\t"),
                              pd.read_csv(inputs / "model_list.tsv", sep="\t"))
    if manifest.get("new_training") is not False or manifest.get("role_corrected_runs_included") is not False:
        raise ValueError("unexpected training provenance")
    declared = {r["model_id"]: r for r in manifest["models"]}
    if len(declared) != 54:
        raise ValueError("checkpoint manifest incomplete")
    shape_reference, records = None, []
    for row in models.itertuples():
        if list(row.source_set) != declared[row.model_id]["sources"] or row.checkpoint != declared[row.model_id]["checkpoint"]:
            raise ValueError("checkpoint snapshot differs from manifest")
        if Path(row.checkpoint).name != "state_dict_2500.ckpt":
            raise ValueError("wrong comparison checkpoint")
        state = torch.load(row.checkpoint, map_location="cpu", weights_only=True)["model"]
        shape = {k: [list(v.shape), str(v.dtype)] for k, v in state.items()}
        if shape_reference is None:
            shape_reference = shape
        if shape != shape_reference:
            raise ValueError("historical architectures differ")
        records.append({**declared[row.model_id], "weights_sha256": model_digest(state)})
    return records


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--reference-root", type=Path, default=Path("/dataMeR1/phil/gfm/prodigy-mechanisms/log/target_mechanisms"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.threads <= 8 or torch.cuda.is_available():
        raise ValueError("existing output, invalid threads, or visible GPU")
    torch.set_num_threads(args.threads)
    if args.dry_run:
        validate_lattice(pd.read_csv(args.inputs / "classification_long.tsv", sep="\t"), pd.read_csv(args.inputs / "model_list.tsv", sep="\t"))
        print("45 frozen mixture models, five targets, two streams, 128 episodes; no training.")
        return
    args.output.mkdir(parents=True)
    status = {"status": "verifying_weights", "started": time.time(), "device": "cpu", "threads": args.threads,
              "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}
    write_json(args.output / "pipeline.json", status)
    try:
        inventory = verify_inventory(args.inputs)
        write_json(args.output / "checkpoint_inventory.json", {"models": inventory, "finite_common_architecture": True})
        # Generate the executable mixture list from the verified full inventory.
        model_list = args.output / "verified_mixture_models.tsv"
        model_list.write_text("model_id\tcheckpoint\tsources\n" + "".join(
            f"{r['model_id']}\t{r['checkpoint']}\t{','.join(r['sources'])}\n" for r in inventory if r["source_count"] > 1))
        outputs = {}
        for stream, offset in (("original", 0), ("fresh", 100003)):
            output = args.output / stream
            outputs[stream] = output
            status["status"] = f"replaying_{stream}"
            write_json(args.output / "pipeline.json", status)
            command = [sys.executable, "-u", "-m", "scripts.experiments.setup.target_performance_mechanisms.replay",
                       "--model-list", str(model_list), "--output", str(output), "--datasets", ",".join(sorted(TARGETS)),
                       "--variants", "baseline", "--device", "123", "--threads", str(args.threads),
                       "--eval-episode-seed-offset", str(offset), "--reference-tsv", str(args.inputs / "classification_long.tsv"),
                       "--auc-parity-atol", "0.00001"]
            with (args.output / f"{stream}.log").open("w") as log:
                subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        write_json(args.output / "input_validation.json", compare_cached_inputs(outputs, args.reference_root))
        grid = verify_replay_tables(outputs, inventory)
        status.update(status="complete", completed=time.time())
        write_json(args.output / "pipeline.json", status)
        write_json(args.output / "DONE.json", {"mixture_models": 45, "targets": 5, "streams": 2,
            **grid, "new_training": False, "all_cached_inputs_match": True})
    except BaseException:
        status.update(status="failed", error=traceback.format_exc())
        write_json(args.output / "pipeline.json", status)
        raise


if __name__ == "__main__":
    main()
