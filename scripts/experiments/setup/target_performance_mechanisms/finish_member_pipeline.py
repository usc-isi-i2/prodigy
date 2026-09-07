"""Continue the already-running experiment after its training validity gate.

This is a finite train->verify->evaluate dependency, not a recurring monitor.
It never starts, changes, or interrupts training and never makes GPUs visible.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

TARGETS = ("covid_political", "facebook_page_reference", "election2020", "twibot20", "ukr_rus_suspended")
GATES = ("valid", "research_result", "same_consumed_anchors", "same_final_walk_rng",
         "same_retention_sets_across_role_treatments", "same_initialization_verified")


def require_substantive_receipt(receipt, manifest, revision):
    if not all(receipt.get(k) is True for k in GATES) or receipt.get("models") != 24 or receipt.get("steps_per_model") != 2500:
        raise ValueError("training validity gate did not pass the substantive 24-arm contract")
    if manifest.get("mode") != "training" or manifest.get("device") != "cpu" or len(manifest.get("jobs", [])) != 24:
        raise ValueError("expected the common-device CPU factorial")
    if manifest.get("revision") != revision or receipt.get("launch_revision") != revision:
        raise ValueError("training revision differs from the declared frozen version")


def compare_cached_inputs(outputs, reference_root):
    rows = []
    for stream, output in outputs.items():
        if not (output / "DONE").is_file():
            raise ValueError("replay is incomplete")
        for target in TARGETS:
            prior = "fresh_stage_cpu_20260906" if stream == "fresh" else (
                "specialist_cpu_tail_20260906" if target in {"twibot20", "ukr_rus_suspended"} else "specialist_cpu_20260906")
            current_path = output / target / "cache.json"
            reference_path = reference_root / prior / target / "cache.json"
            a, b = (json.loads(p.read_text()) for p in (current_path, reference_path))
            for key in ("episode_fingerprint", "batch_sha256", "episodes", "graph_path"):
                if a[key] != b[key]:
                    raise ValueError(f"cached {key} differs: {stream}/{target}")
            if len(a["batch_sha256"]) != 32 or a["episodes"] != 128:
                raise ValueError("wrong cached evaluation budget")
            rows.append({"stream": stream, "target": target, "cache_path": str(current_path),
                         "reference_path": str(reference_path), "episode_fingerprint": a["episode_fingerprint"],
                         "batch_sha256": a["batch_sha256"]})
    if set(outputs) != {"original", "fresh"}:
        raise ValueError("both episode streams required")
    return {"all_cached_batches_identical": True, "stream_target_cells": len(rows), "cells": rows}


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--training-run", type=Path, required=True)
    p.add_argument("--training-revision", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reference-root", type=Path, default=Path("/dataMeR1/phil/gfm/prodigy-mechanisms/log/target_mechanisms"))
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--max-wait-hours", type=float, default=12)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if not 1 <= args.threads <= 8 or not 0 < args.max_wait_hours <= 12:
        raise ValueError("invalid bounded execution resources")
    if args.output.exists() or not (args.training_run / "manifest.json").is_file():
        raise ValueError("output exists or training run is unresolved")
    outputs = {s: args.output.resolve() / s for s in ("original", "fresh")}
    verified = args.training_run.resolve() / "verified"
    commands = [[sys.executable, "-u", "-m", "scripts.experiments.setup.target_performance_mechanisms.replay",
                 "--model-list", str(verified / "model_list.tsv"), "--output", str(output),
                 "--datasets", ",".join(TARGETS), "--variants", "baseline", "--device", "123",
                 "--threads", str(args.threads), "--eval-episode-seed-offset", "0" if stream == "original" else "100003"]
                for stream, output in outputs.items()]
    if args.dry_run:
        print(json.dumps({"training_dependency": str(args.training_run), "commands": commands}, indent=2))
        return
    os.environ.update(CUDA_VISIBLE_DEVICES="", WANDB_MODE="offline", OMP_NUM_THREADS=str(args.threads), OPENBLAS_NUM_THREADS="1")
    args.output.mkdir(parents=True)
    status_path = args.output / "pipeline.json"
    status = {"status": "waiting_for_training_validity", "started": time.time(),
              "training_run": str(args.training_run.resolve()), "training_revision": args.training_revision,
              "evaluation_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "commands": commands, "device": "cpu", "threads": args.threads}
    def save():
        status_path.write_text(json.dumps(status, indent=2) + "\n")
    save()
    try:
        deadline = time.monotonic() + args.max_wait_hours * 3600
        receipt = None
        while receipt is None:
            if (verified / "DONE.json").is_file():
                try:
                    receipt = json.loads((verified / "DONE.json").read_text())
                except json.JSONDecodeError:
                    pass  # The training verifier may still be publishing it.
                if receipt is not None:
                    break
            training_status = args.training_run / "status.json"
            if training_status.is_file():
                try:
                    failed = json.loads(training_status.read_text()).get("status") == "failed"
                except json.JSONDecodeError:
                    failed = False  # A concurrent status write is not a failure.
                if failed:
                    raise RuntimeError("training or consumed-stream verification failed; no evaluation")
            if time.monotonic() >= deadline:
                raise TimeoutError("training validity receipt did not arrive within the bounded wait")
            time.sleep(30)
        manifest = json.loads((args.training_run / "manifest.json").read_text())
        require_substantive_receipt(receipt, manifest, args.training_revision)
        with (verified / "model_list.tsv").open() as handle:
            models = list(csv.DictReader(handle, delimiter="\t"))
        if len(models) != 24 or len({r["model_id"] for r in models}) != 24 or any(not Path(r["checkpoint"]).is_file() for r in models):
            raise ValueError("verified model manifest is incomplete")
        for stream, command in zip(outputs, commands):
            status.update(status=f"evaluating_{stream}", training_validity=receipt)
            save()
            with (args.output / f"{stream}.log").open("w") as log:
                subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        comparison = compare_cached_inputs(outputs, args.reference_root)
        (args.output / "input_validation.json").write_text(json.dumps(comparison, indent=2) + "\n")
        status.update(status="complete", completed=time.time(), exact_cached_inputs_verified=True)
        save()
    except BaseException:
        status.update(status="failed", error=traceback.format_exc(), completed=time.time())
        save()
        raise


if __name__ == "__main__":
    main()
