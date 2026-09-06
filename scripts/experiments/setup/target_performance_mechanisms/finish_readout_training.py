"""Finite verified-training to fixed-target-evaluation continuation; no monitoring."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

from experiments.run_shared_graph import write_json
from .finish_member_pipeline import TARGETS, compare_cached_inputs


def run_evaluation(training, output, threads=4, reference_root=Path("/dataMeR1/phil/gfm/prodigy-mechanisms/log/target_mechanisms")):
    receipt = json.loads((training / "verified/DONE.json").read_text())
    if receipt.get("models") != 18 or receipt.get("steps_per_model") != 2500 or not all(receipt.get(k) is True for k in (
            "valid", "research_result", "exact_paired_inputs", "same_initialization", "same_final_walk_rng",
            "exact_frozen_tensors_all_updates_and_checkpoints", "complete_effective_config_checks")):
        raise ValueError("complete substantive training validity gate required")
    if output.exists() or not 1 <= threads <= 8:
        raise ValueError("existing output or invalid thread budget")
    output.mkdir(parents=True)
    status = {"status": "starting", "started": time.time(), "training": str(training),
              "training_validity": receipt, "device": "cpu", "threads": threads}
    try:
        outputs = {}
        for stream, offset in (("original", 0), ("fresh", 100003)):
            destination = output / stream
            outputs[stream] = destination
            status["status"] = f"evaluating_{stream}"
            write_json(output / "pipeline.json", status)
            command = [sys.executable, "-u", "-m", "scripts.experiments.setup.target_performance_mechanisms.replay",
                       "--model-list", str(training / "verified/model_list.tsv"), "--output", str(destination),
                       "--datasets", ",".join(TARGETS), "--variants", "baseline", "--device", "123",
                       "--threads", str(threads), "--eval-episode-seed-offset", str(offset)]
            with (output / f"{stream}.log").open("w") as log:
                subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        write_json(output / "input_validation.json", compare_cached_inputs(outputs, reference_root))
        status.update(status="complete", completed=time.time())
        write_json(output / "pipeline.json", status)
        write_json(output / "DONE.json", {"models": 18, "streams": 2, "targets": 5,
            "all_cached_batches_identical": True, "training_constraint_verified": True})
    except BaseException:
        status.update(status="failed", error=traceback.format_exc())
        write_json(output / "pipeline.json", status)
        raise


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--reference-root", type=Path, default=Path("/dataMeR1/phil/gfm/prodigy-mechanisms/log/target_mechanisms"))
    args = parser.parse_args()
    run_evaluation(args.training.resolve(), args.output.resolve(), args.threads, args.reference_root)


if __name__ == "__main__":
    main()
