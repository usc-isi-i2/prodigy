"""Reuse all nine completed corrected-sampler runs; never modify their pipeline."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

import pandas as pd
import torch
import yaml

from experiments.run_shared_graph import write_json
from scripts.experiments.setup.final_core.core_plan import SOURCES
from scripts.experiments.analysis.graphs.transfer_prediction.target_performance_mechanisms.analyze_trajectories import STEPS, DECODERS, PANEL
from .finish_member_pipeline import compare_cached_inputs
from .prepare_campaign_cls import FORWARD
from .verify_member_training import model_digest

TRAINING_REVISION = "edd1649e8446416213d2e48893d641b3b697eca1"
CONTRACT = {"seed": 0, "task_name": "neighbor_matching", "edge_view": "static_train", "target_edge_view": "static_test",
            "feature_subset": "all", "original_features": True, "n_hop": 2, "neighbor_sampling_hop_sizes": "9,9",
            "neighbor_sampling_node_limit": 101, "neighbor_matching_walk_hops": 1, "neighbor_sampling_strategy": "strict",
            "n_way": 30, "n_shots": 3, "n_query": 4, "batch_size": 4, "learning_rate": .002,
            "weight_decay": .001, "dataset_len_cap": 2500, "epochs": 1, "workers": 2,
            "root": "/dataMeR1/phil/data/merged/graphs",
            "graph_filename": "ukr_rus_covid_midterm_all9_facebook_final_core_split_seed0.pt"}


def validate_config(params, source, checkpoint_contract):
    expected = {**CONTRACT, **FORWARD, "neighbor_sampling_source_subset": source,
                "neighbor_matching_edge_split": True, "neighbor_sampling_episode_source": "graph_id",
                "neighbor_sampling_episode_source_weighting": "balanced", "neighbor_sampling_batch_source_mode": "independent",
                "neighbor_sampling_cross_source_prob": 0., "use_edge_features": False,
                "eval_only": False, "ablate_features": "none", "ablate_edges": "none"}
    for key, value in expected.items():
        if params.get(key) != value:
            raise ValueError(f"corrected training config differs: {source}: {key}")
    for key, value in checkpoint_contract.items():
        if params.get(key) != value:
            raise ValueError(f"saved training contract differs from effective config: {key}")


def verify_training(training_root, output):
    provenance = training_root / "log/final_core_rolefix/launch/provenance.txt"
    prov = dict(line.split("=", 1) for line in provenance.read_text().splitlines() if "=" in line)
    if prov.get("commit") != TRAINING_REVISION or prov.get("seeds") != "0" or prov.get("source_counts") != "1":
        raise ValueError("unexpected corrected training provenance")
    if not subprocess.check_output(["git", "-C", str(training_root), "show", f"{TRAINING_REVISION}:data/dataloader.py"], text=True).count("rng.shuffle(unique_nodes)"):
        raise ValueError("recorded training revision lacks the corrected member selector")
    records, expected_shapes = [], None
    for source in sorted(SOURCES):
        run_name = f"finalcore_ss_{source}_s0_20260906rolefix"
        log = training_root / "log/final_core_rolefix/train" / f"{run_name}.log"
        config_paths = re.findall(r"Saved effective config YAML to W&B files: ([^\r\n]+)", log.read_text())
        if len(config_paths) != 1:
            raise ValueError("effective training config not uniquely located")
        config_path = Path(config_paths[0])
        params = yaml.safe_load(config_path.read_text())["params"]
        root = training_root / "state/final_core_rolefix" / run_name / "checkpoint"
        for step in STEPS:
            path = root / f"state_dict_{step}.ckpt"
            sidecar = root / f"training_state_{step}.ckpt"
            state = torch.load(path, map_location="cpu", weights_only=True)["model"]
            training = torch.load(sidecar, map_location="cpu", weights_only=False)
            meta = training["_training_checkpoint"]
            validate_config(params, source, meta["parameter_contract"])
            if meta["completed_steps"] != step or meta["format_version"] != 1:
                raise ValueError("incorrect saved optimizer step")
            sha = model_digest(state)
            if model_digest(training["model"]) != sha:
                raise ValueError("weights-only checkpoint differs from training-state sidecar")
            optimizer_steps = [float(v["step"]) for v in meta["optimizer"]["state"].values() if "step" in v]
            if not optimizer_steps or min(optimizer_steps) != step or max(optimizer_steps) != step:
                raise ValueError("optimizer step count differs from declared checkpoint")
            shapes = {k: [list(v.shape), str(v.dtype)] for k, v in state.items()}
            if expected_shapes is None:
                expected_shapes = shapes
            if shapes != expected_shapes:
                raise ValueError("corrected checkpoint architectures differ")
            records.append({"model_id": f"corrected_ss_{source}_step{step}", "source": source, "sources": [source],
                "step": step, "training_seed": 0, "checkpoint": str(path), "weights_sha256": sha,
                "training_state": str(sidecar), "completed_steps_verified": step, "optimizer_steps_verified": step,
                "config_path": str(config_path), "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                "effective_config": params, "training_contract": meta["parameter_contract"]})
    if len(records) != 36 or len({r["weights_sha256"] for r in records}) != 36:
        raise ValueError("complete distinct nine-source four-step checkpoint grid required")
    write_json(output / "checkpoint_inventory.json", {"models": records, "training_provenance": prov,
        "training_provenance_path": str(provenance), "training_provenance_sha256": hashlib.sha256(provenance.read_bytes()).hexdigest(),
        "new_training": False, "changes_retention_and_roles_jointly": True, "matched_training_inputs_claim": False})
    pd.DataFrame([{"model_id": r["model_id"], "checkpoint": r["checkpoint"], "sources": r["source"]} for r in records]).to_csv(output / "model_list.tsv", sep="\t", index=False)
    return records


def verify_replay(outputs, inventory):
    lookup = {r["model_id"]: r for r in inventory}
    expected = {(m, d) for m in lookup for d in DECODERS}
    rows_total = 0
    for stream, output in outputs.items():
        if not (output / "DONE").is_file():
            raise ValueError("corrected replay incomplete")
        for target in PANEL:
            cache = json.loads((output / target / "cache.json").read_text())
            rows = [json.loads(line) for line in (output / target / "metrics.jsonl").read_text().splitlines()]
            if len(rows) != 612 or {(r["model_id"], r["decoder"]) for r in rows} != expected:
                raise ValueError("complete corrected checkpoint/decoder grid required")
            for row in rows:
                model = lookup[row["model_id"]]
                if row["dataset"] != target or row["variant"] != "baseline" or row["episodes"] != 128:
                    raise ValueError("unexpected corrected replay protocol")
                if any(row[k] != model[k] for k in ("checkpoint", "sources", "weights_sha256")) or row["episode_fingerprint"] != cache["episode_fingerprint"]:
                    raise ValueError("corrected replay weights/input identity mismatch")
                if not all(torch.isfinite(torch.tensor(row[m])) for m in ("roc_auc", "accuracy", "f1", "nll")):
                    raise ValueError("nonfinite corrected replay metric")
            rows_total += len(rows)
    if set(outputs) != {"original", "fresh"} or rows_total != 6120:
        raise ValueError("both complete corrected streams required")
    return rows_total


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--training-root", type=Path, default=Path("/dataMeR1/phil/gfm/prodigy-roleexposure"))
    parser.add_argument("--reference-root", type=Path, default=Path("/dataMeR1/phil/gfm/prodigy-mechanisms/log/target_mechanisms"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.threads <= 8 or torch.cuda.is_available():
        raise ValueError("new output, bounded CPU threads and hidden GPUs required")
    if args.dry_run:
        print("Read-only reuse: nine corrected sources, four saved steps, five targets, 17 decoders, two streams; 6120 rows.")
        return
    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True)
    status = {"status": "verifying_corrected_training", "started": time.time(),
              "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), "device": "cpu"}
    write_json(args.output / "pipeline.json", status)
    try:
        inventory = verify_training(args.training_root, args.output)
        outputs = {}
        for stream, offset in (("original", 0), ("fresh", 100003)):
            output = args.output / stream
            outputs[stream] = output
            status["status"] = f"replaying_{stream}"
            write_json(args.output / "pipeline.json", status)
            command = [sys.executable, "-u", "-m", "scripts.experiments.setup.target_performance_mechanisms.replay",
                       "--model-list", str(args.output / "model_list.tsv"), "--output", str(output),
                       "--datasets", ",".join(PANEL), "--variants", "baseline", "--device", "123", "--threads", str(args.threads),
                       "--eval-episode-seed-offset", str(offset)]
            with (args.output / f"{stream}.log").open("w") as log:
                subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        write_json(args.output / "input_validation.json", compare_cached_inputs(outputs, args.reference_root))
        total = verify_replay(outputs, inventory)
        status.update(status="complete", completed=time.time())
        write_json(args.output / "pipeline.json", status)
        write_json(args.output / "DONE.json", {"rows": total, "models": 36, "underlying_training_runs": 9,
            "training_seeds": [0], "steps": list(STEPS), "training_sidecars_and_configs_verified": True,
            "all_cached_inputs_match": True, "new_training": False, "matched_training_inputs_claim": False})
    except BaseException:
        status.update(status="failed", error=traceback.format_exc())
        write_json(args.output / "pipeline.json", status)
        raise


if __name__ == "__main__":
    main()
