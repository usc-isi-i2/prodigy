"""Export exact training prefixes once, then run bounded no-update gradients."""
import argparse
import copy
import gzip
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback
from types import SimpleNamespace

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("WANDB_MODE", "offline")

import torch
import wandb

from experiments.layers import get_module_list
from experiments.run_shared_graph import prepare_shared_dataset, shared_storage_report, write_json
from experiments.run_single_experiment import load_dataset, seed_everything
from experiments.trainer import TrainerFS
from models.general_gnn import SingleLayerGeneralGNN
from .replay import batch_hash, clone_batch
from .verify_member_training import model_digest
from .support_identity_gradients import (VARIANTS, MODES, support_plan, label_intervention,
    verify_interventions, gradient_probe, tensor_receipt, probe_summary, gradient_comparison)


DEFAULT_REFERENCE = Path("/dataMeR1/phil/gfm/prodigy-mechanisms-freeze/log/target_mechanisms/readout_constraint_training_20260906")


def make_shell(params, state):
    p = copy.deepcopy(params)
    p["device"] = torch.device("cpu")
    expected = {"layers": "S,U,M", "gnn_type": "sage", "dropout": 0, "text_features_dropout": 0,
        "has_final_back": False, "skip_path": False, "meta_gnn_pos_only": False,
        "no_bn_encoder": False, "no_bn_metagraph": False, "task_name": "neighbor_matching",
        "use_edge_features": False, "attr_regression_weight": 0}
    if any(p.get(k) != v for k, v in expected.items()):
        raise ValueError("unhandled model/training architecture")
    layers = get_module_list(p["layers"], p["emb_dim"], edge_attr_dim=None,
        input_dim=p["input_dim"], dropout=p["dropout"], reset_after_layer=p["reset_after_layer"],
        attention_mask_scheme=p["attention_mask_scheme"], has_final_back=p["has_final_back"],
        msg_pos_only=p["meta_gnn_pos_only"], batch_norm_metagraph=not p["no_bn_metagraph"],
        batch_norm_encoder=not p["no_bn_encoder"], encoder_gnn_type=p["gnn_type"])
    label_dim = state["initial_label_mlp.weight"].shape[1]
    model = SingleLayerGeneralGNN(torch.nn.ModuleList(layers), initial_label_mlp=torch.nn.Linear(label_dim, p["emb_dim"]),
        params=p, text_dropout=torch.nn.Dropout(p["text_features_dropout"]))
    if not p["not_freeze_learned_label_embedding"]:
        model.learned_label_embedding.weight.requires_grad = False
    model.load_state_dict(state, strict=True)
    shell = SimpleNamespace(model=model, parameter=p, loss=torch.nn.CrossEntropyLoss(), is_regression=False,
        is_multiway=True, device=torch.device("cpu"), get_aux_loss=lambda g: 0.)
    shell.get_loss_and_acc = lambda y, pred: TrainerFS.get_loss_and_acc(shell, y, pred)
    shell._restore_rng_state = lambda rng: TrainerFS._restore_rng_state(shell, rng)
    return shell


def reference_jobs(root):
    verified = json.loads((root/"verified/DONE.json").read_text())
    if not verified["research_result"] or not verified["exact_paired_inputs"]:
        raise ValueError("unverified reference training")
    arms = [r for r in json.loads((root/"verified/arms.json").read_text()) if r["condition"] == "free"]
    if len(arms) != 9 or {(r["source"], r["seed"]) for r in arms} != {
        (s, i) for s in ("ukr_rus", "cp_hk", "covid") for i in range(3)}:
        raise ValueError("incomplete reference arm grid")
    plans = {p["prefix"]: (i, p) for i, p in enumerate(json.loads((root/"manifest.json").read_text())["jobs"])}
    result = []
    for arm in arms:
        index, plan = plans[arm["model_id"]]
        params = json.loads((root/f"job_{index:03d}/effective_config.json").read_text())
        if params != plan:
            raise ValueError("reference manifest/config differ")
        result.append({"arm": arm, "params": params})
    return result


def export_inputs(dataset, job, output, threads):
    arm, original = job["arm"], job["params"]
    params = copy.deepcopy(original)
    params.update(device=torch.device("cpu"), state_dir=str(output/"state"), log_dir=str(output/"log"),
        exp_name="support_identity_input_export", prefix="support_identity_input_export")
    seed_everything(params)
    trainer = TrainerFS(dataset, params)
    initial_path = Path(arm["checkpoint"]).with_name("training_state_0.ckpt")
    actual = torch.load(initial_path, map_location="cpu", weights_only=False)
    saved = actual["_training_checkpoint"]
    if saved["completed_steps"] != 0 or trainer._resume_parameter_contract() != saved["parameter_contract"]:
        raise ValueError("exact step-zero reconstruction contract differs")
    trainer.model.load_state_dict(actual["model"], strict=True)
    trainer.optimizer.load_state_dict(saved["optimizer"])
    trainer._training_batch_sampler().load_state_dict(saved["train_batch_sampler"])
    trainer._restore_rng_state(saved["rng"])
    if model_digest(trainer.model.state_dict()) != arm["initial_sha256"]:
        raise ValueError("initial model mismatch")
    reference = Path(original["log_dir"])/original["exp_name"]/"data/constraint_inputs.jsonl.gz"
    with gzip.open(reference, "rt") as handle:
        expected = [json.loads(next(handle)) for _ in range(4)]
    source_id = list(dataset.graph.source_graph_names).index(arm["source"])
    iterator = iter(trainer.train_dataloader)
    observed = []
    try:
        for step in range(1, 5):
            batch = clone_batch(next(iterator))
            record = {"step": step, "batch_sha256": batch_hash(batch)}
            if record != expected[step-1]:
                raise ValueError("actual full training input hash differs")
            nodes = batch[0].global_node_ids
            if not (dataset.graph.graph_id[nodes[nodes>=0]] == source_id).all():
                raise ValueError("wrong source node in reconstructed batch")
            torch.save(batch, output/f"batch_{step:03d}.pt")
            observed.append(record)
            del batch
    finally:
        if hasattr(iterator, "_shutdown_workers"):
            iterator._shutdown_workers()
    write_json(output/"params.json", trainer.parameter)
    # The lightweight second stage must reproduce the actual TrainerFS forward,
    # gradients and changed running buffers, not just load matching key shapes.
    batch = torch.load(output/"batch_001.pt", map_location="cpu", weights_only=False)
    torch.use_deterministic_algorithms(True)
    shell = make_shell(trainer.parameter, actual["model"])
    if [(n, p.requires_grad) for n, p in trainer.model.named_parameters()] != [(n, p.requires_grad) for n, p in shell.model.named_parameters()]:
        raise ValueError("reconstructed trainability differs")
    first = gradient_probe(trainer, actual["model"], batch, "training", saved["rng"])
    second = gradient_probe(shell, actual["model"], batch, "training", saved["rng"])
    if tensor_receipt(first) != tensor_receipt(second):
        raise ValueError("lightweight model differs from actual trainer")
    write_json(output/"DONE.json", {"complete": True, "arm": arm, "inputs": observed,
        "full_prefix_hashes_verified": True, "initial_checkpoint": str(initial_path),
        "actual_trainer_reconstruction_bit_exact": True, "reference_probe": tensor_receipt(first),
        "source_members_and_contexts_verified": True, "prefix_not_full_training": True,
        "params_path": str(output/"params.json")})
    wandb.finish()


def run_probes(dataset, job, output, threads):
    cache = Path(job["cache"])
    done = json.loads((cache/"DONE.json").read_text())
    if not done["complete"] or not done["full_prefix_hashes_verified"] or not done["actual_trainer_reconstruction_bit_exact"]:
        raise ValueError("unverified training input cache")
    arm = done["arm"]
    params = json.loads((cache/"params.json").read_text())
    initial = torch.load(done["initial_checkpoint"], map_location="cpu", weights_only=False)
    terminal = torch.load(arm["checkpoint"], map_location="cpu", weights_only=True)["model"]
    states = {0: initial["model"], 2500: terminal}
    if (model_digest(states[0]), model_digest(states[2500])) != (arm["initial_sha256"], arm["final_sha256"]):
        raise ValueError("reference weights changed")
    shell = make_shell(params, states[0])
    torch.use_deterministic_algorithms(True)
    rows = []
    for input_step in range(1, 2 if job["smoke"] else 5):
        batch = torch.load(cache/f"batch_{input_step:03d}.pt", map_location="cpu", weights_only=False)
        digest = batch_hash(batch)
        if digest != done["inputs"][input_step-1]["batch_sha256"]:
            raise ValueError("cached input was changed")
        plan = support_plan(batch, 99101+arm["seed"]*100+input_step)
        audits = verify_interventions(batch, plan)
        for checkpoint, state in states.items():
            for mode in MODES:
                base = gradient_probe(shell, state, batch, mode, initial["_training_checkpoint"]["rng"])
                repeat = gradient_probe(shell, state, batch, mode, initial["_training_checkpoint"]["rng"])
                if tensor_receipt(base) != tensor_receipt(repeat):
                    raise ValueError("deterministic baseline repeat differs")
                if checkpoint == 0 and input_step == 1 and mode == "training" and tensor_receipt(base) != done["reference_probe"]:
                    raise ValueError("cached actual-trainer reference changed")
                del repeat
                tensors = {"baseline": base}
                for variant in VARIANTS[1:]:
                    treated = label_intervention(batch, plan, variant)
                    tensors[variant] = gradient_probe(shell, state, treated, mode, initial["_training_checkpoint"]["rng"])
                    del treated
                name = f"input{input_step}_ckpt{checkpoint}_{mode}"
                tensor_path = output/f"{name}.pt"
                torch.save({"conditions": tensors, "plan": plan, "input_sha256": digest,
                    "weights_sha256": model_digest(state)}, tensor_path)
                for variant in VARIANTS:
                    summary = probe_summary(tensors[variant], base, plan)
                    if mode == "meta_frozen" and summary["post_query_max_difference"] != 0:
                        raise ValueError("support labels changed query vectors with frozen metagraph BN")
                    row = {"model_id": arm["model_id"], "source": arm["source"], "seed": arm["seed"],
                        "input_step": input_step, "checkpoint_step": checkpoint, "mode": mode, "variant": variant,
                        "input_sha256": digest, "weights_sha256": model_digest(state), "tensor_path": str(tensor_path),
                        "intervention_audit": audits, "baseline_repeat_bit_exact": True, **summary}
                    rows.append(row)
                    with (output/"cells.jsonl").open("a") as handle:
                        handle.write(json.dumps(row)+"\n")
                write_json(output/f"{name}_actual_vs_null.json", gradient_comparison(
                    tensors["identity_soft"]["gradients"], tensors["permuted_soft"]["gradients"]))
                print(f"Completed {arm['model_id']} {name}", flush=True)
                del tensors, base
        if batch_hash(batch) != digest:
            raise ValueError("original cached batch mutated")
        del batch
    shell.model.load_state_dict(states[0], strict=True)
    if model_digest(shell.model.state_dict()) != arm["initial_sha256"]:
        raise ValueError("could not restore exact model state")
    write_json(output/"DONE.json", {"complete": True, "model_id": arm["model_id"], "smoke": job["smoke"],
        "cells": len(rows), "no_optimizer_updates": True, "all_baseline_repeats_bit_exact": True})


def child_worker(function, dataset, job, output, threads):
    os.setsid()
    output = Path(output)
    with (output/"console.log").open("a", buffering=1) as stream:
        os.dup2(stream.fileno(), 1)
        os.dup2(stream.fileno(), 2)
        try:
            torch.set_num_threads(threads)
            torch.multiprocessing.set_sharing_strategy("file_system")
            if torch.cuda.is_available():
                raise ValueError("GPU must be hidden")
            function(dataset, job, output, threads)
        except BaseException:
            write_json(output/"FAILED.json", {"error": traceback.format_exc()})
            traceback.print_exc()
            raise


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--phase", choices=("inputs", "probe"), required=True)
    p.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    p.add_argument("--inputs", type=Path)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--models", type=int, default=3)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if args.run_dir.exists() or not 1 <= args.models <= 3 or not 1 <= args.threads <= 8 or torch.cuda.is_available():
        raise ValueError("new output and bounded CPU resources required")
    if args.phase == "inputs":
        if args.smoke or args.inputs:
            raise ValueError("export all nine exact four-batch prefixes once")
        jobs = reference_jobs(args.reference)
    else:
        if not args.inputs or not (args.inputs/"DONE.json").is_file():
            raise ValueError("complete prefix export required")
        jobs = []
        for root in sorted(args.inputs.glob("job_*")):
            d = json.loads((root/"DONE.json").read_text())
            if not args.smoke or (d["arm"]["source"], d["arm"]["seed"]) == ("cp_hk", 0):
                jobs.append({"cache": str(root), "smoke": args.smoke})
        if len(jobs) != (1 if args.smoke else 9):
            raise ValueError("incomplete cached model grid")
    print(json.dumps({"phase": args.phase, "jobs": len(jobs), "smoke": args.smoke,
        "full_gradients_cells": 12 if args.smoke else 432, "no_optimizer_updates": True}), flush=True)
    if args.dry_run:
        return
    mem = {r.split(':')[0]: int(r.split()[1]) for r in Path('/proc/meminfo').read_text().splitlines()}
    if args.phase == "inputs":
        shm = os.statvfs('/dev/shm')
        if mem['MemAvailable'] < 512*1024**2 or shm.f_bavail*shm.f_frsize < 200*1024**3:
            raise ValueError("insufficient graph/shared memory headroom")
    elif mem['MemAvailable'] < 96*1024**2:
        raise ValueError("insufficient gradient-probe memory headroom")
    args.run_dir.mkdir(parents=True)
    write_json(args.run_dir/"protocol.json", {**vars(args), "revision": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip(), "no_target_outcomes": True,
        "inputs_are_actual_prefix_not_full_training": True, "no_optimizer_updates": True})
    torch.set_num_threads(4)
    torch.multiprocessing.set_sharing_strategy("file_system")
    dataset = None
    if args.phase == "inputs":
        params = copy.deepcopy(jobs[0]["params"])
        params["device"] = torch.device("cpu")
        dataset = prepare_shared_dataset(load_dataset(params))
        if not all(shared_storage_report(dataset).values()):
            raise ValueError("graph storage not shared")
    context = torch.multiprocessing.get_context("spawn")
    active, next_job = {}, 0
    function = export_inputs if args.phase == "inputs" else run_probes
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    try:
        while next_job < len(jobs) or active:
            while next_job < len(jobs) and len(active) < args.models:
                output = args.run_dir/f"job_{next_job:03d}"
                output.mkdir()
                process = context.Process(target=child_worker, args=(function, dataset, jobs[next_job], str(output), args.threads))
                process.start()
                active[next_job] = process
                next_job += 1
            for index, process in list(active.items()):
                if not process.is_alive():
                    process.join()
                    del active[index]
                    if process.exitcode or not (args.run_dir/f"job_{index:03d}/DONE.json").is_file():
                        raise RuntimeError(f"job {index} failed, exit {process.exitcode}")
                    print(f"Completed job {index}", flush=True)
            time.sleep(1)
        write_json(args.run_dir/"DONE.json", {"complete": True, "phase": args.phase, "jobs": len(jobs),
            "smoke": args.smoke, "no_optimizer_updates": True})
    except BaseException:
        for process in active.values():
            if process.is_alive():
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    process.terminate()
        for process in active.values():
            process.join(timeout=5)
        write_json(args.run_dir/"FAILED.json", {"error": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
