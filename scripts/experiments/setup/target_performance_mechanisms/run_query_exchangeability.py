"""No-update CPU audit on 36 verified actual NM input batches and 9 saved models."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import traceback

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("WANDB_MODE", "offline")

import numpy as np
import torch
from torch_geometric.data import Batch

from experiments.run_shared_graph import write_json
from .query_exchangeability import cyclic_permutation, symmetrized_groups, group_totals
from .replay import batch_hash, clone_batch, trace_stages
from .run_support_identity_gradients import make_shell
from .verify_member_training import model_digest


def reset_model(shell, state, mode, rng):
    shell.model.load_state_dict(state, strict=True)
    shell.model.train()
    if mode == "meta_frozen":
        for m in shell.model.layer_list[2].modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                m.eval()
    elif mode != "training":
        raise ValueError("unknown normalization mode")
    shell._restore_rng_state(rng)


def nm_suffix(model, batch, pre):
    if (len(model.layer_list) != 3 or not model.params["ignore_label_embeddings"]
        or model.params["zero_label_embeddings"] or model.params["skip_path"]
        or model.params["zero_shot"] or model.layer_list[2].num_gnn_layers != 1
        or model.layer_list[2].gnn_layers_back is not None):
        raise ValueError("unexpected NM model recipe")
    labels = model.learned_label_embedding(torch.arange(len(batch[1])))
    x, labels = model.forward_metagraph(model.layer_list[2], pre, labels, *batch[3:9])
    logits = model.decode(model.final_input_mlp(x), model.final_label_mlp(labels), batch[3]).reshape(batch[2].shape)
    q = batch[5].reshape(-1, batch[2].shape[1])[:, 0].bool()
    return {"post": torch.cat([x, labels]), "logits": logits[q]}


def whole_permutation(batch, p):
    """Move whole sampled subgraphs, preserving every collator-level field."""
    graphs = batch[0].to_data_list()
    g = Batch.from_data_list([graphs[i] for i in p.tolist()])
    # Fields added after Batch.from_data_list are metadata, not sampled contents.
    for k, v in batch[0]:
        if k not in batch[0]._slice_dict and k not in ("batch", "ptr"):
            g[k] = v.clone() if isinstance(v, torch.Tensor) else v
    return [g, *[x.clone() for x in batch[1:]]]


def maximum_difference(a, b):
    if a.shape != b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("invalid compared tensor")
    return float((a-b).abs().max()) if a.numel() else 0.


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    if args.run_dir.exists():
        raise ValueError("output already exists; preserve earlier evidence")
    args.run_dir.mkdir(parents=True)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        raise ValueError("CPU-only audit")
    ref = json.loads((args.probes/"verified/DONE.json").read_text())
    if not ref["complete"] or ref["smoke"] or not ref["every_input_full_hash_reverified"]:
        raise ValueError("unverified gradient tensor campaign")
    protocol = {"revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "inputs": str(args.inputs), "probes": str(args.probes), "smoke": args.smoke,
        "threads": args.threads, "torch": torch.__version__, "permutation": "one cyclic shift per identity within episode",
        "whole_graph_inputs": "first actual batch of every model; both checkpoints and normalization modes",
        "whole_graph_atol": 1e-4, "whole_graph_rtol": 1e-5,
        "no_optimizer_updates": True, "no_target_outcomes": True, "population_bound_assumption":
        "conditionally exchangeable context draws; code has no anchor argument, not proof of PRNG independence"}
    write_json(args.run_dir/"protocol.json", protocol)
    cells, groups, audits, witnesses = [], [], [], []
    jobs = [3] if args.smoke else range(9)
    for job in jobs:
        cache = args.inputs/f"job_{job:03d}"
        receipt = json.loads((cache/"DONE.json").read_text())
        if not all(receipt[k] for k in ("complete", "full_prefix_hashes_verified", "actual_trainer_reconstruction_bit_exact")):
            raise ValueError("unverified actual-input prefix")
        arm = receipt["arm"]
        initial = torch.load(receipt["initial_checkpoint"], map_location="cpu", weights_only=False)
        states = {0: initial["model"], 2500: torch.load(arm["checkpoint"], map_location="cpu", weights_only=True)["model"]}
        if (model_digest(states[0]), model_digest(states[2500])) != (arm["initial_sha256"], arm["final_sha256"]):
            raise ValueError("source checkpoint changed")
        params = json.loads((cache/"params.json").read_text())
        shell = make_shell(params, states[0])
        rng = initial["_training_checkpoint"]["rng"]
        for step in ([1] if args.smoke else range(1, 5)):
            batch = torch.load(cache/f"batch_{step:03d}.pt", map_location="cpu", weights_only=False)
            digest = batch_hash(batch)
            if digest != receipt["inputs"][step-1]["batch_sha256"]:
                raise ValueError("full input changed")
            q = batch[5].reshape(-1, batch[2].shape[1])[:, 0].bool()
            ids = batch[0].global_node_ids[batch[0].ptr[:-1]]
            tasks = batch[0].task_id_per_sample
            qp = torch.from_numpy(cyclic_permutation(ids[q].numpy(), tasks[q].numpy()))
            p = torch.arange(len(ids))
            p[q] = torch.where(q)[0][qp]
            if not torch.equal(ids[p], ids) or not torch.equal(tasks[p], tasks) or not torch.equal(q[p], q):
                raise ValueError("permutation crossed identity, episode or role")
            if not torch.equal(batch[4].reshape(len(ids), -1)[p], batch[4].reshape(len(ids), -1)):
                raise ValueError("query metagraph attributes are not exchangeable")
            if not torch.equal(batch[3][1].reshape(len(ids), -1)[p], batch[3][1].reshape(len(ids), -1)):
                raise ValueError("query metagraph label neighborhoods differ")
            if step == 1:
                reconstructed = whole_permutation(batch, torch.arange(len(ids)))
                if batch_hash(reconstructed) != digest:
                    raise ValueError("whole-graph identity roundtrip changed input")
                del reconstructed
                permuted = whole_permutation(batch, p)
                for a, b in zip(batch[1:], permuted[1:]):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                torch.testing.assert_close(permuted[0].global_node_ids[permuted[0].ptr[:-1]], ids, rtol=0, atol=0)
            for ckpt, state in states.items():
                for mode in ("training", "meta_frozen"):
                    path = args.probes/f"job_{job:03d}"/f"input{step}_ckpt{ckpt}_{mode}.pt"
                    saved = torch.load(path, map_location="cpu", weights_only=False)
                    if saved["input_sha256"] != digest or saved["weights_sha256"] != model_digest(state):
                        raise ValueError("baseline tensor provenance differs")
                    base = saved["conditions"]["baseline"]
                    torch.testing.assert_close(base["truth"], batch[2][q], rtol=0, atol=0)
                    key = {"model_id": arm["model_id"], "source": arm["source"], "seed": arm["seed"],
                        "input_step": step, "checkpoint_step": ckpt, "mode": mode}
                    rows = symmetrized_groups(base["logits"].numpy(), base["truth"].argmax(1).numpy(), ids[q].numpy(), tasks[q].numpy())
                    groups.extend({**key, **r} for r in rows)
                    for cohort in ("all", "unique", "repeated"):
                        selected = rows if cohort == "all" else [r for r in rows if r["cohort"] == cohort]
                        cells.append({**key, "cohort": cohort, **group_totals(selected)})
                    with torch.no_grad():
                        reset_model(shell, state, mode, rng)
                        suffix_base = nm_suffix(shell.model, batch, base["pre"])
                        for k in ("logits", "post"):
                            torch.testing.assert_close(suffix_base[k], base[k], rtol=0, atol=0)
                        reset_model(shell, state, mode, rng)
                        suffix_swap = nm_suffix(shell.model, batch, base["pre"][p])
                        post_p = torch.cat([p, torch.arange(len(p), len(base["post"]))])
                        audit = {**key, "input_sha256": digest, "weights_sha256": model_digest(state),
                            "suffix_baseline_bit_exact": True, "moved_queries": int((qp != torch.arange(len(qp))).sum()),
                            "suffix_logit_error": maximum_difference(suffix_swap["logits"], base["logits"][qp]),
                            "suffix_post_error": maximum_difference(suffix_swap["post"], base["post"][post_p])}
                        # Training BN can change finite-precision reduction order even
                        # for a mathematical row permutation. Never round it to zero.
                        for k, expected in (("logits", base["logits"][qp]), ("post", base["post"][post_p])):
                            torch.testing.assert_close(suffix_swap[k], expected, rtol=1e-5, atol=1e-4)
                        if step == 1:
                            reset_model(shell, state, mode, rng)
                            b = clone_batch(batch)
                            with trace_stages(shell.model, b[0]) as trace:
                                truth, logits, _ = shell.model(*b)
                            torch.testing.assert_close(logits, base["logits"], rtol=0, atol=0)
                            torch.testing.assert_close(trace["U1_pre_meta"], base["pre"], rtol=0, atol=0)
                            del b, trace
                            reset_model(shell, state, mode, rng)
                            b = clone_batch(permuted)
                            with trace_stages(shell.model, b[0]) as trace:
                                truth, logits, _ = shell.model(*b)
                            torch.testing.assert_close(truth, base["truth"], rtol=0, atol=0)
                            expected_pre = base["pre"][p]
                            expected_logits = base["logits"][qp]
                            for a, expected in ((trace["U1_pre_meta"], expected_pre), (logits, expected_logits)):
                                torch.testing.assert_close(a, expected, rtol=1e-5, atol=1e-4)
                            audit.update(whole_baseline_bit_exact=True,
                                whole_pre_error=maximum_difference(trace["U1_pre_meta"], expected_pre),
                                whole_logit_error=maximum_difference(logits, expected_logits),
                                whole_vs_suffix_error=maximum_difference(logits, suffix_swap["logits"]),
                                whole_argmax_mismatches=int((logits.argmax(1) != expected_logits.argmax(1)).sum()))
                            witnesses.append({**key, "permutation": p, "baseline_logits": base["logits"],
                                "suffix_swapped_logits": suffix_swap["logits"], "whole_swapped_logits": logits.clone(),
                                "whole_swapped_pre": trace["U1_pre_meta"], "expected_pre": expected_pre,
                                "truth": truth, "input_sha256": digest})
                            del b, trace
                        audits.append(audit)
                    del saved, base
            if batch_hash(batch) != digest:
                raise ValueError("original input mutated")
            print(f"Completed {arm['source']} seed={arm['seed']} input={step}", flush=True)
            if step == 1:
                del permuted
            del batch
        write_csv(args.run_dir/"cells.csv", cells)
        write_csv(args.run_dir/"groups.csv", groups)
        write_csv(args.run_dir/"audits.csv", audits)
        torch.save(witnesses, args.run_dir/"whole_input_witnesses.pt")
    write_json(args.run_dir/"DONE.json", {"complete": True, "smoke": args.smoke, "cells": len(cells),
        "identity_group_rows": len(groups), "suffix_audits": len(audits), "whole_input_audits": len(witnesses),
        "no_optimizer_updates": True, "no_target_outcomes": True, "all_full_input_hashes_reverified": True,
        "all_suffix_baselines_bit_exact": True, "all_whole_baselines_bit_exact": True,
        "symmetrized_experiment_not_realized_input_bound": True})


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--probes", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    existed = args.run_dir.exists()
    try:
        run(args)
    except BaseException:
        if not existed and args.run_dir.exists():
            write_json(args.run_dir/"FAILED.json", {"error": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
