"""Require exact paired examples/initialization and a maintained readout constraint."""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path

import torch

from experiments.episode_audit import digest
from experiments.run_shared_graph import write_json
from .build_readout_interventions import READOUT_KEYS
from .readout_training_constraint import CONDITIONS, SOURCES
from .run_readout_training import make_plans
from .verify_member_training import audit_consumed, model_digest


def input_hashes(path, steps):
    hashes = []
    with gzip.open(path, "rt") as handle:
        for step, line in enumerate(handle, 1):
            row = json.loads(line)
            value = row.get("batch_sha256", "")
            if row.get("step") != step or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("invalid full-input audit sequence or digest")
            hashes.append(value)
    if len(hashes) != steps:
        raise ValueError("full-input audit incomplete")
    return hashes


def check_pairs(records):
    expected = {(source, seed, condition) for source in SOURCES for seed in range(3) for condition in CONDITIONS}
    if len(records) != 18 or {(r["source"], r["seed"], r["condition"]) for r in records} != expected:
        raise ValueError("complete 18-arm grid required")
    grouped = defaultdict(dict)
    for row in records:
        grouped[row["source"], row["seed"]][row["condition"]] = row
    for group in grouped.values():
        a, b = (group[c] for c in CONDITIONS)
        for key in ("initial_sha256", "anchor_hashes", "member_set_hashes", "member_order_hashes",
                    "input_hashes", "final_walk_rng_sha256", "summary"):
            if a[key] != b[key]:
                raise ValueError(f"paired training differs in {key}")
    for seed in range(3):
        if len({r["initial_sha256"] for r in records if r["seed"] == seed}) != 1:
            raise ValueError("initialization differs across sources within seed")
    if len({r["initial_sha256"] for r in records}) != 3:
        raise ValueError("three distinct initializations required")


def verify_checkpoint_states(initial, states, condition):
    if condition not in CONDITIONS or not READOUT_KEYS <= initial.keys():
        raise ValueError("unknown readout constraint or missing tensors")
    model_digest(initial)
    changed = []
    for step, state in states.items():
        if set(state) != set(initial):
            raise ValueError("checkpoint layout differs")
        model_digest(state)
        if condition == "frozen" and any(not torch.equal(state[k], initial[k]) for k in READOUT_KEYS):
            raise ValueError("frozen tensor changed at a saved checkpoint")
        if step:
            other_parameters = [k for k in initial if k not in READOUT_KEYS and
                                k.endswith((".weight", ".bias", "logit_scale"))]
            if not any(not torch.equal(state[k], initial[k]) for k in other_parameters):
                raise ValueError("other network weights did not update")
            if condition == "free" and not all(not torch.equal(state[k], initial[k]) for k in READOUT_KEYS):
                raise ValueError("control readout did not update")
        changed.append(step)
    return sorted(changed)


def compare_prior_controls(records, prior_run, steps):
    """Check earlier member identities/RNG and report, rather than hide, weight drift."""
    arms = json.loads((prior_run / "verified/arms.json").read_text())
    jobs = json.loads((prior_run / "manifest.json").read_text())["jobs"]
    params_by_id = {p["prefix"]: p for p in jobs}
    output = []
    for current in records:
        if current["condition"] != "free" or current["source"] not in {"ukr_rus", "cp_hk"}:
            continue
        matches = [r for r in arms if r["source"] == current["source"] and r["seed"] == current["seed"] and r["policy"] == "lowest_sorted"]
        if len(matches) != 1:
            raise ValueError("previous control not uniquely resolved")
        prior = matches[0]
        p = params_by_id[prior["model_id"]]
        consumed = audit_consumed(Path(p["log_dir"]) / p["exp_name"] / "data/consumed_episodes.jsonl.gz", p, steps)
        for key in ("anchor_hashes", "member_order_hashes", "member_set_hashes", "summary"):
            if consumed[key] != current[key]:
                raise ValueError(f"new free control changed prior consumed {key}")
        if current["initial_sha256"] != prior["initial_sha256"] or current["final_walk_rng_sha256"] != prior["final_walk_rng_sha256"]:
            raise ValueError("new free control changed prior initialization or walk RNG")
        old_state = torch.load(prior["checkpoint"], map_location="cpu", weights_only=True)["model"]
        new_state = torch.load(current["checkpoint"], map_location="cpu", weights_only=True)["model"]
        if model_digest(old_state) != prior["final_sha256"]:
            raise ValueError("previous verified checkpoint changed")
        changed = [k for k in old_state if not torch.equal(old_state[k], new_state[k])]
        output.append({"model_id": current["model_id"], "previous_model_id": prior["model_id"],
                       "source": current["source"], "seed": current["seed"],
                       "same_initialization_and_member_stream": True, "same_final_walk_rng": True,
                       "old_full_input_hashes_available": False,
                       "previous_weights_sha256": prior["final_sha256"], "weights_sha256": current["final_sha256"],
                       "weights_bit_exact": not changed, "changed_keys": changed,
                       "maximum_absolute_tensor_difference": max(float((new_state[k].double() - old_state[k].double()).abs().max()) for k in old_state)})
    if len(output) != 6:
        raise ValueError("all six historical matched controls required")
    return output


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-steps", type=int, default=2500)
    parser.add_argument("--allow-smoke", action="store_true")
    parser.add_argument("--prior-training", type=Path, default=Path("/dataMeR1/phil/gfm/prodigy-mechanisms-train/log/target_mechanisms/member_cpu_training_20260906"))
    args = parser.parse_args()
    if args.output.exists() or (not args.allow_smoke and args.expected_steps != 2500):
        raise ValueError("existing output or wrong substantive budget")
    manifest = json.loads((args.run_dir / "manifest.json").read_text())
    status = json.loads((args.run_dir / "status.json").read_text())
    if status["status"] != "complete" or len(manifest["jobs"]) != 18 or manifest["device"] != "cpu":
        raise ValueError("requires complete CPU training")
    if (manifest["mode"] == "smoke") != args.allow_smoke:
        raise ValueError("smoke/substantive contract mismatch")
    source_names = manifest["source_names"]
    if len(source_names) != len(set(source_names)) or not set(SOURCES) <= set(source_names):
        raise ValueError("shared graph source registry invalid")
    workers = manifest["jobs"][0]["workers"]
    expected = make_plans(args.run_dir, workers, args.expected_steps if args.allow_smoke else 0)
    ignored = {"timestamp", "exp_name"}
    records = []
    for index, (plan, recipe) in enumerate(zip(manifest["jobs"], expected)):
        job = args.run_dir / f"job_{index:03d}"
        p = json.loads((job / "effective_config.json").read_text())
        result = json.loads((job / "result.json").read_text())
        recipe = json.loads(json.dumps(recipe, default=str))
        if p != plan or {k: v for k, v in p.items() if k not in ignored} != {k: v for k, v in recipe.items() if k not in ignored}:
            raise ValueError("effective configuration differs from declared recipe/manifest")
        if result["status"] != "complete" or result["steps_observed"] != args.expected_steps:
            raise ValueError("incomplete training")
        condition, source, seed = p["readout_training_condition"], p["neighbor_sampling_source_subset"], p["seed"]
        audit = result["training_constraint"]
        if (audit["condition"] != condition or set(audit["readout_keys"]) != READOUT_KEYS
                or audit["steps_checked"] != args.expected_steps or audit["inputs_hashed"] != args.expected_steps
                or audit["other_trainable_parameters_unchanged"] is not True):
            raise ValueError("per-update constraint audit incomplete")
        names = sum(audit["optimizer_parameter_names"], [])
        if len(names) != len(set(names)) or (set(names) & READOUT_KEYS) != (READOUT_KEYS if condition == "free" else set()):
            raise ValueError("readout optimizer inventory differs")
        ckpt = Path(result["checkpoint_dir"])
        schedule = [0, args.expected_steps] if args.allow_smoke else [0, 100, 300, 900, 2500]
        states = {step: torch.load(ckpt / f"state_dict_{step}.ckpt", map_location="cpu", weights_only=True)["model"] for step in schedule}
        verified_steps = verify_checkpoint_states(states[0], states, condition)
        initial_sha = model_digest(states[0])
        if initial_sha != audit["initial_model_sha256"] or model_digest({k: states[0][k] for k in READOUT_KEYS}) != audit["initial_readout_sha256"]:
            raise ValueError("saved initialization differs from actual training constraint")
        terminal = states[args.expected_steps]
        final_sha = model_digest(terminal)
        full = torch.load(ckpt / f"training_state_{args.expected_steps}.ckpt", map_location="cpu", weights_only=False)
        training = full["_training_checkpoint"]
        if training["completed_steps"] != args.expected_steps or model_digest(full["model"]) != final_sha:
            raise ValueError("terminal and full-state checkpoint disagree")
        groups = training["optimizer"]["param_groups"]
        if [len(g["params"]) for g in groups] != [len(g) for g in audit["optimizer_parameter_names"]]:
            raise ValueError("saved optimizer inventory differs")
        if sorted(sum([g["params"] for g in groups], [])) != list(range(len(names))):
            raise ValueError("unexpected saved optimizer parameter IDs")
        task_state = training["train_batch_sampler"]["task_sampling_state"]
        if task_state["member_policy"] != "lowest_sorted" or task_state["member_sampling_seed"] != 280100 + seed:
            raise ValueError("wrong saved sampling policy")
        log = Path(p["log_dir"]) / p["exp_name"] / "data"
        consumed = audit_consumed(log / "consumed_episodes.jsonl.gz", p, args.expected_steps)
        if {str(k): v for k, v in consumed["summary"]["source_ids"].items()} != {str(source_names.index(source)): args.expected_steps * p["batch_size"]}:
            raise ValueError("consumed source labels differ")
        record = {"model_id": p["prefix"], "source": source, "seed": seed, "condition": condition,
                  "initial_sha256": initial_sha, "final_sha256": final_sha,
                  "checkpoint": str(ckpt / f"state_dict_{args.expected_steps}.ckpt"),
                  "final_walk_rng_sha256": digest(task_state["generators"]["walk"]),
                  "input_hashes": input_hashes(log / "constraint_inputs.jsonl.gz", args.expected_steps),
                  "verified_checkpoints": verified_steps, "training_constraint": audit, **consumed}
        records.append(record)
        print(json.dumps({k: record[k] for k in ("model_id", "source", "seed", "condition", "verified_checkpoints")}), flush=True)
    check_pairs(records)
    prior = compare_prior_controls(records, args.prior_training, args.expected_steps) if not args.allow_smoke else None
    args.output.mkdir(parents=True)
    write_json(args.output / "arms.json", [{k: v for k, v in r.items() if not k.endswith("hashes")} for r in records])
    write_json(args.output / "paired_input_hashes.json", [{k: r[k] for k in ("model_id", "source", "seed", "condition", "input_hashes")} for r in records])
    if not args.allow_smoke:
        write_json(args.output / "control_reproduction.json", prior)
        (args.output / "model_list.tsv").write_text("model_id\tcheckpoint\tsources\n" + "".join(
            f"{r['model_id']}\t{r['checkpoint']}\t{r['source']}\n" for r in records))
    write_json(args.output / "DONE.json", {"valid": True, "research_result": not args.allow_smoke,
        "models": 18, "paired_comparisons": 9, "steps_per_model": args.expected_steps,
        "exact_paired_inputs": True, "same_initialization": True, "same_final_walk_rng": True,
        "exact_frozen_tensors_all_updates_and_checkpoints": True, "complete_effective_config_checks": True,
        "launch_revision": manifest["revision"]})


if __name__ == "__main__":
    main()
