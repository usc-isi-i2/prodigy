"""Fail-closed validity audit of consumed matched-policy training runs."""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path

import torch

from experiments.episode_audit import digest
from scripts.experiments.setup.target_performance_mechanisms.make_member_configs import POLICIES


def model_digest(state):
    result = hashlib.sha256()
    for key, value in sorted(state.items()):
        if not torch.isfinite(value).all():
            raise ValueError(f"nonfinite model tensor: {key}")
        result.update(json.dumps([key, str(value.dtype), list(value.shape)]).encode())
        result.update(value.cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def audit_consumed(path, params, steps):
    anchors, sets, orders, source_ids = [], [], [], Counter()
    members_seen, anchors_seen = Counter(), set()
    repeat_total, context_sum, member_total, episode_count = 0., 0, 0, 0
    n_way = params["n_way"]
    n_members = params["n_shots"] + params["n_query"]
    roles = [0] * params["n_shots"] + [1] * params["n_query"]
    with gzip.open(path, "rt") as handle:
        for expected_step, line in enumerate(handle, 1):
            row = json.loads(line)
            if row["step"] != expected_step or row["query_roles"] != roles:
                raise ValueError("consumed step sequence or role pattern is invalid")
            ids = torch.tensor(row["member_ids"], dtype=torch.long)
            anchor = torch.tensor(row["anchor_ids"], dtype=torch.long)
            if ids.shape != (params["batch_size"], n_way, n_members) or anchor.shape != (params["batch_size"], n_way):
                raise ValueError("unexpected consumed episode dimensions")
            if (row["anchor_sha256"], row["member_set_sha256"], row["member_order_sha256"]) != (digest(anchor), digest(ids.sort(-1).values), digest(ids)):
                raise ValueError("consumed episode hash mismatch")
            if torch.any(ids.sort(-1).values.diff(dim=-1) == 0):
                raise ValueError("member repeated within one pseudo-class")
            if any(len(set(part)) != n_way for part in row["anchor_ids"]):
                raise ValueError("anchors repeated within one episode")
            anchors.append(row["anchor_sha256"])
            sets.append(row["member_set_sha256"])
            orders.append(row["member_order_sha256"])
            source_ids.update(row["source_ids"])
            members_seen.update(ids.reshape(-1).tolist())
            anchors_seen.update(anchor.reshape(-1).tolist())
            repeat_total += sum(1 - len(part.unique()) / part.numel() for part in ids)
            context = torch.tensor(row["context_node_counts"])
            if context.shape != ids.shape or torch.any(context < 1):
                raise ValueError("invalid consumed context occupancy")
            context_sum += int(context.sum())
            member_total += ids.numel()
            episode_count += len(ids)
    if len(anchors) != steps or episode_count != steps * params["batch_size"] or len(source_ids) != 1:
        raise ValueError("incomplete consumed audit or mixed training source")
    count = torch.tensor(list(members_seen.values()), dtype=torch.float64)
    return {"anchor_hashes": anchors, "member_set_hashes": sets, "member_order_hashes": orders,
            "summary": {"steps": steps, "episodes": episode_count, "member_positions": member_total,
                        "unique_members": len(members_seen), "unique_anchors": len(anchors_seen),
                        "effective_member_count": float(count.sum().square() / count.square().sum()),
                        "cross_class_duplicate_fraction": repeat_total / episode_count,
                        "context_nodes_mean": context_sum / member_total, "source_ids": dict(source_ids)}}


def check_groups(records, require_initial=True):
    grouped = defaultdict(dict)
    for row in records:
        key = (row["source"], row["seed"])
        if row["policy"] in grouped[key]:
            raise ValueError("duplicate source/seed/policy")
        grouped[key][row["policy"]] = row
    for group in grouped.values():
        if set(group) != set(POLICIES):
            raise ValueError("incomplete factorial group")
        first = group[POLICIES[0]]
        for row in group.values():
            if require_initial and row["initial_sha256"] != first["initial_sha256"]:
                raise ValueError("initial weights differ within matched group")
            for key in ("anchor_hashes", "final_walk_rng_sha256"):
                if row[key] != first[key]:
                    raise ValueError(f"matched walk/anchor stream diverged: {key}")
        for retention in ("lowest", "uniform"):
            if group[retention + "_sorted"]["member_set_hashes"] != group[retention + "_shuffled"]["member_set_hashes"]:
                raise ValueError("role-only treatment changed retained sets")
    return len(grouped)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--expected-steps", type=int, default=2500)
    p.add_argument("--allow-smoke", action="store_true")
    args = p.parse_args()
    manifest = json.loads((args.run_dir / "manifest.json").read_text())
    status = json.loads((args.run_dir / "status.json").read_text())
    if status["status"] != "complete" or len(manifest["jobs"]) != 24:
        raise ValueError("requires completed 24-arm launch")
    smoke = manifest["mode"] == "smoke"
    if smoke != args.allow_smoke or (not smoke and args.expected_steps != 2500):
        raise ValueError("smoke/production contract mismatch")
    if args.output.exists():
        raise ValueError("refusing to overwrite an existing validity receipt")
    records, model_list = [], []
    for index, planned in enumerate(manifest["jobs"]):
        job = args.run_dir / f"job_{index:03d}"
        params = json.loads((job / "effective_config.json").read_text())
        result = json.loads((job / "result.json").read_text())
        if result["status"] != "complete" or params["epochs"] * params["dataset_len_cap"] != args.expected_steps:
            raise ValueError("training did not complete the exact budget")
        source, seed, policy = params["neighbor_sampling_source_subset"], params["seed"], params["neighbor_matching_member_policy"]
        if source not in {"ukr_rus", "cp_hk"} or seed not in {0, 1, 2} or policy not in POLICIES:
            raise ValueError("unexpected arm")
        if params["neighbor_matching_member_seed"] != 280100 + seed or not params["train_episode_audit"]:
            raise ValueError("wrong matched sampler seed or no audit")
        for key in ("prefix", "seed", "neighbor_matching_member_policy", "neighbor_sampling_source_subset"):
            if params[key] != planned[key]:
                raise ValueError("effective config differs from launch plan")
        ckpt = Path(result["checkpoint_dir"])
        terminal_path = ckpt / f"state_dict_{args.expected_steps}.ckpt"
        terminal = torch.load(terminal_path, map_location="cpu", weights_only=True)["model"]
        final_sha = model_digest(terminal)
        initial_path = ckpt / "state_dict_0.ckpt"
        initial_sha = model_digest(torch.load(initial_path, map_location="cpu", weights_only=True)["model"]) if initial_path.exists() else None
        if not smoke and (initial_sha is None or initial_sha == final_sha):
            raise ValueError("missing initialization or no updated weights")
        full = torch.load(ckpt / f"training_state_{args.expected_steps}.ckpt", map_location="cpu", weights_only=False)
        training = full["_training_checkpoint"]
        if training["completed_steps"] != args.expected_steps or model_digest(full["model"]) != final_sha:
            raise ValueError("full-state sidecar disagrees with terminal checkpoint")
        task_state = training["train_batch_sampler"]["task_sampling_state"]
        if task_state["member_policy"] != policy or task_state["member_sampling_seed"] != 280100 + seed:
            raise ValueError("saved sampler contract differs")
        consumed = audit_consumed(Path(params["log_dir"]) / params["exp_name"] / "data/consumed_episodes.jsonl.gz", params, args.expected_steps)
        row = {"model_id": params["prefix"], "source": source, "seed": seed, "policy": policy,
               "initial_sha256": initial_sha, "final_sha256": final_sha,
               "final_walk_rng_sha256": digest(task_state["generators"]["walk"]),
               "checkpoint": str(terminal_path), **consumed}
        records.append(row)
        model_list.append({"model_id": params["prefix"], "checkpoint": str(terminal_path), "sources": source})
        print(json.dumps({k: v for k, v in row.items() if not k.endswith("hashes")}), flush=True)
    verify_initial = all(r["initial_sha256"] is not None for r in records)
    if check_groups(records, require_initial=verify_initial) != 6:
        raise ValueError("expected six complete source/seed groups")
    args.output.mkdir(parents=True)
    compact = [{k: v for k, v in r.items() if not k.endswith("hashes")} for r in records]
    (args.output / "arms.json").write_text(json.dumps(compact, indent=2))
    if not smoke:
        with (args.output / "model_list.tsv").open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=["model_id", "checkpoint", "sources"], delimiter="\t")
            writer.writeheader()
            writer.writerows(model_list)
    (args.output / "DONE.json").write_text(json.dumps({"valid": True, "models": 24, "steps_per_model": args.expected_steps,
        "research_result": not smoke, "launch_revision": manifest["revision"], "same_consumed_anchors": True,
        "same_final_walk_rng": True, "same_retention_sets_across_role_treatments": True,
        "same_initialization_verified": verify_initial}, indent=2))


if __name__ == "__main__":
    main()
