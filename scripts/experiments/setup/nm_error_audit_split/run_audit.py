#!/usr/bin/env python3
"""Paired NM exports on the checkpoint's canonical 70/15/15 graph split.

Reuses final-core planning, replay, strict checkpoint loading, and evaluation.
All raw outputs are private. No training or shared graph writes are performed.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts/experiments/setup/final_core"))
sys.path.insert(0, str(ROOT))
from evaluate_fixed_grid import (
    AuditedLoader, ReplayLoader, atomic_json, fingerprint_plan, git_commit,
    iter_batch_tensors, load_checkpoint_strict, make_frozen_loader,
    reset_fixed_eval_rng, resolved_params, seed_everything, load_dataset, TrainerFS,
)

TARGETS = {"ukr_rus": "ukr_rus_twitter", "cp_hk": "cp_hk_twitter"}


def membership(sampler, pairs):
    """Exact membership in the actual undirected sampling adjacency."""
    rowptr, col, _ = sampler.whole_adj.csr()
    rowptr, col = rowptr.numpy(), col.numpy()
    return np.array([b in col[rowptr[a]:rowptr[a+1]] for a, b in pairs], dtype=bool)


def check_plan(dataset, batches, split):
    pairs = sorted({(int(a), int(b)) for episodes, _ in batches
                    for episode in episodes for a, members in episode.items() for b in members})
    samplers = {"train": dataset.neighbor_sampler,
                "val": dataset.nm_validation_neighbor_sampler,
                "test": dataset.nm_test_neighbor_sampler}
    counts = {name: int(membership(sampler, pairs).sum()) for name, sampler in samplers.items()}
    assert counts[split] == len(pairs), (split, counts, len(pairs))
    assert all(counts[name] == 0 for name in counts if name != split), (split, counts)
    return {"unique_anchor_member_pairs": len(pairs), "membership_counts": counts,
            "includes_support_and_query_pairs": True, "direction": "undirected"}


def full_hash(batches):
    digest = hashlib.sha256()
    for batch in batches:
        for tensor in iter_batch_tensors(batch):
            array = tensor.detach().cpu().contiguous().numpy()
            digest.update(str((array.dtype, array.shape)).encode())
            digest.update(array.tobytes())
    return digest.hexdigest()


def remap_export(src, dest, target, offset, count):
    """Retain core ids alongside reversible source-local ids for existing bio joins."""
    def local(node):
        assert offset <= node < offset + count, (target, node, offset, count)
        return int(node - offset)
    n = 0
    with src.open() as fin, dest.open("x") as fout:
        for line in fin:
            row = json.loads(line)
            row["core_query_node_id"] = row["query_node_id"]
            row["core_gt"] = row["gt"]
            row["core_prediction"] = row["prediction"]
            row["source_node_offset"] = offset
            row["core_dataset"] = row["dataset"]
            row["dataset"] = TARGETS[target]
            for key in ["query_node_id", "gt", "prediction"]:
                row[key] = local(row[key])
            for key in ["episode_label_map", "context_node_ids"]:
                row[key] = [local(v) for v in row[key]]
            for support in row["supports"]:
                support["node_id"] = local(support["node_id"])
                support["label"] = local(support["label"])
                support["context_node_ids"] = [local(v) for v in support["context_node_ids"]]
            fout.write(json.dumps(row) + "\n")
            n += 1
    return n


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--training-state-root", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--episodes", type=int, default=512)
    p.add_argument("--threads", type=int, default=12)
    p.add_argument("--targets", nargs="+", choices=list(TARGETS), default=list(TARGETS))
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    assert args.episodes > 0 and args.episodes % args.batch_size == 0
    args.config = ROOT / "scripts/experiments/setup/final_core/training.yaml"
    args.plan_config = None
    args.evaluation_run_stamp = args.out_dir.name
    args.evaluation_state_root = args.out_dir / "state"
    args.evaluation_log_root = args.out_dir / "logs"
    args.worker_index = 0
    args.batch_count = args.episodes // args.batch_size
    ckpts = {name: args.training_state_root / f"finalcore_ss_{name}_s0_20260807/checkpoint/state_dict_2500.ckpt"
             for name in TARGETS}
    for checkpoint in ckpts.values():
        assert checkpoint.is_file(), checkpoint
    params = resolved_params(args, seed=0, model_id="ss_ukr_rus", target=args.targets[0],
                             checkpoint=ckpts["ukr_rus"])
    params.update(eval_only_split="both", val_len_cap=args.batch_count,
                  export_predictions=True, prediction_context_neighbors=3,
                  prediction_support_per_label=3)
    assert params["neighbor_matching_edge_split"] is True
    assert params["edge_view"] == "static_train" and params["target_edge_view"] == "static_test"
    assert params["neighbor_matching_walk_hops"] == 1
    if args.dry_run:
        print(json.dumps({"params": params, "checkpoints": {k: str(v) for k,v in ckpts.items()}}, default=str, indent=2))
        return
    args.out_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    atomic_json(args.out_dir / "effective_config.json", json.loads(json.dumps(params, default=str)))
    seed_everything(params)
    dataset = load_dataset(params)
    assert dataset.nm_background_edge_view == "static_train"
    assert dataset.nm_holdout_edge_view == "static_test"
    trainer = TrainerFS(dataset, params)
    summary = {"protocol": "canonical_split_paired_nm_v1", "commit": git_commit(),
               "graph": str(Path(params["root"]) / params["graph_filename"]),
               "episodes_per_target_split": args.episodes, "batch_size": args.batch_size,
               "n_way": 30, "n_shots": 3, "n_query": 4, "targets": {}}
    paths = {}
    for target in args.targets:
        source_id = list(dataset.graph.source_graph_names).index(target)
        nodes = torch.where(dataset.graph.graph_id == source_id)[0]
        offset, count = int(nodes[0]), len(nodes)
        assert int(nodes[-1]) == offset + count - 1
        del nodes
        target_dir = args.out_dir / target
        target_dir.mkdir()
        probe_ids = np.random.default_rng(7).choice(count, min(count, 256), replace=False)
        np.savez(target_dir / "source_feature_probe.npz", ids=probe_ids,
                 x=dataset.graph.x[torch.tensor(probe_ids + offset)].cpu().numpy(), source_node_count=count)
        summary["targets"][target] = {"source_node_offset": offset, "source_node_count": count, "splits": {}}
        paths[TARGETS[target]] = {"paths": {m: str(target_dir / name / "predictions_test_step2500.jsonl")
                                          for m, name in [("ukr", "ukr_rus"), ("hk", "cp_hk")]}}
        for split in ["test", "val"]:
            reset_fixed_eval_rng(target)
            trainer.parameter["neighbor_sampling_source_subset"] = target
            loaders = trainer._build_dataloaders(dataset, trainer.dataset_name)
            loader = loaders[3 if split == "test" else 2]
            batches = list(loader.batch_sampler)
            plan_hash, episodes = fingerprint_plan(target, batches, expected_batch_size=args.batch_size, dataset=dataset)
            assert episodes == args.episodes
            torch.save(batches, target_dir / f"{split}_plan_private.pt")
            check = check_plan(dataset, batches, split)
            print(f"SPLIT_CHECK {target} {split} {check}", flush=True)
            reset_fixed_eval_rng(target)
            audited = AuditedLoader(make_frozen_loader(dataset, loader.collate_fn, batches), args.batch_size)
            cache = list(audited)
            assert audited.episode_count == args.episodes
            identity_hash = audited.fingerprint
            tensor_hash = full_hash(cache)
            cell = {"edge_audit": check, "plan_sha256": plan_hash, "observed_identity_sha256": identity_hash,
                    "all_input_tensors_sha256": tensor_hash, "models": {}}
            summary["targets"][target]["splits"][split] = cell
            # Published test fingerprints provide an independent episode check.
            if split == "test" and args.episodes == 512 and args.batch_size == 32:
                ref = ROOT / "scripts/experiments/analysis/transfer/matrices/cross_model/final_core/data/prodigy_final_core/auc/results/seed_0/ss_cp_hk" / f"{target}.json"
                reference = json.loads(ref.read_text())
                cell["published_plan_match"] = plan_hash == reference["episode_plan_fingerprint"]
                cell["published_identity_match"] = identity_hash == reference["observed_episode_fingerprint"]
                assert cell["published_plan_match"] and cell["published_identity_match"], cell
            for model, checkpoint in ckpts.items():
                dest = target_dir / model
                dest.mkdir(exist_ok=True)
                core = dest / "core_ids"
                core.mkdir(exist_ok=True)
                trainer.logging_dir = str(core)
                trainer.parameter.update(pretrained_model_run=str(checkpoint), exp_name=f"corrected_{model}_to_{target}")
                load_checkpoint_strict(trainer, checkpoint)
                trainer.model.eval()
                replay = AuditedLoader(ReplayLoader(cache), args.batch_size)
                started = time.monotonic()
                with torch.no_grad():
                    metrics = trainer.do_eval(replay, split_name=split, step=2500)
                assert replay.fingerprint == identity_hash and replay.episode_count == args.episodes
                assert full_hash(cache) == tensor_hash, "evaluation mutated frozen inputs"
                filename = f"predictions_{split}_step2500.jsonl"
                n = remap_export(core / filename, dest / filename, target, offset, count)
                assert n == args.episodes * 120, n
                cell["models"][model] = {"checkpoint": str(checkpoint), "accuracy": float(metrics[1]),
                                         "query_occurrences": n, "seconds": time.monotonic() - started,
                                         "path": str(dest / filename), "identical_input_replay": True}
                print(f"DONE {target} {split} {model}: {cell['models'][model]}", flush=True)
                atomic_json(args.out_dir / "protocol_summary.json", summary)
            del cache, batches, loaders, loader
            gc.collect()
    atomic_json(args.out_dir / "nm_ukr_vs_cp_hk_test_summary.json", paths)
    print("AUDIT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
