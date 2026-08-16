#!/usr/bin/env python3
"""Evaluate one target shard on 500 paired 10-shot CLS episodes."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys

import torch
import wandb

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
sys.path[:0] = [str(REPO_ROOT), str(HERE)]

from experiments.params import get_params  # noqa: E402
from experiments.trainer import TrainerFS  # noqa: E402
from make_plan import TARGETS, evaluation_rows  # noqa: E402
from run_train import complete  # noqa: E402
from scripts.experiments.setup.icl_arch_matrix.common_protocol import (  # noqa: E402
    build_classification_dataset, classification_targets, reset_episode_rng,
)
from scripts.experiments.setup.icl_arch_matrix.evaluate_prodigy import AuditedLoader  # noqa: E402

EPISODES = 500
BATCH_SIZE = 4


def params_for(args, target_name, target, graph_path, checkpoint, prefix):
    return get_params([
        "--config", str(HERE / "train.yaml"),
        "--dataset", target_name, "--root", str(graph_path.parent),
        "--graph_filename", graph_path.name, "--task_name", "classification",
        "--feature_subset", "emb_only", "--original_features", "True",
        "--edge_view", "default", "--target_edge_view", "default",
        "--neighbor_matching_edge_split", "False",
        "--n_way", "2", "--n_shots", "10", "--n_query", str(target["n_query"]),
        "--batch_size", str(BATCH_SIZE), "--dataset_len_cap", str(EPISODES // BATCH_SIZE),
        "--val_len_cap", str(EPISODES // BATCH_SIZE),
        "--test_len_cap", str(EPISODES // BATCH_SIZE),
        "--workers", "0", "--eval_only", "True", "--eval_only_split", "test",
        "--eval_test_before_train", "False", "--eval_val_before_train", "False",
        "--ignore_label_embeddings", "False", "--linear_probe", "False",
        "--pretrained_model_run", str(checkpoint), "--device", str(args.device),
        "--prefix", f"labmix500_eval_{prefix}_to_{target_name}",
        "--timestamp", args.run_stamp, "--state_dir", str(args.eval_state_root),
        "--log_dir", str(args.log_root), "--override_log", "True",
        "--eval_random_query", str(target["eval_random_query"]),
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--state-root", type=Path, default=REPO_ROOT / "state")
    parser.add_argument("--eval-state-root", type=Path, default=REPO_ROOT / "state_eval_labmix500")
    parser.add_argument("--log-root", type=Path, default=REPO_ROOT / "log_eval_labmix500")
    parser.add_argument("--results", type=Path, default=HERE / "results_seed0.jsonl")
    parser.add_argument("--run-stamp", default="seed0")
    parser.add_argument("--data-root", default="/dataMeR1/phil/data")
    args = parser.parse_args()
    if args.device not in {0, 1}:
        parser.error("only Tucker GPUs 0 and 1 are owned")
    targets = classification_targets(REPO_ROOT / "docs/graph_catalog.json", include_facebook=True)
    target_names = [target for i, target in enumerate(TARGETS) if i % args.num_shards == args.shard_index]
    args.results.parent.mkdir(parents=True, exist_ok=True)
    shard_result = args.results.with_name(f"{args.results.stem}_shard{args.shard_index}.jsonl")
    completed = set()
    if shard_result.is_file():
        for line in shard_result.read_text().splitlines():
            row = json.loads(line)
            completed.add((row["target"], row["model_id"]))

    with shard_result.open("a", encoding="utf-8") as handle:
        for target_name in target_names:
            target = targets[target_name]
            dataset, _, graph_path = build_classification_dataset(
                dataset_name=target_name, data_root=args.data_root, target=target
            )
            plan = [row for row in evaluation_rows() if row["target"] == target_name]
            for index, row in enumerate(plan, 1):
                key = (target_name, str(row["prefix"]))
                if key in completed:
                    print(f"SKIP {key}", flush=True)
                    continue
                checkpoint = complete(args.state_root, str(row["prefix"]))
                if checkpoint is None:
                    raise FileNotFoundError(f"missing checkpoint for {row['prefix']}")
                print(f"[{target_name} {index}/{len(plan)}] {row['prefix']}", flush=True)
                params = params_for(args, target_name, target, graph_path, checkpoint, str(row["prefix"]))
                torch.manual_seed(0)
                torch.cuda.manual_seed_all(0)
                trainer = TrainerFS(dataset, params)
                audited = AuditedLoader(
                    trainer.test_dataloader, n_query=int(target["n_query"]),
                    equal_query_counts=not target["eval_random_query"],
                )
                try:
                    trainer.model.eval()
                    reset_episode_rng()
                    with torch.no_grad():
                        trainer.do_eval(audited, split_name="test", step=500)
                    if audited.episodes != EPISODES:
                        raise RuntimeError(f"expected {EPISODES} episodes, got {audited.episodes}")
                    metrics_path = Path(trainer.logging_dir) / "metrics_test_step500.json"
                    metrics = json.loads(metrics_path.read_text())
                    metrics = {key.removeprefix("test_"): value for key, value in metrics.items()}
                    payload = {
                        "target": target_name, "model_id": row["prefix"],
                        "mixture_size": row["mixture_size"], "donors": list(row["donors"]),
                        "training_steps": 500, "training_seed": 0,
                        "eval_episodes": audited.episodes,
                        "episode_fingerprint": audited.fingerprint,
                        "checkpoint": str(checkpoint), **metrics,
                    }
                    handle.write(json.dumps(payload, sort_keys=True) + "\n")
                    handle.flush()
                finally:
                    wandb.finish()
                    del trainer, audited
                    gc.collect()
                    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
