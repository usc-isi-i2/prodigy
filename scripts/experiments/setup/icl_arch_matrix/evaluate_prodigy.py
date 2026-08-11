#!/usr/bin/env python3
"""Evaluate all seed-0 PRODIGY matrix models on the shared CLS episodes."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import wandb

from experiments.params import get_params
from experiments.trainer import TrainerFS
from scripts.experiments.setup.final_core.core_plan import build_models
from scripts.experiments.setup.icl_arch_matrix.common_protocol import (
    EVAL_BATCH_SIZE,
    EVAL_EPISODES,
    EVAL_N_SHOT,
    EVAL_N_WAY,
    TRAIN_STEPS,
    build_classification_dataset,
    classification_targets,
    iter_episodes,
    new_fingerprint,
    reset_episode_rng,
    update_episode_fingerprint,
)


class AuditedLoader:
    """Fingerprint the exact episode stream consumed by PRODIGY."""

    def __init__(self, loader, *, n_query: int, equal_query_counts: bool):
        self.loader = loader
        self.n_query = n_query
        self.equal_query_counts = equal_query_counts
        self.hasher = new_fingerprint()
        self.episodes = 0

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        for batch in self.loader:
            for episode in iter_episodes(
                batch,
                n_way=EVAL_N_WAY,
                n_shot=EVAL_N_SHOT,
                n_query=self.n_query,
                equal_query_counts=self.equal_query_counts,
            ):
                update_episode_fingerprint(self.hasher, episode)
                self.episodes += 1
            yield batch

    @property
    def fingerprint(self):
        return self.hasher.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).with_name("prodigy_training.yaml")))
    parser.add_argument("--state-root", required=True)
    parser.add_argument("--eval-state-root")
    parser.add_argument("--data-root", default="/dataMeR1/phil/data")
    parser.add_argument("--catalog", default="docs/graph_catalog.json")
    parser.add_argument("--log-root", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--run-stamp", default="20260810")
    parser.add_argument("--device", default="0")
    parser.add_argument("--model-ids", default="")
    parser.add_argument("--datasets", default="")
    parser.add_argument(
        "--random-init",
        action="store_true",
        help="Evaluate one deterministically initialized, untrained PRODIGY model.",
    )
    return parser.parse_args()


def resolved_params(args, dataset_name, target, graph_path, checkpoint, model_id):
    eval_state_root = args.eval_state_root or str(Path(args.state_root) / "eval")
    argv = [
        "--config", args.config,
        "--dataset", dataset_name,
        "--root", str(graph_path.parent),
        "--graph_filename", graph_path.name,
        "--task_name", "classification",
        "--feature_subset", "emb_only",
        "--original_features", "True",
        "--edge_view", "default",
        "--target_edge_view", "default",
        "--neighbor_matching_edge_split", "False",
        "--n_way", str(EVAL_N_WAY),
        "--n_shots", str(EVAL_N_SHOT),
        "--n_query", str(target["n_query"]),
        "--batch_size", str(EVAL_BATCH_SIZE),
        "--dataset_len_cap", str(EVAL_EPISODES // EVAL_BATCH_SIZE),
        "--val_len_cap", str(EVAL_EPISODES // EVAL_BATCH_SIZE),
        "--test_len_cap", str(EVAL_EPISODES // EVAL_BATCH_SIZE),
        "--workers", "0",
        "--seed", "0",
        "--eval_episode_seed_offset", "0",
        "--eval_only", "True",
        "--eval_only_split", "test",
        "--eval_test_before_train", "False",
        "--eval_val_before_train", "False",
        "--ignore_label_embeddings", "False",
        "--linear_probe", "False",
        "--device", str(args.device),
        "--prefix", f"archmatrix_prodigy_eval_{model_id}_{dataset_name}",
        "--timestamp", args.run_stamp,
        "--state_dir", eval_state_root,
        "--log_dir", args.log_root,
        "--override_log", "True",
    ]
    if checkpoint is not None:
        argv.extend(["--pretrained_model_run", str(checkpoint)])
    if target["eval_random_query"]:
        argv.extend(["--eval_random_query", "True"])
    return get_params(argv)


def main() -> int:
    args = parse_args()
    torch.set_num_threads(16)
    selected = set(filter(None, args.model_ids.split(",")))
    if args.random_init:
        if selected:
            raise ValueError("--model-ids cannot be combined with --random-init")
        models = [SimpleNamespace(model_id="random_init", sources=())]
    else:
        models = [model for model in build_models() if not selected or model.model_id in selected]
        if selected and selected != {model.model_id for model in models}:
            raise ValueError(f"unknown model ids: {sorted(selected - {m.model_id for m in models})}")
    result_path = Path(args.results)
    if result_path.exists():
        raise FileExistsError(f"refusing to overwrite results: {result_path}")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    Path(args.log_root).mkdir(parents=True, exist_ok=True)
    expected_fingerprints = {}
    targets = classification_targets(args.catalog)
    selected_datasets = set(filter(None, args.datasets.split(",")))
    if selected_datasets:
        missing = selected_datasets - targets.keys()
        if missing:
            raise ValueError(f"unknown classification datasets: {sorted(missing)}")
        targets = {name: target for name, target in targets.items() if name in selected_datasets}

    with result_path.open("w", encoding="utf-8") as handle:
        for dataset_name, target in targets.items():
            dataset, _, graph_path = build_classification_dataset(
                dataset_name=dataset_name, data_root=args.data_root, target=target
            )
            for plan_model in models:
                checkpoint = None
                checkpoint_step = 0 if args.random_init else TRAIN_STEPS
                if not args.random_init:
                    checkpoint = (
                        Path(args.state_root)
                        / "prodigy"
                        / f"archmatrix_prodigy_{plan_model.model_id}_s0_{args.run_stamp}"
                        / "checkpoint"
                        / f"state_dict_{TRAIN_STEPS}.ckpt"
                    )
                    if not checkpoint.is_file():
                        raise FileNotFoundError(checkpoint)
                params = resolved_params(
                    args, dataset_name, target, graph_path, checkpoint, plan_model.model_id
                )
                torch.manual_seed(0)
                torch.cuda.manual_seed_all(0)
                trainer = TrainerFS(dataset, params)
                audited = AuditedLoader(
                    trainer.test_dataloader,
                    n_query=int(target["n_query"]),
                    equal_query_counts=not target["eval_random_query"],
                )
                try:
                    trainer.model.eval()
                    reset_episode_rng()
                    with torch.no_grad():
                        trainer.do_eval(audited, split_name="test", step=checkpoint_step)
                    metrics_path = (
                        Path(trainer.logging_dir) / f"metrics_test_step{checkpoint_step}.json"
                    )
                    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                    metrics = {key.removeprefix("test_"): value for key, value in metrics.items()}
                    if audited.episodes != EVAL_EPISODES:
                        raise RuntimeError(
                            f"expected {EVAL_EPISODES} episodes, observed {audited.episodes}"
                        )
                    prior = expected_fingerprints.setdefault(dataset_name, audited.fingerprint)
                    if audited.fingerprint != prior:
                        raise RuntimeError(f"episode drift on {dataset_name}")
                    row = {
                        "architecture": "prodigy",
                        "model_id": plan_model.model_id,
                        "sources": list(plan_model.sources),
                        "seed": 0,
                        "checkpoint_step": checkpoint_step,
                        "baseline": "random_init" if args.random_init else "pretrained",
                        "task": "classification",
                        "dataset": dataset_name,
                        "n_way": EVAL_N_WAY,
                        "n_shot": EVAL_N_SHOT,
                        "n_query": int(target["n_query"]),
                        "episodes": audited.episodes,
                        "queries": audited.episodes * EVAL_N_WAY * int(target["n_query"]),
                        "episode_fingerprint": audited.fingerprint,
                        **metrics,
                    }
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                    handle.flush()
                    print(json.dumps(row, sort_keys=True), flush=True)
                finally:
                    wandb.finish()
                    del trainer, audited
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
