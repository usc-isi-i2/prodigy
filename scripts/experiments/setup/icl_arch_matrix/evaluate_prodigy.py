#!/usr/bin/env python3
"""Evaluate all seed-0 PRODIGY matrix models on the shared CLS episodes."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import wandb

from experiments.params import get_params
from scripts.experiments.setup.final_core.core_plan import ORDERS, build_models
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


def model_for_ladder(order: str, rung: int):
    wanted = frozenset(ORDERS[order][:rung])
    matches = [model for model in build_models() if frozenset(model.sources) == wanted]
    if len(matches) != 1:
        raise AssertionError(f"expected one physical model for {order}/rung {rung}")
    return matches[0]


def ladder_model_ids() -> set[str]:
    return {
        model_for_ladder(order, rung).model_id
        for order in ORDERS
        for rung in range(1, 10)
    }


def checkpoint_path(args, training_seed: int, model_id: str) -> Path:
    if args.checkpoint_layout == "architecture-matrix":
        return (
            Path(args.state_root)
            / "prodigy"
            / f"archmatrix_prodigy_{model_id}_s0_{args.run_stamp}"
            / "checkpoint"
            / f"state_dict_{args.checkpoint_step}.ckpt"
        )
    return (
        Path(args.state_root)
        / f"finalcore_{model_id}_s{training_seed}_{args.run_stamp}"
        / "checkpoint"
        / f"state_dict_{args.checkpoint_step}.ckpt"
    )


def load_reference_fingerprints(path: str | None) -> dict[str, str]:
    if not path:
        return {}
    fingerprints: dict[str, set[str]] = {}
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        try:
            fingerprints.setdefault(row["dataset"], set()).add(row["episode_fingerprint"])
        except KeyError as error:
            raise ValueError(f"invalid reference row {line_number} in {path}: missing {error}")
    drift = {dataset: values for dataset, values in fingerprints.items() if len(values) != 1}
    if drift:
        raise ValueError(f"reference episode fingerprint drift: {drift}")
    return {dataset: next(iter(values)) for dataset, values in fingerprints.items()}


def load_existing_results(
    path: Path,
    *,
    expected_keys: set[tuple[int, str, str]],
    checkpoint_step: int,
    baseline: str,
) -> tuple[set[tuple[int, str, str]], dict[str, str]]:
    completed: set[tuple[int, str, str]] = set()
    fingerprints: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        key = int(row["training_seed"]), row["model_id"], row["dataset"]
        if key not in expected_keys:
            raise ValueError(f"unexpected existing result {key} in {path}:{line_number}")
        if key in completed:
            raise ValueError(f"duplicate existing result {key} in {path}:{line_number}")
        if int(row.get("checkpoint_step", -1)) != checkpoint_step:
            raise ValueError(f"wrong checkpoint step in {path}:{line_number}")
        if row.get("baseline") != baseline or int(row.get("episodes", -1)) != EVAL_EPISODES:
            raise ValueError(f"wrong protocol in {path}:{line_number}")
        dataset = row["dataset"]
        fingerprint = row["episode_fingerprint"]
        prior = fingerprints.setdefault(dataset, fingerprint)
        if prior != fingerprint:
            raise ValueError(f"episode fingerprint drift in {path}:{line_number}")
        completed.add(key)
    return completed, fingerprints


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
    parser.add_argument("--checkpoint-step", default=TRAIN_STEPS, type=int)
    parser.add_argument(
        "--checkpoint-layout",
        choices=("architecture-matrix", "final-core", "saturation"),
        default="architecture-matrix",
    )
    parser.add_argument("--training-seeds", default="0")
    parser.add_argument("--ladder-only", action="store_true")
    parser.add_argument("--worker-index", default=0, type=int)
    parser.add_argument("--worker-count", default=1, type=int)
    parser.add_argument("--reference-results")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--include-facebook",
        action="store_true",
        help="Add facebook_page_reference to the original four-target panel.",
    )
    parser.add_argument("--eval-episode-seed-offset", type=int, default=0)
    parser.add_argument(
        "--random-init",
        action="store_true",
        help="Evaluate one deterministically initialized, untrained PRODIGY model.",
    )
    return parser.parse_args()


def checkpoint_path(args, training_seed: int, model_id: str) -> Path:
    if args.checkpoint_layout == "architecture-matrix":
        return (
            Path(args.state_root)
            / "prodigy"
            / f"archmatrix_prodigy_{model_id}_s0_{args.run_stamp}"
            / "checkpoint"
            / f"state_dict_{args.checkpoint_step}.ckpt"
        )
    if args.checkpoint_layout == "final-core":
        return (
            Path(args.state_root)
            / f"finalcore_{model_id}_s{training_seed}_{args.run_stamp}"
            / "checkpoint"
            / f"state_dict_{args.checkpoint_step}.ckpt"
        )
    return (
        Path(args.state_root)
        / "prodigy"
        / f"archsat_prodigy_{model_id}_s{training_seed}_{args.run_stamp}"
        / "checkpoint"
        / f"state_dict_{args.checkpoint_step}.ckpt"
    )


def evaluation_prefix(
    model_id: str,
    dataset_name: str,
    training_seed: int,
    checkpoint_step: int,
) -> str:
    return (
        f"archmatrix_prodigy_eval_{model_id}_s{training_seed}_"
        f"step{checkpoint_step}_{dataset_name}"
    )


def resolved_params(
    args,
    dataset_name,
    target,
    graph_path,
    checkpoint,
    model_id,
    training_seed,
    checkpoint_step,
):
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
        "--seed", str(training_seed),
        "--eval_episode_seed_offset", str(args.eval_episode_seed_offset),
        "--eval_only", "True",
        "--eval_only_split", "test",
        "--eval_test_before_train", "False",
        "--eval_val_before_train", "False",
        "--ignore_label_embeddings", "False",
        "--linear_probe", "False",
        "--device", str(args.device),
        "--prefix", evaluation_prefix(
            model_id, dataset_name, training_seed, checkpoint_step
        ),
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
    from experiments.trainer import TrainerFS

    args = parse_args()
    cpu_threads = int(os.environ.get("FINAL_CORE_CPU_THREADS", "16"))
    if cpu_threads <= 0:
        raise ValueError("FINAL_CORE_CPU_THREADS must be positive")
    torch.set_num_threads(cpu_threads)
    torch.set_num_interop_threads(1)
    selected = set(filter(None, args.model_ids.split(",")))
    if args.random_init:
        if args.checkpoint_step != TRAIN_STEPS:
            raise ValueError("--checkpoint-step cannot be combined with --random-init")
        if selected:
            raise ValueError("--model-ids cannot be combined with --random-init")
        models = [SimpleNamespace(model_id="random_init", sources=())]
    else:
        models = [model for model in build_models() if not selected or model.model_id in selected]
        if selected and selected != {model.model_id for model in models}:
            raise ValueError(f"unknown model ids: {sorted(selected - {m.model_id for m in models})}")
        if args.ladder_only:
            models = [model for model in models if model.model_id in ladder_model_ids()]
            if not selected and len(models) != 25:
                raise AssertionError(f"expected 25 physical ladder models, got {len(models)}")
    training_seeds = tuple(int(part) for part in args.training_seeds.split(",") if part)
    if not training_seeds or len(training_seeds) != len(set(training_seeds)):
        raise ValueError(f"invalid --training-seeds {args.training_seeds!r}")
    if args.checkpoint_layout == "architecture-matrix" and training_seeds != (0,):
        raise ValueError("architecture-matrix checkpoints exist only for seed 0")
    if not 0 <= args.worker_index < args.worker_count:
        raise ValueError("worker-index must be in [0, worker-count)")
    jobs = [
        (training_seed, model)
        for training_seed in training_seeds
        for model in models
    ]
    jobs = [job for index, job in enumerate(jobs) if index % args.worker_count == args.worker_index]
    if not jobs:
        raise ValueError(f"worker {args.worker_index} has no assigned jobs")
    if not args.random_init:
        for training_seed, model in jobs:
            checkpoint = checkpoint_path(args, training_seed, model.model_id)
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
    result_path = Path(args.results)
    if result_path.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite results: {result_path}")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    Path(args.log_root).mkdir(parents=True, exist_ok=True)
    targets = classification_targets(args.catalog, include_facebook=args.include_facebook)
    selected_datasets = set(filter(None, args.datasets.split(",")))
    if selected_datasets:
        missing = selected_datasets - targets.keys()
        if missing:
            raise ValueError(f"unknown classification datasets: {sorted(missing)}")
        targets = {name: target for name, target in targets.items() if name in selected_datasets}
    reference_fingerprints = load_reference_fingerprints(args.reference_results)
    missing_references = set(targets) - set(reference_fingerprints) if reference_fingerprints else set()
    if missing_references:
        raise ValueError(f"reference results omit datasets: {sorted(missing_references)}")
    expected_keys = {
        (training_seed, model.model_id, dataset_name)
        for training_seed, model in jobs
        for dataset_name in targets
    }
    completed: set[tuple[int, str, str]] = set()
    expected_fingerprints: dict[str, str] = {}
    if result_path.exists():
        completed, expected_fingerprints = load_existing_results(
            result_path,
            expected_keys=expected_keys,
            checkpoint_step=0 if args.random_init else args.checkpoint_step,
            baseline="random_init" if args.random_init else "pretrained",
        )
        for dataset_name, observed in expected_fingerprints.items():
            if reference_fingerprints and observed != reference_fingerprints[dataset_name]:
                raise ValueError(
                    f"existing result differs from published episode fingerprint on {dataset_name}"
                )

    with result_path.open("a" if result_path.exists() else "w", encoding="utf-8") as handle:
        for dataset_name, target in targets.items():
            dataset, _, graph_path = build_classification_dataset(
                dataset_name=dataset_name, data_root=args.data_root, target=target
            )
            for training_seed, plan_model in jobs:
                result_key = training_seed, plan_model.model_id, dataset_name
                if result_key in completed:
                    print(f"SKIP seed={training_seed} model={plan_model.model_id} dataset={dataset_name}")
                    continue
                checkpoint = None
                checkpoint_step = 0 if args.random_init else args.checkpoint_step
                if not args.random_init:
                    checkpoint = checkpoint_path(args, training_seed, plan_model.model_id)
                    if not checkpoint.is_file():
                        raise FileNotFoundError(checkpoint)
                params = resolved_params(
                    args,
                    dataset_name,
                    target,
                    graph_path,
                    checkpoint,
                    plan_model.model_id,
                    training_seed,
                    checkpoint_step,
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
                    if (
                        reference_fingerprints
                        and audited.fingerprint != reference_fingerprints[dataset_name]
                    ):
                        raise RuntimeError(f"published episode fingerprint mismatch on {dataset_name}")
                    row = {
                        "architecture": "prodigy",
                        "model_id": plan_model.model_id,
                        "sources": list(plan_model.sources),
                        "seed": 0,
                        "training_seed": training_seed,
                        "evaluation_seed": 0,
                        "eval_episode_seed_offset": args.eval_episode_seed_offset,
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
