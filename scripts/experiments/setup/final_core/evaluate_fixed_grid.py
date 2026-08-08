#!/usr/bin/env python3
"""Evaluate assigned final-core checkpoints on nine fixed target streams.

One long-lived process loads the all-nine graph exactly once.  For each target
it samples and collates the 16 fixed CPU batches once, replays cloned batches
across every assigned checkpoint, and releases that target cache before moving
to the next target.  There is no validation or checkpoint selection path.
"""

from __future__ import annotations

import argparse
import gc
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import resource
import struct
import sys
import time
from typing import Any, Iterable, Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader
import wandb

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(HERE))

from auc_contract import METRIC_CONTRACT, load_metric_sidecar, metric_sidecar_path  # noqa: E402
from core_plan import SOURCES  # noqa: E402
from experiments.params import get_params  # noqa: E402
from experiments.run_single_experiment import load_dataset, seed_everything  # noqa: E402
from experiments.trainer import TrainerFS, _to_float  # noqa: E402
from fixed_test_plan import (  # noqa: E402
    CHECKPOINT_STEP,
    EPISODE_COUNT,
    PROTOCOL,
    checkpoint_path,
    physical_jobs,
)


FIXED_EVAL_SEED = 271828


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def git_commit() -> str:
    import subprocess

    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def mem_available_gib() -> float:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / (1024 ** 2)
    raise RuntimeError("/proc/meminfo has no MemAvailable entry")


def max_rss_gib() -> float:
    # Linux reports ru_maxrss in KiB.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)


def update_int(digest: "hashlib._Hash", value: int) -> None:
    digest.update(struct.pack("<q", int(value)))


class FrozenBatchSampler:
    def __init__(self, batches: list[Any]):
        self.batches = batches

    def __iter__(self):
        return iter(self.batches)

    def __len__(self) -> int:
        return len(self.batches)


def clone_batch(batch: Any) -> Any:
    """Clone a collated CPU batch so TrainerFS may move it without mutating cache."""
    if isinstance(batch, torch.Tensor):
        return batch.clone()
    if isinstance(batch, tuple):
        return tuple(clone_batch(value) for value in batch)
    if isinstance(batch, list):
        return [clone_batch(value) for value in batch]
    clone = getattr(batch, "clone", None)
    if callable(clone):
        return clone()
    raise TypeError(f"cached batch contains unsupported value {type(batch)!r}")


def iter_batch_tensors(value: Any) -> Iterator[torch.Tensor]:
    """Yield tensors recursively, including tensors stored on PyG data objects."""
    if isinstance(value, torch.Tensor):
        yield value
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            yield from iter_batch_tensors(item)
        return
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        for item in to_dict().values():
            yield from iter_batch_tensors(item)


def assert_cpu_batches(batches: Iterable[Any]) -> None:
    for batch_index, batch in enumerate(batches):
        for tensor in iter_batch_tensors(batch):
            if tensor.device.type != "cpu":
                raise AssertionError(
                    f"cached batch {batch_index} escaped CPU onto {tensor.device}"
                )


class ReplayLoader:
    """Replay immutable cached CPU batches through fresh clones on every pass."""

    def __init__(self, batches: list[Any]):
        self.batches = batches

    def __iter__(self):
        for batch in self.batches:
            yield clone_batch(batch)

    def __len__(self) -> int:
        return len(self.batches)


class AuditedLoader:
    """Count and hash the episode stream actually handed to the evaluator."""

    def __init__(self, loader: DataLoader, expected_batch_size: int):
        self.loader = loader
        self.expected_batch_size = expected_batch_size
        self.batch_count = 0
        self.episode_count = 0
        self._digest = hashlib.sha256()
        self._digest.update(b"final-core-observed-episode-stream-v1\0")

    @staticmethod
    def _tensor_bytes(value: torch.Tensor) -> bytes:
        return value.detach().cpu().contiguous().numpy().tobytes(order="C")

    def __iter__(self):
        for batch_index, batch in enumerate(self.loader):
            graph = batch[0]
            self._audit_indices(batch_index, batch)
            task_ids = graph.task_id_per_sample.detach().cpu().long()
            episodes = int(torch.unique(task_ids).numel())
            if episodes != self.expected_batch_size:
                raise AssertionError(
                    f"batch {batch_index} has {episodes} episodes; "
                    f"expected {self.expected_batch_size}"
                )
            self.batch_count += 1
            self.episode_count += episodes
            update_int(self._digest, batch_index)
            for name, tensor in (
                ("task_label_map", graph.task_label_map),
                ("center_node_idx", graph.center_node_idx),
                ("task_id_per_sample", graph.task_id_per_sample),
                ("query_mask", batch[5]),
            ):
                self._digest.update(name.encode("utf-8") + b"\0")
                update_int(self._digest, tensor.numel())
                self._digest.update(self._tensor_bytes(tensor))
            yield batch

    @staticmethod
    def _audit_indices(batch_index: int, batch: Any) -> None:
        graph = batch[0]
        num_nodes = int(graph.num_nodes)
        if int(graph.ptr[0]) != 0 or int(graph.ptr[-1]) != num_nodes:
            raise AssertionError(
                f"batch {batch_index}: invalid ptr bounds for {num_nodes} nodes"
            )
        if torch.any(graph.ptr[1:] < graph.ptr[:-1]):
            raise AssertionError(f"batch {batch_index}: graph.ptr is not monotonic")
        if int(graph.x.size(0)) != num_nodes:
            raise AssertionError(
                f"batch {batch_index}: x has {graph.x.size(0)} rows for {num_nodes} nodes"
            )
        for name in ("edge_index", "edge_index_supernode", "edge_index_from_supernode"):
            index = getattr(graph, name)
            if index.numel() == 0:
                continue
            low = int(index.min())
            high = int(index.max())
            if low < 0 or high >= num_nodes:
                raise AssertionError(
                    f"batch {batch_index}: {name} range [{low}, {high}] is outside "
                    f"[0, {num_nodes})"
                )
        graph_sizes = graph.ptr[1:] - graph.ptr[:-1]
        if graph.supernode.numel() != graph_sizes.numel():
            raise AssertionError(
                f"batch {batch_index}: {graph.supernode.numel()} supernodes for "
                f"{graph_sizes.numel()} sampled graphs"
            )
        if torch.any(graph.supernode < 0) or torch.any(graph.supernode >= graph_sizes):
            raise AssertionError(f"batch {batch_index}: invalid local supernode index")
        global_supernodes = graph.supernode + graph.ptr[:-1]
        if torch.any(global_supernodes >= num_nodes):
            raise AssertionError(f"batch {batch_index}: invalid global supernode index")

        metagraph_edge_index = batch[3]
        metagraph_nodes = int(graph.supernode.numel() + batch[1].size(0))
        if metagraph_edge_index.numel():
            low = int(metagraph_edge_index.min())
            high = int(metagraph_edge_index.max())
            if low < 0 or high >= metagraph_nodes:
                raise AssertionError(
                    f"batch {batch_index}: metagraph edge range [{low}, {high}] is "
                    f"outside [0, {metagraph_nodes})"
                )

    def __len__(self) -> int:
        return len(self.loader)

    @property
    def fingerprint(self) -> str:
        return self._digest.hexdigest()


def fingerprint_plan(
    target: str,
    batches: list[Any],
    *,
    expected_batch_size: int,
    dataset,
) -> tuple[str, int]:
    digest = hashlib.sha256()
    digest.update(b"final-core-raw-episode-plan-v1\0")
    digest.update(target.encode("utf-8") + b"\0")
    source_names = list(dataset.graph.source_graph_names)
    if source_names.count(target) != 1:
        raise ValueError(f"target {target!r} is not unique in source registry {source_names}")
    target_id = source_names.index(target)
    episode_count = 0
    referenced_nodes: list[int] = []
    for batch_index, (episodes, params) in enumerate(batches):
        if len(episodes) != expected_batch_size or params.batch_size != expected_batch_size:
            raise AssertionError(f"target {target} batch {batch_index} is not full-sized")
        if (params.n_way, params.n_shot, params.n_query, params.n_member) != (30, 3, 4, 7):
            raise AssertionError(f"unexpected episode parameters: {asdict(params)}")
        update_int(digest, batch_index)
        update_int(digest, len(episodes))
        for episode in episodes:
            if len(episode) != 30:
                raise AssertionError(f"target {target} episode has {len(episode)} labels")
            episode_count += 1
            update_int(digest, len(episode))
            for label_center, members in episode.items():
                if len(members) != 7:
                    raise AssertionError("each label must have 3 supports and 4 queries")
                update_int(digest, label_center)
                referenced_nodes.append(int(label_center))
                update_int(digest, len(members))
                for member in members:
                    update_int(digest, member)
                    referenced_nodes.append(int(member))
    graph_ids = dataset.graph.graph_id[
        torch.as_tensor(referenced_nodes, dtype=torch.long)
    ].detach().cpu()
    observed_ids = set(int(value) for value in torch.unique(graph_ids).tolist())
    if observed_ids != {target_id}:
        raise AssertionError(
            f"target {target} stream crossed graph_id boundary: {sorted(observed_ids)}"
        )
    return digest.hexdigest(), episode_count


def reset_fixed_eval_rng(target: str) -> None:
    seed = FIXED_EVAL_SEED + list(SOURCES).index(target)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def resolved_params(
    args: argparse.Namespace,
    *,
    seed: int,
    model_id: str,
    target: str,
    checkpoint: Path,
) -> dict[str, Any]:
    argv = [
        "--config", str(args.config),
        "--device", "0",
        "--seed", str(seed),
        "--prefix", f"finalcore_fixed_{model_id}_s{seed}",
        "--timestamp", args.evaluation_run_stamp,
        "--state_dir", str(args.evaluation_state_root),
        "--log_dir", str(args.evaluation_log_root),
        "--neighbor_sampling_source_subset", target,
        "--pretrained_model_run", str(checkpoint),
        "--eval_only", "True",
        "--eval_only_split", "test",
        "--eval_test_before_train", "False",
        "--eval_val_before_train", "False",
        "--batch_size", str(args.batch_size),
        "--test_len_cap", str(args.batch_count),
        "--workers", "0",
        "--override_log", "True",
    ]
    params = get_params(argv)
    params["exp_name"] = (
        f"fixedgrid_w{args.worker_index}_{model_id}_s{seed}_{args.evaluation_run_stamp}"
    )
    return params


def wait_at_load_barrier(args: argparse.Namespace) -> None:
    if args.ready_dir is None:
        return
    marker = args.ready_dir / f"worker_{args.worker_index}.json"
    atomic_json(
        marker,
        {
            "worker_index": args.worker_index,
            "pid": os.getpid(),
            "ready_utc": utc_now(),
            "max_rss_gib": max_rss_gib(),
            "mem_available_gib": mem_available_gib(),
        },
    )
    deadline = time.monotonic() + args.barrier_timeout_seconds
    while len(list(args.ready_dir.glob("worker_*.json"))) < args.expected_workers:
        if time.monotonic() >= deadline:
            raise TimeoutError("not every persistent worker reached the graph-load barrier")
        time.sleep(2)
    available = mem_available_gib()
    if available < args.min_host_reserve_gib:
        raise MemoryError(
            f"only {available:.1f} GiB remains after concurrent graph loads; "
            f"minimum reserve is {args.min_host_reserve_gib:.1f} GiB"
        )


def make_frozen_loader(dataset, collate_fn, batches: list[Any]) -> DataLoader:
    return DataLoader(
        dataset,
        batch_sampler=FrozenBatchSampler(batches),
        num_workers=0,
        collate_fn=collate_fn,
    )


def materialize_target_batches(
    dataset,
    target_plan: dict[str, Any],
    *,
    target: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Collate one target stream once and keep immutable CPU templates."""
    reset_fixed_eval_rng(target)
    loader = make_frozen_loader(
        dataset, target_plan["collate_fn"], target_plan["batches"]
    )
    audited = AuditedLoader(loader, args.batch_size)
    cached_batches: list[Any] = []
    min_available = mem_available_gib()
    started = time.monotonic()
    for batch in audited:
        cached_batches.append(batch)
        available = mem_available_gib()
        min_available = min(min_available, available)
        if available < args.min_host_reserve_gib:
            raise MemoryError(
                f"target {target} materialization left {available:.1f} GiB available; "
                f"required reserve is {args.min_host_reserve_gib:.1f} GiB"
            )
    elapsed = time.monotonic() - started
    if audited.batch_count != args.batch_count or audited.episode_count != EPISODE_COUNT:
        raise AssertionError(
            f"materialized {audited.batch_count} batches/{audited.episode_count} "
            f"episodes for {target}; expected {args.batch_count}/{EPISODE_COUNT}"
        )
    assert_cpu_batches(cached_batches)
    payload = {
        "batches": cached_batches,
        "observed_fingerprint": audited.fingerprint,
        "elapsed_seconds": elapsed,
        "min_mem_available_gib": min_available,
        "max_rss_gib": max_rss_gib(),
    }
    print(
        f"CACHE target={target} batches={len(cached_batches)} "
        f"episodes={audited.episode_count} seconds={elapsed:.1f} "
        f"min_mem_available_gib={min_available:.1f} "
        f"max_rss_gib={payload['max_rss_gib']:.1f} "
        f"fingerprint={audited.fingerprint}",
        flush=True,
    )
    return payload


def validate_existing(
    payload: dict[str, Any],
    *,
    job,
    target: str,
    checkpoint: Path,
    args: argparse.Namespace,
    plan_fingerprint: str,
) -> None:
    expected = {
        "protocol": PROTOCOL,
        "model_id": job.model.model_id,
        "seed": job.seed,
        "target": target,
        "checkpoint_step": CHECKPOINT_STEP,
        "checkpoint": str(checkpoint),
        "batch_size": args.batch_size,
        "batch_count": args.batch_count,
        "episode_count": EPISODE_COUNT,
        "episode_plan_fingerprint": plan_fingerprint,
        "edge_view": "static_train",
        "target_edge_view": "static_test",
        "metric_contract": METRIC_CONTRACT,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(
                f"existing result mismatch for {job.key}/{target}: "
                f"{key} expected {value!r}, got {payload.get(key)!r}"
            )
    for key in ("accuracy", "f1_macro", "roc_auc_ovr_macro"):
        if not math.isfinite(float(payload.get(key, float("nan")))):
            raise ValueError(
                f"existing result mismatch for {job.key}/{target}: "
                f"missing or non-finite {key}"
            )


def load_checkpoint_strict(trainer: TrainerFS, checkpoint: Path) -> None:
    """Reload all evaluator modules without allowing prior-model state to leak."""
    state = torch.load(checkpoint, map_location=trainer.device)
    missing_modules = sorted(set(trainer.all_saveable_modules) - set(state))
    if missing_modules:
        raise KeyError(f"checkpoint {checkpoint} is missing modules {missing_modules}")
    for key, module in trainer.all_saveable_modules.items():
        module.load_state_dict(state[key], strict=True)
    del state


def evaluate_cell(
    trainer: TrainerFS,
    target_plan: dict[str, Any],
    target_cache: dict[str, Any],
    *,
    target: str,
    job,
    checkpoint: Path,
    result_path: Path,
    args: argparse.Namespace,
) -> None:
    if result_path.is_file():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        validate_existing(
            payload,
            job=job,
            target=target,
            checkpoint=checkpoint,
            args=args,
            plan_fingerprint=target_plan["fingerprint"],
        )
        print(f"SKIP {job.model.model_id} seed={job.seed} target={target}", flush=True)
        return

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    assert_cpu_batches(target_cache["batches"])
    loader = ReplayLoader(target_cache["batches"])
    audited = AuditedLoader(loader, args.batch_size)
    metrics_path = metric_sidecar_path(trainer.logging_dir, target, CHECKPOINT_STEP)
    metrics_path.unlink(missing_ok=True)
    started = time.monotonic()
    with torch.no_grad():
        trainer.model.eval()
        loss, score, score_std, aux_loss, ranks = trainer.do_eval(
            audited, split_name=f"test_{target}", step=CHECKPOINT_STEP
        )
    elapsed = time.monotonic() - started
    metrics = load_metric_sidecar(trainer.logging_dir, target, CHECKPOINT_STEP)
    if audited.batch_count != args.batch_count or audited.episode_count != EPISODE_COUNT:
        raise AssertionError(
            f"consumed {audited.batch_count} batches/{audited.episode_count} episodes; "
            f"expected {args.batch_count}/{EPISODE_COUNT}"
        )
    if audited.fingerprint != target_cache["observed_fingerprint"]:
        raise AssertionError(
            f"cached replay fingerprint changed for {job.key}/{target}: "
            f"expected {target_cache['observed_fingerprint']}, got {audited.fingerprint}"
        )
    assert_cpu_batches(target_cache["batches"])
    numeric = {
        "score": _to_float(score),
        "score_std": _to_float(score_std),
        "loss": _to_float(loss),
        "aux_loss": _to_float(aux_loss),
        **metrics,
    }
    if not all(math.isfinite(float(value)) for value in numeric.values()):
        raise ValueError(f"non-finite result for {job.key}/{target}: {numeric}")
    if not math.isclose(numeric["score"], numeric["accuracy"], abs_tol=1e-12):
        raise ValueError(
            f"score/accuracy mismatch for {job.key}/{target}: "
            f"{numeric['score']} versus {numeric['accuracy']}"
        )
    payload = {
        "protocol": PROTOCOL,
        "created_utc": utc_now(),
        "evaluation_commit": git_commit(),
        "worker_index": args.worker_index,
        "model_id": job.model.model_id,
        "aliases": list(job.model.aliases),
        "sources": list(job.model.sources),
        "seed": job.seed,
        "target": target,
        "checkpoint_step": CHECKPOINT_STEP,
        "checkpoint": str(checkpoint),
        "split": "test",
        "edge_view": "static_train",
        "target_edge_view": "static_test",
        "metric_contract": METRIC_CONTRACT,
        "batch_size": args.batch_size,
        "batch_count": audited.batch_count,
        "episode_count": audited.episode_count,
        "episode_plan_fingerprint": target_plan["fingerprint"],
        "observed_episode_fingerprint": audited.fingerprint,
        "batch_replay_mode": "materialized_cpu_clone_v1",
        "cache_materialization_seconds": target_cache["elapsed_seconds"],
        "cache_min_mem_available_gib": target_cache["min_mem_available_gib"],
        "cache_max_rss_gib": target_cache["max_rss_gib"],
        "elapsed_seconds": elapsed,
        "max_rss_gib": max_rss_gib(),
        "peak_cuda_allocated_gib": (
            torch.cuda.max_memory_allocated() / (1024 ** 3)
            if torch.cuda.is_available() else 0.0
        ),
        **numeric,
    }
    if ranks:
        payload["ranks"] = {key: _to_float(value) for key, value in ranks.items()}
    atomic_json(result_path, payload)
    print(
        f"DONE model={job.model.model_id} seed={job.seed} target={target} "
        f"score={numeric['score']:.6f} auc={numeric['roc_auc_ovr_macro']:.6f} "
        f"seconds={elapsed:.1f}",
        flush=True,
    )


def parse_targets(text: str) -> list[str]:
    targets = [part.strip() for part in text.split(",") if part.strip()]
    if not targets:
        targets = list(SOURCES)
    if len(targets) != len(set(targets)):
        raise ValueError(f"duplicate targets: {targets}")
    unknown = [target for target in targets if target not in SOURCES]
    if unknown:
        raise ValueError(f"unknown targets {unknown}; expected {list(SOURCES)}")
    return targets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-index", required=True, type=int)
    parser.add_argument("--worker-count", required=True, type=int)
    parser.add_argument("--max-checkpoints", type=int)
    parser.add_argument("--specialists-only", action="store_true")
    parser.add_argument("--targets", default=",".join(SOURCES))
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--episode-count", default=EPISODE_COUNT, type=int)
    parser.add_argument("--config", type=Path, default=HERE / "training.yaml")
    parser.add_argument("--training-state-root", required=True, type=Path)
    parser.add_argument("--training-run-stamp", default="20260807")
    parser.add_argument("--evaluation-state-root", required=True, type=Path)
    parser.add_argument("--evaluation-log-root", required=True, type=Path)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--evaluation-run-stamp", required=True)
    parser.add_argument("--ready-dir", type=Path)
    parser.add_argument("--expected-workers", default=1, type=int)
    parser.add_argument("--barrier-timeout-seconds", default=1800, type=int)
    parser.add_argument("--min-host-reserve-gib", default=256.0, type=float)
    args = parser.parse_args()
    if not 0 <= args.worker_index < args.worker_count:
        parser.error("worker-index must be in [0, worker-count)")
    if args.episode_count != EPISODE_COUNT:
        parser.error(f"episode-count is frozen at {EPISODE_COUNT}")
    if args.batch_size <= 0 or EPISODE_COUNT % args.batch_size:
        parser.error("batch-size must be a positive divisor of 512")
    args.batch_count = EPISODE_COUNT // args.batch_size
    return args


def main() -> int:
    args = parse_args()
    cpu_threads = int(os.environ.get("FINAL_CORE_CPU_THREADS", "24"))
    if cpu_threads <= 0:
        raise ValueError("FINAL_CORE_CPU_THREADS must be positive")
    torch.set_num_threads(cpu_threads)
    torch.set_num_interop_threads(1)
    print(
        f"CPU_THREADS intraop={torch.get_num_threads()} "
        f"interop={torch.get_num_interop_threads()}",
        flush=True,
    )
    targets = parse_targets(args.targets)
    jobs = physical_jobs()
    if args.specialists_only:
        jobs = [job for job in jobs if job.model.model_id.startswith("ss_")]
        if len(jobs) != 27:
            raise AssertionError(f"expected 27 specialist checkpoints, got {len(jobs)}")
    assigned = [
        job for index, job in enumerate(jobs)
        if index % args.worker_count == args.worker_index
    ]
    if args.max_checkpoints is not None:
        assigned = assigned[:args.max_checkpoints]
    if not assigned:
        raise ValueError(f"worker {args.worker_index} has no assigned checkpoints")
    for job in assigned:
        checkpoint = checkpoint_path(args.training_state_root, job, args.training_run_stamp)
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)

    first_job = assigned[0]
    first_checkpoint = checkpoint_path(
        args.training_state_root, first_job, args.training_run_stamp
    )
    base_params = resolved_params(
        args,
        seed=0,
        model_id=first_job.model.model_id,
        target=targets[0],
        checkpoint=first_checkpoint,
    )
    seed_everything(base_params)
    print(
        f"worker={args.worker_index}/{args.worker_count} loading graph once; "
        f"jobs={len(assigned)} targets={targets}",
        flush=True,
    )
    dataset = load_dataset(base_params)
    if getattr(dataset, "nm_background_edge_view", None) != "static_train":
        raise AssertionError("message passing is not restricted to static_train")
    if getattr(dataset, "nm_holdout_edge_view", None) != "static_test":
        raise AssertionError("test positives are not using static_test")
    wait_at_load_barrier(args)

    bootstrap_params = resolved_params(
        args,
        seed=first_job.seed,
        model_id=first_job.model.model_id,
        target=targets[0],
        checkpoint=first_checkpoint,
    )
    seed_everything(bootstrap_params)
    trainer = TrainerFS(dataset, bootstrap_params)
    target_plans: dict[str, dict[str, Any]] = {}
    for target in targets:
        # Positive-member sampling uses the holdout NeighborSampler and therefore
        # consumes torch RNG in addition to BatchSampler's private Python RNG.
        # Reset before materialization so the raw stream is independent of the
        # checkpoint/model initialization that happened earlier in this process.
        reset_fixed_eval_rng(target)
        trainer.parameter["neighbor_sampling_source_subset"] = target
        _, _, _, loader = trainer._build_dataloaders(dataset, trainer.dataset_name)
        batches = list(loader.batch_sampler)
        fingerprint, episode_count = fingerprint_plan(
            target,
            batches,
            expected_batch_size=args.batch_size,
            dataset=dataset,
        )
        if len(batches) != args.batch_count or episode_count != EPISODE_COUNT:
            raise AssertionError(
                f"target {target}: {len(batches)} batches/{episode_count} episodes"
            )
        target_plans[target] = {
            "batches": batches,
            "collate_fn": loader.collate_fn,
            "fingerprint": fingerprint,
        }
        print(
            f"PLAN target={target} batches={len(batches)} episodes={episode_count} "
            f"fingerprint={fingerprint}",
            flush=True,
        )
    trainer.test_dataloader = None

    for target in targets:
        pending: list[tuple[Any, Path, Path]] = []
        existing_observed_fingerprints: set[str] = set()
        for job in assigned:
            checkpoint = checkpoint_path(
                args.training_state_root, job, args.training_run_stamp
            )
            result_path = (
                args.results_root
                / f"seed_{job.seed}"
                / job.model.model_id
                / f"{target}.json"
            )
            if result_path.is_file():
                existing_payload = json.loads(result_path.read_text(encoding="utf-8"))
                validate_existing(
                    existing_payload,
                    job=job,
                    target=target,
                    checkpoint=checkpoint,
                    args=args,
                    plan_fingerprint=target_plans[target]["fingerprint"],
                )
                print(
                    f"SKIP {job.model.model_id} seed={job.seed} target={target}",
                    flush=True,
                )
                existing_observed_fingerprints.add(
                    existing_payload["observed_episode_fingerprint"]
                )
            else:
                pending.append((job, checkpoint, result_path))
        if len(existing_observed_fingerprints) > 1:
            raise AssertionError(
                f"existing results disagree on observed fingerprint for {target}: "
                f"{sorted(existing_observed_fingerprints)}"
            )
        if not pending:
            print(f"TARGET_COMPLETE target={target} existing={len(assigned)}", flush=True)
            continue

        target_cache = materialize_target_batches(
            dataset,
            target_plans[target],
            target=target,
            args=args,
        )
        if target_cache["observed_fingerprint"] != target_plans[target].get(
            "observed_fingerprint", target_cache["observed_fingerprint"]
        ):
            raise AssertionError(f"target {target} cache fingerprint changed")
        target_plans[target]["observed_fingerprint"] = target_cache[
            "observed_fingerprint"
        ]
        if (
            existing_observed_fingerprints
            and existing_observed_fingerprints
            != {target_cache["observed_fingerprint"]}
        ):
            raise AssertionError(
                f"cached stream for {target} does not match resumed results: "
                f"cache={target_cache['observed_fingerprint']} "
                f"existing={sorted(existing_observed_fingerprints)}"
            )

        for job, checkpoint, result_path in pending:
            params = resolved_params(
                args,
                seed=job.seed,
                model_id=job.model.model_id,
                target=target,
                checkpoint=checkpoint,
            )
            seed_everything(params)
            load_checkpoint_strict(trainer, checkpoint)
            trainer.parameter.update(
                seed=job.seed,
                neighbor_sampling_source_subset=target,
                pretrained_model_run=str(checkpoint),
            )
            evaluate_cell(
                trainer,
                target_plans[target],
                target_cache,
                target=target,
                job=job,
                checkpoint=checkpoint,
                result_path=result_path,
                args=args,
            )
        del target_cache
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        available = mem_available_gib()
        if available < args.min_host_reserve_gib:
            raise MemoryError(
                f"after releasing target {target}, only {available:.1f} GiB available; "
                f"required reserve is {args.min_host_reserve_gib:.1f} GiB"
            )
        print(
            f"TARGET_DONE target={target} evaluated={len(pending)} "
            f"mem_available_gib={available:.1f}",
            flush=True,
        )
    wandb.finish()
    print(
        f"WORKER_DONE index={args.worker_index} jobs={len(assigned)} "
        f"max_rss_gib={max_rss_gib():.1f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
