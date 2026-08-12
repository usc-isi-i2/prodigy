#!/usr/bin/env python3
"""Rescore final-core checkpoints on common exact-ID-center-clean episodes."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import torch
import wandb

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
FINAL_CORE = HERE.parent / "final_core"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(FINAL_CORE))

import evaluate_fixed_grid as base  # noqa: E402
from experiments.run_single_experiment import load_dataset, seed_everything  # noqa: E402
from experiments.trainer import TrainerFS, _to_float  # noqa: E402
from fixed_test_plan import CHECKPOINT_STEP, checkpoint_path, physical_jobs  # noqa: E402
from protocol import (  # noqa: E402
    CENTER_PROTOCOL,
    EPISODE_COUNT,
    INDUCED_PROTOCOL,
    TARGETS,
    configure_allowed_episode_centers,
    induced_neighbor_sampler,
    select_center_clean_batches,
    sha256_file,
)


class CleanAuditedLoader(base.AuditedLoader):
    """Fingerprint exact sampled context and quantify residual context overlap."""

    def __init__(
        self,
        loader,
        expected_batch_size: int,
        excluded_mask: torch.Tensor,
        *,
        require_zero_overlap: bool,
    ):
        super().__init__(loader, expected_batch_size)
        if excluded_mask.dtype != torch.bool or excluded_mask.ndim != 1:
            raise ValueError("excluded_mask must be a one-dimensional bool tensor")
        self.excluded_mask = excluded_mask.cpu()
        self.require_zero_overlap = require_zero_overlap
        self.context_node_occurrences = 0
        self.context_overlap_occurrences = 0
        self._context_overlap_nodes: set[int] = set()
        self._digest = hashlib.sha256()
        self._digest.update(b"entity-disjoint-observed-episode-stream-v1\0")

    def __iter__(self):
        for batch in super().__iter__():
            graph = batch[0]
            global_ids = graph.global_node_ids.detach().cpu().long()
            valid = global_ids >= 0
            if valid.any() and int(global_ids[valid].max()) >= self.excluded_mask.numel():
                raise AssertionError("sampled global node ID exceeds graph bounds")
            real_ids = global_ids[valid]
            overlap = self.excluded_mask[real_ids]
            if self.require_zero_overlap and bool(overlap.any()):
                raise AssertionError("induced-subgraph sampler returned an excluded node")
            self.context_node_occurrences += int(real_ids.numel())
            self.context_overlap_occurrences += int(overlap.sum())
            self._context_overlap_nodes.update(int(value) for value in torch.unique(real_ids[overlap]).tolist())
            for name, tensor in (
                ("global_node_ids", graph.global_node_ids),
                ("edge_index", graph.edge_index),
                ("graph_ptr", graph.ptr),
            ):
                self._digest.update(name.encode("utf-8") + b"\0")
                base.update_int(self._digest, tensor.numel())
                self._digest.update(self._tensor_bytes(tensor))
            yield batch

    @property
    def context_unique_overlap_nodes(self) -> int:
        return len(self._context_overlap_nodes)


def load_exclusion(
    path: Path,
    target: str,
    graph_nodes: int,
    *,
    graph_sha256: str,
    identity_db_sha256: str,
    target_graph_nodes: int,
) -> tuple[set[int], dict[str, Any]]:
    payload = torch.load(path, map_location="cpu")
    if payload.get("protocol") != "exact_id_exclusion_union3_v1":
        raise ValueError(f"{path}: wrong exclusion protocol")
    if payload.get("target") != target:
        raise ValueError(f"{path}: target mismatch")
    expected_comparisons = sorted(name for name in TARGETS if name != target)
    provenance_checks = {
        "graph_sha256": graph_sha256,
        "identity_db_sha256": identity_db_sha256,
        "target_graph_nodes": target_graph_nodes,
    }
    for field, wanted in provenance_checks.items():
        if payload.get(field) != wanted:
            raise ValueError(f"{path}: stale exclusion {field}")
    if sorted(payload.get("comparison_sources", [])) != expected_comparisons:
        raise ValueError(f"{path}: stale comparison_sources")
    indices = payload.get("excluded_node_indices")
    if not isinstance(indices, torch.Tensor) or indices.dtype != torch.long:
        raise ValueError(f"{path}: excluded_node_indices must be a long tensor")
    if indices.numel() and (int(indices.min()) < 0 or int(indices.max()) >= graph_nodes):
        raise ValueError(f"{path}: exclusion indices are outside graph bounds")
    if int(torch.unique(indices).numel()) != int(indices.numel()):
        raise ValueError(f"{path}: exclusion indices are not unique")
    metadata = {
        "exclusion_artifact": str(path),
        "exclusion_artifact_sha256": sha256_file(path),
        "excluded_node_count": int(indices.numel()),
        "target_graph_nodes": int(payload["target_graph_nodes"]),
        "comparison_sources": list(payload["comparison_sources"]),
    }
    return set(int(value) for value in indices.tolist()), metadata


def evaluate_cell(
    trainer: TrainerFS,
    dataset,
    target_plan: dict[str, Any],
    *,
    target: str,
    job,
    checkpoint: Path,
    result_path: Path,
    args: argparse.Namespace,
) -> None:
    protocol = INDUCED_PROTOCOL if args.exclusion_level == "induced_subgraph" else CENTER_PROTOCOL
    if result_path.is_file():
        existing = json.loads(result_path.read_text())
        checks = {
            "protocol": protocol,
            "model_id": job.model.model_id,
            "seed": job.seed,
            "target": target,
            "checkpoint_step": CHECKPOINT_STEP,
            "exclusion_level": args.exclusion_level,
            "episode_plan_fingerprint": target_plan["fingerprint"],
            "unfiltered_prefix_plan_fingerprint": target_plan["unfiltered_prefix_fingerprint"],
            "exclusion_artifact_sha256": target_plan["exclusion"]["exclusion_artifact_sha256"],
            **target_plan["induced_adjacency"],
        }
        for field, wanted in checks.items():
            if existing.get(field) != wanted:
                raise ValueError(f"existing {result_path} has mismatched {field}")
        print(f"SKIP {job.model.model_id} seed={job.seed} target={target}", flush=True)
        return

    base.reset_fixed_eval_rng(target)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    loader = base.make_frozen_loader(
        target_plan["dataset"], target_plan["collate_fn"], target_plan["batches"]
    )
    audited = CleanAuditedLoader(
        loader,
        args.batch_size,
        target_plan["context_forbidden_mask"],
        require_zero_overlap=args.exclusion_level == "induced_subgraph",
    )
    started = time.monotonic()
    with torch.no_grad():
        trainer.model.eval()
        loss, score, score_std, aux_loss, ranks = trainer.do_eval(
            audited,
            split_name=f"test_{args.exclusion_level}_{target}",
            step=CHECKPOINT_STEP,
        )
    elapsed = time.monotonic() - started
    if audited.batch_count != args.batch_count or audited.episode_count != EPISODE_COUNT:
        raise AssertionError("clean evaluator did not consume the frozen episode plan")
    numeric = {
        "score": _to_float(score),
        "score_std": _to_float(score_std),
        "loss": _to_float(loss),
        "aux_loss": _to_float(aux_loss),
    }
    if not all(math.isfinite(float(value)) for value in numeric.values()):
        raise ValueError(f"non-finite result for {job.key}/{target}: {numeric}")
    payload = {
        "protocol": protocol,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_commit": base.git_commit(),
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
        "exclusion_level": args.exclusion_level,
        "exclusion_scope": "target IDs recurring in either other full exact-ID graph",
        "batch_size": args.batch_size,
        "batch_count": audited.batch_count,
        "episode_count": audited.episode_count,
        "episode_plan_fingerprint": target_plan["fingerprint"],
        "unfiltered_prefix_plan_fingerprint": target_plan["unfiltered_prefix_fingerprint"],
        "observed_episode_fingerprint": audited.fingerprint,
        "elapsed_seconds": elapsed,
        "max_rss_gib": base.max_rss_gib(),
        "peak_cuda_allocated_gib": (
            torch.cuda.max_memory_allocated() / (1024 ** 3)
            if torch.cuda.is_available() else 0.0
        ),
        "sampled_context_node_occurrences": audited.context_node_occurrences,
        "sampled_context_overlap_occurrences": audited.context_overlap_occurrences,
        "sampled_context_unique_overlap_nodes": audited.context_unique_overlap_nodes,
        **target_plan["exclusion"],
        **target_plan["sampling"],
        **target_plan["induced_adjacency"],
        **numeric,
    }
    if ranks:
        payload["ranks"] = {key: _to_float(value) for key, value in ranks.items()}
    base.atomic_json(result_path, payload)
    print(
        f"DONE model={job.model.model_id} seed={job.seed} target={target} "
        f"score={numeric['score']:.6f} seconds={elapsed:.1f}", flush=True
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-index", required=True, type=int)
    parser.add_argument("--worker-count", required=True, type=int)
    parser.add_argument("--max-checkpoints", type=int)
    parser.add_argument("--targets", default=",".join(TARGETS))
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--candidate-multiplier", default=8, type=int)
    parser.add_argument(
        "--exclusion-level",
        choices=("episode_centers", "induced_subgraph"),
        default="episode_centers",
    )
    parser.add_argument(
        "--model-ids",
        default="ss_ukr_rus,ss_covid,ss_midterm",
        help="Comma-separated physical model IDs; diagnostic defaults to the 3 exact-ID specialists.",
    )
    parser.add_argument("--config", type=Path, default=FINAL_CORE / "training.yaml")
    parser.add_argument("--training-state-root", required=True, type=Path)
    parser.add_argument("--training-run-stamp", default="20260807")
    parser.add_argument("--evaluation-state-root", required=True, type=Path)
    parser.add_argument("--evaluation-log-root", required=True, type=Path)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--evaluation-run-stamp", required=True)
    parser.add_argument("--exclusion-root", required=True, type=Path)
    parser.add_argument("--graph-sha256", required=True)
    parser.add_argument("--identity-db-sha256", required=True)
    parser.add_argument("--original-results-root", required=True, type=Path)
    parser.add_argument("--ready-dir", type=Path)
    parser.add_argument("--expected-workers", default=1, type=int)
    parser.add_argument("--barrier-timeout-seconds", default=1800, type=int)
    parser.add_argument("--min-host-reserve-gib", default=128.0, type=float)
    args = parser.parse_args()
    if not 0 <= args.worker_index < args.worker_count:
        parser.error("worker-index must be in [0, worker-count)")
    if args.batch_size <= 0 or EPISODE_COUNT % args.batch_size:
        parser.error("batch-size must be a positive divisor of 512")
    if args.candidate_multiplier < 2:
        parser.error("candidate-multiplier must be at least 2")
    args.batch_count = EPISODE_COUNT // args.batch_size
    targets = [item.strip() for item in args.targets.split(",") if item.strip()]
    if not targets or len(targets) != len(set(targets)) or any(t not in TARGETS for t in targets):
        parser.error(f"targets must be a unique subset of {TARGETS}")
    args.targets_parsed = targets
    model_ids = [item.strip() for item in args.model_ids.split(",") if item.strip()]
    known = {job.model.model_id for job in physical_jobs()}
    if not model_ids or len(model_ids) != len(set(model_ids)) or any(m not in known for m in model_ids):
        parser.error("model-ids must be unique physical final-core model IDs")
    args.model_ids_parsed = model_ids
    return args


def main() -> int:
    args = parse_args()
    cpu_threads = int(os.environ.get("ENTITY_DISJOINT_CPU_THREADS", "24"))
    torch.set_num_threads(cpu_threads)
    torch.set_num_interop_threads(1)
    assigned = [
        job for index, job in enumerate(physical_jobs())
        if job.model.model_id in args.model_ids_parsed
        and index % args.worker_count == args.worker_index
    ]
    if args.max_checkpoints is not None:
        assigned = assigned[: args.max_checkpoints]
    if not assigned:
        raise ValueError("worker has no assigned checkpoints")
    for job in assigned:
        path = checkpoint_path(args.training_state_root, job, args.training_run_stamp)
        if not path.is_file():
            raise FileNotFoundError(path)

    plan_args = copy.copy(args)
    plan_args.batch_count = args.batch_count * args.candidate_multiplier
    first_job = assigned[0]
    first_checkpoint = checkpoint_path(args.training_state_root, first_job, args.training_run_stamp)
    base_params = base.resolved_params(
        plan_args, seed=0, model_id=first_job.model.model_id,
        target=args.targets_parsed[0], checkpoint=first_checkpoint,
    )
    seed_everything(base_params)
    dataset = load_dataset(base_params)
    if getattr(dataset, "nm_background_edge_view", None) != "static_train":
        raise AssertionError("message passing is not restricted to static_train")
    if getattr(dataset, "nm_holdout_edge_view", None) != "static_test":
        raise AssertionError("test positives are not using static_test")
    base.wait_at_load_barrier(args)

    bootstrap_params = base.resolved_params(
        plan_args,
        seed=first_job.seed,
        model_id=first_job.model.model_id,
        target=args.targets_parsed[0],
        checkpoint=first_checkpoint,
    )
    seed_everything(bootstrap_params)
    trainer = TrainerFS(dataset, bootstrap_params)
    target_plans: dict[str, dict[str, Any]] = {}
    for target in args.targets_parsed:
        excluded, exclusion_meta = load_exclusion(
            args.exclusion_root / f"{target}.pt",
            target,
            int(dataset.graph.num_nodes),
            graph_sha256=args.graph_sha256,
            identity_db_sha256=args.identity_db_sha256,
            target_graph_nodes=int((dataset.graph.graph_id == list(dataset.graph.source_graph_names).index(target)).sum()),
        )
        base.reset_fixed_eval_rng(target)
        trainer.parameter["neighbor_sampling_source_subset"] = target
        trainer.parameter["test_len_cap"] = plan_args.batch_count
        _, _, _, unfiltered_loader = trainer._build_dataloaders(dataset, trainer.dataset_name)
        unfiltered_batches = list(unfiltered_loader.batch_sampler)
        unfiltered_prefix_fingerprint, unfiltered_count = base.fingerprint_plan(
            target,
            unfiltered_batches[: args.batch_count],
            expected_batch_size=args.batch_size,
            dataset=dataset,
        )
        if unfiltered_count != EPISODE_COUNT:
            raise AssertionError("unfiltered prefix does not contain 512 episodes")
        for job in assigned:
            if target in job.model.sources:
                continue
            original_path = (
                args.original_results_root
                / f"seed_{job.seed}"
                / job.model.model_id
                / f"{target}.json"
            )
            if not original_path.is_file():
                raise FileNotFoundError(original_path)
            original = json.loads(original_path.read_text())
            if original.get("episode_plan_fingerprint") != unfiltered_prefix_fingerprint:
                raise ValueError(
                    f"preflight failed: target {target} does not reproduce the frozen "
                    f"original plan for {job.key}"
                )
        base.reset_fixed_eval_rng(target)
        target_dataset = dataset
        target_id = list(dataset.graph.source_graph_names).index(target)
        allowed_mask = dataset.graph.graph_id == target_id
        excluded_tensor = torch.tensor(sorted(excluded), dtype=torch.long)
        allowed_mask[excluded_tensor] = False
        allowed = torch.nonzero(allowed_mask, as_tuple=False).flatten().long()
        induced_adjacency: dict[str, int] = {}
        if args.exclusion_level == "induced_subgraph":
            target_dataset = copy.copy(dataset)
            target_dataset.neighbor_sampler, background_meta = induced_neighbor_sampler(
                dataset.neighbor_sampler, allowed_mask
            )
            target_dataset.nm_test_neighbor_sampler, positive_meta = induced_neighbor_sampler(
                dataset.nm_test_neighbor_sampler, allowed_mask
            )
            target_dataset.nm_holdout_neighbor_sampler = target_dataset.nm_test_neighbor_sampler
            induced_adjacency = {
                "background_original_adjacency_nnz": background_meta["original_adjacency_nnz"],
                "background_induced_adjacency_nnz": background_meta["induced_adjacency_nnz"],
                "positive_original_adjacency_nnz": positive_meta["original_adjacency_nnz"],
                "positive_induced_adjacency_nnz": positive_meta["induced_adjacency_nnz"],
                "allowed_target_nodes": int(allowed.numel()),
            }
        _, _, _, loader = trainer._build_dataloaders(target_dataset, trainer.dataset_name)
        allowed_set = configure_allowed_episode_centers(loader, allowed)
        candidate_batches = list(loader.batch_sampler)
        clean_batches, sampling = select_center_clean_batches(
            candidate_batches, excluded,
            episode_count=EPISODE_COUNT, batch_size=args.batch_size,
        )
        fingerprint, count = base.fingerprint_plan(
            target, clean_batches, expected_batch_size=args.batch_size, dataset=dataset
        )
        if count != EPISODE_COUNT:
            raise AssertionError("clean plan does not contain 512 episodes")
        target_plans[target] = {
            "batches": clean_batches,
            "dataset": target_dataset,
            "collate_fn": loader.collate_fn,
            "fingerprint": fingerprint,
            "unfiltered_prefix_fingerprint": unfiltered_prefix_fingerprint,
            "exclusion": exclusion_meta,
            "sampling": sampling,
            "context_forbidden_mask": (
                ~allowed_mask
                if args.exclusion_level == "induced_subgraph"
                else ~allowed_mask & (dataset.graph.graph_id == target_id)
            ),
            "induced_adjacency": induced_adjacency,
        }
        print(
            f"CLEAN_PLAN target={target} fingerprint={fingerprint} "
            f"unfiltered_prefix={unfiltered_prefix_fingerprint} "
            f"sampling={sampling} excluded_nodes={len(excluded)} "
            f"allowed_nodes={len(allowed_set)} induced_adjacency={induced_adjacency}", flush=True
        )
        del excluded, excluded_tensor, allowed, allowed_mask, allowed_set
    trainer.test_dataloader = None

    for job_index, job in enumerate(assigned):
        checkpoint = checkpoint_path(args.training_state_root, job, args.training_run_stamp)
        if job_index > 0:
            wandb.finish()
            del trainer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            params = base.resolved_params(
                args, seed=job.seed, model_id=job.model.model_id,
                target=args.targets_parsed[0], checkpoint=checkpoint,
            )
            seed_everything(params)
            trainer = TrainerFS(dataset, params)
            trainer.test_dataloader = None
        for target in args.targets_parsed:
            # Training on the target itself cannot be made entity-disjoint without
            # retraining. The diagnostic is the six off-diagonal specialist cells.
            if target in job.model.sources:
                continue
            result_path = args.results_root / f"seed_{job.seed}" / job.model.model_id / f"{target}.json"
            evaluate_cell(
                trainer, dataset, target_plans[target], target=target, job=job,
                checkpoint=checkpoint, result_path=result_path, args=args,
            )
    wandb.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
