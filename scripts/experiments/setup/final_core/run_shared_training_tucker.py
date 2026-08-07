#!/usr/bin/env python3
"""Train the final core queue while sharing one immutable graph/CSR image via fork."""

from __future__ import annotations

import argparse
from collections import deque
import multiprocessing as mp
import os
from pathlib import Path
import random
import sys
import time
import traceback

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT))

from core_plan import build_models  # noqa: E402
from data.data_loader_wrapper import get_dataset_wrap  # noqa: E402
from experiments.params import get_params  # noqa: E402
from experiments.trainer import TrainerFS  # noqa: E402


def resolve_params(config: Path, overrides: list[str]) -> dict:
    previous = sys.argv
    sys.argv = ["run_single_experiment.py", "--config", str(config), *overrides]
    try:
        return get_params()
    finally:
        sys.argv = previous


def load_shared_dataset(params: dict):
    """Mirror run_single_experiment's CPU-only dataset construction exactly."""
    return get_dataset_wrap(
        root=params["root"], dataset=params["dataset"], force_cache=params["force_cache"],
        small_dataset=params["small_dataset"], invalidate_cache=None,
        original_features=params["original_features"], n_shot=params["n_shots"],
        n_query=params["n_query"], bert=None if params["original_features"] else params["bert_emb_model"],
        bert_device=params["device"], val_len_cap=params["val_len_cap"],
        test_len_cap=params["test_len_cap"], dataset_len_cap=params["dataset_len_cap"],
        n_way=params["n_way"], rel_sample_rand_seed=params["rel_sample_random_seed"],
        calc_ranks=params["calc_ranks"],
        kg_emb_model=params["kg_emb_model"] if params["kg_emb_model"] else None,
        task_name=params["task_name"], shuffle_index=params["shuffle_index"],
        node_graph=params["task_name"] == "sn_neighbor_matching",
        csv_filename=params["csv_filename"], label_type=params["label_type"],
        max_users=params["max_users"], pkl_filename=params["facebook_pkl_filename"],
        facebook_edges_filename=params["facebook_edges_filename"],
        facebook_node_features_filename=params["facebook_node_features_filename"],
        facebook_data_source=params["facebook_data_source"],
        facebook_use_edge_features=params["facebook_use_edge_features"],
        facebook_edge_feature_columns=params["facebook_edge_feature_columns"],
        source_pkl_path=params["facebook_source_pkl_path"],
        facebook_embeddings_path=params["facebook_embeddings_path"],
        facebook_embedding_ids_path=params["facebook_embedding_ids_path"],
        facebook_text_emb_model=params["facebook_text_emb_model"],
        facebook_target_dim=params["facebook_target_dim"],
        facebook_filter_to_uk_ru=params["facebook_filter_to_uk_ru"],
        max_posts=params["facebook_max_posts"], n_hop=params["n_hop"],
        neighbor_sampling_hop_sizes=params["neighbor_sampling_hop_sizes"],
        neighbor_sampling_node_limit=params["neighbor_sampling_node_limit"],
        neighbor_matching_walk_hops=params["neighbor_matching_walk_hops"],
        graph_filename=params["graph_filename"], target_feature=params["target_feature"],
        target_feature_keep_in_x=params["target_feature_keep_in_x"],
        target_transform=params["target_transform"], feature_subset=params["feature_subset"],
        midterm_label_downsample=params["midterm_label_downsample"],
        edge_view=params["edge_view"], target_edge_view=params["target_edge_view"],
        edge_feature_subset=params["edge_feature_subset"],
        neighbor_sampling_strategy=params["neighbor_sampling_strategy"],
        neighbor_sampling_strata=params["neighbor_sampling_strata"],
        neighbor_matching_edge_split=params["neighbor_matching_edge_split"], seed=params["seed"],
    )


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def run_child(
    dataset,
    config: str,
    state_root: str,
    log_root: str,
    run_stamp: str,
    model_id: str,
    sources: str,
    seed: int,
    gpu: int,
    log_path: str,
) -> None:
    with open(log_path, "w", buffering=1) as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        try:
            torch.set_num_threads(4)
            torch.autograd.set_detect_anomaly(False)
            prefix = f"finalcore_{model_id}_s{seed}"
            params = resolve_params(Path(config), [
                "--device", str(gpu), "--seed", str(seed), "--prefix", prefix,
                "--timestamp", run_stamp, "--state_dir", state_root, "--log_dir", log_root,
                "--neighbor_sampling_source_subset", sources,
            ])
            seed_everything(seed)
            trainer = TrainerFS(dataset, params)
            trainer.train()
        except BaseException:
            traceback.print_exc()
            raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "training.yaml")
    parser.add_argument("--state-root", type=Path, default=REPO_ROOT / "state/final_core")
    parser.add_argument("--log-root", type=Path, default=REPO_ROOT / "log/final_core")
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--slots-per-gpu", type=int, default=3)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--run-stamp", default="20260807")
    parser.add_argument("--max-jobs", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    gpu_ids = [int(x) for x in args.gpus.split(",") if x]
    seeds = [int(x) for x in args.seeds.split(",") if x]
    if not gpu_ids or any(gpu not in {0, 1, 2, 3} for gpu in gpu_ids):
        raise ValueError("GPUs must be a non-empty subset of Tucker-owned GPUs 0-3")
    if set(seeds) - {0, 1, 2}:
        raise ValueError("training seeds must be drawn from 0,1,2")
    jobs = deque(
        (model.model_id, ",".join(model.sources), seed)
        for model in build_models() for seed in seeds
    )
    if args.max_jobs is not None:
        jobs = deque(list(jobs)[:args.max_jobs])
    if args.dry_run:
        for model_id, sources, seed in jobs:
            print(f"model={model_id} seed={seed} sources={sources}")
        return 0

    args.state_root.mkdir(parents=True, exist_ok=True)
    (args.log_root / "train").mkdir(parents=True, exist_ok=True)
    (args.log_root / "launch").mkdir(parents=True, exist_ok=True)
    base = resolve_params(args.config, ["--seed", "0", "--device", "0"])
    print("Loading one shared graph and three CSR samplers before forking workers...", flush=True)
    started = time.time()
    dataset = load_shared_dataset(base)
    print(f"Shared dataset ready in {time.time() - started:.1f}s", flush=True)
    if torch.cuda.is_initialized():
        raise RuntimeError("parent initialized CUDA before fork; refusing unsafe launch")

    slots = deque(gpu for gpu in gpu_ids for _ in range(args.slots_per_gpu))
    ctx = mp.get_context("fork")
    active: dict[int, tuple[mp.Process, int, str, int, Path]] = {}
    failed = False
    while jobs or active:
        while jobs and slots and not failed:
            model_id, sources, seed = jobs.popleft()
            gpu = slots.popleft()
            prefix = f"finalcore_{model_id}_s{seed}"
            run_name = f"{prefix}_{args.run_stamp}"
            checkpoint = args.state_root / run_name / "checkpoint/state_dict_2500.ckpt"
            if checkpoint.is_file():
                print(f"SKIP complete {run_name}", flush=True)
                slots.append(gpu)
                continue
            run_dir = args.state_root / run_name
            if run_dir.exists():
                raise FileExistsError(f"refusing incomplete existing run {run_dir}")
            log_path = args.log_root / "train" / f"{run_name}.log"
            process = ctx.Process(
                target=run_child,
                args=(dataset, str(args.config), str(args.state_root), str(args.log_root),
                      args.run_stamp, model_id, sources, seed, gpu, str(log_path)),
            )
            process.start()
            active[process.pid] = (process, gpu, model_id, seed, checkpoint)
            print(f"START pid={process.pid} gpu={gpu} model={model_id} seed={seed}", flush=True)
        for pid, (process, gpu, model_id, seed, checkpoint) in list(active.items()):
            if process.is_alive():
                continue
            process.join()
            del active[pid]
            slots.append(gpu)
            if process.exitcode != 0 or not checkpoint.is_file():
                print(f"FAIL pid={pid} model={model_id} seed={seed} exit={process.exitcode}", flush=True)
                failed = True
            else:
                print(f"DONE pid={pid} model={model_id} seed={seed}", flush=True)
        if failed and not active:
            break
        if active:
            time.sleep(1)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
