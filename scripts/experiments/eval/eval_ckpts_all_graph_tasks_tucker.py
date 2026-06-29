#!/usr/bin/env python3
"""Run checkpoint evaluation on Tucker without Slurm.

The runner targets the GTE-compatible graph artifacts under /dataMeR1/phil/data.
It inspects each graph and skips classification when no labels are present, and
skips temporal LP when no future-edge target view is present.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from typing import Any


DEFAULT_MODEL_LIST = (
    "scripts/experiments/covid_ukr/merged_ukr_rus_covid_nm_eval_model_list.txt"
)


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    root_name: str
    graph_filename: str
    supports_lp: bool
    nm_n_query: int
    pl_n_query: int
    val_cap: int
    test_cap: int
    eval_random_query: bool = False


DATASETS = {
    "midterm": DatasetConfig(
        name="midterm",
        root_name="midterm",
        graph_filename="retweet_graph_parquet.pt",
        supports_lp=True,
        nm_n_query=12,
        pl_n_query=12,
        val_cap=500,
        test_cap=500,
    ),
    "covid19_twitter": DatasetConfig(
        name="covid19_twitter",
        root_name="covid19_twitter",
        graph_filename="retweet_graph_parquet.pt",
        supports_lp=True,
        nm_n_query=12,
        pl_n_query=12,
        val_cap=500,
        test_cap=500,
    ),
    "ukr_rus_twitter": DatasetConfig(
        name="ukr_rus_twitter",
        root_name="ukr_rus_twitter",
        graph_filename="retweet_graph_parquet.pt",
        supports_lp=True,
        nm_n_query=12,
        pl_n_query=12,
        val_cap=500,
        test_cap=500,
    ),
    "merged_ukr_rus_covid": DatasetConfig(
        name="merged_ukr_rus_covid",
        root_name="merged",
        graph_filename="ukr_rus_covid_retweet_graph.pt",
        supports_lp=False,
        nm_n_query=12,
        pl_n_query=12,
        val_cap=500,
        test_cap=500,
    ),
    "covid_political": DatasetConfig(
        name="covid_political",
        root_name="covid_political",
        graph_filename="retweet_graph.pt",
        supports_lp=False,
        nm_n_query=12,
        pl_n_query=12,
        val_cap=500,
        test_cap=500,
        eval_random_query=True,
    ),
    "election2020": DatasetConfig(
        name="election2020",
        root_name="election2020",
        graph_filename="retweet_graph.pt",
        supports_lp=False,
        nm_n_query=1,
        pl_n_query=1,
        val_cap=500,
        test_cap=500,
    ),
    "ukr_rus_suspended": DatasetConfig(
        name="ukr_rus_suspended",
        root_name="ukr_rus_suspended",
        graph_filename="retweet_graph.pt",
        supports_lp=False,
        nm_n_query=1,
        pl_n_query=1,
        val_cap=500,
        test_cap=500,
    ),
}

TASK_ALIASES = {
    "nm": "neighbor_matching",
    "lp": "temporal_link_prediction",
    "pl": "classification",
    "neighbor_matching": "neighbor_matching",
    "temporal_link_prediction": "temporal_link_prediction",
    "classification": "classification",
}


def load_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required to inspect graph files") from exc
    return torch


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_model_list(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) == 1:
            ckpt = parts[0]
            model_name = Path(ckpt).parent.parent.name
        elif len(parts) == 2:
            model_name, ckpt = parts
        else:
            raise ValueError(f"Invalid model-list line: {raw_line!r}")
        rows.append((model_name, ckpt))
    if not rows:
        raise ValueError(f"No checkpoints found in {path}")
    return rows


def parse_checkpoint_step(path: Path) -> int | None:
    match = re.fullmatch(r"state_dict_(\d+)\.ckpt", path.name)
    if match is None:
        return None
    return int(match.group(1))


def discover_checkpoint_run_dir(run_dir: Path, name_prefix: str | None = None) -> list[tuple[str, str]]:
    checkpoint_dir = run_dir / "checkpoint"
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_dir}")

    checkpoints: list[tuple[int, Path]] = []
    for path in checkpoint_dir.glob("state_dict_*.ckpt"):
        step = parse_checkpoint_step(path)
        if step is not None:
            checkpoints.append((step, path))
    checkpoints.sort(key=lambda item: item[0])

    if not checkpoints:
        raise ValueError(f"No state_dict_<step>.ckpt checkpoints found in {checkpoint_dir}")

    prefix = name_prefix or run_dir.name
    return [
        (f"{prefix}_step{step}", checkpoint_path.as_posix())
        for step, checkpoint_path in checkpoints
    ]


def torch_load_graph(torch_mod: Any, path: Path) -> dict[str, Any]:
    try:
        return torch_mod.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch_mod.load(path, map_location="cpu")


def has_classification_labels(raw: dict[str, Any], torch_mod: Any) -> bool:
    label_names = list(raw.get("label_names") or [])
    if not label_names:
        return False
    y = raw.get("y")
    if y is None and "data" in raw:
        y = getattr(raw["data"], "y", None)
    if y is None:
        return False
    if torch_mod.is_floating_point(y):
        return False
    return bool((y >= 0).any().item())


def classification_label_count(raw: dict[str, Any]) -> int:
    label_names = list(raw.get("label_names") or [])
    if label_names:
        return len(label_names)
    y = raw.get("y")
    if y is None and "data" in raw:
        y = getattr(raw["data"], "y", None)
    if y is None:
        return 0
    valid = y[y >= 0]
    if valid.numel() == 0:
        return 0
    return int(valid.max().item()) + 1


def has_future_edges(raw: dict[str, Any]) -> bool:
    future = raw.get("future_edge_index")
    if future is not None and getattr(future, "numel", lambda: 0)() > 0:
        return True
    views = raw.get("target_edge_index_views") or {}
    for key in ("temporal_new", "future", "future_edge_index"):
        value = views.get(key)
        if value is not None and getattr(value, "numel", lambda: 0)() > 0:
            return True
    return False


def graph_info(torch_mod: Any, path: Path) -> dict[str, Any]:
    raw = torch_load_graph(torch_mod, path)
    x = raw.get("x")
    if x is None and "data" in raw:
        x = getattr(raw["data"], "x", None)
    x_dim = int(x.shape[1]) if x is not None and x.dim() == 2 else -1
    return {
        "x_dim": x_dim,
        "has_labels": has_classification_labels(raw, torch_mod),
        "label_count": classification_label_count(raw),
        "has_future_edges": has_future_edges(raw),
    }


def zero_shot(shots: str) -> str:
    return "True" if str(shots) == "0" else "False"


def task_tag(task: str) -> str:
    return {
        "neighbor_matching": "nm",
        "temporal_link_prediction": "lp",
        "classification": "pl",
    }[task]


def build_command(
    args: argparse.Namespace,
    dataset: DatasetConfig,
    model_name: str,
    ckpt_path: str,
    task: str,
    shots: str,
    classification_n_way: int | None = None,
) -> list[str]:
    root = Path(args.data_root) / dataset.root_name / "graphs"
    common = [
        args.python,
        "experiments/run_single_experiment.py",
        "--dataset",
        dataset.name,
        "--root",
        root.as_posix(),
        "--graph_filename",
        dataset.graph_filename,
        "--feature_subset",
        "emb_only",
        "--input_dim",
        "768",
        "--original_features",
        "True",
        "--workers",
        str(args.workers),
        "--batch_size",
        str(args.batch_size),
        "--device",
        str(args.device),
        "--seed",
        str(args.seed),
        "--eval_only",
        "True",
        "--eval_test_before_train",
        "True",
        "--eval_val_before_train",
        "True",
        "--save_roc_curve",
        "True",
        "--pretrained_model_run",
        ckpt_path,
    ]
    if dataset.eval_random_query:
        common.extend(["--eval_random_query", "True"])

    tag = task_tag(task)
    prefix = f"eval_{model_name}_to_{dataset.name}_{tag}_{shots}shot"
    if task == "neighbor_matching":
        # encode n_way so 3-way and 30-way evals don't collide in the same run dir
        prefix = f"{prefix}_{args.nm_n_way}way"
        extra = [
            "--task_name",
            "neighbor_matching",
            "--n_way",
            str(args.nm_n_way),
            "--n_shots",
            shots,
            "--n_query",
            str(dataset.nm_n_query),
            "--zero_shot",
            zero_shot(shots),
            "--dataset_len_cap",
            str(args.nm_dataset_len_cap),
            "--val_len_cap",
            str(dataset.val_cap),
            "--test_len_cap",
            str(dataset.test_cap),
            "--epochs",
            str(args.epochs),
            "--eval_step",
            str(args.eval_step),
            "--checkpoint_step",
            str(args.checkpoint_step),
            "--prefix",
            prefix,
        ]
    elif task == "temporal_link_prediction":
        extra = [
            "--task_name",
            "temporal_link_prediction",
            "--edge_view",
            "temporal_history",
            "--target_edge_view",
            "temporal_new",
            "--n_way",
            "1",
            "--n_shots",
            shots,
            "--n_query",
            str(args.lp_n_query),
            "--zero_shot",
            zero_shot(shots),
            "--dataset_len_cap",
            str(args.lp_dataset_len_cap),
            "--val_len_cap",
            str(args.parquet_val_cap),
            "--test_len_cap",
            str(args.parquet_test_cap),
            "--epochs",
            str(args.epochs),
            "--eval_step",
            str(args.eval_step),
            "--checkpoint_step",
            str(args.checkpoint_step),
            "--prefix",
            prefix,
        ]
    elif task == "classification":
        n_way = classification_n_way if classification_n_way is not None else 2
        extra = [
            "--task_name",
            "classification",
            "--ignore_label_embeddings",
            "False",
            "--linear_probe",
            "False",
            "--n_way",
            str(n_way),
            "--n_shots",
            shots,
            "--n_query",
            str(dataset.pl_n_query),
            "--zero_shot",
            zero_shot(shots),
            "--dataset_len_cap",
            str(args.pl_dataset_len_cap),
            "--val_len_cap",
            str(dataset.val_cap),
            "--test_len_cap",
            str(dataset.test_cap),
            "--epochs",
            str(args.epochs),
            "--eval_step",
            str(args.eval_step),
            "--checkpoint_step",
            str(args.checkpoint_step),
            "--midterm_label_downsample",
            "50:50",
            "--prefix",
            prefix,
        ]
    else:
        raise ValueError(f"Unsupported task: {task}")
    return common + extra + args.extra_args


def run_job_queue(
    jobs: list[tuple[list[str], str]],
    *,
    gpus: list[str],
    dry_run: bool,
    continue_on_error: bool,
) -> int:
    if not jobs:
        print("[progress] no eval jobs selected", flush=True)
        return 0
    if not gpus:
        gpus = [""]

    print(
        f"[progress] starting eval queue jobs={len(jobs)} parallel_slots={len(gpus)} "
        f"gpus={','.join(gpus) if any(gpus) else '<inherit>'}",
        flush=True,
    )
    running: list[tuple[subprocess.Popen, str, str]] = []
    failures = 0
    next_job = 0
    completed = 0

    while next_job < len(jobs) or running:
        while next_job < len(jobs) and len(running) < len(gpus):
            cmd, label = jobs[next_job]
            gpu = gpus[next_job % len(gpus)]
            env = os.environ.copy()
            if gpu:
                env["CUDA_VISIBLE_DEVICES"] = gpu
            printable = " ".join(shlex.quote(part) for part in cmd)
            prefix = f"CUDA_VISIBLE_DEVICES={gpu} " if gpu else ""
            print(
                f"[launch {next_job + 1}/{len(jobs)}] gpu={gpu or '<inherit>'} {label}",
                flush=True,
            )
            print(f"[cmd] {prefix}{printable}", flush=True)
            next_job += 1
            if dry_run:
                continue
            process = subprocess.Popen(cmd, env=env)
            running.append((process, label, gpu))

        if dry_run:
            continue

        process, label, gpu = running.pop(0)
        returncode = process.wait()
        completed += 1
        if returncode != 0:
            failures += 1
            print(
                f"[error] returncode={returncode} gpu={gpu or '<inherit>'} {label}",
                flush=True,
            )
            if not continue_on_error:
                for other_process, _, _ in running:
                    other_process.terminate()
                return returncode
        else:
            print(
                f"[done {completed}/{len(jobs)}] gpu={gpu or '<inherit>'} {label}",
                flush=True,
            )

    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-list", default=DEFAULT_MODEL_LIST)
    parser.add_argument(
        "--checkpoint-run-dir",
        default="",
        help=(
            "Training run directory containing checkpoint/state_dict_<step>.ckpt files. "
            "When provided, checkpoints are discovered in numeric step order and "
            "--model-list is ignored."
        ),
    )
    parser.add_argument(
        "--checkpoint-name-prefix",
        default="",
        help="Optional model-name prefix for --checkpoint-run-dir rows.",
    )
    parser.add_argument("--data-root", default="/dataMeR1/phil/data")
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--tasks", default="neighbor_matching,temporal_link_prediction,classification")
    parser.add_argument("--shots", default="0,3,10")
    parser.add_argument("--python", default="python3")
    parser.add_argument(
        "--gpus",
        default="",
        help=(
            "Comma-separated physical GPU ids for parallel eval, e.g. 1,3. "
            "Each subprocess sees one GPU and uses --device 0."
        ),
    )
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--eval-step", type=int, default=2000)
    parser.add_argument("--checkpoint-step", type=int, default=2000)
    parser.add_argument("--nm-n-way", type=int, default=3,
                        help="N-way for NM eval episodes. 3-way 3-shot is near-ceiling; "
                             "use 30 for a more discriminative comparison.")
    parser.add_argument("--nm-dataset-len-cap", type=int, default=5000)
    parser.add_argument("--lp-dataset-len-cap", type=int, default=2500)
    parser.add_argument("--pl-dataset-len-cap", type=int, default=2000)
    parser.add_argument("--parquet-val-cap", type=int, default=500)
    parser.add_argument("--parquet-test-cap", type=int, default=500)
    parser.add_argument("--lp-n-query", type=int, default=12)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("extra_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.extra_args and args.extra_args[0] == "--":
        args.extra_args = args.extra_args[1:]

    selected_datasets = parse_csv(args.datasets)
    selected_tasks = [TASK_ALIASES[task] for task in parse_csv(args.tasks)]
    shots = parse_csv(args.shots)
    if args.checkpoint_run_dir:
        models = discover_checkpoint_run_dir(
            Path(args.checkpoint_run_dir),
            args.checkpoint_name_prefix or None,
        )
    else:
        models = parse_model_list(Path(args.model_list))
    print(
        "[progress] eval selection "
        f"datasets={selected_datasets} tasks={selected_tasks} shots={shots} "
        f"models={len(models)}",
        flush=True,
    )
    torch_mod = load_torch()

    jobs: list[tuple[list[str], str]] = []
    for dataset_name in selected_datasets:
        if dataset_name not in DATASETS:
            raise ValueError(f"Unknown dataset {dataset_name!r}. Known: {sorted(DATASETS)}")
        dataset = DATASETS[dataset_name]
        graph_path = Path(args.data_root) / dataset.root_name / "graphs" / dataset.graph_filename
        if not graph_path.exists():
            print(f"[skip] {dataset.name}: missing graph {graph_path}", flush=True)
            continue
        print(f"[progress] inspecting {dataset.name} graph={graph_path}", flush=True)
        info = graph_info(torch_mod, graph_path)
        print(
            f"[progress] {dataset.name}: x_dim={info['x_dim']} "
            f"labels={info['has_labels']} label_count={info['label_count']} "
            f"future_edges={info['has_future_edges']}",
            flush=True,
        )
        if info["x_dim"] != 768:
            print(f"[warn] {dataset.name}: graph x_dim={info['x_dim']} expected 768", flush=True)

        for model_name, ckpt_path in models:
            if not Path(ckpt_path).exists():
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
            for task in selected_tasks:
                if task == "temporal_link_prediction":
                    if not dataset.supports_lp or not info["has_future_edges"]:
                        print(f"[skip] {dataset.name}: no temporal LP target edges", flush=True)
                        continue
                if task == "classification" and not info["has_labels"]:
                    print(f"[skip] {dataset.name}: no classification labels", flush=True)
                    continue
                for shot in shots:
                    cmd = build_command(
                        args,
                        dataset,
                        model_name,
                        ckpt_path,
                        task,
                        shot,
                        classification_n_way=info["label_count"] if task == "classification" else None,
                    )
                    label = f"dataset={dataset.name} model={model_name} task={task} shot={shot}"
                    jobs.append((cmd, label))

    return run_job_queue(
        jobs,
        gpus=parse_csv(args.gpus),
        dry_run=args.dry_run,
        continue_on_error=args.continue_on_error,
    )


if __name__ == "__main__":
    raise SystemExit(main())
