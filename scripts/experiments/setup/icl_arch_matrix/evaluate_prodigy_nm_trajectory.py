#!/usr/bin/env python3
"""Evaluate one PRODIGY specialist's native NM trajectory on its source graph."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
import wandb

from experiments.params import get_params
from experiments.run_single_experiment import load_dataset, seed_everything
from experiments.trainer import TrainerFS
from scripts.experiments.setup.final_core.core_plan import build_models
from scripts.experiments.setup.icl_arch_matrix.common_protocol import reset_episode_rng


def parse_steps(text: str) -> list[int]:
    steps = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not steps or len(steps) != len(set(steps)):
        raise ValueError(f"checkpoint steps must be a non-empty unique list, got {text!r}")
    unsupported = sorted(set(steps) - {0, 20, 60, 100, 300, 900, 2000, 2500})
    if unsupported:
        raise ValueError(f"unsupported checkpoint steps: {unsupported}")
    return sorted(steps)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("prodigy_training.yaml"))
    parser.add_argument("--state-root", required=True, type=Path)
    parser.add_argument("--eval-state-root", required=True, type=Path)
    parser.add_argument("--log-root", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--run-stamp", default="20260810")
    parser.add_argument("--eval-run-stamp", default="20260815")
    parser.add_argument("--checkpoint-steps", default="0,20,60,100")
    parser.add_argument("--device", default="0")
    parser.add_argument("--eval-episode-seed-offset", type=int, default=0)
    parser.add_argument("--training-seed", type=int, default=0)
    parser.add_argument(
        "--checkpoint-layout",
        choices=("architecture-matrix", "saturation"),
        default="architecture-matrix",
    )
    return parser.parse_args()


def specialist(model_id: str):
    matches = [model for model in build_models() if model.model_id == model_id]
    if len(matches) != 1 or len(matches[0].sources) != 1:
        raise ValueError(f"expected one registered single-source model, got {model_id!r}")
    return matches[0]


def checkpoint_path(args: argparse.Namespace, model_id: str, step: int) -> Path | None:
    if step == 0:
        return None
    run_name = (
        f"archmatrix_prodigy_{model_id}_s0_{args.run_stamp}"
        if args.checkpoint_layout == "architecture-matrix"
        else f"archsat_prodigy_{model_id}_s{args.training_seed}_{args.run_stamp}"
    )
    return (
        args.state_root
        / "prodigy"
        / run_name
        / "checkpoint"
        / f"state_dict_{step}.ckpt"
    )


def resolved_params(
    args: argparse.Namespace,
    *,
    model_id: str,
    source: str,
    step: int,
    checkpoint: Path | None,
) -> dict:
    argv = [
        "--config", str(args.config),
        "--device", str(args.device),
        "--seed", str(args.training_seed),
        "--eval_episode_seed_offset", str(args.eval_episode_seed_offset),
        "--prefix", f"archmatrix_prodigy_nm_{model_id}_step{step}",
        "--timestamp", args.eval_run_stamp,
        "--state_dir", str(args.eval_state_root),
        "--log_dir", str(args.log_root),
        "--override_log", "True",
        "--neighbor_sampling_source_subset", source,
        "--eval_only", "True",
        "--eval_only_split", "test",
        "--eval_test_before_train", "False",
        "--eval_val_before_train", "False",
    ]
    if checkpoint is not None:
        argv.extend(["--pretrained_model_run", str(checkpoint)])
    return get_params(argv)


def main() -> int:
    args = parse_args()
    torch.set_num_threads(16)
    model = specialist(args.model_id)
    source = model.sources[0]
    steps = parse_steps(args.checkpoint_steps)
    checkpoints = {step: checkpoint_path(args, model.model_id, step) for step in steps}
    missing = [str(path) for path in checkpoints.values() if path is not None and not path.is_file()]
    if missing:
        raise FileNotFoundError("missing checkpoints:\n" + "\n".join(missing))
    if args.results.exists():
        raise FileExistsError(f"refusing to overwrite results: {args.results}")

    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.eval_state_root.mkdir(parents=True, exist_ok=True)
    args.log_root.mkdir(parents=True, exist_ok=True)

    base_params = resolved_params(
        args,
        model_id=model.model_id,
        source=source,
        step=steps[0],
        checkpoint=checkpoints[steps[0]],
    )
    seed_everything(base_params)
    dataset = load_dataset(base_params)

    with args.results.open("w", encoding="utf-8") as handle:
        for step in steps:
            checkpoint = checkpoints[step]
            params = resolved_params(
                args,
                model_id=model.model_id,
                source=source,
                step=step,
                checkpoint=checkpoint,
            )
            seed_everything(params)
            trainer = TrainerFS(dataset, params)
            try:
                reset_episode_rng()
                trainer.model.eval()
                with torch.no_grad():
                    loss, score, score_std, aux_loss, ranks = trainer.do_eval(
                        trainer.test_dataloader, split_name="test", step=step
                    )
                metrics_path = Path(trainer.logging_dir) / f"metrics_test_step{step}.json"
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                row = {
                    "architecture": "prodigy",
                    "model_id": model.model_id,
                    "sources": list(model.sources),
                    "seed": args.training_seed,
                    "training_seed": args.training_seed,
                    "eval_episode_seed_offset": args.eval_episode_seed_offset,
                    "checkpoint_step": step,
                    "checkpoint": str(checkpoint) if checkpoint is not None else None,
                    "baseline": "random_init" if step == 0 else "pretrained",
                    "task": "neighbor_matching",
                    "dataset": source,
                    "n_way": int(params["n_way"]),
                    "n_shot": int(params["n_shots"]),
                    "n_query": int(params["n_query"]),
                    "episodes": int(params["test_len_cap"]) * int(params["batch_size"]),
                    "score": float(score),
                    "score_std": float(score_std),
                    "loss": float(loss),
                    "aux_loss": float(aux_loss),
                    **{key.removeprefix("test_"): value for key, value in metrics.items()},
                }
                if ranks:
                    row["ranks"] = {key: float(value) for key, value in ranks.items()}
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                handle.flush()
                print(json.dumps(row, sort_keys=True), flush=True)
            finally:
                wandb.finish()
                del trainer
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
