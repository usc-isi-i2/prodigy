#!/usr/bin/env python3
"""Validation-select and test the three final-core radius arms without test peeking."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

import torch
import wandb

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(HERE))

from experiments.params import get_params  # noqa: E402
from experiments.run_single_experiment import load_dataset, seed_everything  # noqa: E402
from experiments.trainer import TrainerFS, _to_float  # noqa: E402
from radius_plan import (  # noqa: E402
    ARMS,
    CHECKPOINT_STEPS,
    PANELS,
    get_arm,
    get_panel,
    select_validation_checkpoint,
)
from shared_eval import score_models_on_shared_batches  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def checkpoint_path(args: argparse.Namespace, step: int) -> Path:
    run_name = f"{args.training_prefix}_{args.arm}_s{args.seed}_{args.training_run_stamp}"
    return (
        args.training_state_root
        / run_name
        / "checkpoint"
        / f"state_dict_{step}.ckpt"
    )


def resolved_params(
    args: argparse.Namespace,
    panel_id: str,
    split: str,
    checkpoint: Path,
) -> dict[str, Any]:
    arm = get_arm(args.arm)
    panel = get_panel(panel_id)
    radii = ",".join(panel.radii)
    weights = ",".join("1" for _ in panel.radii)
    argv = [
        "--config", str(arm.config),
        "--device", str(args.device),
        "--seed", str(args.seed),
        "--prefix", f"radiusfc_eval_{args.arm}_s{args.seed}_{panel_id}_{split}",
        "--timestamp", args.evaluation_run_stamp,
        "--state_dir", str(args.evaluation_state_root),
        "--log_dir", str(args.evaluation_log_root),
        "--neighbor_sampling_center_radii", radii,
        "--neighbor_sampling_center_radius_weights", weights,
        "--neighbor_sampling_episode_source", "graph_id" if panel.source_confined else "",
        "--neighbor_sampling_episode_source_weighting", "balanced",
        "--neighbor_sampling_strata", "",
        "--neighbor_sampling_cross_source_prob", "0.0",
        "--pretrained_model_run", str(checkpoint),
        "--eval_only", "True",
        "--eval_only_split", split,
        "--eval_test_before_train", "False",
        "--eval_val_before_train", "False",
    ]
    if args.eval_batch_count is not None:
        argv.extend([
            "--val_len_cap", str(args.eval_batch_count),
            "--test_len_cap", str(args.eval_batch_count),
        ])
    if args.workers is not None:
        argv.extend(["--workers", str(args.workers)])
    return get_params(argv)


def evaluate_checkpoint(
    dataset,
    base_params: dict[str, Any],
    checkpoint: Path,
    split: str,
    step: int,
    panel_id: str,
) -> dict[str, Any]:
    params = deepcopy(base_params)
    params["pretrained_model_run"] = str(checkpoint)
    params["eval_only_split"] = split
    params["exp_name"] = (
        f"radiusfc_eval_{params['prefix'].removeprefix('radiusfc_eval_')}"
        f"_step{step}_{utc_now().replace(':', '').replace('+', '_')}"
    )
    seed_everything(params)
    trainer = TrainerFS(dataset, params)
    try:
        dataloader = trainer.val_dataloader if split == "val" else trainer.test_dataloader
        if dataloader is None:
            raise RuntimeError(f"{split} dataloader was not built")
        with torch.no_grad():
            trainer.model.eval()
            loss, score, score_std, aux_loss, ranks = trainer.do_eval(
                dataloader, split_name=split, step=step
            )
        payload = {
            "checkpoint": str(checkpoint),
            "checkpoint_step": step,
            "panel": panel_id,
            "split": split,
            "score": _to_float(score),
            "score_std": _to_float(score_std),
            "loss": _to_float(loss),
            "aux_loss": _to_float(aux_loss),
        }
        if ranks:
            payload["ranks"] = {
                key: _to_float(value) for key, value in ranks.items()
            }
        if not all(
            math.isfinite(float(payload[key]))
            for key in ("score", "score_std", "loss")
        ):
            raise ValueError(f"non-finite evaluation result: {payload}")
        return payload
    finally:
        wandb.finish()
        del trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def evaluate_checkpoints_shared(
    dataset,
    base_params: dict[str, Any],
    checkpoints: dict[int, Path],
    checkpoint_steps: tuple[int, ...],
    split: str,
    panel_id: str,
) -> list[dict[str, Any]]:
    """Score all checkpoints on one sampled batch stream.

    Episode construction and PyG collation dominate this evaluation.  The eval
    stream is deterministic for a (split, panel), so rebuilding it once per
    checkpoint only repeats CPU work.  Keep four tiny checkpoint models on the
    same GPU and forward every sampled batch through each model instead.
    """
    steps = list(checkpoint_steps)
    first_step = steps[0]
    params = deepcopy(base_params)
    params["pretrained_model_run"] = str(checkpoints[first_step])
    params["eval_only_split"] = split
    params["exp_name"] = (
        f"radiusfc_eval_{params['prefix'].removeprefix('radiusfc_eval_')}"
        f"_shared_{utc_now().replace(':', '').replace('+', '_')}"
    )
    seed_everything(params)
    trainer = TrainerFS(dataset, params)
    models: dict[int, torch.nn.Module] = {first_step: trainer.model}
    try:
        if trainer.calc_ranks:
            raise ValueError("shared checkpoint evaluation does not support calc_ranks")
        if params.get("export_predictions", False):
            raise ValueError(
                "shared checkpoint evaluation requires export_predictions=False"
            )
        dataloader = (
            trainer.val_dataloader if split == "val" else trainer.test_dataloader
        )
        if dataloader is None:
            raise RuntimeError(f"{split} dataloader was not built")

        for step in steps[1:]:
            model = deepcopy(trainer.model)
            state_dict = TrainerFS._torch_load_checkpoint(
                checkpoints[step], map_location=trainer.device
            )
            if "model" not in state_dict:
                raise KeyError(f"checkpoint has no model state: {checkpoints[step]}")
            model.load_state_dict(state_dict["model"], strict=False)
            model.eval()
            models[step] = model
        for model in models.values():
            model.eval()

        metrics = score_models_on_shared_batches(
            models=models,
            steps=steps,
            dataloader=dataloader,
            device=trainer.device,
            get_loss_and_score=trainer.get_loss_and_acc,
            get_aux_loss=trainer.get_aux_loss,
        )
        results = []
        for step in steps:
            payload = {
                "checkpoint": str(checkpoints[step]),
                "checkpoint_step": step,
                "panel": panel_id,
                "split": split,
                **metrics[step],
            }
            if not all(
                math.isfinite(float(payload[key]))
                for key in ("score", "score_std", "loss")
            ):
                raise ValueError(f"non-finite evaluation result: {payload}")
            results.append(payload)
        return results
    finally:
        wandb.finish()
        models.clear()
        del trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def result_dir(args: argparse.Namespace) -> Path:
    return args.results_root / f"seed_{args.seed}" / args.arm


def verify_selection(selection: dict[str, Any], args: argparse.Namespace) -> None:
    expected = {
        "arm": args.arm,
        "seed": args.seed,
        "training_run_stamp": args.training_run_stamp,
        "protocol": "radius_panel_macro_validation_then_frozen_test",
    }
    for key, value in expected.items():
        if selection.get(key) != value:
            raise ValueError(
                f"selection {key} mismatch: expected {value!r}, got {selection.get(key)!r}"
            )


def run_validation(args: argparse.Namespace) -> Path:
    selection_path = result_dir(args) / "selection.json"
    if selection_path.exists():
        verify_selection(
            json.loads(selection_path.read_text(encoding="utf-8")), args
        )
        print(f"SKIP frozen selection {selection_path}")
        return selection_path

    checkpoint_steps = args.checkpoint_steps_eval
    checkpoints = {step: checkpoint_path(args, step) for step in checkpoint_steps}
    missing = [str(path) for path in checkpoints.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing training checkpoints:\n" + "\n".join(missing))

    primary_panels = [panel for panel in PANELS if panel.primary]
    first_panel = PANELS[0]
    first_step = checkpoint_steps[0]
    first_params = resolved_params(
        args, first_panel.panel_id, "val", checkpoints[first_step]
    )
    dataset = load_dataset(first_params)
    validations = []
    # Record trajectories on all four panels. The within-source compatibility
    # panel remains excluded from checkpoint selection, but is no longer test-only.
    for panel in PANELS:
        params = resolved_params(args, panel.panel_id, "val", checkpoints[first_step])
        if args.validation_mode == "shared":
            validations.extend(
                evaluate_checkpoints_shared(
                    dataset,
                    params,
                    checkpoints,
                    checkpoint_steps,
                    "val",
                    panel.panel_id,
                )
            )
        else:
            for step in checkpoint_steps:
                step_params = resolved_params(
                    args, panel.panel_id, "val", checkpoints[step]
                )
                validations.append(
                    evaluate_checkpoint(
                        dataset,
                        step_params,
                        checkpoints[step],
                        "val",
                        step,
                        panel.panel_id,
                    )
                )

    primary_validations = [
        row for row in validations if get_panel(str(row["panel"])).primary
    ]
    selection_summary = select_validation_checkpoint(
        primary_validations, checkpoint_steps=checkpoint_steps
    )
    selected_step = int(selection_summary["selected"]["checkpoint_step"])
    payload = {
        "protocol": "radius_panel_macro_validation_then_frozen_test",
        "created_utc": utc_now(),
        "evaluation_commit": git_commit(),
        "arm": args.arm,
        "training_radii": list(get_arm(args.arm).radii),
        "seed": args.seed,
        "training_run_stamp": args.training_run_stamp,
        "checkpoint_steps": list(checkpoint_steps),
        "validation_panels": [panel.panel_id for panel in PANELS],
        "primary_panels": [panel.panel_id for panel in primary_panels],
        "validation_results": validations,
        "checkpoint_summaries": selection_summary["checkpoint_summaries"],
        "selection_rule": (
            "maximum macro mean over radius2, radius3, and global validation panels; "
            "earliest checkpoint breaks exact ties"
        ),
        "selected_checkpoint_step": selected_step,
        "selected_checkpoint": str(checkpoints[selected_step]),
    }
    atomic_json(selection_path, payload)
    print(selection_path)
    return selection_path


def run_test(args: argparse.Namespace) -> Path:
    directory = result_dir(args)
    selection_path = directory / "selection.json"
    result_path = directory / "result.json"
    if not selection_path.is_file():
        raise FileNotFoundError(
            f"test is locked until validation selection exists: {selection_path}"
        )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    verify_selection(selection, args)
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("selection_created_utc") != selection["created_utc"]:
            raise ValueError("test result does not match the current frozen selection")
        print(f"SKIP complete test {result_path}")
        return result_path

    step = int(selection["selected_checkpoint_step"])
    checkpoint = Path(selection["selected_checkpoint"])
    expected_checkpoint = checkpoint_path(args, step)
    if checkpoint != expected_checkpoint or not checkpoint.is_file():
        raise ValueError(
            f"frozen checkpoint is missing or unexpected: {checkpoint}; "
            f"expected {expected_checkpoint}"
        )

    first_panel = PANELS[0]
    first_params = resolved_params(args, first_panel.panel_id, "test", checkpoint)
    dataset = load_dataset(first_params)
    test_results = []
    for panel in PANELS:
        params = resolved_params(args, panel.panel_id, "test", checkpoint)
        test_results.append(
            evaluate_checkpoint(
                dataset, params, checkpoint, "test", step, panel.panel_id
            )
        )
    payload = {
        "protocol": "radius_panel_macro_validation_then_frozen_test",
        "created_utc": utc_now(),
        "evaluation_commit": git_commit(),
        "arm": args.arm,
        "training_radii": list(get_arm(args.arm).radii),
        "seed": args.seed,
        "training_run_stamp": args.training_run_stamp,
        "selection_created_utc": selection["created_utc"],
        "selected_checkpoint_step": step,
        "selected_checkpoint": str(checkpoint),
        "test_results": test_results,
    }
    atomic_json(result_path, payload)
    print(result_path)
    return result_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=["validation", "test"])
    parser.add_argument("--arm", required=True, choices=[arm.arm_id for arm in ARMS])
    parser.add_argument("--seed", required=True, type=int, choices=[0, 1, 2])
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--training-state-root", required=True, type=Path)
    parser.add_argument("--training-run-stamp", default="20260807")
    parser.add_argument("--training-prefix", default="radiusfc")
    parser.add_argument(
        "--checkpoint-steps",
        default=",".join(str(step) for step in CHECKPOINT_STEPS),
        help="Comma-separated completed-step checkpoints to validate.",
    )
    parser.add_argument("--evaluation-state-root", required=True, type=Path)
    parser.add_argument("--evaluation-log-root", required=True, type=Path)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--evaluation-run-stamp", default="20260807")
    parser.add_argument(
        "--validation-mode", choices=["shared", "legacy"], default="shared"
    )
    parser.add_argument(
        "--eval-batch-count",
        type=int,
        help="Override val/test batch count (intended for equivalence smoke tests).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="Override dataloader worker count.",
    )
    args = parser.parse_args()
    try:
        args.checkpoint_steps_eval = tuple(
            int(part.strip())
            for part in args.checkpoint_steps.split(",")
            if part.strip()
        )
    except ValueError as exc:
        parser.error(f"invalid --checkpoint-steps: {exc}")
    if not args.checkpoint_steps_eval or any(
        step <= 0 for step in args.checkpoint_steps_eval
    ):
        parser.error("--checkpoint-steps must contain positive integers")
    if tuple(sorted(set(args.checkpoint_steps_eval))) != args.checkpoint_steps_eval:
        parser.error("--checkpoint-steps must be unique and increasing")
    return args


def main() -> int:
    args = parse_args()
    if args.eval_batch_count is not None and args.eval_batch_count <= 0:
        raise ValueError("--eval-batch-count must be positive")
    if args.workers is not None and args.workers < 0:
        raise ValueError("--workers must be non-negative")
    get_arm(args.arm)
    if args.phase == "validation":
        run_validation(args)
    else:
        run_test(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
