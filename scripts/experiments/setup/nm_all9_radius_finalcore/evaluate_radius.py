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
    run_name = f"radiusfc_{args.arm}_s{args.seed}_{args.training_run_stamp}"
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

    checkpoints = {step: checkpoint_path(args, step) for step in CHECKPOINT_STEPS}
    missing = [str(path) for path in checkpoints.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing training checkpoints:\n" + "\n".join(missing))

    primary_panels = [panel for panel in PANELS if panel.primary]
    first_panel = primary_panels[0]
    first_step = CHECKPOINT_STEPS[0]
    first_params = resolved_params(
        args, first_panel.panel_id, "val", checkpoints[first_step]
    )
    dataset = load_dataset(first_params)
    validations = []
    for panel in primary_panels:
        for step in CHECKPOINT_STEPS:
            params = resolved_params(args, panel.panel_id, "val", checkpoints[step])
            validations.append(
                evaluate_checkpoint(
                    dataset, params, checkpoints[step], "val", step, panel.panel_id
                )
            )

    selection_summary = select_validation_checkpoint(validations)
    selected_step = int(selection_summary["selected"]["checkpoint_step"])
    payload = {
        "protocol": "radius_panel_macro_validation_then_frozen_test",
        "created_utc": utc_now(),
        "evaluation_commit": git_commit(),
        "arm": args.arm,
        "training_radii": list(get_arm(args.arm).radii),
        "seed": args.seed,
        "training_run_stamp": args.training_run_stamp,
        "checkpoint_steps": list(CHECKPOINT_STEPS),
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
    parser.add_argument("--evaluation-state-root", required=True, type=Path)
    parser.add_argument("--evaluation-log-root", required=True, type=Path)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--evaluation-run-stamp", default="20260807")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    get_arm(args.arm)
    if args.phase == "validation":
        run_validation(args)
    else:
        run_test(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
