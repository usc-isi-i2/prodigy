#!/usr/bin/env python3
"""Select one final-core checkpoint on validation, then evaluate it on test.

The two phases are deliberately separate.  ``validation`` writes selection.json
without iterating the test dataloader.  ``test`` refuses to run until that frozen
selection exists and never compares alternative checkpoints on the test split.
"""

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

from core_plan import build_models, select_validation_checkpoint  # noqa: E402
from experiments.params import get_params  # noqa: E402
from experiments.run_single_experiment import load_dataset, seed_everything  # noqa: E402
from experiments.trainer import TrainerFS, _to_float  # noqa: E402


CHECKPOINT_STEPS = (100, 300, 900, 2500)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def registered_model(model_id: str, sources: str):
    matches = [model for model in build_models() if model.model_id == model_id]
    if len(matches) != 1:
        raise ValueError(f"unknown final-core model {model_id!r}")
    model = matches[0]
    supplied = tuple(part for part in sources.split(",") if part)
    if supplied != model.sources:
        raise ValueError(
            f"source mismatch for {model_id}: registered={model.sources}, supplied={supplied}"
        )
    return model


def checkpoint_path(args: argparse.Namespace, step: int) -> Path:
    run_name = f"finalcore_{args.model_id}_s{args.seed}_{args.training_run_stamp}"
    return args.training_state_root / run_name / "checkpoint" / f"state_dict_{step}.ckpt"


def resolved_params(args: argparse.Namespace, split: str, checkpoint: Path) -> dict[str, Any]:
    argv = [
        "--config", str(args.config),
        "--device", str(args.device),
        "--seed", str(args.seed),
        "--prefix", f"finalcore_eval_{args.model_id}_s{args.seed}_{split}",
        "--timestamp", args.evaluation_run_stamp,
        "--state_dir", str(args.evaluation_state_root),
        "--log_dir", str(args.evaluation_log_root),
        "--neighbor_sampling_source_subset", args.sources,
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
) -> dict[str, Any]:
    params = deepcopy(base_params)
    params["pretrained_model_run"] = str(checkpoint)
    params["eval_only_split"] = split
    params["exp_name"] = (
        f"finalcore_eval_{base_params['prefix'].removeprefix('finalcore_eval_')}"
        f"_step{step}_{split}_{utc_now().replace(':', '').replace('+', '_')}"
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
            "split": split,
            "score": _to_float(score),
            "score_std": _to_float(score_std),
            "loss": _to_float(loss),
            "aux_loss": _to_float(aux_loss),
        }
        if ranks:
            payload["ranks"] = {key: _to_float(value) for key, value in ranks.items()}
        if not all(math.isfinite(float(payload[key])) for key in ("score", "score_std", "loss")):
            raise ValueError(f"non-finite evaluation result: {payload}")
        return payload
    finally:
        wandb.finish()
        del trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def verify_selection(selection: dict[str, Any], args: argparse.Namespace) -> None:
    expected = {
        "model_id": args.model_id,
        "seed": args.seed,
        "sources": args.sources.split(","),
        "training_run_stamp": args.training_run_stamp,
    }
    for key, value in expected.items():
        if selection.get(key) != value:
            raise ValueError(f"selection {key} mismatch: expected {value!r}, got {selection.get(key)!r}")
    if selection.get("protocol") != "validation_select_then_single_test":
        raise ValueError("selection file has an unknown protocol")


def run_validation(args: argparse.Namespace, model) -> Path:
    result_dir = args.results_root / f"seed_{args.seed}" / args.model_id
    selection_path = result_dir / "selection.json"
    if selection_path.exists():
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        verify_selection(selection, args)
        print(f"SKIP frozen selection {selection_path}")
        return selection_path

    checkpoints = {step: checkpoint_path(args, step) for step in CHECKPOINT_STEPS}
    missing = [str(path) for path in checkpoints.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing training checkpoints:\n" + "\n".join(missing))

    base_params = resolved_params(args, "val", checkpoints[CHECKPOINT_STEPS[0]])
    dataset = load_dataset(base_params)
    validations = [
        evaluate_checkpoint(dataset, base_params, checkpoints[step], "val", step)
        for step in CHECKPOINT_STEPS
    ]
    # Highest validation score wins.  The earlier checkpoint is the deterministic
    # tie-break, favoring the lower-compute model without consulting test results.
    selected = select_validation_checkpoint(validations)
    payload = {
        "protocol": "validation_select_then_single_test",
        "created_utc": utc_now(),
        "evaluation_commit": git_commit(),
        "model_id": args.model_id,
        "aliases": list(model.aliases),
        "seed": args.seed,
        "sources": list(model.sources),
        "training_run_stamp": args.training_run_stamp,
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "validation_results": validations,
        "selection_rule": "maximum validation score; earliest checkpoint breaks exact ties",
        "selected_checkpoint_step": selected["checkpoint_step"],
        "selected_checkpoint": selected["checkpoint"],
    }
    atomic_json(selection_path, payload)
    print(selection_path)
    return selection_path


def run_test(args: argparse.Namespace, model) -> Path:
    result_dir = args.results_root / f"seed_{args.seed}" / args.model_id
    selection_path = result_dir / "selection.json"
    result_path = result_dir / "result.json"
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
            f"frozen selected checkpoint is missing or unexpected: {checkpoint}; "
            f"expected {expected_checkpoint}"
        )
    base_params = resolved_params(args, "test", checkpoint)
    dataset = load_dataset(base_params)
    test_result = evaluate_checkpoint(dataset, base_params, checkpoint, "test", step)
    payload = {
        "protocol": "validation_select_then_single_test",
        "created_utc": utc_now(),
        "evaluation_commit": git_commit(),
        "model_id": args.model_id,
        "aliases": list(model.aliases),
        "seed": args.seed,
        "sources": list(model.sources),
        "training_run_stamp": args.training_run_stamp,
        "selection_created_utc": selection["created_utc"],
        "selected_checkpoint_step": step,
        "selected_checkpoint": str(checkpoint),
        "test_result": test_result,
    }
    atomic_json(result_path, payload)
    print(result_path)
    return result_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=["validation", "test"])
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--sources", required=True)
    parser.add_argument("--seed", required=True, type=int, choices=[0, 1, 2])
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--config", type=Path, default=HERE / "training.yaml")
    parser.add_argument("--training-state-root", required=True, type=Path)
    parser.add_argument("--training-run-stamp", default="20260807")
    parser.add_argument("--evaluation-state-root", required=True, type=Path)
    parser.add_argument("--evaluation-log-root", required=True, type=Path)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--evaluation-run-stamp", default="20260807")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model = registered_model(args.model_id, args.sources)
    if args.phase == "validation":
        run_validation(args, model)
    else:
        run_test(args, model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
