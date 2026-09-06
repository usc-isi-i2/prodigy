"""Tiny paired integration on Tucker; never substantive evidence."""
import argparse
import json
from pathlib import Path

import torch

from experiments.params import get_params
from experiments.run_single_experiment import seed_everything
from experiments.tests.test_shared_graph_training import tiny_dataset
from experiments.trainer import TrainerFS
from .readout_training_constraint import CONDITIONS, configure_trainer
from .verify_member_training import audit_consumed, model_digest
from .verify_readout_training import input_hashes, verify_checkpoint_states


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    rows = []
    for condition in CONDITIONS:
        params = get_params([
            "--config", "scripts/experiments/setup/final_core/training.yaml", "--device", "123",
            "--neighbor_matching_member_policy", "lowest_sorted", "--neighbor_matching_member_seed", "991",
            "--neighbor_sampling_source_subset", "a", "--train_episode_audit", "True",
            "--emb_dim", "16", "--n_way", "2", "--batch_size", "2", "--workers", "0",
            "--epochs", "1", "--dataset_len_cap", "3", "--val_len_cap", "1", "--test_len_cap", "1",
            "--checkpoint_steps", "0,3", "--eval_after_train", "False",
            "--eval_val_before_train", "False", "--eval_test_before_train", "False",
            "--state_dir", str(args.output / "state"), "--log_dir", str(args.output / "log"),
            "--prefix", f"toy_readout_{condition}", "--timestamp", "fixed"])
        params["readout_training_condition"] = condition
        dataset = tiny_dataset()
        dataset.nm_validation_neighbor_sampler = dataset.neighbor_sampler
        dataset.nm_test_neighbor_sampler = dataset.neighbor_sampler
        dataset.nm_holdout_neighbor_sampler = dataset.neighbor_sampler
        seed_everything(params)
        trainer = TrainerFS(dataset, params)
        configure_trainer(trainer)
        trainer.train()
        states = {step: torch.load(Path(trainer.ckpt_dir) / f"state_dict_{step}.ckpt", weights_only=True)["model"] for step in (0, 3)}
        verify_checkpoint_states(states[0], states, condition)
        if trainer.training_constraint_receipt["steps_checked"] != 3:
            raise ValueError("missing per-update constraint checks")
        rows.append({"initial": model_digest(states[0]),
                     "inputs": input_hashes(Path(trainer.logging_dir) / "constraint_inputs.jsonl.gz", 3),
                     "consumed": audit_consumed(Path(trainer.logging_dir) / "consumed_episodes.jsonl.gz", params, 3)})
    if rows[0] != rows[1]:
        raise ValueError("paired toy initialization or training examples differ")
    result = {"research_result": False, "models": 2, "steps_per_model": 3,
              "exact_initial_and_inputs": True, "frozen_readout_and_other_updates_verified": True}
    (args.output / "DONE.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
