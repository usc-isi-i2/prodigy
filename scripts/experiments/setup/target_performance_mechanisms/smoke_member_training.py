"""Tiny CPU training integration check. Run on Tucker, never as research evidence."""
import argparse
import gzip
import json
from pathlib import Path

import torch

from experiments.params import get_params
from experiments.run_single_experiment import seed_everything
from experiments.tests.test_shared_graph_training import tiny_dataset
from experiments.trainer import TrainerFS
from scripts.experiments.setup.target_performance_mechanisms.make_member_configs import POLICIES


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    records, initials = {}, {}
    for policy in POLICIES:
        params = get_params([
            "--config", "scripts/experiments/setup/final_core/training.yaml", "--device", "123",
            "--neighbor_matching_member_policy", policy, "--neighbor_matching_member_seed", "991",
            "--neighbor_sampling_source_subset", "a", "--train_episode_audit", "True",
            "--emb_dim", "16", "--n_way", "2", "--batch_size", "2", "--workers", "0",
            "--epochs", "1", "--dataset_len_cap", "3", "--val_len_cap", "1", "--test_len_cap", "1",
            "--checkpoint_steps", "0,3", "--eval_after_train", "False",
            "--eval_val_before_train", "False", "--eval_test_before_train", "False",
            "--state_dir", str(args.output / "state"), "--log_dir", str(args.output / "log"),
            "--prefix", f"toy_{policy}", "--timestamp", "fixed"])
        dataset = tiny_dataset()
        dataset.nm_validation_neighbor_sampler = dataset.neighbor_sampler
        dataset.nm_test_neighbor_sampler = dataset.neighbor_sampler
        dataset.nm_holdout_neighbor_sampler = dataset.neighbor_sampler
        seed_everything(params)
        trainer = TrainerFS(dataset, params)
        trainer.train()
        with gzip.open(Path(trainer.logging_dir) / "consumed_episodes.jsonl.gz", "rt") as f:
            rows = [json.loads(line) for line in f]
        if [r["step"] for r in rows] != [1, 2, 3]:
            raise RuntimeError("consumed step audit is incomplete")
        records[policy] = rows
        initials[policy] = torch.load(Path(trainer.ckpt_dir) / "state_dict_0.ckpt", weights_only=True)["model"]
        terminal = torch.load(Path(trainer.ckpt_dir) / "state_dict_3.ckpt", weights_only=True)["model"]
        if not all(torch.isfinite(v).all() for v in terminal.values()):
            raise RuntimeError("nonfinite trained checkpoint")
        if all(torch.equal(v, terminal[k]) for k, v in initials[policy].items()):
            raise RuntimeError("weights did not update")
    reference = initials[POLICIES[0]]
    for policy in POLICIES:
        if not all(torch.equal(v, initials[policy][k]) for k, v in reference.items()):
            raise RuntimeError("factorial arms did not share initialization")
        if [r["anchor_sha256"] for r in records[policy]] != [r["anchor_sha256"] for r in records[POLICIES[0]]]:
            raise RuntimeError("consumed anchors drifted across treatments")
    for prefix in ("lowest", "uniform"):
        if [r["member_set_sha256"] for r in records[prefix + "_sorted"]] != [r["member_set_sha256"] for r in records[prefix + "_shuffled"]]:
            raise RuntimeError("role-only control changed retained member sets")
    result = {"models": 4, "steps_per_model": 3, "finite_updated_weights": True,
              "same_initialization": True, "same_consumed_anchors": True,
              "same_retention_sets_across_role_treatments": True, "research_result": False}
    (args.output / "DONE.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
