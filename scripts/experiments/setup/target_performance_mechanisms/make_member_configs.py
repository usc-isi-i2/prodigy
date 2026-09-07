"""Generate a predeclared retention x role-order control from final-core NM."""
import argparse
from pathlib import Path

import yaml

from scripts.experiments.setup.final_core.core_plan import SOURCES


POLICIES = ("lowest_sorted", "lowest_shuffled", "uniform_sorted", "uniform_shuffled")
HERE = Path(__file__).resolve().parent


def plans(sources=("ukr_rus", "cp_hk"), seeds=(0, 1, 2)):
    if len(set(sources)) != len(sources) or not set(sources) <= set(SOURCES):
        raise ValueError("unknown or duplicate sources")
    if len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("invalid or duplicate seeds")
    base = yaml.safe_load((HERE.parent / "final_core/training.yaml").read_text())
    for source in sources:
        for seed in seeds:
            for policy in POLICIES:
                result = {**base, "seed": seed, "neighbor_sampling_source_subset": source,
                          "neighbor_matching_member_policy": policy,
                          "neighbor_matching_member_seed": 280100 + seed,
                          "train_episode_audit": True, "checkpoint_steps": "0,100,300,900,2500",
                          "eval_test_before_train": False, "eval_val_before_train": False,
                          "eval_after_train": False, "val_len_cap": 2, "test_len_cap": 2,
                          "prefix": f"memberctl_{source}_{policy}_s{seed}"}
                yield result


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--sources", default="ukr_rus,cp_hk")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--output", type=Path, default=HERE / "member_configs")
    args = p.parse_args()
    configs = list(plans(args.sources.split(","), tuple(map(int, args.seeds.split(",")))))
    args.output.mkdir(parents=True, exist_ok=False)
    for i, config in enumerate(configs):
        (args.output / f"{i:02d}_{config['prefix']}.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    print(f"Generated {len(configs)} models: {len(configs)*2500} updates, {len(configs)*10000} episodes")


if __name__ == "__main__":
    main()
