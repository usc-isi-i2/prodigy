"""Verify exact inputs and intervention accounting for the bounded HK training pair."""
import argparse
import gzip
import json
from pathlib import Path

import torch

from scripts.experiments.setup.target_performance_mechanisms.verify_member_training import model_digest


ALLOWED_CONFIG_DIFFERENCES = {
    "config", "exp_name", "prefix", "log_dir", "state_dir",
    "neighbor_matching_overlap_aware_support_edges",
}
MASK_FIELDS = {"overlap_masked_negative_edges", "overlap_message_keep_sha256"}


def read_records(path):
    with gzip.open(path, "rt") as handle:
        return [json.loads(line) for line in handle]


def checkpoint(run, step):
    matches = list(run.glob(f"state/**/checkpoint/state_dict_{step}.ckpt"))
    if len(matches) != 1:
        raise ValueError(f"expected one step-{step} checkpoint under {run}, found {len(matches)}")
    return matches[0]


def config(run):
    matches = list(run.glob("log/**/data/effective_config.json"))
    if not matches:
        matches = list(run.glob("log/**/effective_config.json"))
    if len(matches) != 1:
        raise ValueError(f"expected one effective config under {run}, found {len(matches)}")
    return json.loads(matches[0].read_text())


def records(run):
    matches = list(run.glob("log/**/data/consumed_episodes.jsonl.gz"))
    if len(matches) != 1:
        raise ValueError(f"expected one consumed stream under {run}, found {len(matches)}")
    return read_records(matches[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--treatment", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-steps", type=int, default=2500)
    args = parser.parse_args()

    base_config, treated_config = config(args.baseline), config(args.treatment)
    differing = {key for key in set(base_config) | set(treated_config) if base_config.get(key) != treated_config.get(key)}
    if not differing <= ALLOWED_CONFIG_DIFFERENCES:
        raise ValueError(f"uncontrolled config differences: {sorted(differing - ALLOWED_CONFIG_DIFFERENCES)}")
    if base_config["neighbor_matching_overlap_aware_support_edges"] or not treated_config["neighbor_matching_overlap_aware_support_edges"]:
        raise ValueError("baseline/treatment flags are reversed")

    base, treated = records(args.baseline), records(args.treatment)
    if len(base) != args.expected_steps or len(treated) != args.expected_steps:
        raise ValueError("incomplete consumed episode stream")
    mismatch = []
    masked = []
    for left, right in zip(base, treated):
        stripped = {key: value for key, value in right.items() if key not in MASK_FIELDS}
        if left != stripped:
            mismatch.append(left["step"])
        count = right.get("overlap_masked_negative_edges")
        if count is None or "overlap_message_keep_sha256" not in right:
            raise ValueError("treatment lacks overlap-mask audit")
        masked.append(int(count))
    if mismatch:
        raise ValueError(f"training inputs differ at steps {mismatch[:10]}")
    if any(key in row for row in base for key in MASK_FIELDS):
        raise ValueError("baseline unexpectedly contains an overlap mask")

    initial = {}
    terminal = {}
    for name, run in (("baseline", args.baseline), ("treatment", args.treatment)):
        initial_state = torch.load(checkpoint(run, 0), map_location="cpu", weights_only=False)["model"]
        terminal_state = torch.load(checkpoint(run, args.expected_steps), map_location="cpu", weights_only=False)["model"]
        initial[name] = model_digest(initial_state)
        terminal[name] = model_digest(terminal_state)
    if initial["baseline"] != initial["treatment"]:
        raise ValueError("initial model states differ")

    report = {
        "complete": True,
        "expected_steps": args.expected_steps,
        "controlled_config_differences": sorted(differing),
        "exact_consumed_episode_inputs": True,
        "initial_model_sha256": initial["baseline"],
        "terminal_model_sha256": terminal,
        "terminal_models_differ": terminal["baseline"] != terminal["treatment"],
        "masked_negative_edges": {
            "total": sum(masked),
            "mean_per_batch": sum(masked) / len(masked),
            "min_per_batch": min(masked),
            "max_per_batch": max(masked),
            "batches_with_any": sum(value > 0 for value in masked),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
