"""Summarize post-hoc saved-logit agreement; no outcome-selected checkpoints."""
import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).parent / "data"
    if not (args.input / "DONE").is_file():
        raise ValueError("incomplete cue alignment")
    cells = pd.read_json(args.input / "metrics.json")
    keys = ["stream", "source", "step", "decoder", "cue"]
    if len(cells) != 648 or cells.duplicated(keys).any():
        raise ValueError("incomplete or duplicated agreement grid")
    if set(cells.stream) != {"original", "fresh"} or set(cells.step) != {100, 300, 900, 2500}:
        raise ValueError("unexpected streams or checkpoints")
    trajectory = pd.read_csv(root / "trajectory_cells.csv")
    metric = "mean_within_episode_spearman"
    changes = cells.pivot(index=["stream", "source", "decoder", "cue"], columns="step", values=metric)
    changes["change_100_to_2500"] = changes[2500] - changes[100]
    changes = changes.reset_index()
    summary = changes.groupby(["stream", "decoder", "cue"])["change_100_to_2500"].agg(
        mean="mean", minimum="min", maximum="max", positive=lambda x: int((x > 0).sum()),
        negative=lambda x: int((x < 0).sum()), sources="count").reset_index()
    # Weight and episode provenance must match the independently validated replay.
    reference = trajectory[trajectory.dataset == "twibot20"].drop_duplicates(
        ["stream", "model_id", "decoder"])
    joined = cells.merge(reference[["stream", "model_id", "decoder", "weights_sha256", "episode_fingerprint"]],
                         on=["stream", "model_id", "decoder"], validate="many_to_one", suffixes=("", "_reference"))
    for field in ("weights_sha256", "episode_fingerprint"):
        if len(joined) != 648 or not (joined[field] == joined[f"{field}_reference"]).all():
            raise ValueError(f"trajectory provenance mismatch: {field}")
    cells.to_csv(root / "twibot_cue_alignment_cells.csv", index=False)
    changes.to_csv(root / "twibot_cue_alignment_changes.csv", index=False)
    summary.to_csv(root / "twibot_cue_alignment_summary.csv", index=False)
    protocol = json.loads((args.input / "protocol.json").read_text())
    protocol.update({"validated_against_trajectory_cells": True, "rows": len(cells),
                     "minimum_valid_episodes": int(cells.valid_episodes.min()),
                     "independent_training_seeds": 1})
    (root / "twibot_cue_alignment_validation.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
