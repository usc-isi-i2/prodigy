"""Matched fresh-episode sensitivity checks and a no-prompt-label sanity test."""
import json
from pathlib import Path

import pandas as pd


def main():
    root = Path(__file__).parent / "data"
    interventions = pd.DataFrame([json.loads(line) for path in (root / "fresh_interventions").glob("*.jsonl")
                                  for line in path.read_text().splitlines()])
    fresh = pd.read_csv(root / "fresh_replay_cells.csv")
    old = pd.read_csv(root / "replay_cells.csv")
    if len(interventions) != 60 or interventions.duplicated(["dataset", "model_id", "variant"]).any():
        raise ValueError("incomplete intervention grid")
    base = fresh[fresh.decoder == "full_model"]
    result = interventions.merge(base[["dataset", "model_id", "roc_auc", "weights_sha256", "episode_fingerprint"]],
        on=["dataset", "model_id"], suffixes=("", "_baseline"), validate="many_to_one")
    if not (result.weights_sha256 == result.weights_sha256_baseline).all() or not (result.episode_fingerprint == result.episode_fingerprint_baseline).all():
        raise ValueError("interventions changed weights or cached input stream")
    result["delta_auc_fresh"] = result.roc_auc - result.roc_auc_baseline
    old_full = old[old.decoder == "full_model"]
    old_pair = old_full.merge(old_full[old_full.variant == "baseline"][["dataset", "model_id", "roc_auc"]],
        on=["dataset", "model_id"], suffixes=("", "_baseline"), validate="many_to_one")
    old_pair["delta_auc_original"] = old_pair.roc_auc - old_pair.roc_auc_baseline
    result = result.merge(old_pair[["dataset", "model_id", "variant", "delta_auc_original"]],
        on=["dataset", "model_id", "variant"], how="left", validate="one_to_one")
    result.to_csv(root / "fresh_intervention_deltas.csv", index=False)
    joint = result[result.variant == "zero_support_and_label_text"]
    if len(joint) != 15 or (joint.roc_auc - .5).abs().max() > 1e-6:
        raise ValueError("joint label erasure did not give chance AUC")
    print(result[result.model_id == "ss_ukr_rus"][["dataset", "variant", "roc_auc_baseline", "roc_auc", "delta_auc_original", "delta_auc_fresh"]].round(4).to_string(index=False))
    (root / "fresh_intervention_validation.json").write_text(json.dumps({"cells": len(result),
        "same_weights_and_episode_fingerprints": True, "joint_erasure_cells": len(joint),
        "max_joint_erasure_auc_deviation_from_chance": float((joint.roc_auc - .5).abs().max()),
        "training_seed_replication": False}, indent=2) + "\n")


if __name__ == "__main__":
    main()
