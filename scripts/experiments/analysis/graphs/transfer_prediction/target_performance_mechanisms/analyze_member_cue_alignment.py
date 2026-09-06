"""Exploratory secondary factorial effects on TwiBot cue agreement, not mediation."""
import argparse
import json
from pathlib import Path

import pandas as pd

from .analyze_member_intervention import POLICIES, verify_receipt


def cue_contrasts(cells):
    rows = []
    keys = ["stream", "source", "seed", "decoder", "cue"]
    for group_keys, group in cells.groupby(keys):
        if len(group) != 4 or set(group.policy) != POLICIES:
            raise ValueError("incomplete policy contrast")
        row = dict(zip(keys, group_keys))
        row["minimum_valid_episodes"] = int(group.valid_episodes.min())
        for metric in ("spearman", "pearson", "decision_agreement"):
            v = group.set_index("policy")[f"mean_within_episode_{metric}"]
            ls, lr, us, ur = (v[p] for p in ("lowest_sorted", "lowest_shuffled", "uniform_sorted", "uniform_shuffled"))
            # A constant model may have undefined rank agreement. Propagate that
            # missing value rather than dropping an arm or averaging fewer cells.
            row[f"retention_{metric}"] = .5 * ((us - ls) + (ur - lr))
            row[f"role_shuffle_{metric}"] = .5 * ((lr - ls) + (ur - us))
            row[f"interaction_{metric}"] = (ur - us) - (lr - ls)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path(__file__).parent / "data")
    args = parser.parse_args()
    if not (args.input / "DONE").is_file():
        raise ValueError("incomplete cue comparison")
    receipt = json.loads((args.data / "member_training_verified/DONE.json").read_text())
    arms = pd.read_json(args.data / "member_training_verified/arms.json")
    verify_receipt(receipt, arms)
    # Require completed primary analysis, not a selective first-result follow-up.
    validity = json.loads((args.data / "member_intervention_validation.json").read_text())
    if validity.get("replay_rows") != 4080 or validity.get("models") != 24:
        raise ValueError("complete primary factorial analysis required first")
    reference = pd.read_csv(args.data / "member_replay_cells.csv")
    reference = reference[reference.dataset == "twibot20"]
    cells = pd.read_json(args.input / "metrics.json")
    keys = ["stream", "model_id", "decoder", "cue"]
    if len(cells) != 432 or cells.duplicated(keys).any() or set(cells.step) != {2500}:
        raise ValueError("incomplete or unexpected secondary grid")
    expected = {(s, m, d, c) for s in ("original", "fresh") for m in arms.model_id
                for d in ("S0_pool/ridge", "U1_pre_meta/ridge", "full_model")
                for c in ("center_indegree", "raw_center", "raw_context")}
    if set(map(tuple, cells[keys].to_numpy())) != expected:
        raise ValueError("secondary model/stage/cue grid differs")
    provenance = ["weights_sha256", "episode_fingerprint", "source", "seed", "policy"]
    joined = cells.merge(reference[["stream", "model_id", "decoder"] + provenance],
                         on=["stream", "model_id", "decoder"], validate="many_to_one", suffixes=("", "_reference"))
    if len(joined) != 432 or any(not (joined[p] == joined[f"{p}_reference"]).all() for p in provenance):
        raise ValueError("secondary provenance differs from validated primary replay")
    cells.to_csv(args.data / "member_cue_alignment_cells.csv", index=False)
    contrasts = cue_contrasts(cells)
    contrasts.to_csv(args.data / "member_cue_alignment_contrasts.csv", index=False)
    print(contrasts[(contrasts.decoder == "full_model") & (contrasts.cue == "center_indegree")].to_string(index=False))
    (args.data / "member_cue_alignment_validation.json").write_text(json.dumps({
        "rows": len(cells), "training_seeds": [0, 1, 2], "steps": 2500,
        "matched_to_complete_primary_replay": True, "exploratory_secondary_endpoint": True,
        "causal_mediation_claim": False, "constant_scores_preserved_as_missing": True,
        "query_labels_fitted": False}, indent=2) + "\n")


if __name__ == "__main__":
    main()
