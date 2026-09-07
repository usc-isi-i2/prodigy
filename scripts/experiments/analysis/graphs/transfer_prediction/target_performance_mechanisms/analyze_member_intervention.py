"""Prospective fixed-step factorial contrasts, with all seeds and targets retained."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .analyze_trajectories import DECODERS, PANEL, KEY, read_jsonl

POLICIES = {"lowest_sorted", "lowest_shuffled", "uniform_sorted", "uniform_shuffled"}
PRIMARY = {"covid_political", "facebook_page_reference", "twibot20"}


def verify_receipt(receipt, arms):
    gates = ("valid", "research_result", "same_consumed_anchors", "same_final_walk_rng",
             "same_retention_sets_across_role_treatments", "same_initialization_verified")
    if not all(receipt.get(k) is True for k in gates) or receipt.get("steps_per_model") != 2500 or receipt.get("models") != 24:
        raise ValueError("substantive consumed-stream validity receipt missing or failed")
    if len(arms) != 24 or arms.model_id.duplicated().any() or arms.duplicated(["source", "seed", "policy"]).any():
        raise ValueError("invalid factorial arm inventory")
    expected = {(s, seed, policy) for s in ("ukr_rus", "cp_hk") for seed in range(3) for policy in POLICIES}
    if set(map(tuple, arms[["source", "seed", "policy"]].to_numpy())) != expected:
        raise ValueError("missing or unexpected factorial arm")
    if any(s["steps"] != 2500 or s["episodes"] != 10000 for s in arms.summary):
        raise ValueError("consumed exposure budget differs")
    if (arms.initial_sha256 == arms.final_sha256).any() or arms.initial_sha256.isna().any():
        raise ValueError("missing initialization or unchanged model")
    if (arms.groupby(["source", "seed"]).initial_sha256.nunique() != 1).any():
        raise ValueError("initial weights not matched")


def validate_replay(cells, arms, reference, stream):
    expected = pd.MultiIndex.from_product([PANEL, arms.model_id, sorted(DECODERS)], names=KEY)
    if cells.empty or cells.duplicated(KEY).any() or set(pd.MultiIndex.from_frame(cells[KEY])) != set(expected):
        raise ValueError("incomplete factorial replay grid")
    if set(cells.variant) != {"baseline"} or set(cells.episodes) != {128}:
        raise ValueError("unexpected intervention or episode count")
    if not np.isfinite(cells[["roc_auc", "accuracy", "f1", "nll"]].to_numpy()).all():
        raise ValueError("nonfinite evaluation metric")
    joined = cells.merge(arms[["model_id", "source", "seed", "policy", "checkpoint", "final_sha256"]],
                         on="model_id", suffixes=("", "_verified"), validate="many_to_one")
    if not (joined.weights_sha256 == joined.final_sha256).all() or not (joined.checkpoint == joined.checkpoint_verified).all():
        raise ValueError("replay did not use the verified terminal weights")
    if not all(s == [t] for s, t in zip(joined.sources, joined.source)):
        raise ValueError("source manifest mismatch")
    for target, group in joined.groupby("dataset"):
        ref = reference[reference.dataset == target]
        if group.episode_fingerprint.nunique() != 1 or set(group.episode_fingerprint) != set(ref.episode_fingerprint):
            raise ValueError("episode fingerprint differs from established stream")
        if set(group.queries) != set(ref.queries):
            raise ValueError("query count differs from established stream")
        for decoder, raw in group[group.decoder.str.startswith("raw_")].groupby("decoder"):
            ref_auc = ref[ref.decoder == decoder].roc_auc
            if raw.roc_auc.max() - raw.roc_auc.min() > 1e-8 or len(ref_auc) == 0 or abs(raw.roc_auc.iloc[0] - ref_auc.iloc[0]) > 1e-5:
                raise ValueError("raw input signal changed across models or from reference")
    return joined.drop(columns=["checkpoint_verified", "final_sha256"]).assign(stream=stream)


def factorial_contrasts(cells):
    rows = []
    for keys, group in cells.groupby(["stream", "source", "seed", "dataset", "decoder"]):
        if len(group) != 4 or set(group.policy) != POLICIES:
            raise ValueError("incomplete four-policy contrast")
        row = dict(zip(["stream", "source", "seed", "dataset", "decoder"], keys))
        for metric in ("roc_auc", "accuracy", "f1", "nll"):
            v = group.set_index("policy")[metric]
            ls, lr, us, ur = (v[p] for p in ("lowest_sorted", "lowest_shuffled", "uniform_sorted", "uniform_shuffled"))
            row[f"retention_{metric}"] = .5 * ((us - ls) + (ur - lr))
            row[f"role_shuffle_{metric}"] = .5 * ((lr - ls) + (ur - us))
            row[f"interaction_{metric}"] = (ur - us) - (lr - ls)
        rows.append(row)
    return pd.DataFrame(rows)


def primary_contrast(contrasts):
    panel = contrasts[(contrasts.decoder == "full_model") & contrasts.dataset.isin(PRIMARY)]
    if not (panel.groupby(["stream", "source", "seed"]).dataset.nunique() == 3).all():
        raise ValueError("primary three-target panel incomplete")
    per_source = panel.groupby(["stream", "source", "seed"]).retention_roc_auc.mean().unstack("source")
    if set(per_source.columns) != {"cp_hk", "ukr_rus"} or per_source.isna().any().any():
        raise ValueError("source contrast unmatched")
    per_source["hong_kong_minus_ukraine_retention_effect"] = per_source.cp_hk - per_source.ukr_rus
    return per_source.reset_index()


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--data", type=Path, default=Path(__file__).parent / "data")
    args = p.parse_args()
    receipt = json.loads((args.data / "member_training_verified/DONE.json").read_text())
    arms = pd.read_json(args.data / "member_training_verified/arms.json")
    verify_receipt(receipt, arms)
    contract = json.loads((args.data / "member_training_contract_validation.json").read_text())
    if contract.get("models") != 24 or not all(contract.get(k) is True for k in (
            "declared_cpu_recipe_matches", "common_non_treatment_settings", "consumed_episode_source_labels_match")):
        raise ValueError("independent effective-config/source-label contract audit required")
    input_receipt = json.loads((args.data / "member_training_verified/input_validation.json").read_text())
    if input_receipt.get("all_cached_batches_identical") is not True or input_receipt.get("stream_target_cells") != 10:
        raise ValueError("exact cached-input validation missing")
    old = pd.read_csv(args.data / "replay_cells.csv")
    references = {"original": old[old.variant == "baseline"], "fresh": pd.read_csv(args.data / "fresh_replay_cells.csv")}
    streams = [validate_replay(read_jsonl(args.data / f"member_replay_{s}"), arms, ref, s) for s, ref in references.items()]
    paired = streams[0].merge(streams[1], on=KEY, suffixes=("_original", "_fresh"), validate="one_to_one")
    if not (paired.weights_sha256_original == paired.weights_sha256_fresh).all() or (paired.episode_fingerprint_original == paired.episode_fingerprint_fresh).any():
        raise ValueError("cross-stream weights or inputs disagree")
    cells = pd.concat(streams, ignore_index=True)
    cells.assign(sources=cells.sources.map(json.dumps)).to_csv(args.data / "member_replay_cells.csv", index=False)
    contrasts = factorial_contrasts(cells)
    contrasts.to_csv(args.data / "member_factorial_contrasts.csv", index=False)
    primary = primary_contrast(contrasts)
    primary.to_csv(args.data / "member_primary_contrast.csv", index=False)
    exposure = pd.concat([arms.drop(columns="summary"), pd.json_normalize(arms.summary)], axis=1)
    exposure.to_csv(args.data / "member_consumed_exposure.csv", index=False)
    print("Primary fixed-step 2500 contrast, each training seed and stream separately:")
    print(primary.to_string(index=False))
    print("Descriptive seed means (three seeds; not an independent-domain estimate):")
    print(primary.groupby("stream")[["cp_hk", "ukr_rus", "hong_kong_minus_ukraine_retention_effect"]].mean().to_string())
    (args.data / "member_intervention_validation.json").write_text(json.dumps({
        "replay_rows": len(cells), "models": 24, "training_seeds": [0, 1, 2],
        "steps": 2500, "episode_offsets": [0, 100003], "matched_training_receipt": receipt,
        "training_contract": contract,
        "same_weights_across_streams": True, "target_fingerprints_match_reference": True,
        "best_checkpoint_selected": False, "independent_domains": False,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
