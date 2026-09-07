"""Validate all existing three-seed controls and the exact support-label pathway."""
import argparse
from itertools import product
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .analyze_mixture_complementarity import complete_grid, METRICS, STREAMS

KEY = ["stream", "model_id"]


def audit_paths(audits, ids):
    complete_grid(audits, KEY + ["batch"], set(product(STREAMS, ids, range(32))), "all 384 label-path checks required")
    for field in ("query_pre_and_post_bit_exact", "label_only_recomposition_bit_exact"):
        if not audits[field].eq(True).all():
            raise ValueError("query or label-path invariance failed")
    if not audits.label_vectors.eq(8).all() or not audits.changed_label_vectors.eq(8).all():
        raise ValueError("every episode label representation must be checked")
    if not np.isfinite(audits[["label_max_abs_change", "mean_label_cosine"]]).all().all():
        raise ValueError("nonfinite label-path audit")
    if not audits.label_max_abs_change.gt(0).all() or not audits.mean_label_cosine.between(-1, 1).all():
        raise ValueError("invalid label changes")


def paired_effects(cells):
    baseline = cells[cells.condition.eq("baseline")].set_index(KEY)
    altered = cells[cells.condition.eq("edges_support")].set_index(KEY)
    rows = []
    for key, row in baseline.iterrows():
        other = altered.loc[key]
        rows.append(dict(zip(KEY, key)) | {k: row[k] for k in ("source", "seed")} |
                    {"baseline_" + m: float(row[m]) for m in METRICS} |
                    {"altered_" + m: float(other[m]) for m in METRICS} |
                    {"delta_" + m: float(other[m] - row[m]) for m in METRICS})
    effects = pd.DataFrame(rows)
    gaps = []
    for (stream, seed), group in effects.groupby(["stream", "seed"]):
        g = group.set_index("source")
        before = g.loc["ukr_rus", "baseline_roc_auc"] - g.loc["cp_hk", "baseline_roc_auc"]
        after = g.loc["ukr_rus", "altered_roc_auc"] - g.loc["cp_hk", "altered_roc_auc"]
        gaps.append(dict(stream=stream, seed=int(seed), baseline_auc_gap=float(before), altered_auc_gap=float(after),
                         fraction_gap_reduced=float(1 - after / before) if before else None))
    return effects, pd.DataFrame(gaps)


def validate(root, data):
    read = lambda name: json.loads((root / (name + ".json")).read_text())
    done, protocol = read("DONE"), read("protocol")
    expected = dict(complete=True, cells=24, label_path_checks=384, all_query_embeddings_unchanged=True,
                    all_logits_recomposed_from_label_change=True, all_baseline_metrics_match_saved=True, all_weights_unchanged=True)
    if done != expected or protocol["new_training"] is not False:
        raise ValueError("complete fixed-model study required")
    arms = pd.DataFrame(json.loads((data / "member_training_verified" / "arms.json").read_text()))
    arms = arms[arms.policy.eq("lowest_sorted")]
    complete_grid(arms, ["source", "seed"], set(product(("cp_hk", "ukr_rus"), range(3))), "all six existing seed controls required")
    ids = arms.model_id.tolist()
    cells, audits = pd.DataFrame(read("metrics")), pd.DataFrame(read("label_path_audit"))
    complete_grid(cells, KEY + ["condition"], set(product(STREAMS, ids, ("baseline", "edges_support"))), "all 24 cells required")
    audit_paths(audits, ids)
    if not cells.target.eq("covid_political").all() or not np.isfinite(cells[list(METRICS)]).all().all():
        raise ValueError("invalid target or scores")
    if not cells.nll.ge(0).all() or not cells[["accuracy", "roc_auc", "f1"]].apply(lambda c: c.between(0, 1)).all().all():
        raise ValueError("invalid bounded metrics")
    arms = arms.set_index("model_id")
    for row in cells.itertuples():
        arm = arms.loc[row.model_id]
        if any(getattr(row, k) != arm[k] for k in ("source", "seed", "checkpoint")) or row.weights_sha256 != arm.final_sha256:
            raise ValueError("model identity differs from verified training receipt")
    inputs = pd.DataFrame(json.loads((data / "role_context_replay" / "input_inventory.json").read_text()))
    inputs = inputs[inputs.target.eq("covid_political")].set_index("stream")
    for row in audits.itertuples():
        if row.batch_sha256 != inputs.loc[row.stream, "batch_sha256"][row.batch]:
            raise ValueError("label-path inputs differ from complete role-factorial inputs")
        if row.source != arms.loc[row.model_id, "source"] or row.seed != arms.loc[row.model_id, "seed"]:
            raise ValueError("label-path audit model identity differs")
    for row in cells.itertuples():
        if row.episode_fingerprint != inputs.loc[row.stream, "episode_fingerprint"]:
            raise ValueError("metric input identity differs")
    ref = pd.read_csv(data / "member_replay_cells.csv")
    ref = ref[ref.dataset.eq("covid_political") & ref.decoder.eq("full_model") & ref.policy.eq("lowest_sorted")]
    matched = cells[cells.condition.eq("baseline")].merge(ref, on=KEY, suffixes=("", "_reference"), validate="one_to_one")
    if len(matched) != 12:
        raise ValueError("all 12 independently saved baseline references required")
    for k in ("checkpoint", "weights_sha256", "episode_fingerprint"):
        if not matched[k].eq(matched[k + "_reference"]).all():
            raise ValueError("independent saved baseline identity differs")
    if any((matched[m] - matched[m + "_reference"]).abs().max() > 1e-6 for m in METRICS):
        raise ValueError("independent saved baseline metrics differ")
    return cells, audits


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--data", type=Path, default=Path(__file__).parent / "data")
    args = p.parse_args()
    cells, audits = validate(args.data / "support_path_seeds", args.data)
    effects, gaps = paired_effects(cells)
    effects.to_csv(args.data / "support_path_seed_effects.csv", index=False)
    gaps.to_csv(args.data / "support_path_seed_gaps.csv", index=False)
    hk = effects[effects.source.eq("cp_hk")]
    result = dict(cells=len(cells), label_path_checks=len(audits), independent_initialization_seeds=3,
                  all_baseline_metrics_match_saved=True, all_queries_unchanged_through_metagraph=True,
                  all_altered_logits_exactly_recomposed_from_label_change=True,
                  hong_kong_auc_gain_every_seed_both_streams=bool(hk.delta_roc_auc.gt(0).all()),
                  hong_kong_auc_gain_min=float(hk.delta_roc_auc.min()), hong_kong_auc_gain_max=float(hk.delta_roc_auc.max()),
                  gap_reduction_fraction_min=float(gaps.fraction_gap_reduced.min()), gap_reduction_fraction_max=float(gaps.fraction_gap_reduced.max()),
                  training_source_causal_effect_established=False)
    (args.data / "support_path_validation.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
