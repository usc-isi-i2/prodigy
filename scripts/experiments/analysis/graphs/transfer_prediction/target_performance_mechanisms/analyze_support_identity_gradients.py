"""Complete-grid analysis of the no-update, prefix-only training gradient audit."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
BASE = ["model_id", "source", "seed", "input_step", "checkpoint_step", "mode"]
KEY = BASE + ["variant"]
BLOCKS = {"all", "encoder", "readout", "metagraph", "label_input", "logit_scale"}


def validate(root):
    done = json.loads((root/"DONE.json").read_text())
    if done["smoke"] or not all(done[k] for k in ("complete", "every_reported_tensor_digest_and_gradient_statistic_recomputed",
        "every_input_full_hash_reverified", "pre_metagraph_identical_under_label_interventions",
        "frozen_meta_bn_query_vectors_bit_exact", "no_target_outcomes", "no_optimizer_updates")):
        raise ValueError("requires complete verified non-smoke diagnostic")
    cells, gradients, binding, groups = (pd.read_csv(root/f"{n}.csv") for n in ("cells", "gradients", "binding_gradients", "groups"))
    model_ids = {f"readouttrain_{s}_free_s{i}" for s in ("ukr_rus", "cp_hk", "covid") for i in range(3)}
    expected = set(itertools.product(model_ids, range(1, 5), (0, 2500), ("training", "meta_frozen"),
        ("baseline", "identity_soft", "permuted_soft")))
    grid = ["model_id", "input_step", "checkpoint_step", "mode", "variant"]
    if len(cells) != 432 or set(cells[grid].itertuples(index=False, name=None)) != expected:
        raise ValueError("incomplete gradient cell grid")
    if not (cells.all_queries == 480).all() or not cells.pre_bit_exact.all() or not cells.query_truth_blind.all():
        raise ValueError("invalid input/cohort audit")
    if not cells.null_matches_every_class_vector_multiset.all() or not cells.only_support_label_edge_values_changed.all():
        raise ValueError("intervention matching failed")
    if not (cells[cells["mode"]=="meta_frozen"].post_query_max_difference == 0).all():
        raise ValueError("query vectors changed with frozen metagraph normalization")
    np.testing.assert_allclose(cells.squared_perturbation, cells.null_squared_perturbation, rtol=1e-14, atol=1e-12)
    if (cells.maximum_label_mass_roundoff > 1e-5).any():
        raise ValueError("support label mass not conserved")
    if not (cells.query_support_conflict_queries+cells.no_query_support_conflict_queries == cells.all_queries).all():
        raise ValueError("cohort counts do not conserve")
    for metric in ("nll", "accuracy"):
        for cohort in ("query_support_conflict", "no_query_support_conflict"):
            if not ((cells[cohort+"_queries"] == 0) == cells[cohort+"_"+metric].isna()).all():
                raise ValueError("empty/nonempty cohort metric mismatch")
        weighted = (cells["query_support_conflict_"+metric].fillna(0)*cells.query_support_conflict_queries+
            cells["no_query_support_conflict_"+metric].fillna(0)*cells.no_query_support_conflict_queries)/cells.all_queries
        np.testing.assert_allclose(cells["all_"+metric], weighted, rtol=8*np.finfo(np.float32).eps, atol=1e-8)
    for _, g in cells.groupby(["model_id", "input_step"]):
        for k in ("input_sha256", "groups", "affected_support_positions", "squared_perturbation", "query_support_conflict_queries"):
            if g[k].nunique(dropna=False) != 1:
                raise ValueError("fixed input/intervention changed across conditions")
    for _, g in cells.groupby(["model_id", "checkpoint_step"]):
        if g.weights_sha256.nunique() != 1:
            raise ValueError("weights changed across probes")
    expected_gradient = {(k, b) for k in cells[KEY].itertuples(index=False, name=None) for b in BLOCKS}
    if len(gradients) != 2592 or {(tuple(r[:-1]), r[-1]) for r in gradients[KEY+["block"]].itertuples(index=False, name=None)} != expected_gradient:
        raise ValueError("incomplete parameter-gradient grid")
    expected_binding = {(k, b) for k in cells[BASE].drop_duplicates().itertuples(index=False, name=None) for b in BLOCKS}
    if len(binding) != 864 or {(tuple(r[:-1]), r[-1]) for r in binding[BASE+["block"]].itertuples(index=False, name=None)} != expected_binding:
        raise ValueError("incomplete identity-versus-null gradient grid")
    for data in (gradients, binding):
        for k in ("baseline_norm", "changed_norm", "difference_norm"):
            if not np.isfinite(data[k]).all() or (data[k] < 0).any():
                raise ValueError("nonfinite/negative gradient norm")
        if (data.cosine.dropna().abs() > 1+1e-12).any():
            raise ValueError("invalid gradient cosine")
    if groups.duplicated(KEY+["kind", "group"]).any():
        raise ValueError("duplicate gradient-cancellation group")
    counted = groups.groupby(KEY+["kind"]).agg(count=("group", "size"), members=("members", "sum")).reset_index()
    expected_groups = {(*k, kind) for k in cells[cells.groups>0][KEY].itertuples(index=False, name=None)
        for kind in ("identity_group_gradients", "null_group_gradients")}
    if set(counted[KEY+["kind"]].itertuples(index=False, name=None)) != expected_groups:
        raise ValueError("missing cancellation group inventory")
    for _, row in counted.iterrows():
        same = cells
        for k in KEY:
            same = same[same[k] == row[k]]
        if len(same) != 1 or row["count"] != same.iloc[0]["groups"] or row["members"] != same.iloc[0].affected_support_positions:
            raise ValueError("group cancellation inventory differs from intervention")
    return cells, gradients, binding, groups, done


def derive(cells, binding, groups):
    pairs = []
    for key, rows in cells.groupby(BASE):
        variants = rows.set_index("variant")
        base, actual, null = (variants.loc[v] for v in ("baseline", "identity_soft", "permuted_soft"))
        pairs.append({**dict(zip(BASE, key)), "affected_support_fraction": base.affected_support_positions/base.total_support_positions,
            "null_coherent_fraction_of_affected": (base.null_fully_identity_coherent_positions/base.affected_support_positions
                if base.affected_support_positions else None),
            "actual_minus_baseline_nll": actual.all_nll-base.all_nll,
            "null_minus_baseline_nll": null.all_nll-base.all_nll,
            "actual_minus_null_nll": actual.all_nll-null.all_nll,
            "actual_minus_null_accuracy": actual.all_accuracy-null.all_accuracy,
            "actual_query_post_difference": actual.post_query_max_difference,
            "null_query_post_difference": null.post_query_max_difference})
    contrasts = pd.DataFrame(pairs)
    keys = KEY+["group"]
    actual = groups[groups.kind=="identity_group_gradients"].drop(columns="kind")
    null = groups[groups.kind=="null_group_gradients"].drop(columns="kind")
    matched = actual.merge(null, on=keys, suffixes=("_identity", "_null"), validate="one_to_one")
    if len(matched)*2 != len(groups) or not (matched.members_identity == matched.members_null).all():
        raise ValueError("cancellation groups not size matched")
    for k in ("shared_gradient_ratio", "pair_cosine_mean"):
        matched["identity_minus_null_"+k] = matched[k+"_identity"]-matched[k+"_null"]
    # First reduce within actual input batch, then across its fixed prefix. Do
    # not let sources with more repeated identities count as more replications.
    cancellation = matched.groupby(KEY)[["identity_minus_null_shared_gradient_ratio", "identity_minus_null_pair_cosine_mean"]].mean().reset_index()
    return contrasts, cancellation


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--input", type=Path, default=HERE/"data/support_identity_gradients")
    p.add_argument("--output", type=Path, default=HERE/"data")
    args = p.parse_args()
    cells, gradients, binding, groups, done = validate(args.input)
    contrasts, cancellation = derive(cells, binding, groups)
    contrasts.to_csv(args.output/"support_identity_loss_contrasts.csv", index=False)
    cancellation.to_csv(args.output/"support_identity_cancellation_contrasts.csv", index=False)
    summary = contrasts.groupby(["source", "seed", "checkpoint_step", "mode"]).mean(numeric_only=True).reset_index()
    summary.to_csv(args.output/"support_identity_seed_summary.csv", index=False)
    (args.output/"support_identity_validation.json").write_text(json.dumps({**done,
        "local_complete_grid_and_aggregate_checks": True, "training_prefix_batches_per_model": 4,
        "reused_checkpoints_not_independent_training_runs": True, "exploratory_not_transfer_evidence": True}, indent=2))
    print(summary.groupby(["source", "checkpoint_step", "mode"])[["actual_minus_baseline_nll",
        "null_minus_baseline_nll", "actual_minus_null_nll", "actual_query_post_difference"]].mean().to_string())


if __name__ == "__main__":
    main()
