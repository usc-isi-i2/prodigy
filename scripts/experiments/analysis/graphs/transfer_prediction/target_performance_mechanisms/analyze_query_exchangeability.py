"""Validate complete query-context symmetry audit and separate loss components."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
KEY = ["model_id", "source", "seed", "input_step", "checkpoint_step", "mode"]
MAP = {"original_accuracy": "original_correct", "symmetrized_accuracy": "symmetrized_correct",
    "accuracy_ceiling": "correct_ceiling", "original_nll": "original_nll_sum",
    "symmetrized_nll": "symmetrized_nll_sum", "nll_floor": "nll_floor_sum",
    "mass_deficit": "mass_deficit_sum", "within_group_imbalance": "within_group_imbalance_sum"}


def validate_frames(cells, groups, audits):
    expected = {(f"readouttrain_{s}_free_s{i}", s, i, b, c, m)
        for s, i, b, c, m in itertools.product(("ukr_rus", "cp_hk", "covid"), range(3), range(1, 5),
            (0, 2500), ("training", "meta_frozen"))}
    if len(audits) != 144 or set(audits[KEY].itertuples(index=False, name=None)) != expected:
        raise ValueError("incomplete 144-cell audit grid")
    if len(cells) != 432 or set(cells[KEY+["cohort"]].itertuples(index=False, name=None)) != {
        (*k, cohort) for k in expected for cohort in ("all", "unique", "repeated")}:
        raise ValueError("incomplete cohort grid")
    if not audits.suffix_baseline_bit_exact.all():
        raise ValueError("baseline suffix mismatch")
    if not (audits[audits["mode"] == "meta_frozen"][["suffix_logit_error", "suffix_post_error"]] == 0).all().all():
        raise ValueError("frozen normalization is not exactly permutation-equivariant")
    whole = audits[audits.input_step == 1]
    if len(whole) != 36 or not (whole.whole_baseline_bit_exact == True).all():
        raise ValueError("missing whole-input baseline checks")
    if audits[audits.input_step != 1].whole_baseline_bit_exact.notna().any():
        raise ValueError("unexpected whole-input scope")
    for col in ("suffix_logit_error", "suffix_post_error", "whole_pre_error", "whole_logit_error", "whole_vs_suffix_error"):
        values = audits[col].dropna()
        if not np.isfinite(values).all() or (values < 0).any() or (values > 1e-4).any():
            raise ValueError("unacceptable permutation roundoff")
    if groups.duplicated(KEY+["episode", "center_id"]).any():
        raise ValueError("duplicate identity group")
    if groups["size"].lt(1).any() or not groups["size"].eq(groups["size"].astype(int)).all():
        raise ValueError("invalid group sizes")
    if not groups.cohort.eq(np.where(groups["size"] > 1, "repeated", "unique")).all():
        raise ValueError("invalid cohort label")
    if not np.isfinite(groups[list(MAP.values())]).all().all():
        raise ValueError("nonfinite group statistics")
    if not (groups.correct_ceiling == 1).all() or (groups.symmetrized_correct > 1+1e-12).any():
        raise ValueError("accuracy bound failed")
    if (groups.original_correct < 0).any() or (groups.original_correct > groups["size"]).any():
        raise ValueError("invalid original correct count")
    np.testing.assert_allclose(groups.nll_floor_sum, groups["size"]*np.log(groups["size"]), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(groups.symmetrized_nll_sum,
        groups.nll_floor_sum+groups.mass_deficit_sum+groups.within_group_imbalance_sum, rtol=1e-12, atol=1e-10)
    if (groups[["mass_deficit_sum", "within_group_imbalance_sum"]] < -1e-10).any().any():
        raise ValueError("negative decomposition component")
    if set(groups[KEY].itertuples(index=False, name=None)) != expected:
        raise ValueError("missing group cell")
    indexed = cells.set_index(KEY+["cohort"])
    for key, g in groups.groupby(KEY):
        for cohort in ("all", "unique", "repeated"):
            h = g if cohort == "all" else g[g.cohort == cohort]
            row = indexed.loc[(*key, cohort)]
            n = h["size"].sum()
            if n != row.queries or len(h) != row.identity_groups:
                raise ValueError("cohort count mismatch")
            for metric, total in MAP.items():
                if n:
                    np.testing.assert_allclose(row[metric], h[total].sum()/n, atol=1e-11, rtol=1e-12)
                elif not pd.isna(row[metric]):
                    raise ValueError("empty cohort has a metric")
        if g["size"].sum() != 480 or set(g.episode) != set(range(4)):
            raise ValueError("not the actual four-episode NM batch")
    identity = ["model_id", "input_step", "episode", "center_id"]
    if not (groups.groupby(identity)["size"].nunique() == 1).all():
        raise ValueError("identity grouping changed across checkpoints/modes")
    for _, g in audits.groupby(["model_id", "input_step"]):
        if g.input_sha256.nunique() != 1 or g.moved_queries.nunique() != 1:
            raise ValueError("input or permutation changed")
    for _, g in audits.groupby(["model_id", "checkpoint_step"]):
        if g.weights_sha256.nunique() != 1:
            raise ValueError("weights changed")


def analyze(root, output, baseline_path):
    done = json.loads((root/"DONE.json").read_text())
    if done["smoke"] or not all(done[k] for k in ("complete", "no_optimizer_updates", "no_target_outcomes",
        "all_full_input_hashes_reverified", "all_suffix_baselines_bit_exact", "all_whole_baselines_bit_exact",
        "symmetrized_experiment_not_realized_input_bound")):
        raise ValueError("requires a complete non-smoke experiment")
    cells, groups, audits = (pd.read_csv(root/f"{name}.csv") for name in ("cells", "groups", "audits"))
    validate_frames(cells, groups, audits)
    if (done["cells"], done["identity_group_rows"], done["suffix_audits"], done["whole_input_audits"]) != (
        len(cells), len(groups), len(audits), 36):
        raise ValueError("completion inventory mismatch")
    baseline = pd.read_csv(baseline_path)
    baseline = baseline[baseline.variant == "baseline"]
    joined = cells[cells.cohort == "all"].merge(baseline, on=KEY, validate="one_to_one")
    if len(joined) != 144:
        raise ValueError("incomplete prior-baseline comparison")
    for metric in ("nll", "accuracy"):
        np.testing.assert_allclose(joined["original_"+metric], joined["all_"+metric], rtol=8*np.finfo(np.float32).eps, atol=1e-7)
    seed_rows = []
    for key, g in groups.groupby(["source", "seed", "checkpoint_step", "mode"]):
        for cohort in ("all", "unique", "repeated"):
            h = g if cohort == "all" else g[g.cohort == cohort]
            n = h["size"].sum()
            row = dict(zip(["source", "seed", "checkpoint_step", "mode"], key))
            row.update(cohort=cohort, queries=n, identity_groups=len(h))
            row.update({metric: h[total].sum()/n if n else np.nan for metric, total in MAP.items()})
            row["original_minus_symmetrized_nll"] = row["original_nll"]-row["symmetrized_nll"]
            row["original_minus_symmetrized_accuracy"] = row["original_accuracy"]-row["symmetrized_accuracy"]
            row["floor_over_symmetrized_nll"] = row["nll_floor"]/row["symmetrized_nll"]
            seed_rows.append(row)
    seeds = pd.DataFrame(seed_rows)
    sources = seeds.groupby(["source", "checkpoint_step", "mode", "cohort"]).mean(numeric_only=True).drop(columns="seed").reset_index()
    # These are descriptive fixed-prefix means, not episode-as-independent-seed CIs.
    output.mkdir(exist_ok=True, parents=True)
    seeds.to_csv(output/"query_exchangeability_seeds.csv", index=False)
    sources.to_csv(output/"query_exchangeability_sources.csv", index=False)
    report = {"complete": True, "cells": len(cells), "identity_group_rows": len(groups),
        "suffix_checks": len(audits), "whole_input_checks": 36, "baseline_matches_prior_gradient_audit": True,
        "whole_argmax_mismatches": int(audits.whole_argmax_mismatches.sum()),
        **{c+"_max": float(audits[c].max()) for c in ("suffix_logit_error", "suffix_post_error",
            "whole_pre_error", "whole_logit_error", "whole_vs_suffix_error")},
        "frozen_suffix_all_bit_exact": True, "all_cohort_and_loss_decompositions_recomputed": True,
        "scope": "36 actual training-prefix batches, not held-out NM or target transfer",
        "symmetrized_bound_not_realized_input_ceiling": True}
    (output/"query_exchangeability_validation.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))
    print(sources[(sources.checkpoint_step == 2500) & (sources["mode"] == "training")].to_string(index=False))
    return cells, groups, audits, seeds, sources


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--input", type=Path, default=HERE/"data/query_exchangeability")
    p.add_argument("--output", type=Path, default=HERE/"data")
    p.add_argument("--baseline", type=Path, default=HERE/"data/support_identity_gradients/cells.csv")
    a = p.parse_args()
    analyze(a.input, a.output, a.baseline)


if __name__ == "__main__":
    main()
