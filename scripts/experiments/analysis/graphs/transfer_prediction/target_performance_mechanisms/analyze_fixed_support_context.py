"""Complete-grid validation of fixed-account natural support context effects."""
import argparse
import hashlib
from itertools import product
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .analyze_mixture_complementarity import complete_grid, METRICS, STREAMS
from .analyze_trajectories import PANEL


HERE = Path(__file__).resolve().parent
KEY = ["stream", "target", "model_id"]


def validate_plans(plans, *, batches=32, draws=8):
    keys = {(s, t) for s in STREAMS for t in PANEL}
    if len(plans) != 10 or {(p["stream"], p["target"]) for p in plans} != keys:
        raise ValueError("incomplete context plan grid")
    checked = 0
    for p in plans:
        payload = {k: v for k, v in p.items() if k != "sha256"}
        if hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest() != p["sha256"]:
            raise ValueError("plan hash changed")
        if not all(p[k] for k in ("graph_file_predates_original_cache", "every_cached_and_new_feature_and_edge_verified", "query_truth_and_feature_tamper_passed")):
            raise ValueError("source graph or query-blind sampling audit failed")
        sampler = p["same_sampler"]
        if sampler["hops"] != 2 or sampler["hop_sizes"] != [9, 9] or sampler["node_limit"] != 101:
            raise ValueError("changed neighborhood sampler")
        rows = pd.DataFrame(p["rows"])
        complete_grid(rows, ["batch", "draw"], set(product(range(batches), range(draws))), "complete draw grid required")
        for bi, g in rows.groupby("batch"):
            original = g[g.draw == 0].iloc[0]
            if len(original.support_centers) != 80:
                raise ValueError("wrong support count")
            for r in g.itertuples():
                if r.support_centers != original.support_centers:
                    raise ValueError("support identity changed across draws")
                if r.input_sha256 != p["original"]["batch_sha256"][bi]:
                    raise ValueError("original input receipt changed")
                if not all(getattr(r, k) for k in ("query_graphs_bit_exact", "support_centers_and_center_features_bit_exact", "all_metagraph_and_truth_tensors_bit_exact")):
                    raise ValueError("replacement invariance failed")
                if r.draw == 0:
                    if not pd.isna(r.seed):
                        raise ValueError("original draw resampled")
                else:
                    seed = 96060000+(10000000 if p["stream"] == "fresh" else 0)+sum((i+1)*ord(c) for i, c in enumerate(p["target"]))*1000+bi*10+r.draw
                    if r.seed != seed:
                        raise ValueError("fixed draw seed changed")
                checked += 80
    return checked


def validate_frames(cells, episodes, cohorts, audits, arms):
    ids = arms.model_id.tolist()
    grid = set(product(STREAMS, PANEL, ids))
    conditions = [("draw", d) for d in range(8)]+[("probability_ensemble", -1)]
    complete_grid(cells, KEY+["condition", "draw"], {(*k, *v) for k in grid for v in conditions}, "complete 540-cell metric grid required")
    complete_grid(audits, KEY, grid, "complete 60-cell audit required")
    if not audits.baseline_batches_bit_exact.eq(32).all() or not audits.query_invariance_batches.eq(256).all() or not audits.direct_suffix_checks.eq(256).all():
        raise ValueError("incomplete exact query/full-suffix checks")
    if not audits.maximum_direct_error.eq(0).all() or not audits.all_support_ids_and_labels_unchanged.all() or not audits.all_weights_unchanged.all():
        raise ValueError("failed input/model invariants")
    if not np.isfinite(cells[list(METRICS)]).all().all() or not cells.nll.ge(0).all() or not cells[["roc_auc", "accuracy", "f1"]].apply(lambda v: v.between(0, 1)).all().all():
        raise ValueError("invalid scores")
    expected = {(*k, ep, draw) for k in grid for ep in range(128) for draw in range(8)}
    complete_grid(episodes, KEY+["episode", "draw"], expected, "incomplete episode/draw accounting")
    if not np.isfinite(episodes.query_nll).all() or not episodes.query_nll.ge(0).all() or not episodes.query_correct.between(0, episodes.queries).all():
        raise ValueError("invalid episode result")
    base = cells[cells.condition == "draw"].set_index(KEY+["draw"])
    for key, g in episodes.groupby(KEY+["draw"]):
        r = base.loc[key]
        if g.queries.sum() != r.queries:
            raise ValueError("episode query count mismatch")
        np.testing.assert_allclose(g.query_correct.sum()/g.queries.sum(), r.accuracy, rtol=0, atol=1e-12)
        np.testing.assert_allclose(np.average(g.query_nll, weights=g.queries), r.nll, rtol=8*np.finfo(np.float32).eps, atol=1e-7)
        if key[-1] == 0 and g.prediction_changes_vs_original.sum():
            raise ValueError("original draw differs from itself")
    complete_grid(cohorts[cohorts.cohort == "all"], KEY, grid, "missing sensitivity cohorts")
    if cohorts.duplicated(KEY+["cohort"]).any() or not set(cohorts.cohort) <= {"all", "no_context", "has_context"}:
        raise ValueError("invalid cohort inventory")
    if not np.isfinite(cohorts.mean_probability_variance).all() or not cohorts.mean_probability_variance.between(0, .25).all():
        raise ValueError("invalid binary probability variance")
    if not (cohorts.always_correct+cohorts.always_wrong+cohorts.correctness_flips).eq(cohorts.queries).all():
        raise ValueError("correctness cohorts do not conserve")
    for key, g in cohorts.groupby(KEY):
        total = g[g.cohort == "all"].iloc[0]
        parts = g[g.cohort != "all"]
        if parts.queries.sum() != total.queries:
            raise ValueError("context cohorts do not partition queries")
        np.testing.assert_allclose(np.average(parts.mean_probability_variance, weights=parts.queries), total.mean_probability_variance, rtol=1e-12, atol=1e-14)
    registry = arms.set_index("model_id")
    for r in pd.concat([cells, audits], ignore_index=True).itertuples():
        arm = registry.loc[r.model_id]
        if r.weights_sha256 != arm.final_sha256 or any(getattr(r, k) != arm[k] for k in ("source", "seed", "checkpoint")):
            raise ValueError("model identity changed")


def summarize(cells, cohorts):
    effects = []
    for key, g in cells.groupby(KEY):
        draws = g[g.condition == "draw"]
        original = draws[draws.draw == 0].iloc[0]
        ensemble = g[g.condition == "probability_ensemble"].iloc[0]
        effects.append({**dict(zip(KEY, key)), "source": original.source, "seed": int(original.seed),
            **{"original_"+m: original[m] for m in METRICS},
            **{"mean_draw_"+m: draws[m].mean() for m in METRICS},
            **{"ensemble_minus_original_"+m: ensemble[m]-original[m] for m in METRICS},
            **{"ensemble_minus_mean_draw_"+m: ensemble[m]-draws[m].mean() for m in METRICS}})
    all_cohorts = cohorts[cohorts.cohort == "all"]
    contrast = all_cohorts.pivot(index=["stream", "target", "seed"], columns="source", values="mean_probability_variance")
    if len(contrast) != 30 or set(contrast.columns) != {"cp_hk", "ukr_rus"}:
        raise ValueError("incomplete paired source sensitivity contrasts")
    contrast["hong_kong_minus_ukraine"] = contrast.cp_hk-contrast.ukr_rus
    contrast["hong_kong_over_ukraine"] = contrast.cp_hk/contrast.ukr_rus.replace(0, np.nan)
    contrast = contrast.reset_index()
    political = contrast[contrast.target == "covid_political"]
    return pd.DataFrame(effects), contrast, {"primary_hong_kong_more_sensitive_every_seed_both_streams": bool(political.hong_kong_minus_ukraine.gt(0).all()),
        "primary_positive_comparisons": int(political.hong_kong_minus_ukraine.gt(0).sum()), "primary_comparisons": 6,
        "same_labeled_accounts_not_same_inference_compute": True, "unseen_target_validation": False,
        "source_training_causal_effect_established": False, "ensemble_jensen_gain_not_a_primary_prediction": True}


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--data", type=Path, default=HERE/"data")
    a = p.parse_args()
    root = a.data/"fixed_support_context"
    read = lambda name: json.loads((root/f"{name}.json").read_text())
    done = read("DONE")
    expected = {"complete": True, "phase": "evaluate", "smoke": False, "cells": 60, "metric_cells": 540,
        "episode_draw_cells": 61440, "direct_suffix_checks": 15360, "all_query_inputs_and_vectors_bit_exact": True,
        "same_labeled_support_accounts": True, "no_optimizer_updates": True, "no_additional_labeled_accounts": True}
    if any(done.get(k) != v for k, v in expected.items()):
        raise ValueError("requires complete non-smoke result")
    plans = read("plans")
    checked = validate_plans(plans)
    cells, cohorts, audits = (pd.DataFrame(read(n)) for n in ("metrics", "cohorts", "audits"))
    episodes = pd.read_csv(root/"episode_scores.csv")
    arms = pd.DataFrame(json.loads((a.data/"member_training_verified/arms.json").read_text()))
    arms = arms[arms.policy == "lowest_sorted"]
    complete_grid(arms, ["source", "seed"], set(product(("ukr_rus", "cp_hk"), range(3))), "complete control model grid required")
    validate_frames(cells, episodes, cohorts, audits, arms)
    lookup = {(p["stream"], p["target"]): p for p in plans}
    for r in pd.concat([cells, audits], ignore_index=True).itertuples():
        plan = lookup[r.stream, r.target]
        if r.plan_sha256 != plan["sha256"] or r.episode_fingerprint != plan["original"]["episode_fingerprint"]:
            raise ValueError("model result mismatches input plan")
    prior = pd.read_csv(a.data/"member_replay_cells.csv")
    prior = prior[(prior.policy == "lowest_sorted") & (prior.decoder == "full_model")].rename(columns={"dataset": "target"})
    joined = cells[(cells.condition == "draw") & (cells.draw == 0)].merge(prior, on=KEY, suffixes=("", "_prior"), validate="one_to_one")
    if len(joined) != 60:
        raise ValueError("missing independent baseline")
    for m in METRICS:
        np.testing.assert_allclose(joined[m], joined[m+"_prior"], rtol=0, atol=1e-6)
    verified = read("independent_verification")
    if not verified["complete"] or verified["prediction_cells"] != 60 or verified["support_draw_batches_rehashed"] != 2560:
        raise ValueError("missing independent raw-tensor verification")
    effects, contrasts, result = summarize(cells, cohorts)
    for name, frame in (("effects", effects), ("sensitivity", contrasts), ("cohorts", cohorts)):
        frame.to_csv(a.data/f"fixed_support_context_{name}.csv", index=False)
    result.update(complete=True, support_positions_verified=checked, metric_cells=540, direct_suffix_checks=15360)
    (a.data/"fixed_support_context_validation.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))
    print(contrasts[contrasts.target == "covid_political"].to_string(index=False))


if __name__ == "__main__":
    main()
