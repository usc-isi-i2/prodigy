"""Independent validation and fixed primary tests for natural support replay."""
import argparse
import hashlib
from itertools import product
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd

from .analyze_mixture_complementarity import complete_grid, METRICS, STREAMS
from .analyze_trajectories import PANEL

KEY = ["stream", "target", "model_id"]


def validate_plan(plan, expected_draws=8, expected_batches=32, expected_shots=10):
    payload = {k: v for k, v in plan.items() if k != "sha256"}
    if hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest() != plan["sha256"]:
        raise ValueError("sampling plan content hash differs")
    if plan["draws"] != expected_draws or len(plan["inputs"]) != expected_batches or len(plan["mappings"]) != expected_batches:
        raise ValueError("incomplete support sampling plan")
    pool, identity_labels, offset, ep_offset = [], {}, 0, 0
    for bi, spec in enumerate(plan["inputs"]):
        q, task, center, label, classes = [np.asarray(spec[k]) for k in ("query", "tasks", "centers", "support_labels", "classes")]
        if (spec["offset"] != offset or spec["episode_offset"] != ep_offset or q.dtype != bool
                or classes.ndim != 2 or classes.shape[1] != 2 or (classes[:, 0] == classes[:, 1]).any()
                or any(a.shape != q.shape for a in (task, center, label)) or (center < 0).any()
                or set(task) != set(range(len(classes))) or not (label[q] == -1).all()
                or not np.isin(label[~q], [0, 1]).all()):
            raise ValueError("invalid query-blind input metadata")
        for row in np.flatnonzero(~q):
            value = int(classes[task[row], label[row]])
            identity = int(center[row])
            if identity in identity_labels and identity_labels[identity] != value:
                raise ValueError("conflicting support labels for one identity")
            identity_labels[identity] = value
            pool.append(dict(flat=offset+int(row), center=identity, global_label=value, episode=ep_offset+int(task[row])))
        offset += len(q)
        ep_offset += len(classes)
    if len(pool) != plan["pool_occurrences"] or len(identity_labels) != plan["pool_unique_identities"]:
        raise ValueError("pool inventory differs")
    audits = {(r["batch"], r["episode"], r["draw"], r["global_class"]): r for r in plan["audits"]}
    if len(audits) != len(plan["audits"]) or len(audits) != ep_offset * 2 * expected_draws:
        raise ValueError("incomplete unique support-draw audit")
    rng = random.Random(plan["seed"])
    checked = 0
    for bi, spec in enumerate(plan["inputs"]):
        q, task, center, y, classes = [np.asarray(spec[k]) for k in ("query", "tasks", "centers", "support_labels", "classes")]
        mapping = np.asarray(plan["mappings"][bi])
        if mapping.shape != (expected_draws, len(q)) or not (mapping[:, q] == (spec["offset"] + np.flatnonzero(q))[None]).all():
            raise ValueError("query identity changed or mapping incomplete")
        for ep in range(len(classes)):
            origin = spec["episode_offset"] + ep
            forbidden = set(center[(task == ep) & q])
            for local, global_class in enumerate(classes[ep]):
                slots = np.flatnonzero((task == ep) & ~q & (y == local))
                if len(slots) != expected_shots:
                    raise ValueError("support count differs from fixed shot budget")
                eligible = {}
                for record in pool:
                    if record["global_label"] == global_class and record["episode"] != origin and record["center"] not in forbidden:
                        eligible.setdefault(record["center"], []).append(record)
                if len(eligible) < len(slots):
                    raise ValueError("insufficient unique eligible identities")
                for draw in range(expected_draws):
                    ids = rng.sample(sorted(eligible), len(slots))
                    chosen = [rng.choice(eligible[i])["flat"] for i in ids]
                    audit = audits[(bi, origin, draw, int(global_class))]
                    if (chosen != mapping[draw, slots].tolist() or chosen != audit["selected"]
                            or audit["eligible_unique_ids"] != len(eligible) or audit["shots"] != len(slots)
                            or audit["overlap_with_original"] != len(set(center[slots]).intersection(ids))):
                        raise ValueError("sampling differs from fixed query-blind RNG/eligibility contract")
                    checked += len(chosen)
    return dict(checked_support_positions=checked, query_occurrences=sum(sum(s["query"]) for s in plan["inputs"]))


def validate_episode_accounting(episodes, cells):
    expected = {(s, t, m, e, d) for s, t, m in cells[KEY].drop_duplicates().itertuples(index=False, name=None)
                for e in range(128) for d in range(8)}
    complete_grid(episodes, KEY + ["episode", "draw"], expected, "complete episode/draw accounting required")
    if (not np.isfinite(episodes[["support_cv_nll", "query_nll"]]).all().all()
            or not episodes[["support_cv_nll", "query_nll"]].ge(0).all().all()
            or not episodes.queries.gt(0).all() or not episodes.query_correct.between(0, episodes.queries).all()):
        raise ValueError("invalid episode scores or counts")
    indexed_cells = cells.set_index(KEY).sort_index()
    for key, group in episodes.groupby(KEY):
        counts = group.groupby("episode").queries.nunique()
        if not counts.eq(1).all():
            raise ValueError("query cohort changes across natural draws")
        cv = group.pivot(index="episode", columns="draw", values="support_cv_nll").sort_index(axis=1)
        chosen = cv.idxmin(1)
        expected_selected = group.episode.map(chosen).to_numpy() == group.draw.to_numpy()
        if not np.array_equal(group.selected.to_numpy(dtype=bool), expected_selected):
            raise ValueError("support selection differs from fixed support-only rule")
        relevant = indexed_cells.loc[key].reset_index(drop=True)
        groups = [("draw", draw, part) for draw, part in group.groupby("draw")]
        groups.append(("support_cv_selected", -1, group[group.selected]))
        for condition, draw, part in groups:
            row = relevant[relevant.condition.eq(condition) & relevant.draw.eq(draw)].iloc[0]
            total = part.queries.sum()
            if total != row.queries or not np.isclose(part.query_correct.sum() / total, row.accuracy, rtol=0, atol=1e-12):
                raise ValueError("query count/correctness differs from saved metrics")
            if not np.isclose(np.average(part.query_nll, weights=part.queries), row.nll, rtol=0, atol=1e-6):
                raise ValueError("query loss differs from saved metrics")


def validate(root, data):
    read = lambda name: json.loads((root / (name + ".json")).read_text())
    done, protocol = read("DONE"), read("protocol")
    expected = dict(complete=True, smoke=False, model_target_stream_cells=60, metric_cells=660,
                    episode_draw_cells=61440, plans=10, draws=8, direct_checks=60,
                    all_query_vectors_unchanged=True, all_selectors_query_blind=True, all_weights_unchanged=True)
    if any(done.get(k) != v for k, v in expected.items()) or protocol["new_training"] is not False or protocol["smoke"] is not False:
        raise ValueError("complete non-smoke natural-support study required")
    arms = pd.DataFrame(json.loads((data / "member_training_verified/arms.json").read_text()))
    arms = arms[arms.policy.eq("lowest_sorted")]
    complete_grid(arms, ["source", "seed"], set(product(("cp_hk", "ukr_rus"), range(3))), "all six verified source/seed controls required")
    ids = arms.model_id.tolist()
    cells, audits, cohorts, plans = [pd.DataFrame(read(n)) for n in ("metrics", "audits", "cohorts", "plan_inventory")]
    grid = set(product(STREAMS, PANEL, ids))
    conditions = [("draw", d) for d in range(8)] + [(v, -1) for v in ("baseline", "support_cv_selected", "probability_ensemble")]
    complete_grid(cells, KEY + ["condition", "draw"], {(*k, *v) for k in grid for v in conditions}, "all 660 metrics required")
    complete_grid(audits, KEY, grid, "all 60 model/input checks required")
    complete_grid(plans, KEY[:2], set(product(STREAMS, PANEL)), "all 10 shared support plans required")
    if (not audits.baseline_batches_bit_exact.eq(32).all() or not audits.query_invariance_batches.eq(256).all()
            or not audits.support_cv_query_blind.eq(True).all() or not audits.all_weights_unchanged.eq(True).all()
            or not audits.direct_checks.eq(1).all() or not audits.max_direct_error.between(0, 1e-5).all()
            or done["maximum_direct_error"] != audits.max_direct_error.max()):
        raise ValueError("missing or failed model/factorization validity checks")
    if not np.isfinite(cells[list(METRICS)]).all().all() or not cells.nll.ge(0).all() or not cells[["roc_auc", "accuracy", "f1"]].apply(lambda x: x.between(0, 1)).all().all():
        raise ValueError("invalid output scores")
    by_model, by_plan = arms.set_index("model_id"), plans.set_index(KEY[:2])
    for row in pd.concat([cells, audits], ignore_index=True).itertuples():
        arm = by_model.loc[row.model_id]
        if any(getattr(row, k) != arm[k] for k in ("checkpoint", "source", "seed")) or row.weights_sha256 != arm.final_sha256:
            raise ValueError("model differs from independently verified training")
        if row.plan_sha256 != by_plan.loc[(row.stream, row.target), "plan_sha256"]:
            raise ValueError("models use different support plans")
    ref = pd.read_csv(data / "member_replay_cells.csv")
    ref = ref[ref.policy.eq("lowest_sorted") & ref.decoder.eq("full_model")].rename(columns={"dataset": "target"})
    match = cells[cells.condition.eq("baseline")].merge(ref, on=KEY, suffixes=("", "_reference"), validate="one_to_one")
    if len(match) != 60 or any(not match[k].eq(match[k + "_reference"]).all() for k in ("checkpoint", "weights_sha256", "episode_fingerprint", "queries")):
        raise ValueError("independent baseline identity differs")
    if any((match[m] - match[m + "_reference"]).abs().max() > 1e-6 for m in METRICS):
        raise ValueError("independent baseline scores differ")
    for key, group in cells.groupby(KEY):
        if group.episode_fingerprint.nunique() != 1 or group.queries.nunique() != 1:
            raise ValueError("query inputs change across support conditions")
    audit_match = audits.merge(cells[cells.condition.eq("baseline")], on=KEY, suffixes=("", "_cell"), validate="one_to_one")
    if any(not audit_match[k].eq(audit_match[k + "_cell"]).all() for k in ("queries", "episode_fingerprint", "plan_sha256")):
        raise ValueError("model audit uses different queries or support plan")
    checked = 0
    for p in plans.itertuples():
        plan = json.loads((root / "plans" / f"{p.stream}_{p.target}.json").read_text())
        seed = 9102026 + (100003 if p.stream == "fresh" else 0) + sum((i+1)*ord(c) for i, c in enumerate(p.target))
        if plan["sha256"] != p.plan_sha256 or plan["seed"] != seed:
            raise ValueError("support plan identity or deterministic seed differs")
        check = validate_plan(plan)
        expected_n = cells[cells.stream.eq(p.stream) & cells.target.eq(p.target)].queries.unique()
        if len(expected_n) != 1 or check["query_occurrences"] != expected_n[0]:
            raise ValueError("plan query cohort differs from metric cohort")
        checked += check["checked_support_positions"]
    episodes = pd.read_csv(root / "episode_scores.csv")
    validate_episode_accounting(episodes, cells)
    if cohorts.duplicated(KEY + ["cohort"]).any() or set(cohorts.cohort) != {"all", "no_context", "has_context"}:
        raise ValueError("invalid natural-support sensitivity cohorts")
    complete_grid(cohorts[cohorts.cohort.eq("all")], KEY, grid, "all 60 complete sensitivity cells required")
    counts = cohorts[["queries", "correctness_flips", "always_correct", "always_wrong"]]
    if not np.isfinite(counts).all().all() or not counts.ge(0).all().all() or not counts.eq(counts.astype(int)).all().all():
        raise ValueError("sensitivity counts must be nonnegative integers")
    if not cohorts.mean_probability_variance.between(0, .25).all() or not (cohorts.correctness_flips + cohorts.always_correct + cohorts.always_wrong).eq(cohorts.queries).all():
        raise ValueError("invalid sensitivity or correctness partition")
    variance_residuals = []
    for key, group in cohorts.groupby(KEY):
        total = group[group.cohort.eq("all")].iloc[0]
        parts = group[~group.cohort.eq("all")]
        if any(parts[k].sum() != total[k] for k in ("queries", "correctness_flips", "always_correct", "always_wrong")):
            raise ValueError("context strata do not sum to full query cohort")
        weighted = np.average(parts.mean_probability_variance, weights=parts.queries)
        variance_residuals.append(abs(weighted - total.mean_probability_variance))
        # Runtime means/variances were reduced in float32. Repartitioning the
        # reduction can differ by a few rounding units; the full-model forward
        # parity tolerance is unchanged. Saved-prediction float64 audit confirms
        # the largest observed summary residual is rounding (1.074901e-8).
        if not np.isclose(weighted, total.mean_probability_variance, rtol=4*np.finfo(np.float32).eps, atol=1e-12):
            raise ValueError("context variance does not sum to global variance")
    return cells, episodes, cohorts, dict(metric_cells=len(cells), episode_draw_cells=len(episodes),
                                          support_positions_reproduced=checked, max_direct_error=float(audits.max_direct_error.max()),
                                          max_cohort_variance_reduction_residual=float(max(variance_residuals)))


def summarize(cells, episodes, cohorts):
    effects, relationships = [], []
    for key, group in cells.groupby(KEY):
        row = dict(zip(KEY, key)) | {k: group.iloc[0][k] for k in ("source", "seed")}
        draws = group[group.condition.eq("draw")]
        for condition in ("baseline", "support_cv_selected", "probability_ensemble"):
            result = group[group.condition.eq(condition)].iloc[0]
            effects.append(row | dict(condition=condition) | {m: float(result[m]) for m in METRICS}
                           | {"mean_draw_" + m: float(draws[m].mean()) for m in METRICS}
                           | {"delta_" + m: float(result[m] - draws[m].mean()) for m in METRICS})
    for key, group in episodes.groupby(KEY):
        centered = group[["support_cv_nll", "query_nll"]] - group.groupby("episode")[["support_cv_nll", "query_nll"]].transform("mean")
        a, b = centered.support_cv_nll.to_numpy(), centered.query_nll.to_numpy()
        denominator = np.linalg.norm(a) * np.linalg.norm(b)
        relationships.append(dict(zip(KEY, key)) | dict(within_episode_loss_correlation=float(a @ b / denominator) if denominator else None))
    effects = pd.DataFrame(effects)
    sensitivity = cohorts[cohorts.target.eq("covid_political") & cohorts.cohort.eq("all")]
    wide = sensitivity.pivot(index=["stream", "seed"], columns="source", values="mean_probability_variance")
    if len(wide) != 6 or set(wide.columns) != {"cp_hk", "ukr_rus"}:
        raise ValueError("all matched political sensitivity predictions required")
    contrasts = wide.assign(hong_kong_minus_ukraine=wide.cp_hk-wide.ukr_rus).reset_index()
    selection = effects[effects.source.eq("cp_hk") & effects.target.eq("covid_political") & effects.condition.eq("support_cv_selected")]
    if len(selection) != 6:
        raise ValueError("all political support-selection predictions required")
    result = dict(hong_kong_more_sensitive_every_seed_both_streams=bool(contrasts.hong_kong_minus_ukraine.gt(0).all()),
                  support_cv_selection_lowers_hong_kong_nll_every_seed_both_streams=bool(selection.delta_nll.lt(0).all()),
                  independent_initialization_seeds=3, unseen_target_validation=False, benchmark_fair_10shot_remedy=False,
                  source_training_causal_effect_established=False)
    return effects, pd.DataFrame(relationships), contrasts, result


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--data", type=Path, default=Path(__file__).parent / "data")
    args = p.parse_args()
    cells, episodes, cohorts, validation = validate(args.data / "natural_support_replay", args.data)
    effects, relationships, contrasts, primary = summarize(cells, episodes, cohorts)
    for name, frame in (("cells", cells), ("selection_effects", effects), ("quality_relationships", relationships), ("political_sensitivity", contrasts)):
        frame.to_csv(args.data / f"natural_support_{name}.csv", index=False)
    cohorts.assign(correctness_flip_fraction=cohorts.correctness_flips / cohorts.queries).to_csv(
        args.data / "natural_support_sensitivity_cohorts.csv", index=False)
    result = validation | primary
    (args.data / "natural_support_validation.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
