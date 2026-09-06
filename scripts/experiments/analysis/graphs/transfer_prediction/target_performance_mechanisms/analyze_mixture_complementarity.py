"""Complete-grid mixture diagnostics; associations are descriptive, not causal."""
import argparse
import hashlib
from itertools import product
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experiments.setup.target_performance_mechanisms.prepare_mixture_complementarity import TARGETS, validate_lattice

STREAMS = ("original", "fresh")
KEY = ["stream", "target", "model_id"]
METRICS = ("roc_auc", "accuracy", "f1", "nll")
STRATA = ("all_correct", "all_wrong", "mixed_correctness")
RULES = ("probability", "logit")


def close(actual, expected, message, tol=1e-8):
    if not np.isfinite(actual) or not np.isfinite(expected) or abs(actual - expected) > tol:
        raise ValueError(message)


def complete_grid(frame, keys, expected, message):
    if len(frame) != len(expected) or frame.duplicated(keys).any() or set(map(tuple, frame[keys].to_numpy())) != expected:
        raise ValueError(message)


def validate_artifacts(root, inputs):
    """Recompute algebraic and coverage gates from artifacts, not just DONE flags."""
    read = lambda name: json.loads((root / f"{name}.json").read_text())
    receipt, protocol = read("DONE"), read("protocol")
    if any(receipt.get(k) != v for k, v in {"models": 54, "verified_model_target_stream_cells": 540,
            "mixture_comparisons": 450, "error_strata": 1350}.items()) or not all(receipt.get(k) is True for k in
            ("all_logits_reproduce_metrics", "all_specialist_mixture_inputs_identical", "complete_both_streams")):
        raise ValueError("complete prediction-analysis receipt required")
    if any(protocol.get(k) is not False for k in ("new_training", "ensemble_weights_fitted", "query_labels_used_for_classifier_fitting", "causal_interference_claim")):
        raise ValueError("unfitted historical diagnostic protocol required")
    if protocol.get("training_seeds") != [0] or protocol.get("ensemble_training_and_inference_cost_multiplier") != [2, 8]:
        raise ValueError("single-seed compute contract differs")
    reference = pd.read_csv(inputs / "classification_long.tsv", sep="\t")
    models = validate_lattice(reference, pd.read_csv(inputs / "model_list.tsv", sep="\t"))
    manifest = json.loads((inputs / "manifest.json").read_text())
    for name, field in (("classification_long.tsv", "snapshot_metrics_sha256"), ("model_list.tsv", "snapshot_models_sha256")):
        if hashlib.sha256((inputs / name).read_bytes()).hexdigest() != manifest[field]:
            raise ValueError("frozen historical snapshot changed")
    sources = models.set_index("model_id").source_set.to_dict()
    checkpoint = models.set_index("model_id").checkpoint.to_dict()
    mixtures = {m for m, s in sources.items() if len(s) > 1}
    grids = {"model_metrics": set(product(STREAMS, TARGETS, sources)),
             "comparisons": set(product(STREAMS, TARGETS, mixtures)),
             "prediction_inventory": set(product(STREAMS, TARGETS, sources)),
             "error_strata": set(product(STREAMS, TARGETS, mixtures, STRATA)),
             "input_inventory": set(product(STREAMS, TARGETS))}
    tables = {name: pd.DataFrame(read(name)) for name in grids}
    for name, frame in tables.items():
        keys = ["stream", "target"] if name == "input_inventory" else KEY + (["stratum"] if name == "error_strata" else [])
        complete_grid(frame, keys, grids[name], f"incomplete or duplicated {name} grid")
    inputs_by_key = tables["input_inventory"].set_index(["stream", "target"])
    expected_queries = reference.groupby("dataset").queries.first().to_dict()
    for row in tables["input_inventory"].itertuples():
        if row.episodes != 128 or row.queries != expected_queries[row.target] or len(row.batch_sha256) != 32:
            raise ValueError("cached input count contract differs")
        if row.production_metric_uses_global_binary_labels != (row.target != "facebook_page_reference"):
            raise ValueError("production metric label space differs")
    for target in TARGETS:
        a, b = (inputs_by_key.loc[(s, target)] for s in STREAMS)
        if a.episode_fingerprint == b.episode_fingerprint or a.batch_sha256 == b.batch_sha256 or a.graph_path != b.graph_path:
            raise ValueError("distinct episode streams on the same graph required")
        if a.episode_fingerprint != reference[reference.dataset == target].episode_fingerprint.iloc[0]:
            raise ValueError("original episodes differ from historical reference")
    inventory = tables["prediction_inventory"]
    for row in inventory.itertuples():
        inp = inputs_by_key.loc[(row.stream, row.target)]
        if row.checkpoint != checkpoint[row.model_id] or row.queries != inp.queries or row.episode_fingerprint != inp.episode_fingerprint:
            raise ValueError("prediction identity or target inputs differ")
        for value in (row.weights_sha256, row.prediction_file_sha256):
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("invalid prediction/weight digest")
        if not 0 <= row.metric_reproduction_max_abs_error <= 1e-6:
            raise ValueError("saved logits failed their official metric reproduction")
    if not inventory.groupby("model_id").weights_sha256.nunique().eq(1).all():
        raise ValueError("checkpoint weights differ across targets or streams")
    for name in ("model_metrics", "comparisons", "error_strata"):
        for row in tables[name].itertuples():
            if tuple(row.sources) != sources[row.model_id] or row.source_count != len(sources[row.model_id]):
                raise ValueError("source composition differs from frozen model inventory")
            if name != "model_metrics" and row.target_seen != (row.target in row.sources):
                raise ValueError("target membership differs")
            if name == "comparisons" and row.queries != expected_queries[row.target]:
                raise ValueError("comparison query count differs")
    scores = tables["model_metrics"].set_index(KEY)
    for metric in METRICS:
        values = scores[metric]
        if not np.isfinite(values).all() or (values < 0).any() or (metric != "nll" and (values > 1).any()):
            raise ValueError("invalid model metric")
    # Every original full-model metric is tied back to the frozen aggregate table.
    prior = reference.set_index(["dataset", "model_id"])
    for row in tables["model_metrics"].query("stream == 'original'").itertuples():
        for metric in ("roc_auc", "accuracy", "f1"):
            close(getattr(row, metric), prior.loc[(row.target, row.model_id), metric],
                  "historical official metric parity failed", 1e-5 if metric == "roc_auc" else 1e-6)
    singleton = {s[0]: m for m, s in sources.items() if len(s) == 1}
    strata = tables["error_strata"].set_index(KEY + ["stratum"]).sort_index()
    for row in tables["comparisons"].itertuples():
        key = (row.stream, row.target, row.model_id)
        for metric in METRICS:
            constituent = np.array([scores.loc[(row.stream, row.target, singleton[s]), metric] for s in row.sources])
            for stat, op in (("mean", np.mean), ("min", np.min), ("max", np.max)):
                close(getattr(row, f"constituent_{stat}_{metric}"), op(constituent), "constituent metric mismatch")
            value = getattr(row, f"mixture_{metric}")
            close(value, scores.loc[key, metric], "mixture metric mismatch")
            for rule in RULES:
                ensemble = getattr(row, f"{rule}_ensemble_{metric}")
                if not np.isfinite(ensemble) or ensemble < 0 or (metric != "nll" and ensemble > 1):
                    raise ValueError("invalid ensemble metric")
                close(getattr(row, f"mixture_minus_{rule}_ensemble_{metric}"), value - ensemble, "ensemble delta mismatch")
        subset = strata.loc[key]
        close(subset.queries.sum(), row.queries, "strata do not partition queries")
        for s in subset.itertuples():
            if s.queries < 0 or int(s.queries) != s.queries:
                raise ValueError("invalid stratum count")
            close(s.fraction_of_queries, s.queries / row.queries, "stratum fraction differs")
            for method in ("mixture", "probability_ensemble", "logit_ensemble"):
                correct = getattr(s, f"{method}_correct")
                accuracy = getattr(s, f"{method}_accuracy")
                if not 0 <= correct <= s.queries or int(correct) != correct:
                    raise ValueError("invalid stratum correct count")
                if s.queries:
                    close(accuracy, correct / s.queries, "stratum conditional rate differs")
                elif pd.notna(accuracy):
                    raise ValueError("empty-stratum accuracy must remain undefined")
        for method in ("mixture", "probability_ensemble", "logit_ensemble"):
            close(subset[f"{method}_correct"].sum() / row.queries, getattr(row, f"{method}_accuracy"), "strata do not reproduce total accuracy")
        aw, ac, mixed = (subset.loc[s] for s in ("all_wrong", "all_correct", "mixed_correctness"))
        for rule in RULES:
            if aw[f"{rule}_ensemble_correct"] != 0 or ac[f"{rule}_ensemble_correct"] != ac.queries:
                raise ValueError("binary ensemble unanimity violated")
            agreement = getattr(row, f"mixture_{rule}_ensemble_decision_agreement")
            l1 = getattr(row, f"mixture_{rule}_ensemble_probability_l1")
            if not 0 <= agreement <= 1 or not 0 <= l1 <= 2:
                raise ValueError("invalid model/ensemble agreement")
        close(row.constituent_query_oracle_accuracy, 1 - aw.queries / row.queries, "oracle fraction differs")
        close(row.mixture_fixes_all_wrong_fraction, aw.mixture_correct / row.queries, "shared-error fix fraction differs")
        close(row.mixture_harms_all_correct_fraction, (ac.queries - ac.mixture_correct) / row.queries, "unanimous-correct harm fraction differs")
        close(row.mixed_correctness_fraction, mixed.queries / row.queries, "mixed fraction differs")
        k = row.source_count
        if not 0 <= row.mean_pairwise_disagreement <= row.mixed_correctness_fraction * k / (2 * (k - 1)) + 1e-8:
            raise ValueError("pairwise disagreement exceeds binary bound")
        if k == 2:
            close(row.mean_pairwise_disagreement, row.mixed_correctness_fraction, "pair disagreement must equal mixed correctness")
    return tables


def rank_association(frame, outcome):
    columns = ["mean_pairwise_disagreement", outcome, "constituent_mean_roc_auc", "constituent_auc_range"]
    if len(frame) != 28 or not np.isfinite(frame[columns]).all().all():
        raise ValueError("association requires the complete 28 foreign pairs")
    ranks = frame[columns].rank(method="average").to_numpy()
    x, y = ranks[:, 0], ranks[:, 1]
    corr = lambda a, b: None if np.std(a) < 1e-10 or np.std(b) < 1e-10 else float(np.corrcoef(a, b)[0, 1])
    raw = corr(x, y)
    design = np.column_stack([np.ones(len(frame)), ranks[:, 2:]])
    rank = np.linalg.matrix_rank(design)
    adjusted = None
    reason = "rank-deficient nuisance design" if rank != design.shape[1] else None
    if reason is None:
        rx = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
        ry = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
        adjusted = corr(rx, ry)
        if adjusted is None:
            reason = "zero residual rank variance"
    return {"outcome": outcome, "pairs": len(frame), "spearman": raw, "partial_rank_correlation": adjusted,
            "partial_undefined_reason": reason, "nuisance_design_rank": int(rank),
            "independent_pair_inference": False, "training_seeds": 1}


def summarize(tables):
    comparisons = tables["comparisons"].copy()
    comparisons["constituent_auc_range"] = comparisons.constituent_max_roc_auc - comparisons.constituent_min_roc_auc
    comparisons["mixture_minus_constituent_mean_roc_auc"] = comparisons.mixture_roc_auc - comparisons.constituent_mean_roc_auc
    comparisons["mixture_minus_best_constituent_roc_auc"] = comparisons.mixture_roc_auc - comparisons.constituent_max_roc_auc
    comparisons["shared_error_net_accuracy_contribution"] = comparisons.mixture_fixes_all_wrong_fraction - comparisons.mixture_harms_all_correct_fraction
    comparisons["mixed_error_net_accuracy_contribution"] = comparisons.mixture_minus_probability_ensemble_accuracy - comparisons.shared_error_net_accuracy_contribution
    outputs, associations = [], []
    measures = ["mixture_roc_auc", "constituent_mean_roc_auc", "constituent_max_roc_auc", "probability_ensemble_roc_auc", "logit_ensemble_roc_auc",
                "mixture_minus_constituent_mean_roc_auc", "mixture_minus_best_constituent_roc_auc", "mixture_minus_probability_ensemble_roc_auc",
                "mixture_minus_logit_ensemble_roc_auc", "mixture_minus_probability_ensemble_accuracy", "mixture_minus_probability_ensemble_nll",
                "mean_pairwise_disagreement", "mixture_fixes_all_wrong_fraction", "mixture_harms_all_correct_fraction",
                "shared_error_net_accuracy_contribution", "mixed_error_net_accuracy_contribution"]
    for (stream, target, count), group in comparisons.groupby(["stream", "target", "source_count"]):
        for panel, subset in (("all", group), ("foreign", group[~group.target_seen]), ("target_seen", group[group.target_seen])):
            expected = {2: {"all": 36, "foreign": 28, "target_seen": 8}, 8: {"all": 9, "foreign": 1, "target_seen": 8}}[count][panel]
            if len(subset) != expected:
                raise ValueError("incomplete source-membership panel")
            row = {"stream": stream, "target": target, "source_count": count, "panel": panel, "mixtures": len(subset)}
            for column in measures:
                row[f"mean_{column}"] = float(subset[column].mean())
            for column in ("mixture_minus_probability_ensemble_roc_auc", "mixture_minus_best_constituent_roc_auc"):
                row[f"positive_{column}"] = int(subset[column].gt(0).sum())
                row[f"negative_{column}"] = int(subset[column].lt(0).sum())
            outputs.append(row)
            if panel == "foreign" and count == 2:
                for outcome in ("mixture_minus_constituent_mean_roc_auc", "mixture_minus_probability_ensemble_roc_auc"):
                    associations.append({"stream": stream, "target": target, **rank_association(subset, outcome)})
    matched = comparisons[comparisons.stream == "original"].merge(comparisons[comparisons.stream == "fresh"],
        on=["target", "model_id", "source_count", "target_seen"], suffixes=("_original", "_fresh"), validate="one_to_one")
    if len(matched) != 225:
        raise ValueError("cross-stream comparison incomplete")
    for outcome in ("mixture_minus_probability_ensemble_roc_auc", "mixture_minus_best_constituent_roc_auc"):
        matched[f"{outcome}_positive_both"] = matched[f"{outcome}_original"].gt(0) & matched[f"{outcome}_fresh"].gt(0)
        matched[f"{outcome}_negative_both"] = matched[f"{outcome}_original"].lt(0) & matched[f"{outcome}_fresh"].lt(0)
    return comparisons, pd.DataFrame(outputs), pd.DataFrame(associations), matched


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).parent / "data")
    args = parser.parse_args()
    root = args.data / "mixture_complementarity_predictions"
    tables = validate_artifacts(root, args.data / "mixture_complementarity_inputs")
    comparisons, summary, associations, matched = summarize(tables)
    for name, frame in (("comparisons", comparisons), ("summary", summary), ("associations", associations), ("stream_agreement", matched)):
        frame.to_csv(args.data / f"mixture_complementarity_{name}.csv", index=False)
    validation = {"model_cells": 540, "mixture_comparisons": 450, "strata": 1350, "input_sets": 10,
                  "primary_foreign_pairs_per_stream": 140, "foreign_loo_per_stream": 5, "training_seeds": [0],
                  "primary_ensemble": "fixed equal probability", "causal_interference_claim": False,
                  "rank_associations": "descriptive within-target; no independent-pair inference",
                  "role_corrected_runs_included": False, "undefined_partial_associations": int(associations.partial_rank_correlation.isna().sum()),
                  "raw_artifact_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob("*.json"))}}
    (args.data / "mixture_complementarity_validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    columns = ["stream", "target", "source_count", "mixtures", "mean_mixture_minus_probability_ensemble_roc_auc", "mean_shared_error_net_accuracy_contribution"]
    print(summary[summary.panel == "foreign"][columns].to_string(index=False))


if __name__ == "__main__":
    main()
