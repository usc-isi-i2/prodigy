"""Saved-update budgets are not matched FLOPs, capacity, or training inputs."""
import argparse
import hashlib
from itertools import product
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experiments.setup.final_core.core_plan import SOURCES
from .analyze_trajectories import STEPS
from .analyze_mixture_complementarity import STREAMS, TARGETS, METRICS, STRATA, RULES, close, complete_grid, validate_artifacts

KEY = ["stream", "target", "model_id", "specialist_step"]
TRAIN_FIELDS = {"batch_size": 4, "epochs": 1, "dataset_len_cap": 2500, "seed": 0,
                "task_name": "neighbor_matching", "n_way": 30, "n_shots": 3, "n_query": 4}


def budget_step(count):
    if count not in (2, 8):
        raise ValueError("only pair and LOO budgets are declared")
    return max(step for step in STEPS if count * step <= 2500)


def validate_training_audit(records, models):
    terminal = models[models.step.eq(2500)].set_index("model_id")
    if len(records) != 54 or len({r["model_id"] for r in records}) != 54 or {r["model_id"] for r in records} != set(terminal.index):
        raise ValueError("54 actual training configs required")
    for record in records:
        model, config = terminal.loc[record["model_id"]], record["parameter_contract"]
        if record["checkpoint"] != model.checkpoint or any(config.get(k) != v for k, v in TRAIN_FIELDS.items()):
            raise ValueError("actual training budget or checkpoint differs")
        if sorted(config["neighbor_sampling_source_subset"].split(",")) != sorted(model.sources):
            raise ValueError("actual training source restriction differs")
        if [int(s) for s in config["checkpoint_steps"].split(",")] != list(STEPS) or len(record["config_sha256"]) != 64:
            raise ValueError("actual saved-step schedule or config digest differs")


def model_registry(manifest, inventory, prior):
    if len(manifest) != 36 or manifest.model_id.duplicated().any() or len(inventory) != 36 or inventory.model_id.duplicated().any():
        raise ValueError("36 unique trajectory checkpoints required")
    models = manifest.merge(inventory, on=["model_id", "checkpoint"], validate="one_to_one")
    if len(models) != 36 or not models.finite.eq(True).all():
        raise ValueError("trajectory checkpoint path/finite inventory differs")
    models["step"] = models.checkpoint.str.extract(r"state_dict_(\d+)\.ckpt$").astype(int)
    if set(map(tuple, models[["sources", "step"]].to_numpy())) != set(product(SOURCES, STEPS)):
        raise ValueError("source by saved-step inventory differs")
    models["sources"] = models.sources.map(lambda source: [source])
    models["source_count"] = 1
    old = prior["model_metrics"].drop_duplicates("model_id")[["model_id", "sources", "source_count"]].merge(
        prior["prediction_inventory"].drop_duplicates("model_id")[["model_id", "checkpoint", "weights_sha256"]], on="model_id")
    for row in models[models.step.eq(2500)].itertuples():
        ref = old[old.model_id.eq(row.model_id)]
        if len(ref) != 1 or any(ref.iloc[0][k] != getattr(row, k) for k in ("sources", "checkpoint", "weights_sha256")):
            raise ValueError("terminal singleton identity differs from previous mixture analysis")
    cols = ["model_id", "sources", "source_count", "step", "checkpoint", "weights_sha256"]
    result = pd.concat([models[cols], old[old.source_count.gt(1)].assign(step=2500)[cols]], ignore_index=True)
    if len(result) != 81 or result.model_id.duplicated().any() or result.weights_sha256.nunique() != 81:
        raise ValueError("81 distinct historical model states required")
    return result


def match_reference(actual, reference, keys, message):
    actual = actual.set_index(keys).sort_index()
    reference = reference.set_index(keys).sort_index()
    if actual.index.has_duplicates or not actual.index.equals(reference.index):
        raise ValueError(message + " grid")
    for column in reference:
        for a, b in zip(actual[column], reference[column]):
            if isinstance(b, (float, int, np.number)) and not isinstance(b, (bool, np.bool_)):
                if pd.isna(a) and pd.isna(b):
                    continue
                close(a, b, message + ": " + column, 1e-8)
            elif a != b:
                raise ValueError(message + ": " + column)


def validate_budget(tables, models, prior):
    identifiers = list(models.model_id)
    mixtures = models[models.source_count.gt(1)]
    lookup = models.set_index("model_id")
    expected = set(product(STREAMS, TARGETS, identifiers))
    for name in ("model_metrics", "prediction_inventory"):
        complete_grid(tables[name], KEY[:3], expected, "complete 810-cell " + name)
    complete_grid(tables["comparisons"], KEY, set(product(STREAMS, TARGETS, mixtures.model_id, STEPS)), "complete 1800-cell comparisons")
    complete_grid(tables["error_strata"], KEY + ["stratum"], set(product(STREAMS, TARGETS, mixtures.model_id, STEPS, STRATA)), "complete 5400-cell strata")
    match_reference(tables["input_inventory"], prior["input_inventory"], ["stream", "target"], "cached inputs differ")
    inputs = tables["input_inventory"].set_index(["stream", "target"])
    for row in tables["prediction_inventory"].itertuples():
        model, inp = lookup.loc[row.model_id], inputs.loc[(row.stream, row.target)]
        if any(getattr(row, k) != model[k] for k in ("checkpoint", "weights_sha256")) or row.queries != inp.queries or row.episode_fingerprint != inp.episode_fingerprint:
            raise ValueError("actual prediction checkpoint/input identity differs")
        if len(row.prediction_file_sha256) != 64 or not 0 <= row.metric_reproduction_max_abs_error <= 1e-6:
            raise ValueError("prediction-file or production-metric audit failed")
    scores = tables["model_metrics"].set_index(KEY[:3])
    for row in tables["model_metrics"].itertuples():
        if any(getattr(row, k) != lookup.loc[row.model_id, k] for k in ("sources", "source_count", "step")):
            raise ValueError("model source or step differs")
        for metric in METRICS:
            value = getattr(row, metric)
            if not np.isfinite(value) or value < 0 or (metric != "nll" and value > 1):
                raise ValueError("invalid model metric")
    singles = {(r.sources[0], r.step): r.model_id for r in models[models.source_count.eq(1)].itertuples()}
    strata = tables["error_strata"].set_index(KEY + ["stratum"]).sort_index()
    for row in tables["comparisons"].itertuples():
        model = lookup.loc[row.model_id]
        if row.sources != model.sources or row.source_count != model.source_count or row.target_seen != (row.target in row.sources):
            raise ValueError("comparison composition differs")
        if (row.total_specialist_updates != row.source_count * row.specialist_step or row.mixture_updates != 2500
                or row.budget_selected != (row.specialist_step == budget_step(row.source_count))
                or row.specialist_training_episodes != 4 * row.total_specialist_updates or row.mixture_training_episodes != 10000
                or row.inference_model_multiplier != row.source_count):
            raise ValueError("saved-update or inference budget differs")
        key = (row.stream, row.target, row.model_id)
        for metric in METRICS:
            constituents = [scores.loc[(row.stream, row.target, singles[(s, row.specialist_step)]), metric] for s in row.sources]
            for name, op in (("mean", np.mean), ("min", np.min), ("max", np.max)):
                close(getattr(row, f"constituent_{name}_{metric}"), op(constituents), "constituent arithmetic differs")
            close(getattr(row, f"mixture_{metric}"), scores.loc[key, metric], "fixed mixture metric differs")
            for rule in RULES:
                ensemble = getattr(row, f"{rule}_ensemble_{metric}")
                if not np.isfinite(ensemble) or ensemble < 0 or (metric != "nll" and ensemble > 1):
                    raise ValueError("invalid ensemble metric")
                close(getattr(row, f"mixture_minus_{rule}_ensemble_{metric}"), scores.loc[key, metric] - ensemble, "ensemble delta differs")
        subset = strata.loc[key + (row.specialist_step,)]
        if row.queries != inputs.loc[(row.stream, row.target), "queries"] or subset.queries.sum() != row.queries:
            raise ValueError("query strata/count mismatch")
        for s in subset.itertuples():
            if s.sources != row.sources or s.source_count != row.source_count or s.target_seen != row.target_seen or s.queries < 0 or int(s.queries) != s.queries:
                raise ValueError("stratum metadata/count differs")
            close(s.fraction_of_queries, s.queries / row.queries, "stratum fraction differs")
            for method in ("mixture", "probability_ensemble", "logit_ensemble"):
                correct, accuracy = getattr(s, method + "_correct"), getattr(s, method + "_accuracy")
                if not 0 <= correct <= s.queries or int(correct) != correct:
                    raise ValueError("invalid stratum correct count")
                if s.queries:
                    close(accuracy, correct / s.queries, "stratum conditional accuracy differs")
                elif pd.notna(accuracy):
                    raise ValueError("empty-stratum accuracy must be undefined")
        for method in ("mixture", "probability_ensemble", "logit_ensemble"):
            close(subset[method + "_correct"].sum() / row.queries, getattr(row, method + "_accuracy"), "error strata fail total accuracy")
        aw, ac = subset.loc["all_wrong"], subset.loc["all_correct"]
        for rule in RULES:
            if aw[rule + "_ensemble_correct"] != 0 or ac[rule + "_ensemble_correct"] != ac.queries:
                raise ValueError("binary ensemble unanimity violated")
        close(row.mixture_fixes_all_wrong_fraction, aw.mixture_correct / row.queries, "shared-error fixes differ")
        close(row.mixture_harms_all_correct_fraction, (ac.queries - ac.mixture_correct) / row.queries, "unanimous harms differ")
        close(row.constituent_query_oracle_accuracy, 1 - aw.queries / row.queries, "oracle fraction differs")
        close(row.mixed_correctness_fraction, subset.loc["mixed_correctness", "queries"] / row.queries, "mixed fraction differs")
        k = row.source_count
        if not 0 <= row.mean_pairwise_disagreement <= row.mixed_correctness_fraction * k / (2 * (k - 1)) + 1e-8:
            raise ValueError("pairwise disagreement bound violated")
        if k == 2:
            close(row.mean_pairwise_disagreement, row.mixed_correctness_fraction, "pair disagreement differs")
    for name in ("comparisons", "error_strata"):
        frame = tables[name]
        match_reference(frame[frame.specialist_step.eq(2500)], prior[name], KEY[:3] + (["stratum"] if name == "error_strata" else []), "terminal " + name + " reproduction")
    match_reference(tables["model_metrics"][tables["model_metrics"].step.eq(2500)], prior["model_metrics"], KEY[:3], "terminal model reproduction")


def summarize_budget(cells):
    cells = cells.copy()
    cells["ensemble_minus_mixture_auc"] = -cells.mixture_minus_probability_ensemble_roc_auc
    cells["ensemble_minus_best_constituent_auc"] = cells.probability_ensemble_roc_auc - cells.constituent_max_roc_auc
    outputs = []
    for key, group in cells.groupby(["stream", "target", "source_count", "specialist_step"]):
        for panel, subset in (("all", group), ("foreign", group[~group.target_seen]), ("target_seen", group[group.target_seen])):
            count = key[2]
            if len(subset) != {2: {"all": 36, "foreign": 28, "target_seen": 8}, 8: {"all": 9, "foreign": 1, "target_seen": 8}}[count][panel]:
                raise ValueError("complete budget panel required")
            row = dict(zip(("stream", "target", "source_count", "specialist_step"), key), panel=panel, cases=len(subset),
                       total_specialist_updates=count * key[3], budget_selected=key[3] == budget_step(count))
            for metric in ("mixture_roc_auc", "probability_ensemble_roc_auc", "logit_ensemble_roc_auc", "ensemble_minus_mixture_auc",
                           "ensemble_minus_best_constituent_auc", "mixture_minus_probability_ensemble_accuracy", "mixture_minus_probability_ensemble_nll"):
                row["mean_" + metric] = float(subset[metric].mean())
            row["ensemble_better"] = int(subset.ensemble_minus_mixture_auc.gt(0).sum())
            outputs.append(row)
    return cells, pd.DataFrame(outputs)


def main():
    root = Path(__file__).parent
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--data", type=Path, default=root / "data")
    args = parser.parse_args()
    export = args.data / "mixture_budget_predictions"
    read = lambda name: json.loads((export / f"{name}.json").read_text())
    receipt, protocol = read("DONE"), read("protocol")
    if any(receipt.get(k) != v for k, v in {"models": 81, "prediction_cells": 810, "comparisons": 1800, "error_strata": 5400,
           "complete_both_streams": True, "terminal_reference_reproduced": True, "training_configs_verified": True}.items()):
        raise ValueError("complete budget prediction receipt required")
    if protocol.get("steps") != list(STEPS) or any(protocol.get(k) is not False for k in
            ("new_training", "ensemble_weights_fitted", "matched_training_inputs", "matched_flops", "causal_interference_claim")):
        raise ValueError("budget protocol differs")
    prior = validate_artifacts(args.data / "mixture_complementarity_predictions", args.data / "mixture_complementarity_inputs")
    model_list = Path(__file__).resolve().parents[6] / "scripts/experiments/setup/target_performance_mechanisms/data/trajectory_model_list.tsv"
    models = model_registry(pd.read_csv(model_list, sep="\t"), pd.read_json(args.data / "trajectory_checkpoint_inventory.json"), prior)
    validate_training_audit(read("training_budget_audit"), models)
    tables = {name: pd.DataFrame(read(name)) for name in ("model_metrics", "prediction_inventory", "comparisons", "error_strata", "input_inventory")}
    validate_budget(tables, models, prior)
    cells, summary = summarize_budget(tables["comparisons"])
    cells.to_csv(args.data / "mixture_budget_comparisons.csv", index=False)
    summary.to_csv(args.data / "mixture_budget_summary.csv", index=False)
    primary = cells[cells.target.eq("facebook_page_reference") & ~cells.target_seen & cells.source_count.eq(8) & cells.budget_selected]
    if len(primary) != 2 or set(primary.stream) != set(STREAMS):
        raise ValueError("complete primary Facebook LOO endpoint required")
    validation = {**receipt, "primary_prediction_passed": bool(primary.ensemble_minus_mixture_auc.gt(0).all()),
                  "primary_cells": primary[["stream", "mixture_roc_auc", "probability_ensemble_roc_auc", "ensemble_minus_mixture_auc"]].to_dict("records"),
                  "no_target_checkpoint_selection": True, "matched_flops": False, "training_seeds": [0],
                  "artifact_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(export.glob("*.json"))}}
    (args.data / "mixture_budget_validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    print(summary[summary.panel.eq("foreign") & summary.budget_selected].to_string(index=False))


if __name__ == "__main__":
    main()
