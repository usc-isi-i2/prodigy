"""Read verified logits, not models; compare mixture decisions with fixed ensembles."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import torch
import torch.nn.functional as F

from experiments.run_shared_graph import write_json
from .prepare_mixture_complementarity import TARGETS
from .replay import batch_hash
from .run_mixture_complementarity import verify_replay_tables

METRICS = ("roc_auc", "accuracy", "f1", "nll")


def input_labels(directory, target):
    cache = json.loads((directory / "cache.json").read_text())
    if cache["episodes"] != 128 or len(cache["batch_sha256"]) != 32:
        raise ValueError("incomplete cached input contract")
    labels, mappings, task_ids, counts = [], [], [], []
    for index in range(32):
        batch = torch.load(directory / "batches" / f"batch_{index:03d}.pt", map_location="cpu", weights_only=False)
        if batch_hash(batch) != cache["batch_sha256"][index]:
            raise ValueError("cached training-independent test tensor changed")
        graph = batch[0]
        if batch[2].ndim != 2 or batch[2].shape[1] != 2 or graph.task_label_map.shape != (4, 2):
            raise ValueError("expected four two-way episodes")
        query = batch[5].reshape(-1, 2)[:, 0].bool()
        yt = batch[2][query].argmax(1).long()
        tasks = graph.task_id_per_sample[query].long()
        mapping = graph.task_label_map[tasks].long()
        if set(tasks.tolist()) != {0, 1, 2, 3} or (mapping[:, 0] == mapping[:, 1]).any():
            raise ValueError("invalid query-to-episode mapping")
        labels.append(yt)
        mappings.append(mapping)
        task_ids.append(tasks + 4 * index)
        counts.append(len(yt))
    label = torch.cat(labels)
    mapping = torch.cat(mappings)
    use_global = int(mapping.max()) + 1 == 2
    if use_global == (target == "facebook_page_reference"):
        raise ValueError("production global/local class-space contract changed")
    return {"target": target, "local_y": label, "mapping": mapping, "use_global": use_global,
            "episode_ids": torch.cat(task_ids), "batch_counts": counts, "cache": cache}


def evaluate_probabilities(probs, labels, nll):
    yt, mapping = labels["local_y"], labels["mapping"]
    if probs.shape != (len(yt), 2) or not torch.isfinite(probs).all() or (probs < 0).any():
        raise ValueError("invalid binary probabilities")
    if not torch.allclose(probs.sum(1), torch.ones(len(yt), dtype=probs.dtype), atol=1e-6, rtol=0):
        raise ValueError("probabilities do not sum to one")
    yp = probs.argmax(1)
    rows = torch.arange(len(yt))
    if labels["use_global"]:
        if not torch.all(mapping.sort(1).values == torch.tensor([0, 1])):
            raise ValueError("global target is not binary")
        ytrue = mapping[rows, yt].numpy()
        ypred = mapping[rows, yp].numpy()
        score = probs[rows, (mapping == 1).long().argmax(1)].numpy()
    else:
        ytrue, ypred, score = yt.numpy(), yp.numpy(), probs[:, 1].numpy()
    return {"roc_auc": float(roc_auc_score(ytrue, score)), "accuracy": float(accuracy_score(ytrue, ypred)),
            "f1": float(f1_score(ytrue, ypred, zero_division=0)), "nll": float(nll)}


def evaluate_logits(logits, labels):
    targets = F.one_hot(labels["local_y"], num_classes=2).to(logits.dtype)
    return evaluate_probabilities(F.softmax(logits, dim=1), labels, F.cross_entropy(logits, targets).item())


def ensemble_scores(logits, labels):
    if logits.ndim != 3 or logits.shape[0] not in (2, 8) or not torch.isfinite(logits).all():
        raise ValueError("complete two- or eight-constituent logits required")
    # Preserve actual float32 constituent probabilities; average them in float64.
    probs = F.softmax(logits, dim=2).double().mean(0)
    stable_log_probs = torch.logsumexp(F.log_softmax(logits.double(), dim=2), dim=0) - np.log(len(logits))
    nll = -stable_log_probs[torch.arange(len(labels["local_y"])), labels["local_y"]].mean()
    mean_logits = logits.double().mean(0)
    return {"probability": (evaluate_probabilities(probs, labels, nll.item()), probs),
            "logit": (evaluate_logits(mean_logits, labels), F.softmax(mean_logits, dim=1))}


def error_strata(constituents, mixture, ensembles, labels):
    yt = labels["local_y"]
    decisions = constituents.argmax(2)
    correct = decisions == yt[None, :]
    mc = mixture.argmax(1) == yt
    ensemble_correct = {name: values[1].argmax(1) == yt for name, values in ensembles.items()}
    strata = {"all_correct": correct.all(0), "all_wrong": (~correct).all(0),
              "mixed_correctness": correct.any(0) & (~correct).any(0)}
    if sum(int(v.sum()) for v in strata.values()) != len(yt):
        raise ValueError("error strata do not partition the queries")
    rows = []
    for name, mask in strata.items():
        count = int(mask.sum())
        row = {"stratum": name, "queries": count, "fraction_of_queries": count / len(yt),
               "mixture_correct": int(mc[mask].sum()),
               "mixture_accuracy": float(mc[mask].double().mean()) if count else None}
        for rule, values in ensemble_correct.items():
            row[f"{rule}_ensemble_correct"] = int(values[mask].sum())
            row[f"{rule}_ensemble_accuracy"] = float(values[mask].double().mean()) if count else None
            # In this binary setting, positive equal-weight averaging cannot overturn unanimity.
            if (name == "all_wrong" and values[mask].any()) or (name == "all_correct" and (~values[mask]).any()):
                raise ValueError("ensemble violated binary unanimous-decision invariant")
        rows.append(row)
    k = len(constituents)
    positives = decisions.double().sum(0)
    disagreement = float((2 * positives * (k - positives) / (k * (k - 1))).mean())
    return rows, {"mean_pairwise_disagreement": disagreement,
                  "constituent_query_oracle_accuracy": float(correct.any(0).double().mean()),
                  "mixture_fixes_all_wrong_fraction": float((mc & strata["all_wrong"]).double().mean()),
                  "mixture_harms_all_correct_fraction": float((~mc & strata["all_correct"]).double().mean()),
                  "mixed_correctness_fraction": float(strata["mixed_correctness"].double().mean())}


def load_predictions(directory, model, labels, metrics_rows):
    matched = [r for r in metrics_rows if r["model_id"] == model["model_id"] and r["variant"] == "baseline" and r["decoder"] == "full_model"]
    if len(matched) != 1:
        raise ValueError("full-model metric row not uniquely resolved")
    row = matched[0]
    if (row["dataset"] != labels["target"] or row["checkpoint"] != model["checkpoint"] or row["weights_sha256"] != model["weights_sha256"]
            or sorted(row["sources"]) != model["sources"] or row["episode_fingerprint"] != labels["cache"]["episode_fingerprint"]):
        raise ValueError("prediction metadata differs from verified model/input inventory")
    path = directory / f"{model['model_id']}__baseline.pt"
    records = torch.load(path, map_location="cpu", weights_only=False)
    if len(records) != 32:
        raise ValueError("prediction export incomplete")
    logits = []
    for index, record in enumerate(records):
        value = record["logits"]["full_model"]
        if record["batch"] != index or record["batch_sha256"] != labels["cache"]["batch_sha256"][index]:
            raise ValueError("exported prediction input hash or batch order differs")
        if value.shape != (labels["batch_counts"][index], 2) or not torch.isfinite(value).all():
            raise ValueError("wrong query order/count or nonfinite prediction")
        logits.append(value)
    full = torch.cat(logits)
    metrics = evaluate_logits(full, labels)
    errors = {k: abs(metrics[k] - row[k]) for k in METRICS}
    if len(full) != row["queries"] or max(errors.values()) > 1e-6:
        raise ValueError(f"saved logits do not reproduce their metric row: {model['model_id']}: {errors}")
    return full, metrics, {"model_id": model["model_id"], "checkpoint": model["checkpoint"],
        "weights_sha256": model["weights_sha256"], "prediction_file": str(path),
        "episode_fingerprint": labels["cache"]["episode_fingerprint"],
        "prediction_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "queries": len(full), "metric_reproduction_max_abs_error": max(errors.values())}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, default=Path("/dataMeR1/phil/gfm/prodigy-mechanisms/log/target_mechanisms"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    receipt = json.loads((args.replay / "DONE.json").read_text())
    if receipt.get("rows") != 7650 or receipt.get("original_official_parity_cells") != 225 or receipt.get("all_cached_inputs_match") is not True:
        raise ValueError("complete verified mixture replay required before interpretation")
    if args.output.exists() or not 1 <= args.threads <= 8 or torch.cuda.is_available():
        raise ValueError("existing output, invalid threads, or visible GPU")
    torch.set_num_threads(args.threads)
    inventory = json.loads((args.replay / "checkpoint_inventory.json").read_text())["models"]
    audit_path = args.replay / "numerical_audit.json"
    numerical_audit = json.loads(audit_path.read_text()) if audit_path.exists() else None
    parity = verify_replay_tables({s: args.replay / s for s in ("original", "fresh")}, inventory, numerical_audit)
    if receipt.get("individually_audited_numerical_cells", 0) != parity["individually_audited_numerical_cells"]:
        raise ValueError("numerical-audit receipt mismatch")
    singles = [r for r in inventory if r["source_count"] == 1]
    mixtures = [r for r in inventory if r["source_count"] > 1]
    if len(singles) != 9 or len(mixtures) != 45:
        raise ValueError("complete model inventory required")
    args.output.mkdir(parents=True)
    metrics_output, comparisons, strata_output, prediction_inventory, input_inventory = [], [], [], [], []
    for stream in ("original", "fresh"):
        for target in sorted(TARGETS):
            mixture_dir = args.replay / stream / target
            prior = "fresh_stage_cpu_20260906" if stream == "fresh" else (
                "specialist_cpu_tail_20260906" if target in {"twibot20", "ukr_rus_suspended"} else "specialist_cpu_20260906")
            single_dir = args.reference_root / prior / target
            labels = input_labels(mixture_dir, target)
            single_cache = json.loads((single_dir / "cache.json").read_text())
            for key in ("episode_fingerprint", "batch_sha256", "episodes", "graph_path"):
                if labels["cache"][key] != single_cache[key]:
                    raise ValueError("specialist and mixture test inputs differ")
            input_inventory.append({"stream": stream, "target": target, "production_metric_uses_global_binary_labels": labels["use_global"],
                                    "queries": len(labels["local_y"]), **labels["cache"]})
            saved_rows = {str(p): [json.loads(line) for line in (p / "metrics.jsonl").read_text().splitlines()]
                          for p in (single_dir, mixture_dir)}
            prediction, scores = {}, {}
            for model in singles + mixtures:
                directory = single_dir if model["source_count"] == 1 else mixture_dir
                value, metrics, audit = load_predictions(directory, model, labels, saved_rows[str(directory)])
                prediction[model["model_id"]] = value
                scores[model["model_id"]] = metrics
                prediction_inventory.append({"stream": stream, "target": target, **audit})
                metrics_output.append({"stream": stream, "target": target, "model_id": model["model_id"],
                                       "sources": model["sources"], "source_count": model["source_count"], **metrics})
            singleton_ids = {r["sources"][0]: r["model_id"] for r in singles}
            for model in mixtures:
                members = [singleton_ids[s] for s in model["sources"]]
                stacked = torch.stack([prediction[m] for m in members])
                ensemble = ensemble_scores(stacked, labels)
                strata, extra = error_strata(stacked, prediction[model["model_id"]], ensemble, labels)
                common = {"stream": stream, "target": target, "model_id": model["model_id"],
                          "sources": model["sources"], "source_count": model["source_count"],
                          "target_seen": target in model["sources"], "queries": len(labels["local_y"])}
                row = {**common, **extra}
                for metric in METRICS:
                    values = [scores[m][metric] for m in members]
                    row[f"constituent_mean_{metric}"] = float(np.mean(values))
                    row[f"constituent_min_{metric}"] = float(np.min(values))
                    row[f"constituent_max_{metric}"] = float(np.max(values))
                    row[f"mixture_{metric}"] = scores[model["model_id"]][metric]
                    for rule, (result, _) in ensemble.items():
                        row[f"{rule}_ensemble_{metric}"] = result[metric]
                        row[f"mixture_minus_{rule}_ensemble_{metric}"] = row[f"mixture_{metric}"] - result[metric]
                for rule, (_, probs) in ensemble.items():
                    mixture_prob = F.softmax(prediction[model["model_id"]], dim=1)
                    row[f"mixture_{rule}_ensemble_decision_agreement"] = float((mixture_prob.argmax(1) == probs.argmax(1)).double().mean())
                    row[f"mixture_{rule}_ensemble_probability_l1"] = float((mixture_prob.double() - probs.double()).abs().sum(1).mean())
                comparisons.append(row)
                strata_output.extend([{**common, **r} for r in strata])
            print(json.dumps({"stream": stream, "target": target, "verified_model_predictions": 54, "mixture_comparisons": 45}), flush=True)
    if (len(metrics_output), len(comparisons), len(strata_output), len(prediction_inventory)) != (540, 450, 1350, 540):
        raise ValueError("incomplete model/comparison/stratum outputs")
    for name, value in (("model_metrics", metrics_output), ("comparisons", comparisons), ("error_strata", strata_output),
                        ("prediction_inventory", prediction_inventory), ("input_inventory", input_inventory)):
        write_json(args.output / f"{name}.json", value)
    if numerical_audit:
        write_json(args.output / "numerical_audit.json", numerical_audit)
    write_json(args.output / "protocol.json", {"replay": str(args.replay), "reference_root": str(args.reference_root),
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), "new_training": False,
        "ensemble_weights_fitted": False, "query_labels_used_for_classifier_fitting": False,
        "query_labels_used_for_error_diagnostics": True, "primary_ensemble": "mean float32 production probabilities in float64",
        "secondary_ensemble": "mean logits in float64", "ensemble_nll": "stable float64 log-sum-exp from constituent logits",
        "training_seeds": [0], "ensemble_training_and_inference_cost_multiplier": [2, 8], "causal_interference_claim": False,
        "strict_original_official_parity_cells": parity["strict_original_official_parity_cells"],
        "individually_audited_numerical_cells": parity["individually_audited_numerical_cells"]})
    write_json(args.output / "DONE.json", {"models": 54, "verified_model_target_stream_cells": 540,
        "mixture_comparisons": 450, "error_strata": 1350, "all_logits_reproduce_metrics": True,
        "all_specialist_mixture_inputs_identical": True, "complete_both_streams": True})


if __name__ == "__main__":
    main()
