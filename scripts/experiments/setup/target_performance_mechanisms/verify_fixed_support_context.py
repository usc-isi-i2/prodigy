"""Independently check saved context batches, predictions, and variance accounting."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import torch

from experiments.run_shared_graph import write_json
from .fixed_support_context import graph_hash, replace_support_contexts, verify_replacement, seal_plan
from .replay import batch_hash


def numpy_metrics(probabilities, log_probabilities, labels):
    y = labels["local_y"].numpy()
    mapping = labels["mapping"].numpy()
    prediction = probabilities.argmax(1)
    nll = -log_probabilities[np.arange(len(y)), y].mean()
    if labels["use_global"]:
        truth = mapping[np.arange(len(y)), y]
        prediction = mapping[np.arange(len(y)), prediction]
        score = probabilities[np.arange(len(y)), (mapping == 1).argmax(1)]
    else:
        truth, score = y, probabilities[:, 1]
    return {"roc_auc": float(roc_auc_score(truth, score)), "accuracy": float(accuracy_score(truth, prediction)),
        "f1": float(f1_score(truth, prediction, zero_division=0)), "nll": float(nll)}


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    output = a.run/"independent_verification.json"
    if output.exists():
        raise ValueError("verification output already exists")
    done = json.loads((a.run/"DONE.json").read_text())
    cache_done = json.loads((a.cache/"DONE.json").read_text())
    if not done["complete"] or done["smoke"] or not cache_done["complete"] or cache_done["smoke"]:
        raise ValueError("complete non-smoke outputs required")
    plans = json.loads((a.cache/"plans.json").read_text())
    checked = 0
    for plan in plans:
        if seal_plan({k: v for k, v in plan.items() if k != "sha256"}) != plan["sha256"]:
            raise ValueError("context plan hash changed")
        spec = {(r["batch"], r["draw"]): r for r in plan["rows"]}
        for bi in range(32):
            b = torch.load(Path(plan["original"]["root"])/"batches"/f"batch_{bi:03d}.pt", map_location="cpu", weights_only=False)
            if batch_hash(b) != plan["original"]["batch_sha256"][bi]:
                raise ValueError("original input hash changed")
            draws = torch.load(Path(plan["root"])/f"supports_{bi:03d}.pt", map_location="cpu", weights_only=False)
            if len(draws) != 8:
                raise ValueError("incomplete saved support contexts")
            for draw, g in enumerate(draws):
                if graph_hash(g) != spec[bi, draw]["support_graph_sha256"]:
                    raise ValueError("support context tensor hash differs")
                changed = replace_support_contexts(b, g)
                verify_replacement(b, changed)
                if draw == 0 and batch_hash(changed) != batch_hash(b):
                    raise ValueError("original support context did not reconstruct")
                checked += 1
                del changed
            del b, draws
        print(f"Rehashed all support draws: {plan['stream']}/{plan['target']}", flush=True)
    cells = pd.DataFrame(json.loads((a.run/"metrics.json").read_text())).set_index(["stream", "target", "model_id", "condition", "draw"])
    cohorts = pd.DataFrame(json.loads((a.run/"cohorts.json").read_text())).set_index(["stream", "target", "model_id", "cohort"])
    paths = sorted((a.run/"predictions").glob("*/*/*.pt"))
    if len(paths) != 60:
        raise ValueError("incomplete prediction grid")
    max_metric_error, max_variance_error = 0., 0.
    for path in paths:
        raw = torch.load(path, map_location="cpu", weights_only=False)
        key = (raw["stream"], raw["target"], raw["model_id"])
        z = raw["predictions"]
        if z.shape != (8, len(raw["labels"]["local_y"]), 2) or not torch.isfinite(z).all():
            raise ValueError("invalid saved predictions")
        # Preserve the exact float32 probability vectors whose variation is
        # measured, but reduce independently with NumPy in float64.
        prob = z.softmax(2).numpy().astype(np.float64)
        logits = z.double().numpy()
        centered = logits-logits.max(2, keepdims=True)
        lp = centered-np.log(np.exp(centered).sum(2, keepdims=True))
        for draw in range(9):
            if draw < 8:
                actual = numpy_metrics(prob[draw], lp[draw], raw["labels"])
                saved = cells.loc[(*key, "draw", draw)]
            else:
                m = lp.max(0)
                mean_lp = m+np.log(np.exp(lp-m[None]).mean(0))
                actual = numpy_metrics(prob.mean(0), mean_lp, raw["labels"])
                saved = cells.loc[(*key, "probability_ensemble", -1)]
            for name, value in actual.items():
                error = abs(value-saved[name])
                max_metric_error = max(max_metric_error, error)
                if error > (1e-6 if name == "nll" else 1e-12):
                    raise ValueError(f"saved metric differs from independent calculation: {key} {name} {error}")
        correct = z.argmax(2).numpy() == raw["labels"]["local_y"].numpy()[None]
        context = raw["context_nodes"].numpy()
        for cohort, mask in (("all", np.ones(len(context), dtype=bool)), ("no_context", context == 0), ("has_context", context > 0)):
            if not mask.any():
                continue
            saved = cohorts.loc[(*key, cohort)]
            variance = float(prob[:, mask, 1].var(0).mean())
            error = abs(saved.mean_probability_variance-variance)
            max_variance_error = max(max_variance_error, error)
            if error > 1e-12:
                raise ValueError("saved probability variance differs")
            count = correct[:, mask].sum(0)
            for name, value in (("queries", int(mask.sum())), ("correctness_flips", int(((count > 0) & (count < 8)).sum())),
                ("always_correct", int((count == 8).sum())), ("always_wrong", int((count == 0).sum()))):
                if saved[name] != value:
                    raise ValueError("saved correctness accounting differs")
    if checked != 2560:
        raise ValueError("incomplete context verification")
    write_json(output, {"complete": True, "support_draw_batches_rehashed": checked,
        "original_input_batches_rehashed": 320, "prediction_cells": len(paths), "all_query_input_and_support_identity_checks_repeated": True,
        "numpy_metric_recomputation": True, "numpy_variance_and_correctness_recomputation": True,
        "maximum_metric_error": max_metric_error, "maximum_variance_error": max_variance_error})


if __name__ == "__main__":
    main()
