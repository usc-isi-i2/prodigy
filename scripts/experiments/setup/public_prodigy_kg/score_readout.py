"""Paired deployment comparison for one task, never pooled local class columns.

The caller must verify episode/checkpoint provenance and native replay parity.
This module only scores predictions; it does not select a readout or its scale.
"""
import numpy as np

from scripts.experiments.setup.trace_health_guided.multiclass_metrics import episode_metrics


def score_task(labels, native_logits, ridge_logits):
    labels = np.asarray(labels)
    native = np.asarray(native_logits, dtype=float)
    ridge = np.asarray(ridge_logits, dtype=float)
    if native.shape != ridge.shape:
        raise ValueError("Readouts must share query order and class columns")
    metrics = {"native": episode_metrics(labels, native),
               "ridge": episode_metrics(labels, ridge)}
    predictions = {"native": native.argmax(1), "ridge": ridge.argmax(1)}
    correct = {name: prediction == labels for name, prediction in predictions.items()}
    groups = {
        "both_correct": correct["native"] & correct["ridge"],
        "ridge_corrects_native": ~correct["native"] & correct["ridge"],
        "ridge_corrupts_native": correct["native"] & ~correct["ridge"],
        "both_wrong": ~correct["native"] & ~correct["ridge"],
    }
    losses = {}
    for name, logits in (("native", native), ("ridge", ridge)):
        shifted = logits - logits.max(1, keepdims=True)
        losses[name] = np.log(np.exp(shifted).sum(1)) - shifted[np.arange(len(labels)), labels]
    examples = []
    for query, label in enumerate(labels):
        examples.append({"query": query, "local_label": int(label),
                         "group": next(name for name, mask in groups.items() if mask[query]),
                         "agreement": bool(predictions["native"][query] == predictions["ridge"][query]),
                         **{f"{name}_prediction": int(predictions[name][query]) for name in predictions},
                         **{f"{name}_nll": float(losses[name][query]) for name in losses}})
    return {"metrics": metrics,
            "ridge_minus_native": {key: metrics["ridge"][key] - metrics["native"][key]
                                   for key in metrics["native"]},
            "counts": {name: int(mask.sum()) for name, mask in groups.items()},
            "examples": examples,
            "nll_note": "Fixed readout scales; NLL differences also reflect calibration, not only discrimination."}
