import numpy as np
import torch

from scripts.experiments.analysis.graphs.transfer_prediction.local_transfer_contrast.analyze_contrast import (
    decoder_audit,
    fit_temperature,
    grouped_bootstrap_difference,
    per_example_ce,
    stream_summary,
)


def test_temperature_reduces_discovery_nll():
    logits = np.array([[8.0, -8.0], [8.0, -8.0], [-8.0, 8.0], [-8.0, 8.0]])
    labels = np.array([0, 1, 1, 0])
    temperature = fit_temperature(logits, labels)
    assert temperature > 1.0
    assert per_example_ce(logits, labels, temperature).mean() < per_example_ce(logits, labels).mean()


def test_summary_counts_complementarity():
    y = np.array([0, 0, 1, 1])
    prob_b = np.array([[.9, .1], [.9, .1], [.2, .8], [.9, .1]])
    prob_c = np.array([[.8, .2], [.2, .8], [.7, .3], [.1, .9]])
    arrays = {
        "y": y,
        "correct_b": np.array([1, 1, 1, 0], dtype=bool),
        "correct_c": np.array([1, 0, 0, 1], dtype=bool),
        "prob_b": prob_b, "prob_c": prob_c,
        "loss_b_raw": -np.log(prob_b[np.arange(4), y]),
        "loss_c_raw": -np.log(prob_c[np.arange(4), y]),
        "loss_b": -np.log(prob_b[np.arange(4), y]),
        "loss_c": -np.log(prob_c[np.arange(4), y]),
    }
    result = stream_summary(arrays)
    assert result["n_b_only"] == 2
    assert result["n_c_only"] == 1
    assert result["n_both_correct"] == 1
    assert result["n_both_wrong"] == 0
    assert result["oracle_accuracy"] == 1.0


def test_decoder_audit_conditions_on_masked_labels():
    y = np.array([0, 1, 0, 1])
    logits_b = torch.tensor([[2., 0.], [0., 2.], [2., 0.], [2., 0.]])
    logits_c = torch.tensor([[2., 0.], [2., 0.], [0., 2.], [0., 2.]])
    record = {"models": {}}
    for name, logits in (("b", logits_b), ("c", logits_c)):
        record["models"][name] = {"logits": {
            "full_model": logits, "raw_joint/ridge": logits_b,
            "U1_pre_meta/ridge": logits, "raw_center/ridge": logits_b,
        }}
    arrays = {
        "y": y,
        "correct_b": logits_b.argmax(1).numpy() == y,
        "correct_c": logits_c.argmax(1).numpy() == y,
    }
    summary, conditional, transitions, identities = decoder_audit(record, "b", "c", arrays, "test")
    assert summary and conditional and transitions
    b_only_full = next(row for row in conditional if row["model"] == "b" and
                       row["decoder"] == "full_model" and row["full_model_outcome"] == "b_only")
    assert b_only_full["conditional_accuracy"] == 1.0
    assert identities["raw_center/ridge"]


def test_grouped_bootstrap_preserves_paired_positive_gain():
    candidate = np.array([1, 1, 0, 1], dtype=bool)
    reference = np.array([0, 1, 0, 0], dtype=bool)
    low, high = grouped_bootstrap_difference(candidate, reference, [0, 0, 1, 1], seed=3, draws=1000)
    assert low > 0
    assert high > 0
