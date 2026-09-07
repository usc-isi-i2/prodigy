import numpy as np

from scripts.experiments.analysis.graphs.transfer_prediction.local_transfer_contrast.analyze_contrast import (
    fit_temperature,
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
