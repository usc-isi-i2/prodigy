import numpy as np
import torch

from scripts.experiments.analysis.graphs.transfer_prediction.local_transfer_contrast.analyze_multi_health import (
    evaluate_selection,
    select_by_support_score,
    select_first_healthy,
)


def test_select_first_healthy_respects_rank_and_fallback():
    selected = select_first_healthy(
        ["a", "b", "c"],
        {"a": np.array([0, 1, 0], bool), "b": np.array([1, 1, 0], bool),
         "c": np.array([1, 1, 0], bool)},
    )
    assert selected.tolist() == ["b", "a", "a"]


def test_evaluate_selection_takes_occurrence_from_selected_model():
    record = {"models": {
        "a": {"logits": {"full_model": torch.tensor([[4., 0.], [4., 0.]])}},
        "b": {"logits": {"full_model": torch.tensor([[0., 4.], [0., 4.]])}},
    }}
    predictions, probabilities = evaluate_selection(
        record, np.array(["a", "b"], dtype=object), {"a": 1., "b": 1.}, np.array([0, 1]),
    )
    assert predictions.tolist() == [0, 1]
    assert np.all(probabilities.max(1) > .9)


def test_support_selector_uses_health_for_ties_and_gating():
    scores = {"a": np.array([.8, .9, .7]), "b": np.array([.8, .8, .9])}
    prior = {"a": .6, "b": .7}
    selected = select_by_support_score(["a", "b"], scores, tie_prior=prior)
    assert selected.tolist() == ["b", "a", "b"]
    gated = select_by_support_score(
        ["a", "b"], scores,
        healthy={"a": np.array([1, 0, 0], bool), "b": np.array([0, 1, 0], bool)},
        tie_prior=prior,
    )
    assert gated.tolist() == ["a", "b", "b"]
