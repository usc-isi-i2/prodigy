import numpy as np
import torch

from scripts.experiments.analysis.graphs.transfer_prediction.local_transfer_contrast.analyze_multi_health import (
    evaluate_selection,
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
