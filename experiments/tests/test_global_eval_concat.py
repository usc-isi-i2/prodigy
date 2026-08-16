import torch

from experiments.trainer import (
    _concat_global_eval_parts,
    _global_eval_matches_local_class_space,
)


def test_concat_global_eval_parts_pads_missing_class_columns():
    parts = [
        {
            "y_true": torch.tensor([0, 1]),
            "y_pred": torch.tensor([0, 1]),
            "probs": torch.tensor([[0.8, 0.2], [0.1, 0.9]]),
        },
        {
            "y_true": torch.tensor([1, 2]),
            "y_pred": torch.tensor([1, 2]),
            "probs": torch.tensor([[0.0, 0.7, 0.3], [0.0, 0.2, 0.8]]),
        },
    ]

    merged = _concat_global_eval_parts(parts)

    assert merged["probs"].shape == (4, 3)
    assert torch.equal(merged["probs"][:2, 2], torch.zeros(2))
    assert torch.equal(merged["y_true"], torch.tensor([0, 1, 1, 2]))
    assert torch.equal(merged["y_pred"], torch.tensor([0, 1, 1, 2]))


def test_multiclass_global_diagnostics_do_not_replace_binary_episode_metrics():
    local_predictions = torch.zeros((8, 2))

    assert not _global_eval_matches_local_class_space(
        {"probs": torch.zeros((8, 30))}, local_predictions
    )
    assert _global_eval_matches_local_class_space(
        {"probs": torch.zeros((8, 2))}, local_predictions
    )
