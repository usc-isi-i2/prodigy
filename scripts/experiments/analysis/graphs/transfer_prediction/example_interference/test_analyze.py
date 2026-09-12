import torch

from scripts.experiments.analysis.graphs.transfer_prediction.example_interference.analyze import (
    expected_labels,
    outcome,
    summarize,
)


def test_outcome_partition_and_net_change():
    labels = torch.tensor([0, 0, 1, 1])
    base = torch.tensor([[2., 0.], [0., 2.], [0., 2.], [2., 0.]])
    pair = torch.tensor([[2., 0.], [2., 0.], [2., 0.], [2., 0.]])
    result = summarize(outcome(base, pair, labels))
    assert result["rescued"] == 1
    assert result["broken"] == 1
    assert result["both_correct"] == 1
    assert result["both_wrong"] == 1
    assert result["net_accuracy_change"] == 0
    assert result["churn"] == 0.5


def test_expected_labels_matches_episode_order():
    assert expected_labels(16).tolist() == [0] * 4 + [1] * 4 + [0] * 4 + [1] * 4
