import torch

from mixture_scaling.objective import random_walk_negative_sampling_loss


def test_better_separated_pairs_have_lower_loss() -> None:
    pairs = torch.tensor([[0], [1]])
    negatives = torch.tensor([[0], [2]])
    good = torch.tensor([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]])
    bad = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [1.0, 0.0]])
    assert random_walk_negative_sampling_loss(good, pairs, negatives) < random_walk_negative_sampling_loss(
        bad, pairs, negatives
    )

