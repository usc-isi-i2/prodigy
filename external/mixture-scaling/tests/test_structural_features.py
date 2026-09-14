import numpy as np
import torch

from mixture_scaling.structural_logistic_baseline import structural_features


def test_structural_features_are_finite_and_label_free():
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 0, 2], [1, 0, 2, 1, 0, 2, 3]])
    features, names = structural_features(edge_index, 5)
    assert features.shape == (5, 32)
    assert len(names) == 32
    assert np.isfinite(features).all()
