import numpy as np
from mixture_scaling.probe_label_budget import support_indices

def test_support_is_balanced_and_deterministic():
    labels = np.repeat([0, 1, 2], 10); first = support_indices(labels, 3, 7)
    assert np.array_equal(first, support_indices(labels, 3, 7))
    assert np.bincount(labels[first]).tolist() == [3, 3, 3]
