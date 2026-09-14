import unittest

import numpy as np
import torch

from scripts.experiments.setup.ogbl_collab_mlp_hardneg.run import (
    FeaturePairModel, choose_hardest, hard_candidate_pool,
)


class HardNegativeProtocolTest(unittest.TestCase):
    def test_interaction_scorer_is_symmetric(self):
        torch.manual_seed(0)
        model = FeaturePairModel("interaction", 4)
        left, right = torch.randn(7, 4), torch.randn(7, 4)
        torch.testing.assert_close(model.logits(left, right), model.logits(right, left))

    def test_pool_retains_uniform_base_stream(self):
        edges = np.array([[0, 1], [1, 2], [2, 3]])
        order, pool, _ = hard_candidate_pool(edges, 4, seed=1, epoch=2)
        from scripts.experiments.setup.ogbl_collab_mlp_lp.run import epoch_training_pairs
        expected_order, base, _ = epoch_training_pairs(edges, 4, 1, 2)
        np.testing.assert_array_equal(order, expected_order)
        np.testing.assert_array_equal(pool[:, 0], base)

    def test_choose_hardest_returns_one_pair_per_row(self):
        torch.manual_seed(0)
        model = FeaturePairModel("cosine", 4)
        x = torch.randn(5, 4)
        candidates = np.array([[[0, 1], [0, 0]], [[2, 3], [2, 2]]])
        selected = choose_hardest(model, x, candidates)
        np.testing.assert_array_equal(selected, np.array([[0, 0], [2, 2]]))


if __name__ == "__main__":
    unittest.main()
