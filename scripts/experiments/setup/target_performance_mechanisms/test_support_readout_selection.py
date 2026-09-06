import unittest

import torch
import torch.nn.functional as F

from .support_readout_selection import fold_pairs, prototype_oof, select_predictions


class SelectorTests(unittest.TestCase):
    def test_prototype_folds_match_explicit_fit(self):
        torch.manual_seed(1)
        y = torch.tensor([0] * 10 + [1] * 10).repeat(2)
        task = torch.arange(2).repeat_interleave(20)
        x = torch.randn(40, 7)
        pairs = fold_pairs(y, task, 2)
        actual = prototype_oof(x, y, task, pairs)
        nx = F.normalize(x, dim=1)
        for pair in pairs:
            for ep in range(2):
                mask = task == ep
                mask[pair[ep]] = False
                p = torch.stack([nx[mask & (y == c)].mean(0) for c in range(2)])
                expected = nx[pair[ep]] @ F.normalize(p, dim=1).T
                torch.testing.assert_close(actual[pair[ep]], expected, rtol=1e-5, atol=1e-6)

    def test_tie_rule_scaling_and_zero_margin(self):
        y = torch.tensor([0] * 10 + [1] * 10)
        tasks = torch.zeros(20, dtype=torch.long)
        oof = torch.stack((1-y, y), 1).float()
        query = torch.tensor([[3., 0.], [0., 3.]])
        r = select_predictions({"raw": query, "full": query*5}, {"raw": oof, "full": oof*5},
            y, tasks, torch.zeros(2, dtype=torch.long))
        self.assertEqual(r["choice"].tolist(), [0])
        torch.testing.assert_close(r["fixed_scaled"][0], r["fixed_scaled"][1])
        r = select_predictions({"raw": query}, {"raw": torch.zeros_like(oof)}, y, tasks, torch.zeros(2, dtype=torch.long))
        self.assertEqual(r["cv_rms"].item(), 1.)
        self.assertEqual(r["cv_auc"].item(), .5)


if __name__ == "__main__":
    unittest.main()
