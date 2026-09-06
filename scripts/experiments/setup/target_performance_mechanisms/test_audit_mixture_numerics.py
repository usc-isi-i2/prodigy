import unittest

import torch

from .audit_mixture_numerics import compare_rank_pairs


class NumericalAuditTests(unittest.TestCase):
    def test_half_pair_tie_decomposition_and_unchanged_decisions(self):
        labels = {"local_y": torch.tensor([0, 1, 0, 1]), "mapping": torch.tensor([[0, 1]] * 4), "use_global": True}
        cpu = torch.tensor([[2., 0.], [2., 0.], [0., 2.], [0., 2.]])
        other = cpu.clone()
        other[1, 1] = .01
        audit = compare_rank_pairs(cpu, other, labels)
        self.assertEqual(audit["auc_half_pair_quantum"], .125)
        self.assertEqual(audit["auc_other_minus_cpu"], .125)
        self.assertEqual(len(audit["changed_cross_label_pairs"]), 1)
        self.assertEqual(audit["changed_decisions"], 0)
        self.assertEqual(audit["changed_cross_label_pairs"][0]["cpu_order"], 0)
        self.assertEqual(audit["changed_cross_label_pairs"][0]["other_order"], 1)


if __name__ == "__main__":
    unittest.main()
