import unittest
import torch
from .audit_trajectory import transitions


class AuditTest(unittest.TestCase):
    def test_all_four_transitions(self):
        self.assertEqual(transitions(torch.tensor([True,True,False,False]),
                                     torch.tensor([True,False,True,False])),
                         dict(correct_both=1,wrong_both=1,corrected=1,corrupted=1))

    def test_reject_unaligned_or_nonboolean(self):
        for a,b in ((torch.ones(2),torch.ones(2)),
                    (torch.ones(2,dtype=torch.bool),torch.ones(3,dtype=torch.bool))):
            with self.assertRaises(ValueError):
                transitions(a,b)


if __name__ == "__main__":
    unittest.main()
