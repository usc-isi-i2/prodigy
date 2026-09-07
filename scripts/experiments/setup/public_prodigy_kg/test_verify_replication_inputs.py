import copy
import unittest

import torch

from .verify_replication_inputs import compare_captures


class ReplicationInputsTest(unittest.TestCase):
    def setUp(self):
        self.capture = dict(input=(torch.arange(4), torch.ones(2, 3)),
                            output=dict(y_true=torch.tensor([0, 1]), logits=torch.zeros(2, 2)))

    def test_predictions_may_differ(self):
        other = copy.deepcopy(self.capture)
        other["output"]["logits"].add_(10)
        self.assertEqual(compare_captures(self.capture, other), [])

    def test_input_truth_and_dtype_changes_rejected(self):
        other = copy.deepcopy(self.capture)
        other["input"][1][0, 0] += 0.001
        other["output"]["y_true"][0] = 1
        self.assertEqual(compare_captures(self.capture, other), ["input_1", "query_truth"])
        other = copy.deepcopy(self.capture)
        other["input"] = (other["input"][0].float(), other["input"][1])
        self.assertEqual(compare_captures(self.capture, other), ["input_0"])


if __name__ == "__main__":
    unittest.main()
