import unittest

import numpy as np
import torch

from .audit_readout_control_states import state_equal


class ControlStateAuditTest(unittest.TestCase):
    def test_nested_rng_states(self):
        a = {"torch": torch.tensor([2, 3]), "numpy": ("MT", np.array([1, 2]), 0),
             "python": (3, (1, 2), None), "optimizer": {"state": {}, "groups": [1, 2]}}
        b = {"torch": torch.tensor([2, 3]), "numpy": ("MT", np.array([1, 2]), 0),
             "python": (3, (1, 2), None), "optimizer": {"state": {}, "groups": [1, 2]}}
        self.assertTrue(state_equal(a, b))
        b["torch"][0] = 9
        self.assertFalse(state_equal(a, b))

    def test_container_and_numpy_changes(self):
        self.assertFalse(state_equal([1, 2], (1, 2)))
        self.assertFalse(state_equal({"a": 1}, {"b": 1}))
        self.assertFalse(state_equal(np.array([1, 2]), np.array([1, 3])))
        self.assertFalse(state_equal([1], [1, 2]))


if __name__ == "__main__":
    unittest.main()
