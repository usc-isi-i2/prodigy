import unittest

import torch

from .audit_update_numerics import differences


class NumericalComparisonTests(unittest.TestCase):
    def test_exact_and_changed_tensor_inventories(self):
        a = {"weight": torch.tensor([1., 2.]), "counter": torch.tensor(2)}
        b = {k: v.clone() for k, v in a.items()}
        self.assertTrue(differences(a, b)["bit_exact"])
        b["weight"][1] += .5
        d = differences(a, b)
        self.assertFalse(d["bit_exact"])
        self.assertEqual(d["changed_keys"], 1)
        self.assertEqual(d["maximum_absolute_difference"], .5)
        self.assertEqual(d["largest_differences"], {"weight": .5})

    def test_missing_tensor_is_not_an_exact_match(self):
        with self.assertRaisesRegex(ValueError, "inventory"):
            differences({"weight": torch.tensor(1.)}, {})


if __name__ == "__main__":
    unittest.main()
