import unittest
import numpy as np
from .inspect_task_switch_bank import fit_rule


class RuleTests(unittest.TestCase):
    def test_normalization_and_sign(self):
        x = np.array([[1., 2], [1, 1], [1, -1], [1, -2]])
        m, v, t, e = fit_rule(x)
        m2, v2, t2, _ = fit_rule(x * np.arange(1, 5)[:, None])
        np.testing.assert_allclose(m, m2, atol=1e-14)
        np.testing.assert_allclose(v, v2, atol=1e-14)
        self.assertAlmostEqual(t, t2)
        self.assertGreater(v[np.argmax(np.abs(v))], 0)
        self.assertGreater(e[-1], e[-2])
        self.assertEqual(int((((x / np.linalg.norm(x, axis=1)[:, None] - m) @ v) > t).sum()), 2)

    def test_degenerate_or_invalid_rejected(self):
        for x in (np.eye(3), np.ones((5, 3)), np.zeros((5, 3))):
            with self.assertRaises(ValueError):
                fit_rule(x)


if __name__ == '__main__':
    unittest.main()
