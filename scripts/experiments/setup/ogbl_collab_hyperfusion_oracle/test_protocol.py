import importlib.util
from pathlib import Path
import unittest
import numpy as np

P = Path(__file__).with_name("run.py")
S = importlib.util.spec_from_file_location("hf_oracle", P)
M = importlib.util.module_from_spec(S); S.loader.exec_module(M)


class ProtocolTest(unittest.TestCase):
    def test_three_base_equal_similarity_weights(self):
        x = np.array([[1., 2., 3.], [1., 2., 3.01], [1., 2., 2.99]])
        weights, h = M.hyper_weights([x, x, x, x])
        np.testing.assert_array_equal(h, np.ones((3, 4)))
        np.testing.assert_array_equal(weights, np.full(3, 12.))

    def test_percentile_is_monotonic_and_common_scale(self):
        p, n = M.percentile_rows(np.array([[20., 10.], [2., 1.]]), np.array([[0.], [0.]]))
        np.testing.assert_array_equal(p, np.array([[1., .5], [1., .5]]))
        np.testing.assert_array_equal(n, np.zeros((2, 1)))

    def test_parameter_cap(self):
        self.assertLess(M.PARAMETERS, 1_000_000)

    def test_hits_is_not_validation_constant(self):
        self.assertEqual(M.hits(np.ones(3), np.ones(100)), 0.0)


if __name__ == "__main__": unittest.main()
