import unittest
import numpy as np
from fusion import cutoff, hits, maximum, transform, hyper_weights


class FusionTests(unittest.TestCase):
    def test_strict_ties(self):
        self.assertEqual(hits(np.ones(4), np.ones(100)), 0)

    def test_frozen_transform(self):
        panel = {'bp': np.array([4.]), 'bn': np.ones(100)}
        scores = {'p': np.array([2.]), 'n': np.zeros(100)}
        first = transform(panel, scores, 2., 1.)
        panel['bn'] *= 100
        scores['n'] += 100
        second = transform(panel, scores, 2., 1.)
        np.testing.assert_array_equal(first['p'], second['p'])
        np.testing.assert_array_equal(maximum(first, 0)['p'], panel['bp']/2)

    def test_two_base_weights(self):
        similar = np.array([[1., 2., 3.], [1., 2., 3.1]])
        w, h = hyper_weights([similar, similar])
        np.testing.assert_array_equal(w, [4., 4.])
        disjoint = np.array([[1., 0., 0.], [0., 1., 0.]])
        w, h = hyper_weights([disjoint, disjoint])
        np.testing.assert_array_equal(w, [0., 0.])
        w, h = hyper_weights([disjoint, similar])
        np.testing.assert_array_equal(w, [2., 2.])


if __name__ == '__main__':
    unittest.main()
