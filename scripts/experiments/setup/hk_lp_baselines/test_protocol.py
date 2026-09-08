import unittest
import numpy as np
from .protocol import degree_bins, edge_keys, make_pairs, measure, pair_keys


class ProtocolTests(unittest.TestCase):
    def test_unique_nonedges_and_degree_match_across_splits(self):
        n = 100
        edges = np.array([(i, (i + delta) % n) for i in range(n) for delta in [1, 2, 3, 4]]).T
        keys = edge_keys(edges, n)
        degrees = np.full(n, 8)
        used = set()
        first, _ = make_pairs(keys[:200], keys, degrees, 50, np.random.default_rng(2), used)
        second, _ = make_pairs(keys[200:], keys, degrees, 50, np.random.default_rng(3), used)
        a = pair_keys(first['u'], first['v'], n)
        b = pair_keys(second['u'], second['v'], n)
        self.assertEqual(len(np.intersect1d(a, b)), 0)
        self.assertFalse(np.isin(a[50:], keys).any())
        self.assertTrue(np.array_equal(first['u'][:50], first['u'][50:]))

    def test_isolates_are_not_degree_one(self):
        self.assertEqual(degree_bins(np.array([0, 1, 2, 3, 4])).tolist(), [-1, 0, 1, 1, 2])

    def test_tied_scores_do_not_get_order_dependent_ap(self):
        score = np.zeros(100)
        y = np.r_[np.ones(50), np.zeros(50)]
        self.assertEqual(measure(y, score), {'auc': .5, 'average_precision': .5})
        self.assertEqual(measure(y[::-1], score), {'auc': .5, 'average_precision': .5})


if __name__ == '__main__':
    unittest.main()
