import unittest
import numpy as np
from scipy import sparse
from run import short_walk_nodes, count_induced_edges, topology_metrics, histogram, compare_histograms


class PilotTests(unittest.TestCase):
    def test_walk_budget_determinism_and_isolate(self):
        a = sparse.csr_matrix(([1,1,1,1,1,1], ([0,1,1,2,2,3],[1,0,2,1,3,2])), shape=(5,5), dtype=bool)
        ids, _ = short_walk_nodes(a, 4, 2, 5, batch_size=3)
        again, _ = short_walk_nodes(a, 4, 2, 5, batch_size=3)
        self.assertTrue(np.array_equal(ids, again))
        self.assertEqual(len(ids), 4)
        all_ids, _ = short_walk_nodes(a, 5, 1, 0, batch_size=3)
        self.assertEqual(all_ids.tolist(), list(range(5)))

    def test_directed_counts_keep_reciprocal_and_self_edges(self):
        edges = np.array([[0,1,0,2,3],[1,0,0,3,4]])
        self.assertEqual(count_induced_edges(edges, np.array([0,1,3]), 5), 3)

    def test_distribution_and_components(self):
        a = sparse.csr_matrix(([True,True], ([0,1],[1,0])), shape=(3,3))
        m = topology_metrics(a, 2)
        self.assertAlmostEqual(m['isolate_fraction'], 1/3)
        self.assertAlmostEqual(m['largest_component_fraction'], 2/3)
        h = histogram(np.array([0,1,1]))
        self.assertEqual(compare_histograms(h,h)['degree_ks'], 0)
        self.assertAlmostEqual(compare_histograms(histogram(np.array([0,0])),histogram(np.array([1,1])))['degree_ks'],1)

if __name__ == '__main__':
    unittest.main()
