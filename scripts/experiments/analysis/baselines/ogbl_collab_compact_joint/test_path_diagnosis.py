import unittest
import numpy as np
from path_diagnosis import make_graph, path_features, match


class PathTests(unittest.TestCase):
    def test_independent_paths_and_recency(self):
        edges = np.array([[0,1],[1,2],[2,5],[0,3],[3,4],[4,5]])
        years = np.array([2017,2017,2017,2014,2017,2017])
        adj, keys, first, last = make_graph(edges, years, 6)
        f = path_features(0,5,adj,keys,first,last,np.eye(6),np.zeros(6))
        self.assertEqual(f['paths3'],2)
        self.assertEqual(f['disjoint_paths3'],2)
        self.assertEqual(f['recent_paths3'],1)
        self.assertEqual(f['independent_recent_paths3'],1)
        self.assertEqual(f['recent_path_fraction'],.5)
        self.assertEqual(f['max_bridge_path_share'],.5)
        self.assertEqual(f['freshest_bottleneck_age'],1)
        reverse = path_features(5,0,adj,keys,first,last,np.eye(6),np.zeros(6))
        self.assertEqual(f,reverse)

    def test_future_edges_rejected(self):
        with self.assertRaises(AssertionError):
            make_graph(np.array([[0,1]]),np.array([2018]),2)

    def test_duplicate_events_preserve_first_and_last(self):
        _, keys, first, last = make_graph(np.array([[0,1],[1,0],[0,1]]),np.array([2012,2017,2015]),2)
        self.assertEqual(keys.tolist(),[1])
        self.assertEqual(first.tolist(),[2012])
        self.assertEqual(last.tolist(),[2017])

    def test_matching_enforces_covariates_and_exact_strata(self):
        pos, neg = np.zeros((3,13)), np.zeros((1,13))
        pos[1,7]=.04
        pos[2,12]=1
        rows=match(pos,neg,np.arange(3),np.arange(1))
        self.assertEqual(rows[0]['positives'],[0])

    def test_all_input_matching_rejects_different_l3(self):
        pos, neg = np.zeros((2,13)), np.zeros((1,13))
        pm, nm = np.zeros((2,14)), np.zeros((1,14))
        pm[1,1] = 1.
        rows = match(pos,neg,np.arange(2),np.arange(1),pm,nm,np.ones(14))
        self.assertEqual(rows[0]['positives'],[0])


if __name__ == '__main__':
    unittest.main()
