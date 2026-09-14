import unittest
import numpy as np
from run import historical_inputs, keys, negatives, hits, warm_positive_mask


class TemporalContractTests(unittest.TestCase):
    def test_warm_filter_requires_both_previously_active_endpoints(self):
        train=np.array([[0,1],[1,2]])
        pos=np.array([[0,2],[0,3],[3,4]])
        self.assertEqual(warm_positive_mask(pos,train,5).tolist(),[True,False,False])

    def test_future_event_weights_do_not_enter_historical_graph(self):
        graph={'edge_index':np.array([[0,1,0,1],[1,0,1,0]]),
               'edge_year':np.array([2014,2014,2017,2017]),
               'edge_weight':np.array([2.,2.,999.,999.])}
        split={'train':{'edge':np.array([[0,1],[0,1]]),'year':np.array([2014,2017])}}
        edges,years,weight,last=historical_inputs(graph,split,2015)
        self.assertEqual(len(edges),1)
        self.assertEqual(years.tolist(),[2014])
        self.assertAlmostEqual(float(weight[0]),1.9,places=6)
        self.assertEqual(last,2014)

    def test_negatives_exclude_undirected_same_year_positives(self):
        pos=np.array([[0,1],[1,2],[3,2]])
        first=negatives(pos,8,2016,count=20)
        second=negatives(pos,8,2016,count=20)
        self.assertTrue(np.array_equal(first,second))
        self.assertEqual(len(np.unique(keys(first,8))),20)
        self.assertEqual(len(np.intersect1d(keys(first,8),keys(pos,8))),0)

    def test_official_strict_negative_threshold(self):
        self.assertEqual(hits(np.array([1.,2.]),np.r_[np.ones(50),np.zeros(50)]),.5)


if __name__=='__main__':
    unittest.main()
