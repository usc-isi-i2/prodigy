import unittest
import numpy as np
from run import historical_inputs, keys, negatives, hits, warm_positive_mask, rescue_eligible, residual_score, direct_features, direct_score


class TemporalContractTests(unittest.TestCase):
    def test_direct_features_preserve_original_inputs_and_zero_base(self):
        x=np.arange(26,dtype=np.float32).reshape(2,13)
        f=direct_features(x,np.array([0.,2.],dtype=np.float32))
        self.assertEqual(f.shape,(2,14))
        np.testing.assert_array_equal(f[:,:13],x)
        self.assertEqual(f[0,-1],0.)
        self.assertAlmostEqual(float(f[1,-1]),float(np.log(3.)),places=6)

    def test_direct_score_has_no_residual_anchor_or_bound(self):
        try:
            import torch
        except ImportError:
            self.skipTest('Torch integration test requires prodigy environment')
        model=torch.nn.Linear(1,1,bias=False)
        with torch.no_grad(): model.weight.fill_(20.)
        f=torch.tensor([[1.],[2.]],requires_grad=True)
        selfpair=torch.tensor([False,True]); eligible=torch.zeros(2,dtype=torch.bool)
        a=direct_score(model,(f,torch.zeros(2),selfpair,eligible))
        b=direct_score(model,(f,torch.full((2,),-100.),selfpair,eligible))
        self.assertTrue(torch.equal(a,b))
        self.assertEqual(a[0].item(),20.)
        self.assertEqual(a[1].item(),-1e9)
        a.sum().backward()
        self.assertGreater(f.grad[0].item(),0.)

    def test_rescue_eligibility(self):
        x=np.zeros((5,13)); x[1,0]=1; x[2,8]=1; x[3,[0,8]]=1
        edges=np.array([[0,1],[0,1],[0,1],[0,1],[2,2]])
        self.assertEqual(rescue_eligible(x,edges).tolist(),[True,False,False,False,False])

    def test_protected_scores_and_gradients(self):
        try:
            import torch
        except ImportError:
            self.skipTest('Torch integration test requires prodigy environment')
        model=torch.nn.Linear(1,1,bias=False)
        with torch.no_grad(): model.weight.fill_(1.)
        features=torch.tensor([[1.],[2.],[3.]],requires_grad=True)
        base=torch.tensor([.2,.3,.4]); selfpair=torch.tensor([False,False,True])
        eligible=torch.tensor([True,False,False])
        values=residual_score(model,(features,base,selfpair,eligible),.25)
        self.assertEqual(values[1].item(),base[1].item())
        self.assertEqual(values[2].item(),-1e9)
        self.assertGreater(values[0].item(),base[0].item())
        values.sum().backward()
        self.assertEqual(features.grad[1].item(),0.)
        self.assertEqual(features.grad[2].item(),0.)

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
