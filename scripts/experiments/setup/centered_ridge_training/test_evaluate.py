import unittest
import torch
from types import SimpleNamespace
from .evaluate import centered,metrics


class EvalTest(unittest.TestCase):
    def test_support_only_scale_one(self):
        torch.manual_seed(9)
        x=torch.randn(8,5);q=torch.tensor([False,False,True,True]*2)
        labels=torch.eye(2).repeat(4,1);graph=SimpleNamespace(task_id_per_sample=torch.tensor([0]*4+[1]*4))
        batch=[graph,None,labels,None,None,q.repeat_interleave(2)]
        a=centered(x,batch)
        altered=labels.clone();altered[q]=float('nan')
        b=centered(x,[graph,None,altered,None,None,batch[5]])
        torch.testing.assert_close(a,b,atol=0,rtol=0)
        self.assertLess(float(a.abs().max()),1)
    def test_global_probability_ties(self):
        labels=dict(local_y=torch.tensor([0,1]),mapping=torch.tensor([[1,0],[1,0]]),use_global=True)
        scores=metrics(torch.zeros(2,2),labels)
        self.assertEqual(scores['f1'],0)
        self.assertEqual(scores['accuracy'],.5)


if __name__=='__main__':unittest.main()
