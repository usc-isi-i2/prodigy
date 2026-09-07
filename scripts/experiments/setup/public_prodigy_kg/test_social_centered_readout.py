import unittest
from types import SimpleNamespace
import torch
from .social_centered_readout import probe, score
from .test_readout_episode import fixture


class SocialReadoutTests(unittest.TestCase):
    def test_task_local_fit_and_query_label_invariance(self):
        x,y,edges,mask=fixture(2,4)
        graph=SimpleNamespace(task_id_per_sample=torch.arange(4).repeat_interleave(14))
        batch=[graph,None,y,edges,None,mask]
        original=probe(x,batch,"support")
        self.assertEqual(original.shape,(32,2))
        truth=y[mask.reshape(56,2)[:,0]].argmax(1)
        self.assertTrue(torch.equal(original.argmax(1),truth))
        changed=y.clone(); changed[mask.reshape(56,2)[:,0]]=float("nan")
        batch[2]=changed
        torch.testing.assert_close(probe(x,batch,"support"),original)

    def test_global_probability_ties(self):
        labels={"local_y":torch.tensor([0,1]),"mapping":torch.tensor([[1,0],[1,0]]),"use_global":True}
        result=score(torch.zeros(2,2),labels)
        self.assertEqual(result["accuracy"],.5)
        self.assertEqual(result["f1"],0)
        self.assertAlmostEqual(result["macro_f1"],1/3)


if __name__ == "__main__":
    unittest.main()
