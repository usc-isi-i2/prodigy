import unittest,copy
import torch
from mixture_scaling.gradient_diagnostic import geometry,snapshot_model,groups
from mixture_scaling.bias_lp_matrix import BiasMLP
class Tests(unittest.TestCase):
 def test_geometry(self):
  x=[torch.tensor([1.,2.]),torch.tensor([3.])];g=geometry(x,[-v for v in x]);self.assertAlmostEqual(g['cosine'],-1);self.assertEqual(g['dot'],-14)
  self.assertIsNone(geometry(x,[torch.zeros_like(v) for v in x])['cosine'])
 def test_disposable_optimizer_has_no_alias_to_checkpoint(self):
  model=BiasMLP('node_neighbors');opt=torch.optim.AdamW(model.parameters(),lr=.0005)
  for p in model.parameters():p.grad=torch.ones_like(p)
  opt.step();ck=copy.deepcopy(dict(model=model.state_dict(),optimizer=opt.state_dict()))
  original=copy.deepcopy(ck);m,o=snapshot_model(ck,torch.device('cpu'))
  for p in m.parameters():p.grad=torch.ones_like(p)
  o.step()
  for k,v in ck['model'].items():self.assertTrue(torch.equal(v,original['model'][k]))
  for k,v in ck['optimizer']['state'].items():
   for field,t in v.items():self.assertTrue(torch.equal(t,original['optimizer']['state'][k][field]))
  g=groups(m);self.assertEqual(sum(len(v) for k,v in g.items() if k!='all'),len(g['all']))
if __name__=='__main__':unittest.main()
