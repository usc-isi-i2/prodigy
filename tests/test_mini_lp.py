import unittest
import torch
from mixture_scaling.mini_lp import ExactNonedges,edge_partition,fixed_neighbors
class MiniLPTests(unittest.TestCase):
 def test_splits_disjoint_and_deduplicated(self):
  e=torch.tensor([[0,1,0,1,2,3,4,4],[1,0,1,2,3,4,5,4]])
  keys,groups=edge_partition(e,6,0)
  self.assertEqual(len(keys),5)
  joined=torch.cat(groups,1);encoded=joined[0]*6+joined[1]
  self.assertTrue(torch.equal(encoded.sort().values,keys))
 def test_exact_negatives_exclude_all_edge_splits_and_reverse(self):
  keys=torch.tensor([1,2,8,15])
  sampler=ExactNonedges(6,keys)
  pairs=sampler.sample(2000,torch.Generator().manual_seed(0))
  self.assertFalse(bool(sampler.forbidden(pairs).any()))
  for u,v in pairs.T.tolist(): self.assertNotIn(min(u,v)*6+max(u,v),keys.tolist());self.assertNotEqual(u,v)
 def test_fixed_neighbors_no_holdout_or_duplicates(self):
  edges=torch.stack((torch.zeros(20,dtype=torch.long),torch.arange(1,21)))
  a=fixed_neighbors(edges,22,10,0);b=fixed_neighbors(edges,22,10,0)
  self.assertTrue(torch.equal(a,b));self.assertEqual(len(a[0].unique()),10)
  self.assertTrue((a[21]==-1).all())
  for u,row in enumerate(a.tolist()):
   for v in row:
    if v>=0:self.assertTrue((u==0 and 1<=v<=20) or (v==0 and 1<=u<=20))
if __name__=='__main__':unittest.main()
