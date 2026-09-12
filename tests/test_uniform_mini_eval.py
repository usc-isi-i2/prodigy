import unittest
import numpy as np
import torch
from mixture_scaling.uniform_mini_eval import uniform_pairs
class UniformEvalTests(unittest.TestCase):
 def test_exact_negatives_ratio_and_fixed_positives(self):
  u=np.array([0,1,2,3]);v=np.array([1,2,3,4]);val=np.array([True,False,False,False])
  keys=torch.tensor([1,8,15,22])
  a=uniform_pairs(u,v,val,6,keys,0)
  b=uniform_pairs(u,v,val,6,keys,0)
  for x,y in zip(a,b):self.assertTrue(np.array_equal(x,y))
  ru,rv,y,mask=a
  self.assertTrue(np.array_equal(ru[:4],u));self.assertTrue(np.array_equal(rv[:4],v));self.assertTrue(np.array_equal(mask[:4],val))
  self.assertEqual(int(((y==0)&~mask).sum()),15)
  self.assertEqual(int(((y==0)&mask).sum()),5)
  for x,z in zip(ru[4:],rv[4:]):self.assertNotEqual(x,z);self.assertNotIn(min(x,z)*6+max(x,z),keys.tolist())
if __name__=='__main__':unittest.main()
