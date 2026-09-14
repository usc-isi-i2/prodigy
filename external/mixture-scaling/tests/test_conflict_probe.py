import unittest
import torch
from mixture_scaling.conflict_probe import corrected,norm
class ConflictTests(unittest.TestCase):
 def test_aligned_unchanged(self):
  a=[torch.tensor([1.,2.])];b=[torch.tensor([3.,1.])]
  self.assertTrue(torch.equal(corrected(a,b)[0],(a[0]+b[0])/2))
 def test_projection_uses_original_vectors(self):
  a=[torch.tensor([1.,0.])];b=[torch.tensor([-1.,1.])]
  c=corrected(a,b)[0]
  self.assertTrue(torch.allclose(c,torch.tensor([.25,.75])))
  self.assertGreaterEqual(float(c@a[0]),0)
  self.assertGreaterEqual(float(c@b[0]),0)
 def test_zero_and_norm(self):
  z=[torch.zeros(2)];a=[torch.tensor([3.,4.])]
  self.assertTrue(torch.equal(corrected(z,a)[0],a[0]/2))
  self.assertEqual(norm(a),5.)
