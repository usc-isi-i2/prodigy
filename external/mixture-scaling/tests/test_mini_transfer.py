import unittest
import torch
from mixture_scaling.mini_transfer import partitions, evaluate, losses
from mixture_scaling.model import MaskedFeatureMLP
class MiniTransferTests(unittest.TestCase):
    def test_disjoint_reproducible_complete_split(self):
        parts=partitions(1000)
        self.assertEqual([len(x) for x in parts],[700,150,150])
        self.assertTrue(torch.equal(torch.cat(parts).sort().values,torch.arange(1000)))
        for a,b in zip(parts,partitions(1000)): self.assertTrue(torch.equal(a,b))
    def test_fixed_validation_does_not_modify_weights(self):
        torch.manual_seed(0); model=MaskedFeatureMLP(8,4,4); x=torch.randn(31,8)
        before={k:v.clone() for k,v in model.state_dict().items()}
        a=evaluate(model,x,torch.arange(31),13,replicates=2,batch_size=7)
        b=evaluate(model,x,torch.arange(31),13,replicates=2,batch_size=7)
        self.assertEqual(a,b)
        for k,v in model.state_dict().items(): self.assertTrue(torch.equal(v,before[k]))
    def test_masked_objective_backward_finite(self):
        model=MaskedFeatureMLP(8,4,4);x=torch.randn(31,8)
        result=losses(model,x,torch.Generator().manual_seed(0))
        result[0].mean().backward()
        self.assertTrue(all(torch.isfinite(v).all() for v in result))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
if __name__=='__main__': unittest.main()
