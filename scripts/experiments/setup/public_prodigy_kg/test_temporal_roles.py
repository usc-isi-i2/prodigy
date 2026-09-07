"""Light CPU synthetic tests; no graph dataset or model experiment."""
import unittest
from types import SimpleNamespace
import torch

from .run_temporal_roles import assemble_rows, validate_roles, clamp_roles, tensor_hash, CONDITIONS


class TemporalRolesTest(unittest.TestCase):
    def setUp(self):
        self.early = torch.arange(18, dtype=torch.float32).reshape(6,3)
        self.late = self.early + 100
        self.mask = torch.tensor([True,False,False,True,False,True])

    def test_shuffled_roles_all_conditions_and_no_mutation(self):
        donors = {2000:self.early,8000:self.late}
        for s,q in CONDITIONS:
            mixed = assemble_rows(donors[s],donors[q],self.mask)
            self.assertTrue(torch.equal(mixed[~self.mask],donors[s][~self.mask]))
            self.assertTrue(torch.equal(mixed[self.mask],donors[q][self.mask]))
            if s == q:
                self.assertTrue(torch.equal(mixed,donors[s]))
            mixed.fill_(0)
        self.assertEqual(float(self.late[0,0]),100)
        self.assertEqual(float(self.early[-1,-1]),17)

    def test_invalid_donors_masks(self):
        cases = [(self.early,self.late[:5],self.mask),
                 (self.early,self.late.double(),self.mask),
                 (self.early,self.late,self.mask.long()),
                 (self.early,self.late,self.mask[:5]),
                 (self.early,self.late,torch.ones(6,dtype=torch.bool)),
                 (self.early,self.late,torch.zeros(6,dtype=torch.bool)),
                 (self.early,self.late*float("nan"),self.mask)]
        for args in cases:
            with self.assertRaises(ValueError):
                assemble_rows(*args)

    def test_query_mask_and_task_order_agree(self):
        args = (SimpleNamespace(ptr=torch.arange(7)),None,torch.eye(2)[torch.tensor([0,0,1,1,0,1])],None,None,self.mask[:,None].expand(6,2).reshape(-1))
        task = dict(support_rows=torch.tensor([1,2,4]),query_rows=torch.tensor([0,3,5]))
        self.assertTrue(torch.equal(validate_roles(args,task),self.mask))
        with self.assertRaises(ValueError):
            validate_roles(args,dict(task,query_rows=torch.tensor([3,0,5])))
        with self.assertRaises(ValueError):
            validate_roles(args,dict(task,support_rows=torch.tensor([1,2,2])))
        bad = list(args)
        bad[5] = bad[5].clone()
        bad[5][1] = False
        with self.assertRaises(ValueError):
            validate_roles(tuple(bad),task)

    def test_hook_checks_natural_but_accepts_mixed_and_preserves_labels(self):
        class Meta(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.gnn_layers = torch.nn.ModuleList()
            def forward(self, *, x, start_right):
                return x
        meta = Meta()
        model = SimpleNamespace(layer_list=[meta])
        labels = torch.tensor([[901.,902.,903.],[904.,905.,906.]])
        x = torch.cat((self.late,labels))
        donor = assemble_rows(self.early,self.late,self.mask)
        with clamp_roles(model,self.late,donor) as audit:
            result = meta(x=x,start_right=6)
        self.assertTrue(torch.equal(result[:6],donor))
        self.assertTrue(torch.equal(result[6:],labels))
        self.assertTrue(torch.equal(x[:6],self.late))
        self.assertEqual(audit["calls"],1)
        self.assertEqual(audit["replacement_sha256"],tensor_hash(donor))
        self.assertFalse(meta._forward_pre_hooks)
        with self.assertRaises(ValueError):
            with clamp_roles(model,self.early,donor):
                meta(x=x,start_right=6)
        self.assertFalse(meta._forward_pre_hooks)
        with self.assertRaises(ValueError):
            with clamp_roles(model,self.late,donor):
                pass
        self.assertFalse(meta._forward_pre_hooks)
        with self.assertRaises(ValueError):
            with clamp_roles(model,self.late,donor):
                meta(x=x,start_right=5)
        self.assertFalse(meta._forward_pre_hooks)
        with self.assertRaises(ValueError):
            with clamp_roles(model,self.late,donor):
                meta(x=x,start_right=6)
                meta(x=x,start_right=6)
        self.assertFalse(meta._forward_pre_hooks)
        with self.assertRaises(ValueError):
            with clamp_roles(model,self.late.double(),donor.double()):
                meta(x=x,start_right=6)
        self.assertFalse(meta._forward_pre_hooks)
        with self.assertRaises(ValueError):
            with clamp_roles(model,self.late,donor):
                meta(x=x*float("nan"),start_right=6)
        self.assertFalse(meta._forward_pre_hooks)


if __name__ == "__main__":
    unittest.main()
