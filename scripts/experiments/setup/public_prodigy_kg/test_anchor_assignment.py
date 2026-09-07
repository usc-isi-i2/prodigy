import unittest
import torch
from .run_anchor_assignment import assignment,clamp_pre_data,EPISODES,WAYS


class Meta(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gnn_layers=torch.nn.ModuleList([])
    def forward(self, *,x,start_right):
        return x


class AssignmentTest(unittest.TestCase):
    def test_exact_clamp_preserves_labels_removes_hook_and_rejects_drift(self):
        model=torch.nn.Module()
        meta=Meta()
        model.layer_list=torch.nn.ModuleList([meta])
        reference=torch.arange(6,dtype=torch.float32).reshape(2,3)
        x=torch.cat([reference+1e-5,torch.ones(2,3)*9])
        before=x.clone()
        with clamp_pre_data(model,reference) as audit:
            result=meta(x=x,start_right=2)
            self.assertTrue(torch.equal(result[:2],reference))
            self.assertTrue(torch.equal(result[2:],x[2:]))
        self.assertGreater(audit["natural_max_abs_drift"],0)
        self.assertTrue(torch.equal(x,before))
        self.assertEqual(len(meta._forward_pre_hooks),0)
        with self.assertRaises(ValueError):
            with clamp_pre_data(model,reference):
                meta(x=x+1,start_right=2)
        self.assertEqual(len(meta._forward_pre_hooks),0)

    def test_confined_and_restored_even_on_error(self):
        table=torch.nn.Parameter(torch.arange(90,dtype=torch.float32).reshape(30,3))
        original=table.detach().clone()
        with assignment(table,False):
            self.assertTrue(torch.equal(table,original))
        with self.assertRaises(RuntimeError):
            with assignment(table,True):
                self.assertTrue(torch.equal(table[:20],original[:20].flip(0)))
                self.assertTrue(torch.equal(table[20:],original[20:]))
                raise RuntimeError("test")
        self.assertTrue(torch.equal(table,original))
        self.assertTrue(table.requires_grad)
        self.assertEqual((EPISODES,WAYS),(32,20))


if __name__=="__main__":
    unittest.main()
