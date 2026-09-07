import unittest
from types import SimpleNamespace
import torch
from .pretraining_control import extract_pre, endpoint_features
from .run_native import capture_forward_state


class ControlTests(unittest.TestCase):
    def test_initialization_buffers_and_early_stop(self):
        class Meta(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.gnn_layers = torch.nn.ModuleList([])
            def forward(self, **kwargs):
                raise AssertionError("Solver must not execute")
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bn = torch.nn.BatchNorm1d(2)
                self.layer_list = torch.nn.ModuleList([Meta()])
            def forward(self, x):
                return self.layer_list[0](x=self.bn(x), start_right=len(x))
        model = Model().eval()
        model.bn.running_mean.fill_(7)
        x = torch.ones(4,2)
        state = capture_forward_state(model, torch)
        state["buffers"]["bn.running_mean"].fill_(40)
        expected = model.bn(x).detach()
        before = capture_forward_state(model, torch)
        actual = extract_pre(model, {"input": (x,), "state": state}, "cpu")
        torch.testing.assert_close(actual, expected)
        self.assertFalse(model.layer_list[0]._forward_pre_hooks)
        for name, value in model.named_buffers():
            self.assertTrue(torch.equal(value, before["buffers"][name]))
        self.assertTrue(torch.equal(torch.get_rng_state(), before["torch_rng"]))

    def test_endpoint_pair_order(self):
        x = torch.arange(6*770,dtype=torch.float32).reshape(6,770)
        graph = SimpleNamespace(x=x, ptr=torch.tensor([0,3,6]),
            edge_index_supernode=torch.tensor([[0,1,3,4],[2,2,5,5]]))
        result = endpoint_features(graph)
        torch.testing.assert_close(result[1], torch.cat([x[3,:768],x[4,:768]]))
        graph.edge_index_supernode[0,0] = 1
        with self.assertRaises(ValueError):
            endpoint_features(graph)


if __name__ == "__main__":
    unittest.main()
