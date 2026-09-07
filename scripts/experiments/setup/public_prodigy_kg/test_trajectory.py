import unittest
from pathlib import Path
import torch
from .run_trajectory import forward_checkpoint, nominated, STEPS, EPISODES
from .run_native import capture_forward_state


class Meta(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gnn_layers = torch.nn.ModuleList([])
    def forward(self, *, x, start_right):
        return x


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm1d(3)
        self.register_buffer("offset", torch.tensor(2.0))
        self.layer_list = torch.nn.ModuleList([Meta()])
    def forward(self, x):
        x.add_(torch.rand_like(x))
        x = self.bn(x) + self.offset
        z = self.layer_list[0](x=x, start_right=len(x))
        return torch.eye(len(x)), z


class TrajectoryTest(unittest.TestCase):
    def test_own_buffers_shared_rng_inputs_state_preserved(self):
        model = Tiny().train()
        state = capture_forward_state(model, torch)
        state["buffers"]["offset"].fill_(999)
        artifact = dict(state=state, input=[torch.randn(4,3)])
        before = {k:v.clone() for k,v in model.state_dict().items()}
        x = artifact["input"][0].clone()
        rng = torch.get_rng_state().clone()
        _, a, pre = forward_checkpoint(model, artifact, "cpu")
        _, b, _ = forward_checkpoint(model, artifact, "cpu")
        torch.testing.assert_close(a,b,atol=0,rtol=0)
        torch.testing.assert_close(a,pre,atol=0,rtol=0)
        self.assertLess(abs(float(a.mean())-2),1e-5)
        for key,value in model.state_dict().items():
            torch.testing.assert_close(value,before[key],atol=0,rtol=0)
        torch.testing.assert_close(artifact["input"][0],x,atol=0,rtol=0)
        self.assertTrue(torch.equal(torch.get_rng_state(),rng))
        model.offset.fill_(5)
        _, c, _ = forward_checkpoint(model, artifact, "cpu")
        torch.testing.assert_close(c-a,torch.full_like(a,3),atol=1e-6,rtol=0)

    def test_nominations(self):
        self.assertEqual(STEPS,(2000,4000,8000))
        self.assertEqual(EPISODES,128)
        self.assertEqual(nominated(Path("/ckpt"))[4000],Path("/ckpt/state_dict_4000.ckpt"))


if __name__ == "__main__":
    unittest.main()
