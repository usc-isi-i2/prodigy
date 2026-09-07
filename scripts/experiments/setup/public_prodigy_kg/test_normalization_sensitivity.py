import unittest
import torch
from .run_normalization_sensitivity import normalization_artifact,STEPS,EPISODES
from .run_trajectory import forward_checkpoint
from .test_trajectory import Tiny
from .run_native import capture_forward_state


class NormalizationTest(unittest.TestCase):
    def test_only_bn_override_own_buffers_restored(self):
        model=Tiny().train()
        model.bn.running_mean.fill_(3)
        state=capture_forward_state(model,torch)
        state["buffers"]["bn.running_mean"].fill_(999)
        original_modes=dict(state["training"])
        artifact=dict(state=state,input=[torch.randn(4,3)])
        modified,inventory=normalization_artifact(model,artifact,True)
        self.assertEqual(state["training"],original_modes)
        for name,flag in modified["state"]["training"].items():
            self.assertEqual(flag,False if name=="bn" else original_modes[name])
        before={k:v.clone() for k,v in model.state_dict().items()}
        _,a,_=forward_checkpoint(model,modified,"cpu")
        _,b,_=forward_checkpoint(model,modified,"cpu")
        torch.testing.assert_close(a,b,atol=0,rtol=0)
        self.assertLess(float(a.abs().max()),20)
        for key,value in model.state_dict().items():torch.testing.assert_close(value,before[key],atol=0,rtol=0)
        self.assertTrue(model.bn.training)
        self.assertTrue(inventory["bn"]["track_running_stats"])
        self.assertEqual((STEPS,EPISODES),((2000,8000),128))


if __name__=="__main__":unittest.main()
