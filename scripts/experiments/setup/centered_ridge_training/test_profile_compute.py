import copy
import unittest
import torch
from models.test_encoder_solver_objective import fixture
from .profile_compute import backward


class ProfileTest(unittest.TestCase):
    def test_early_exit_exact_loss_gradient_and_hook_cleanup(self):
        torch.manual_seed(4);base,batch=fixture('ridge_centered_scaled')
        records={}
        for mode in ('compatibility','early_exit'):
            model=copy.deepcopy(base)
            loss=backward(model,copy.deepcopy(batch),mode)
            records[mode]=(loss,{k:None if v.grad is None else v.grad.clone() for k,v in model.named_parameters()})
            self.assertEqual(len(model.layer_list[2]._forward_pre_hooks),0)
            self.assertFalse(model.encoder_solver_training)
        torch.testing.assert_close(records['compatibility'][0],records['early_exit'][0],atol=0,rtol=0)
        for key,a in records['compatibility'][1].items():
            b=records['early_exit'][1][key]
            self.assertEqual(a is None,b is None)
            if a is not None:torch.testing.assert_close(a,b,atol=0,rtol=0)


if __name__=='__main__':unittest.main()
